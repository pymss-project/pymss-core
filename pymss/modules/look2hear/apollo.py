import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _cached_inference_tensor(module, name, tensor, input, version):
    if tensor is None:
        return None
    if tensor.device == input.device and tensor.dtype == input.dtype:
        return tensor

    key = (name, input.device, input.dtype, version)
    cache = getattr(module, "_apollo_inference_cache", None)
    if cache is None:
        cache = {}
        module._apollo_inference_cache = cache
    cached = cache.get(name)
    if cached is not None and cached[0] == key:
        return cached[1]

    casted = tensor.detach().to(device=input.device, dtype=input.dtype)
    cache[name] = (key, casted)
    return casted


def pointwise_conv1d(input, conv):
    if conv.kernel_size != (1,) or conv.stride != (1,) or conv.padding != (0,) or conv.dilation != (1,) or conv.groups != 1:
        return conv(input)
    weight = conv.weight[:, :, 0]
    bias = conv.bias
    if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16) and not torch.is_grad_enabled():
        weight = _cached_inference_tensor(conv, "pointwise_weight", weight, input, conv.weight._version)
        bias = _cached_inference_tensor(conv, "pointwise_bias", bias, input, bias._version) if bias is not None else None
    output = F.linear(input.transpose(1, 2), weight, bias)
    return output.transpose(1, 2)


class RMSNorm(nn.Module):
    def __init__(self, dimension, groups=1):
        super().__init__()

        self.weight = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, T)
        B, N, T = input.shape
        assert N % self.groups == 0
        if self.groups == 1 and not torch.is_grad_enabled():
            if input.is_cuda and input.dtype in (torch.float16, torch.bfloat16):
                weight = _cached_inference_tensor(self, "rms_weight", self.weight, input, self.weight._version)
                return F.rms_norm(input.transpose(1, 2), (N,), weight, self.eps).transpose(1, 2)
            input_norm = F.rms_norm(input.transpose(1, 2), (N,), None, self.eps).transpose(1, 2)
            return input_norm.type_as(input) * self.weight.reshape(1, -1, 1)

        input_float = input.reshape(B, self.groups, -1, T).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(-2, keepdim=True) + self.eps)

        return input_norm.type_as(input).reshape(B, N, T) * self.weight.reshape(1, -1, 1)


class RMVN(nn.Module):
    """
    Rescaled MVN.
    """

    def __init__(self, dimension, groups=1):
        super(RMVN, self).__init__()

        self.mean = nn.Parameter(torch.zeros(dimension))
        self.std = nn.Parameter(torch.ones(dimension))
        self.groups = groups
        self.eps = 1e-5

    def forward(self, input):
        # input size: (B, N, *)
        B, N = input.shape[:2]
        assert N % self.groups == 0
        input_reshape = input.reshape(B, self.groups, N // self.groups, -1)
        T = input_reshape.shape[-1]

        input_norm = (input_reshape - input_reshape.mean(2).unsqueeze(2)) / (
                    input_reshape.var(2).unsqueeze(2) + self.eps).sqrt()
        input_norm = input_norm.reshape(B, N, T) * self.std.reshape(1, -1, 1) + self.mean.reshape(1, -1, 1)

        return input_norm.reshape(input.shape)


class Roformer(nn.Module):
    """
    Transformer with rotary positional embedding.
    """

    def __init__(self, input_size, hidden_size, num_head=8, theta=10000, window=10000,
                 input_drop=0., attention_drop=0., causal=True):
        super().__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size // num_head
        self.num_head = num_head
        self.theta = theta  # base frequency for RoPE
        self.window = window
        # pre-calculate rotary embeddings
        cos_freq, sin_freq = self._calc_rotary_emb()
        self.register_buffer("cos_freq", cos_freq)  # win, N
        self.register_buffer("sin_freq", sin_freq)  # win, N
        self.register_buffer("reverse_sign", torch.tensor([-1, 1]), persistent=False)
        self._rotary_freq_cache = {}

        self.attention_drop = attention_drop
        self.causal = causal
        self.eps = 1e-5

        self.input_norm = RMSNorm(self.input_size)
        self.input_drop = nn.Dropout(p=input_drop)
        self.weight = nn.Conv1d(self.input_size, self.hidden_size * self.num_head * 3, 1, bias=False)
        self.output = nn.Conv1d(self.hidden_size * self.num_head, self.input_size, 1, bias=False)

        self.MLP = nn.Sequential(RMSNorm(self.input_size),
                                 nn.Conv1d(self.input_size, self.input_size * 8, 1, bias=False),
                                 nn.SiLU()
                                 )
        self.MLP_output = nn.Conv1d(self.input_size * 4, self.input_size, 1, bias=False)

    def _calc_rotary_emb(self):
        freq = 1. / (self.theta ** (
                    torch.arange(0, self.hidden_size, 2)[:(self.hidden_size // 2)] / self.hidden_size))  # theta_i
        freq = freq.reshape(1, -1)  # 1, N//2
        pos = torch.arange(0, self.window).reshape(-1, 1)  # win, 1
        cos_freq = torch.cos(pos * freq)  # win, N//2
        sin_freq = torch.sin(pos * freq)  # win, N//2
        cos_freq = torch.stack([cos_freq] * 2, -1).reshape(self.window, self.hidden_size)  # win, N
        sin_freq = torch.stack([sin_freq] * 2, -1).reshape(self.window, self.hidden_size)  # win, N

        return cos_freq, sin_freq

    def _add_rotary_emb(self, feature, pos):
        # feature shape: ..., N
        N = feature.shape[-1]

        feature_reshape = feature.reshape(-1, N)
        pos = min(pos, self.window - 1)
        cos_freq = self.cos_freq[pos]
        sin_freq = self.sin_freq[pos]
        reverse_sign = self.reverse_sign.to(device=feature.device, dtype=feature.dtype)
        feature_reshape_neg = (
                    torch.flip(feature_reshape.reshape(-1, N // 2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(
            -1, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def _add_rotary_sequence(self, feature):
        # feature shape: ..., T, N
        T, N = feature.shape[-2:]
        feature_reshape = feature.reshape(-1, T, N)

        if feature.is_cuda and feature.dtype in (torch.float16, torch.bfloat16) and not torch.is_grad_enabled():
            key = (T, feature.device, feature.dtype)
            cached = self._rotary_freq_cache.get(key)
            if cached is None:
                cached = (
                    self.cos_freq[:T].to(device=feature.device, dtype=feature.dtype),
                    self.sin_freq[:T].to(device=feature.device, dtype=feature.dtype),
                )
                self._rotary_freq_cache[key] = cached
            cos_freq, sin_freq = cached
            output = torch.empty_like(feature_reshape)
            feature_even = feature_reshape[..., 0::2]
            feature_odd = feature_reshape[..., 1::2]
            cos_freq = cos_freq[..., 0::2].unsqueeze(0)
            sin_freq = sin_freq[..., 0::2].unsqueeze(0)
            output[..., 0::2] = feature_even * cos_freq - feature_odd * sin_freq
            output[..., 1::2] = feature_odd * cos_freq + feature_even * sin_freq
            return output.reshape(feature.shape)

        cos_freq = self.cos_freq[:T]
        sin_freq = self.sin_freq[:T]
        reverse_sign = self.reverse_sign.to(device=feature.device, dtype=feature.dtype)
        feature_reshape_neg = (
                    torch.flip(feature_reshape.reshape(-1, N // 2, 2), [-1]) * reverse_sign.reshape(1, 1, 2)).reshape(
            -1, T, N)
        feature_rope = feature_reshape * cos_freq.unsqueeze(0) + feature_reshape_neg * sin_freq.unsqueeze(0)

        return feature_rope.reshape(feature.shape)

    def forward(self, input):
        # input shape: B, N, T

        B, _, T = input.shape

        weight = pointwise_conv1d(self.input_drop(self.input_norm(input)), self.weight).reshape(
            B, self.num_head, self.hidden_size * 3, T
        ).mT
        Q, K, V = torch.split(weight, self.hidden_size, dim=-1)  # B, num_head, T, N

        # rotary positional embedding
        Q_rot = self._add_rotary_sequence(Q)
        K_rot = self._add_rotary_sequence(K)

        V_attention = V if not torch.is_grad_enabled() else V.contiguous()
        attention_output = F.scaled_dot_product_attention(Q_rot.contiguous(), K_rot.contiguous(), V_attention,
                                                          dropout_p=self.attention_drop,
                                                          is_causal=self.causal)  # B, num_head, T, N
        attention_output = attention_output.mT.reshape(B, -1, T)
        output = pointwise_conv1d(attention_output, self.output) + input

        hidden = self.MLP[0](output)
        hidden = pointwise_conv1d(hidden, self.MLP[1])
        hidden = self.MLP[2](hidden)
        gate, z = hidden.chunk(2, dim=1)
        output = output + pointwise_conv1d(F.silu(gate) * z, self.MLP_output)

        return output, (K_rot, V)


class ConvActNorm1d(nn.Module):
    def __init__(self, in_channel, hidden_channel, kernel=7, causal=False):
        super(ConvActNorm1d, self).__init__()

        self.in_channel = in_channel
        self.kernel = kernel
        self.causal = causal
        if not causal:
            self.conv = nn.Sequential(
                nn.Conv1d(in_channel, in_channel, kernel, padding=(kernel - 1) // 2, groups=in_channel),
                RMSNorm(in_channel),
                nn.Conv1d(in_channel, hidden_channel, 1),
                nn.SiLU(),
                nn.Conv1d(hidden_channel, in_channel, 1)
                )
        else:
            self.conv = nn.Sequential(nn.Conv1d(in_channel, in_channel, kernel, padding=kernel - 1, groups=in_channel),
                                      RMSNorm(in_channel),
                                      nn.Conv1d(in_channel, hidden_channel, 1),
                                      nn.SiLU(),
                                      nn.Conv1d(hidden_channel, in_channel, 1)
                                      )

    def forward(self, input):

        output = self.conv[0](input)
        output = self.conv[1](output)
        output = pointwise_conv1d(output, self.conv[2])
        output = self.conv[3](output)
        output = pointwise_conv1d(output, self.conv[4])
        if self.causal:
            output = output[..., :-self.kernel + 1]
        return input + output


class ICB(nn.Module):
    def __init__(self, in_channel, kernel=7, causal=False):
        super(ICB, self).__init__()

        self.blocks = nn.Sequential(ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal),
                                    ConvActNorm1d(in_channel, in_channel * 4, kernel, causal=causal)
                                    )

    def forward(self, input):
        return self.blocks(input)


class BSNet(nn.Module):
    def __init__(self, feature_dim, kernel=7):
        super(BSNet, self).__init__()

        self.feature_dim = feature_dim

        self.band_net = Roformer(self.feature_dim, self.feature_dim, num_head=8, window=100, causal=False)
        self.seq_net = ICB(self.feature_dim, kernel=kernel)

    def forward(self, input):
        # input shape: B, nband, N, T

        B, nband, N, T = input.shape

        # band comm
        band_input = input.permute(0, 3, 2, 1).reshape(B * T, -1, nband)
        band_output, _ = self.band_net(band_input)
        band_output = band_output.reshape(B, T, -1, nband).permute(0, 3, 2, 1)

        # sequence modeling
        output = self.seq_net(band_output.reshape(B * nband, -1, T)).reshape(B, nband, -1, T)  # B, nband, N, T

        return output


class Apollo(nn.Module):
    def __init__(
            self,
            sr: int,
            win: int,
            feature_dim: int,
            layer: int
    ):
        super().__init__()

        self.sr = sr
        self.win = int(sr * win // 1000)
        self.stride = self.win // 2
        self.enc_dim = self.win // 2 + 1
        self.feature_dim = feature_dim
        self.eps = torch.finfo(torch.float32).eps
        self.register_buffer("window", torch.hann_window(self.win), persistent=False)
        self._packed_cache = {}

        # 80 bands
        bandwidth = int(self.win / 160)
        self.band_width = [bandwidth] * 79
        self.band_width.append(self.enc_dim - np.sum(self.band_width))
        self.nband = len(self.band_width)
        # print(self.band_width, self.nband)

        self.BN = nn.ModuleList([])
        for i in range(self.nband):
            self.BN.append(nn.Sequential(RMSNorm(self.band_width[i] * 2 + 1),
                                         nn.Conv1d(self.band_width[i] * 2 + 1, self.feature_dim, 1))
                           )

        self.net = []
        for _ in range(layer):
            self.net.append(BSNet(self.feature_dim))
        self.net = nn.Sequential(*self.net)

        self.output = nn.ModuleList([])
        for i in range(self.nband):
            self.output.append(nn.Sequential(RMSNorm(self.feature_dim),
                                             nn.Conv1d(self.feature_dim, self.band_width[i] * 4, 1),
                                             nn.GLU(dim=1)
                                             )
                               )

    def _window(self, input):
        return self.window.to(device=input.device)

    def _uniform_band_prefix(self):
        width = self.band_width[0]
        count = 0
        for band_width in self.band_width:
            if band_width != width:
                break
            count += 1
        return count, width

    def _use_packed_band_ops(self):
        if self.training or torch.is_grad_enabled():
            return False
        count, _ = self._uniform_band_prefix()
        return count > 1

    def _cached_packed_bn(self, count):
        conv = self.BN[0][1]
        key = ("bn", count, conv.weight.device, conv.weight.dtype)
        cached = self._packed_cache.get("bn")
        if cached is not None and cached["key"] == key:
            return cached["norm_weight"], cached["conv_weight"], cached["conv_bias"], cached["groups"], cached["eps"]

        modules = list(self.BN[:count])
        norm_weight = torch.stack([module[0].weight.detach() for module in modules], dim=0)
        conv_weight = torch.cat([module[1].weight.detach() for module in modules], dim=0)
        conv_bias = torch.cat([module[1].bias.detach() for module in modules], dim=0) if modules[0][1].bias is not None else None
        cached = {
            "key": key,
            "norm_weight": norm_weight,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "groups": modules[0][0].groups,
            "eps": modules[0][0].eps,
        }
        self._packed_cache["bn"] = cached
        return norm_weight, conv_weight, conv_bias, cached["groups"], cached["eps"]

    def _cached_packed_output(self, count):
        conv = self.output[0][1]
        key = ("output", count, conv.weight.device, conv.weight.dtype)
        cached = self._packed_cache.get("output")
        if cached is not None and cached["key"] == key:
            return cached["norm_weight"], cached["conv_weight"], cached["conv_bias"], cached["groups"], cached["eps"]

        modules = list(self.output[:count])
        norm_weight = torch.stack([module[0].weight.detach() for module in modules], dim=0)
        conv_weight = torch.cat([module[1].weight.detach() for module in modules], dim=0)
        conv_bias = torch.cat([module[1].bias.detach() for module in modules], dim=0) if modules[0][1].bias is not None else None
        cached = {
            "key": key,
            "norm_weight": norm_weight,
            "conv_weight": conv_weight,
            "conv_bias": conv_bias,
            "groups": modules[0][0].groups,
            "eps": modules[0][0].eps,
        }
        self._packed_cache["output"] = cached
        return norm_weight, conv_weight, conv_bias, cached["groups"], cached["eps"]

    @staticmethod
    def _packed_rms_norm(input, weight, groups, eps):
        batch, bands, channels, frames = input.shape
        input_float = input.reshape(batch, bands, groups, channels // groups, frames).float()
        input_norm = input_float * torch.rsqrt(input_float.pow(2).mean(3, keepdim=True) + eps)
        input_norm = input_norm.to(dtype=input.dtype).reshape(batch, bands, channels, frames)
        return input_norm * weight.reshape(1, bands, channels, 1)

    def _packed_bn_prefix(self, input, count):
        batch, bands, channels, frames = input.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_bn(count)
        input = self._packed_rms_norm(input, norm_weight, groups, eps)
        input = input.reshape(batch, bands * channels, frames)
        output = F.conv1d(input, conv_weight, conv_bias, groups=bands)
        return output.reshape(batch, bands, self.feature_dim, frames)

    def _packed_output_prefix(self, feature, count, width):
        batch, bands, channels, frames = feature.shape
        norm_weight, conv_weight, conv_bias, groups, eps = self._cached_packed_output(count)
        feature = self._packed_rms_norm(feature, norm_weight, groups, eps)
        feature = feature.reshape(batch, bands * channels, frames)
        output = F.conv1d(feature, conv_weight, conv_bias, groups=bands)
        output = output.reshape(batch, bands, width * 4, frames)
        left, right = output.chunk(2, dim=2)
        output = left * torch.sigmoid(right)
        return output.reshape(batch, bands, 2, width, frames)

    def spec_band_split(self, input):

        B, nch, nsample = input.shape

        spec = torch.stft(input.view(B * nch, nsample), n_fft=self.win, hop_length=self.stride,
                          window=self._window(input), return_complex=True)

        subband_spec = []
        subband_spec_norm = []
        subband_power = []
        band_idx = 0
        for i in range(self.nband):
            this_spec = spec[:, band_idx:band_idx + self.band_width[i]]
            subband_spec.append(this_spec)  # B, BW, T
            subband_power.append((this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1))  # B, 1, T
            subband_spec_norm.append(
                torch.complex(this_spec.real / subband_power[-1], this_spec.imag / subband_power[-1]))  # B, BW, T
            band_idx += self.band_width[i]
        subband_power = torch.cat(subband_power, 1)  # B, nband, T

        return subband_spec_norm, subband_power

    def _spec_band_split_packed(self, input):
        B, nch, nsample = input.shape
        spec = torch.stft(input.view(B * nch, nsample), n_fft=self.win, hop_length=self.stride,
                          window=self._window(input), return_complex=True)

        count, width = self._uniform_band_prefix()
        prefix_bins = count * width
        prefix_spec = spec[:, :prefix_bins].reshape(B * nch, count, width, -1)
        prefix_power = (prefix_spec.abs().pow(2).sum(2) + self.eps).sqrt()
        prefix_norm = torch.complex(
            prefix_spec.real / prefix_power.unsqueeze(2),
            prefix_spec.imag / prefix_power.unsqueeze(2),
        )

        tail_norm = []
        tail_power = []
        band_idx = prefix_bins
        for i in range(count, self.nband):
            this_spec = spec[:, band_idx:band_idx + self.band_width[i]]
            power = (this_spec.abs().pow(2).sum(1) + self.eps).sqrt().unsqueeze(1)
            tail_power.append(power)
            tail_norm.append(torch.complex(this_spec.real / power, this_spec.imag / power))
            band_idx += self.band_width[i]

        return prefix_norm, prefix_power, tail_norm, tail_power

    def feature_extractor(self, input):
        if self._use_packed_band_ops():
            return self._feature_extractor_packed(input)

        return self._feature_extractor_by_band(input)

    def _feature_extractor_by_band(self, input):

        subband_spec_norm, subband_power = self.spec_band_split(input)

        # normalization and bottleneck
        subband_feature = []
        for i in range(self.nband):
            concat_spec = torch.cat(
                [subband_spec_norm[i].real, subband_spec_norm[i].imag, torch.log(subband_power[:, i].unsqueeze(1))], 1)
            subband_feature.append(self.BN[i](concat_spec))
        subband_feature = torch.stack(subband_feature, 1)  # B, nband, N, T

        return subband_feature

    def _feature_extractor_packed(self, input):
        prefix_norm, prefix_power, tail_norm, tail_power = self._spec_band_split_packed(input)
        count, _ = self._uniform_band_prefix()

        prefix_input = torch.cat(
            [prefix_norm.real, prefix_norm.imag, torch.log(prefix_power).unsqueeze(2)],
            dim=2,
        )
        prefix_feature = self._packed_bn_prefix(prefix_input, count)

        if count == self.nband:
            return prefix_feature

        tail_feature = []
        for offset, subband_norm in enumerate(tail_norm):
            i = count + offset
            concat_spec = torch.cat(
                [subband_norm.real, subband_norm.imag, torch.log(tail_power[offset])],
                1,
            )
            tail_feature.append(self.BN[i](concat_spec))
        return torch.cat([prefix_feature, torch.stack(tail_feature, 1)], dim=1)

    def _estimate_spec_by_band(self, feature, batch_channels):
        est_spec = []
        for i in range(self.nband):
            this_RI = self.output[i](feature[:, i]).view(batch_channels, 2, self.band_width[i], -1)
            est_spec.append(torch.complex(this_RI[:, 0].float(), this_RI[:, 1].float()))
        return torch.cat(est_spec, 1)

    def _estimate_spec_packed(self, feature, batch_channels):
        count, width = self._uniform_band_prefix()
        prefix_RI = self._packed_output_prefix(feature[:, :count], count, width)
        prefix_spec = torch.complex(prefix_RI[:, :, 0].float(), prefix_RI[:, :, 1].float()).reshape(
            batch_channels, count * width, -1
        )

        if count == self.nband:
            return prefix_spec

        est_spec = [prefix_spec]
        for i in range(count, self.nband):
            this_RI = self.output[i](feature[:, i]).view(batch_channels, 2, self.band_width[i], -1)
            est_spec.append(torch.complex(this_RI[:, 0].float(), this_RI[:, 1].float()))
        return torch.cat(est_spec, 1)

    def forward(self, input):

        B, nch, nsample = input.shape

        subband_feature = self.feature_extractor(input)
        feature = self.net(subband_feature)

        if self._use_packed_band_ops():
            est_spec = self._estimate_spec_packed(feature, B * nch)
        else:
            est_spec = self._estimate_spec_by_band(feature, B * nch)
        est_spec = est_spec.to(dtype=torch.complex64)
        output = torch.istft(est_spec, n_fft=self.win, hop_length=self.stride,
                             window=self._window(input), length=nsample).view(B, nch, -1)

        return output
