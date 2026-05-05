from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend
try:
    from sageattention import sageattn
except ImportError:
    sageattn = None

from beartype.typing import Tuple, Optional, List, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from librosa import filters

from .common import forward_roformer_mask_core


# helper functions

def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def pad_at_dim(t, pad, dim=-1, value=0.):
    dims_from_right = (- dim - 1) if dim < 0 else (t.ndim - dim - 1)
    zeros = ((0, 0) * dims_from_right)
    return F.pad(t, (*zeros, *pad), value=value)


def l2norm(t):
    return F.normalize(t, dim=-1, p=2)


def rotate_half(x):
    out = torch.empty_like(x)
    out[..., ::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., ::2]
    return out


def apply_rotary_emb_fast(cos, sin, t):
    return (t * cos) + (rotate_half(t) * sin)


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype, layout):
    cache = getattr(rotary_embed, '_pymss_cos_sin_cache', None)
    if cache is None:
        cache = {}
        rotary_embed._pymss_cos_sin_cache = cache

    key = (seq_len, device.type, device.index, dtype, layout)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0),
        cache_key=f'freqs:{seq_len}|offset:0'
    )

    if layout == 'bnhd':
        freqs = rearrange(freqs, 'n d -> 1 n 1 d')
    elif layout == 'seq_before_head':
        freqs = rearrange(freqs, 'n d -> n 1 d')
    else:
        freqs = rearrange(freqs, 'n d -> 1 1 n d')

    freqs = freqs.to(device=device, dtype=dtype)
    cached = (freqs.cos(), freqs.sin())
    cache[key] = cached
    return cached


def rotate_qk_fast(rotary_embed, q, k):
    seq_dim = rotary_embed.default_seq_dim
    seq_len = q.shape[seq_dim]
    device, dtype = q.device, q.dtype
    layout = 'seq_before_head' if seq_dim == -3 else 'bhnd'
    cos, sin = cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype, layout)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    seq_len = q.shape[1]
    device, dtype = q.device, q.dtype
    cos, sin = cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype, 'bnhd')
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def qkv_to_bnhd(qkv, heads):
    b, n, _ = qkv.shape
    qkv = qkv.view(b, n, 3, heads, -1)
    return qkv.unbind(dim=2)


def qkv_to_bhnd(qkv, heads):
    b, n, _ = qkv.shape
    qkv = qkv.view(b, n, 3, heads, -1).permute(2, 0, 3, 1, 4)
    return qkv.unbind(dim=0)


def dim_input_offsets(dim_inputs):
    offsets = [0]
    for dim_input in dim_inputs:
        offsets.append(offsets[-1] + dim_input)
    return tuple(offsets)


def contiguous_dim_groups(dim_inputs):
    groups = []
    start = 0
    for i in range(1, len(dim_inputs) + 1):
        if i == len(dim_inputs) or dim_inputs[i] != dim_inputs[start]:
            groups.append((start, i, dim_inputs[start]))
            start = i
    return tuple(groups)


def grouped_linear(x, weight, bias):
    group_count, out_features, in_features = weight.shape
    leading_shape = x.shape[:-2]
    x = x.reshape(-1, group_count, in_features).transpose(0, 1)
    out = torch.bmm(x, weight.transpose(1, 2))
    out = out.transpose(0, 1).reshape(*leading_shape, group_count, out_features)
    if bias is not None:
        out = out + bias.to(dtype=out.dtype)
    return out


def set_rmsnorm_fp32(module, use_fp32):
    for child in module.modules():
        if isinstance(child, RMSNorm):
            child.use_fp32 = use_fp32


# norm

class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self.use_fp32 = True
        self._gamma_dtype_cache = {}

    def forward(self, x):
        if not self.training and not self.use_fp32 and x.dtype in (torch.float16, torch.bfloat16):
            key = (x.device.type, x.device.index, x.dtype, self.gamma.data_ptr(), self.gamma._version)
            gamma = self._gamma_dtype_cache.get(key)
            if gamma is None:
                gamma = self.gamma.detach().to(device=x.device, dtype=x.dtype)
                self._gamma_dtype_cache.clear()
                self._gamma_dtype_cache[key] = gamma
            return F.rms_norm(x, (x.shape[-1],), gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


# attention

class FeedForward(Module):
    def __init__(
            self,
            dim,
            mult=4,
            dropout=0.
    ):
        super().__init__()
        dim_inner = int(dim * mult)
        self.net = nn.Sequential(
            RMSNorm(dim),
            nn.Linear(dim, dim_inner),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim_inner, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(
            self,
            dim,
            heads=8,
            dim_head=64,
            dropout=0.,
            rotary_embed=None,
            flash=True,
            sage_attention=False,
            sage_mode=False,
            attention_layout='bhnd'
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head
        self.flash = flash
        self.dropout = dropout
        self.sage_mode = sage_mode
        self.attention_layout = attention_layout

        self.rotary_embed = rotary_embed

        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=False)

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=False),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = self.norm(x)

        if self.attention_layout == 'bnhd':
            q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)

            if exists(self.rotary_embed):
                q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)

            if self.sage_mode and sageattn is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
                out = sageattn(
                    q, k, v,
                    tensor_layout='NHD',
                    is_causal=False,
                    sm_scale=self.scale,
                    smooth_k=False
                )
            elif self.flash:
                out = F.scaled_dot_product_attention(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2),
                    dropout_p=self.dropout if self.training else 0.
                ).transpose(1, 2)
            else:
                out = self.attend(
                    q.transpose(1, 2),
                    k.transpose(1, 2),
                    v.transpose(1, 2)
                ).transpose(1, 2)

            gates = self.to_gates(x)
            out = out * gates.unsqueeze(-1).sigmoid()
            out = out.flatten(start_dim=-2)
            return self.to_out(out)

        q, k, v = qkv_to_bhnd(self.to_qkv(x), self.heads)

        if exists(self.rotary_embed):
            q, k = rotate_qk_fast(self.rotary_embed, q, k)

        if self.sage_mode and sageattn is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            out = sageattn(
                q, k, v,
                tensor_layout='HND',
                is_causal=False,
                sm_scale=self.scale,
                smooth_k=False
            )
        elif self.flash:
            out = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=self.dropout if self.training else 0.
            )
        else:
            out = self.attend(q, k, v)

        gates = self.to_gates(x)
        out = out * gates.transpose(1, 2).unsqueeze(-1).sigmoid()

        out = out.transpose(1, 2).flatten(start_dim=-2)
        return self.to_out(out)


class LinearAttention(Module):
    """
    this flavor of linear attention proposed in https://arxiv.org/abs/2106.09681 by El-Nouby et al.
    """

    @beartype
    def __init__(
            self,
            *,
            dim,
            dim_head=32,
            heads=8,
            scale=8,
            flash=False,
            dropout=0.
    ):
        super().__init__()
        dim_inner = dim_head * heads
        self.norm = RMSNorm(dim)

        self.to_qkv = nn.Sequential(
            nn.Linear(dim, dim_inner * 3, bias=False),
            Rearrange('b n (qkv h d) -> qkv b h d n', qkv=3, h=heads)
        )

        self.temperature = nn.Parameter(torch.ones(heads, 1, 1))

        self.attend = Attend(
            scale=scale,
            dropout=dropout,
            flash=flash
        )

        self.to_out = nn.Sequential(
            Rearrange('b h d n -> b n (h d)'),
            nn.Linear(dim_inner, dim, bias=False)
        )

    def forward(
            self,
            x
    ):
        x = self.norm(x)

        q, k, v = self.to_qkv(x)

        q, k = map(l2norm, (q, k))
        q = q * self.temperature.exp()

        out = self.attend(q, k, v)

        return self.to_out(out)


class Transformer(Module):
    def __init__(
            self,
            *,
            dim,
            depth,
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            ff_mult=4,
            norm_output=True,
            rotary_embed=None,
            flash_attn=True,
            linear_attn=False,
            sage_attention=False,
            sage_mode=False,
            attention_layout='bhnd'
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            if linear_attn:
                attn = LinearAttention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, flash=flash_attn)
            else:
                attn = Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout,
                                 rotary_embed=rotary_embed,
                                 flash=flash_attn,
                                 sage_attention=sage_attention,
                                 sage_mode=sage_mode,
                                 attention_layout=attention_layout)

            self.layers.append(ModuleList([
                attn,
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

        self.norm = RMSNorm(dim) if norm_output else nn.Identity()

    def forward(self, x):

        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)


# bandsplit module

class BandSplit(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...]
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_offsets = dim_input_offsets(dim_inputs)
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_features = ModuleList([])

        for dim_in in dim_inputs:
            net = nn.Sequential(
                RMSNorm(dim_in),
                nn.Linear(dim_in, dim)
            )

            self.to_features.append(net)

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        norms = [self.to_features[i][0] for i in range(start, end)]
        linears = [self.to_features[i][1] for i in range(start, end)]
        gamma = torch.stack([norm.gamma.to(device=device, dtype=dtype) for norm in norms], dim=0)
        weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
        bias = None
        if linears[0].bias is not None:
            bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)

        cached = (gamma, weight, bias)
        self._group_cache[key] = cached
        return cached

    def _forward_grouped(self, x):
        outs = []
        for start, end, dim_in in self._dim_groups:
            offset_start = self._dim_offsets[start]
            offset_end = self._dim_offsets[end]
            group_x = x[..., offset_start:offset_end].reshape(*x.shape[:-1], end - start, dim_in)
            gamma, weight, bias = self._get_group_params(start, end, x.device, x.dtype)
            group_x = F.normalize(group_x, dim=-1) * (dim_in ** 0.5) * gamma
            outs.append(grouped_linear(group_x, weight, bias))

        return torch.cat(outs, dim=-2)

    def warm_group_cache(self, device, dtype):
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)

    def forward(self, x):
        if not self.training and self.use_grouped_forward:
            return self._forward_grouped(x)

        x = x.split(self.dim_inputs, dim=-1)
        to_features = self.to_features

        outs = []
        for split_input, to_feature in zip(x, to_features):
            split_output = to_feature(split_input)
            outs.append(split_output)

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh
):
    dim_hidden = default(dim_hidden, dim_in)

    net = []
    dims = (dim_in, *((dim_hidden,) * depth), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    @beartype
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            net = []

            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth),
                nn.GLU(dim=-1)
            )

            self.to_freqs.append(mlp)

    def _groupable_layers(self, mlp_with_glu):
        if not isinstance(mlp_with_glu, nn.Sequential) or len(mlp_with_glu) != 2:
            return None
        mlp, glu = mlp_with_glu
        if not isinstance(glu, nn.GLU) or not isinstance(mlp, nn.Sequential):
            return None
        layers = []
        for layer in mlp:
            if isinstance(layer, nn.Linear):
                layers.append(('linear', layer))
            elif isinstance(layer, nn.Tanh):
                layers.append(('tanh', None))
            else:
                return None
        if not layers or layers[-1][0] != 'linear':
            return None
        return tuple(layers)

    def _can_group_mlp(self):
        base_signature = None
        for mlp_with_glu in self.to_freqs:
            layers = self._groupable_layers(mlp_with_glu)
            if layers is None:
                return False
            signature = tuple(
                item if kind != 'linear' else (kind, item.in_features, item.out_features, item.bias is not None)
                for kind, item in layers
            )
            if base_signature is None:
                base_signature = signature
            elif signature != base_signature:
                return False
        return True

    def _get_group_params(self, start, end, device, dtype):
        key = (start, end, device.type, device.index, dtype)
        cached = self._group_cache.get(key)
        if cached is not None:
            return cached

        grouped_layers = []
        first_layers = self._groupable_layers(self.to_freqs[start])
        for layer_index, (kind, _) in enumerate(first_layers):
            if kind == 'tanh':
                grouped_layers.append(('tanh', None, None))
                continue

            linears = [self._groupable_layers(self.to_freqs[i])[layer_index][1] for i in range(start, end)]
            weight = torch.stack([linear.weight.to(device=device, dtype=dtype) for linear in linears], dim=0)
            bias = None
            if linears[0].bias is not None:
                bias = torch.stack([linear.bias.to(device=device, dtype=dtype) for linear in linears], dim=0)
            grouped_layers.append(('linear', weight, bias))

        cached = tuple(grouped_layers)
        self._group_cache[key] = cached
        return cached

    def _forward_grouped_mlp(self, x):
        outs = []
        for start, end, _ in self._dim_groups:
            group_x = x[:, :, start:end, :]
            for kind, weight, bias in self._get_group_params(start, end, x.device, x.dtype):
                if kind == 'linear':
                    group_x = grouped_linear(group_x, weight, bias)
                else:
                    group_x = torch.tanh(group_x)
            group_out = F.glu(group_x, dim=-1)
            outs.append(group_out.flatten(start_dim=-2))

        return torch.cat(outs, dim=-1)

    def forward(self, x):
        if not self.training and self.use_grouped_forward and self._can_group_mlp():
            return self._forward_grouped_mlp(x)

        x = x.unbind(dim=-2)

        outs = []

        for band_features, mlp in zip(x, self.to_freqs):
            freq_out = mlp(band_features)
            outs.append(freq_out)

        return torch.cat(outs, dim=-1)

    def warm_group_cache(self, device, dtype):
        if not self._can_group_mlp():
            return
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)


# main class

class MelBandRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,  # needed for mel filter bank from librosa
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            sage_attention_mode='none',
            attention_layout='bhnd',
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.inference_layer_skip = None
        self.inference_time_layer_skip = None
        self.inference_freq_layer_skip = None
        self.inference_grouped_band_ops = True
        self.inference_rmsnorm_fp32 = True
        self._rmsnorm_fp32_state = None

        self.layers = ModuleList([])
        valid_sage_modes = {'none', 'time', 'freq', 'all'}
        if sage_attention_mode not in valid_sage_modes:
            raise ValueError(f"sage_attention_mode must be one of {sorted(valid_sage_modes)}")
        valid_attention_layouts = {'bhnd', 'bnhd'}
        if attention_layout not in valid_attention_layouts:
            raise ValueError(f"attention_layout must be one of {sorted(valid_attention_layouts)}")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            sage_attention=sage_attention,
            attention_layout=attention_layout,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    sage_mode=sage_attention_mode in ('time', 'all'),
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    sage_mode=sage_attention_mode in ('freq', 'all'),
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = nn.Identity()

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self._stft_window_cache = {}

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)
        self.register_buffer(
            'num_bands_per_channel_freq',
            num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1),
            persistent=False
        )

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window

    def _prepare_inference_core_options(self):
        rmsnorm_fp32 = bool(self.inference_rmsnorm_fp32 if not self.training else True)
        if self._rmsnorm_fp32_state is not rmsnorm_fp32:
            set_rmsnorm_fp32(self, rmsnorm_fp32)
            self._rmsnorm_fp32_state = rmsnorm_fp32

        grouped_band_ops = self.inference_grouped_band_ops if not self.training else True
        self.band_split.use_grouped_forward = bool(grouped_band_ops)
        for mask_estimator in self.mask_estimators:
            mask_estimator.use_grouped_forward = bool(grouped_band_ops)

    def _forward_mask_core(self, selected_stft_repr, full_t):
        return forward_roformer_mask_core(
            self,
            selected_stft_repr,
            use_checkpoint=self.training and self.use_torch_checkpoint,
        )

    def _compiled_mask_core(self, selected_stft_repr, full_t):
        mode = self.__dict__.get('_pymss_torch_compile_mode', 'default')
        cache = self.__dict__.setdefault('_pymss_compiled_mask_cores', {})
        key = (
            tuple(selected_stft_repr.shape),
            int(full_t),
            selected_stft_repr.device.type,
            selected_stft_repr.device.index,
            selected_stft_repr.dtype,
            mode,
        )
        compiled = cache.get(key)
        if compiled is None:
            self._prepare_inference_core_options()
            if self.band_split.use_grouped_forward:
                self.band_split.warm_group_cache(selected_stft_repr.device, selected_stft_repr.dtype)
                for mask_estimator in self.mask_estimators:
                    mask_estimator.warm_group_cache(selected_stft_repr.device, selected_stft_repr.dtype)
            compiled = torch.compile(self._forward_mask_core, mode=mode, fullgraph=False)
            cache[key] = compiled
        return compiled(selected_stft_repr, full_t)

    def _forward_mask_core_maybe_compiled(self, selected_stft_repr, full_t):
        if (
            not self.training
            and self.__dict__.get('_pymss_torch_compile_enabled', False)
            and self.__dict__.get('_pymss_torch_compile_scope') == 'core'
            and self.__dict__.get('_pymss_compile_core_this_call', True)
        ):
            return self._compiled_mask_core(selected_stft_repr, full_t)
        return self._forward_mask_core(selected_stft_repr, full_t)

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        stft_audio = raw_audio.reshape(batch * channels, raw_audio_length)

        stft_window = self.stft_window(device)

        stft_repr = torch.stft(stft_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = stft_repr.reshape(batch, channels, *stft_repr.shape[-3:])

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        b, s, f, t, c = stft_repr.shape
        stft_freq_bins = f
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)

        # index out all frequencies for all frequency ranges across bands ascending in one go

        batch_arange = torch.arange(batch, device=device)[..., None]

        # account for stereo

        x = stft_repr[batch_arange, self.freq_indices]

        num_stems = len(self.mask_estimators)
        self._prepare_inference_core_options()
        masks = self._forward_mask_core_maybe_compiled(x, stft_repr.shape[-2])

        # modulate frequency representation

        stft_repr = stft_repr.unsqueeze(1)

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks.contiguous())

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(batch, num_stems, -1, stft_repr.shape[-1])

        masks_summed = stft_repr.new_zeros(batch, num_stems, stft_repr.shape[2], stft_repr.shape[-1])
        masks_summed.scatter_add_(2, scatter_indices, masks)

        denom = self.num_bands_per_channel_freq

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask

        stft_repr = stft_repr * masks_averaged

        # istft

        b, n, fs, t = stft_repr.shape
        stft_repr = stft_repr.reshape(b, n, stft_freq_bins, s, t).permute(0, 1, 3, 2, 4).reshape(
            b * n * s,
            stft_freq_bins,
            t
        )

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False,
                                  length=istft_length)

        recon_audio = recon_audio.reshape(batch, num_stems, channels, recon_audio.shape[-1])

        if num_stems == 1:
            recon_audio = recon_audio[:, 0]

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
