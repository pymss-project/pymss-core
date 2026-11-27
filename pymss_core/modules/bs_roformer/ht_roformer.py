import torch
import torch.nn.functional as F
from torch import nn

from .bands import BandSplit, MaskEstimator
from .common import DEFAULT_FREQS_PER_BANDS, RotaryEmbedding
from .transformer import Attention, FeedForward, RMSNorm, Transformer

#: bridges are inserted after these spectral layers (12 layers -> 4 bridges)
ROFORMER_BLOCKS = (0, 3, 6, 9)
TIME_KERNEL = 16            # encoder conv kernel; conv_transpose uses the matching kernel
TIME_CROP = TIME_KERNEL // 2  # conv_transpose crop that re-aligns frames with the encoder


class DConvBlock(nn.Module):
    """ConvNeXt-ish gated block: C -> C/4 (k3) -> 2C (k1) -> GLU, scaled residual."""

    def __init__(self, channels, dilation=1, scale=1.0):
        super().__init__()
        # each stage stacks one plain and one x2 dilated block (padding == dilation)
        self.dconv_layers = nn.Sequential(
            nn.Conv1d(channels, channels // 4, 3, padding=dilation, dilation=dilation, bias=True),
            nn.GroupNorm(1, channels // 4),
            nn.GELU(),
            nn.Conv1d(channels // 4, channels * 2, 1, bias=True),
        )
        self.norm2 = nn.GroupNorm(1, channels * 2)
        self.res_scale = nn.Parameter(torch.full((channels, 1), float(scale)))

    def forward(self, x):
        # conv -> GroupNorm+GELU -> conv -> GroupNorm -> GLU (no activation after norm2)
        h = self.norm2(self.dconv_layers(x))
        value, gate = h.chunk(2, dim=1)                      # nn.GLU order: first * sigmoid(second)
        return x + self.res_scale.to(dtype=x.dtype) * (value * torch.sigmoid(gate))


class TimeEncoderStage(nn.Module):
    """Zero-pad k/2, stride-kernel conv, GroupNorm+GELU, then two gated blocks."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv = nn.Conv1d(in_channels, out_channels, TIME_KERNEL, stride=stride, bias=True)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dconv_layers = nn.ModuleList([DConvBlock(out_channels, dilation=1), DConvBlock(out_channels, dilation=2)])

    def forward(self, x):
        x = F.pad(x, (TIME_CROP, TIME_CROP))
        x = F.gelu(self.norm(self.conv(x)))
        for block in self.dconv_layers:
            x = block(x)
        return x


class TimeDecoderStage(nn.Module):
    """Transposed conv, crop to the skip length, concat skip, 1x1 fuse, two gated blocks."""

    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv_tr = nn.ConvTranspose1d(in_channels, out_channels, TIME_KERNEL, stride=stride, bias=True)
        self.skip_proj = nn.Conv1d(out_channels * 2, out_channels, 1, bias=True)
        self.norm = nn.GroupNorm(1, out_channels)
        self.dconv_layers = nn.ModuleList([DConvBlock(out_channels, dilation=1), DConvBlock(out_channels, dilation=2)])

    def forward(self, x, skip):
        x = self.conv_tr(x)
        length = skip.shape[-1]
        x = x[..., TIME_CROP:TIME_CROP + length]
        x = torch.cat((x, skip), dim=1)
        x = F.gelu(self.norm(self.skip_proj(x)))
        for block in self.dconv_layers:
            x = block(x)
        return x


class CrossAttention(nn.Module):
    """Bridged attention: queries from one branch, keys/values from the other."""

    def __init__(self, dim, heads=8, dim_head=64, shared_q_bias=None, shared_kv_bias=None,
                 shared_out_bias=None, dropout=0.0):
        super().__init__()
        self.heads, self.dim_head = heads, dim_head
        inner = heads * dim_head
        self.norm = RMSNorm(dim)
        self.norm_ctx = RMSNorm(dim)
        self.q = nn.Linear(dim, inner, bias=shared_q_bias is not None)
        self.kv = nn.Linear(dim, inner * 2, bias=shared_kv_bias is not None)
        if shared_q_bias is not None:
            self.q.bias = shared_q_bias
        if shared_kv_bias is not None:
            self.kv.bias = shared_kv_bias
        self.out = nn.Linear(inner, dim, bias=shared_out_bias is not None)
        if shared_out_bias is not None:
            self.out.bias = shared_out_bias
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x, ctx):
        shape = x.shape[:-1]
        q = self.q(self.norm(x)).reshape(*shape, self.heads, self.dim_head).transpose(-3, -2)
        kv = self.kv(self.norm_ctx(ctx)).reshape(*shape, 2, self.heads, self.dim_head)
        k, v = kv.unbind(dim=-3)
        k, v = k.transpose(-3, -2), v.transpose(-3, -2)
        out = F.scaled_dot_product_attention(q, k, v)                 # scale = dim_head ** -0.5
        out = out.transpose(-3, -2).reshape(*shape, self.heads * self.dim_head)
        return self.out(out) * self.scale


class CrossBridge(nn.Module):
    """Time self-attention + band-mean spectral context, then cross attention both ways."""

    def __init__(self, dim, heads, dim_head, rotary_embed, shared_qkv_bias, shared_out_bias,
                 shared_q_bias, shared_kv_bias, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, flash_attn=True):
        super().__init__()
        self.time_self_attn = Attention(dim=dim, heads=heads, dim_head=dim_head, dropout=attn_dropout,
                                        shared_qkv_bias=shared_qkv_bias, shared_out_bias=shared_out_bias,
                                        rotary_embed=rotary_embed, flash=flash_attn)
        self.time_self_ff = FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
        self.freq_to_time_proj = nn.Sequential(RMSNorm(dim), nn.Linear(dim, dim))
        self.time_cross = CrossAttention(dim, heads, dim_head, shared_q_bias, shared_kv_bias, shared_out_bias, attn_dropout)
        self.freq_cross = CrossAttention(dim, heads, dim_head, shared_q_bias, shared_kv_bias, shared_out_bias, attn_dropout)

    def forward(self, x_time, x_freq):
        t = x_time + self.time_self_attn(x_time)
        t = t + self.time_self_ff(t)
        context = self.freq_to_time_proj(x_freq.mean(dim=2))          # band mean -> time-side context
        x_time = t + self.time_cross(t, context)
        b, n, f, d = x_freq.shape                                     # (batch, time, bands, dim)
        flat = x_freq.permute(0, 2, 1, 3).reshape(b * f, n, d)         # one sequence per band
        freq_ctx = x_time.unsqueeze(2).expand(b, n, f, d).permute(0, 2, 1, 3).reshape(b * f, n, d)
        x_freq = (flat + self.freq_cross(flat, freq_ctx)).reshape(b, f, n, d).permute(0, 2, 1, 3)
        return x_time, x_freq


class HTRoFormer(nn.Module):
    """Demucs-style hybrid: BS-RoFormer spectral branch + time-domain U-Net branch.

    Outputs ``istft(mask * STFT) + time residual`` per stem (6 stems).
    """

    def __init__(self, dim=256, depth=12, stereo=True, num_stems=6, heads=8, dim_head=64,
                 time_transformer_depth=1, freq_transformer_depth=1, attn_dropout=0.0, ff_dropout=0.0,
                 ff_mult=4, flash_attn=True, mlp_expansion_factor=4, mask_estimator_depth=2,
                 freqs_per_bands=DEFAULT_FREQS_PER_BANDS, stft_n_fft=2048, stft_hop_length=1024,
                 stft_win_length=2048, stft_normalized=False, stft_window_fn=None,
                 bridge_layers=ROFORMER_BLOCKS, time_channels=(64, 128, 192, 256),
                 time_strides=(8, 8, 16), time_kernel=TIME_KERNEL, output_mode="sum",
                 use_shared_bias=True, standardize_eps=1e-5, **kwargs):
        super().__init__()
        if kwargs:
            raise TypeError(f"unexpected HTRoFormer config keys: {sorted(kwargs)}")
        self.audio_channels = 2 if stereo else 1
        self.num_stems, self.stereo = num_stems, stereo
        self.output_mode = output_mode
        self.standardize_eps = standardize_eps
        self.time_channels, self.time_strides = tuple(time_channels), tuple(time_strides)
        if int(time_kernel) != TIME_KERNEL:
            raise ValueError(f"time_kernel must be {TIME_KERNEL}")
        self.stft_kwargs = {"n_fft": stft_n_fft, "hop_length": stft_hop_length,
                            "win_length": stft_win_length, "normalized": stft_normalized}
        self.stft_win_length = stft_win_length
        self.stft_window_fn = stft_window_fn or torch.hann_window
        self._stft_window_cache = {}

        # ---- shared attention biases (traced graph shares four of them) ------
        inner = heads * dim_head
        if use_shared_bias:
            self.linear_62_bias_0 = nn.Parameter(torch.ones(inner * 3))    # qkv, every attention
            self.linear_64_bias_0 = nn.Parameter(torch.ones(dim))          # out proj, every attention
            self.linear_78_bias_0 = nn.Parameter(torch.ones(inner))        # cross-attention Q
            self.linear_79_bias_0 = nn.Parameter(torch.ones(inner * 2))    # cross-attention KV
        qkv_bias = getattr(self, "linear_62_bias_0", None)
        out_bias = getattr(self, "linear_64_bias_0", None)
        q_bias = getattr(self, "linear_78_bias_0", None)
        kv_bias = getattr(self, "linear_79_bias_0", None)

        # ---- spectral branch -------------------------------------------------
        self.rotary_embed = RotaryEmbedding(dim_head)      # one table shared by every attention
        self.layers = nn.ModuleList([
            nn.ModuleList([
                Transformer(depth=time_transformer_depth, dim=dim, heads=heads, dim_head=dim_head,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
                            flash_attn=flash_attn, norm_output=False, rotary_embed=self.rotary_embed,
                            shared_qkv_bias=qkv_bias, shared_out_bias=out_bias),
                Transformer(depth=freq_transformer_depth, dim=dim, heads=heads, dim_head=dim_head,
                            attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
                            flash_attn=flash_attn, norm_output=False, rotary_embed=self.rotary_embed,
                            shared_qkv_bias=qkv_bias, shared_out_bias=out_bias),
            ]) for _ in range(depth)])
        self.cross_bridges = nn.ModuleList([
            CrossBridge(dim=dim, heads=heads, dim_head=dim_head, rotary_embed=self.rotary_embed,
                        shared_qkv_bias=qkv_bias, shared_out_bias=out_bias,
                        shared_q_bias=q_bias, shared_kv_bias=kv_bias,
                        attn_dropout=attn_dropout, ff_dropout=ff_dropout, ff_mult=ff_mult,
                        flash_attn=flash_attn) for _ in bridge_layers])
        self.bridge_layers = tuple(int(index) for index in bridge_layers)

        freqs = stft_win_length // 2 + 1
        if sum(freqs_per_bands) != freqs:
            raise ValueError(f"freqs_per_bands must cover {freqs} STFT bins")
        dim_inputs = tuple(2 * f * self.audio_channels for f in freqs_per_bands)
        self.band_split = BandSplit(dim=dim, dim_inputs=dim_inputs)
        self.final_norm = RMSNorm(dim)
        self.mask_estimators = nn.ModuleList([
            MaskEstimator(dim=dim, dim_inputs=dim_inputs, depth=mask_estimator_depth,
                          mlp_expansion_factor=mlp_expansion_factor) for _ in range(num_stems)])

        # ---- time branch -----------------------------------------------------
        c0, c1, c2, c3 = self.time_channels
        self.time_input_conv = nn.Sequential(
            nn.Conv1d(self.audio_channels, c0, 7, padding=3, bias=True),
            nn.GroupNorm(1, c0), nn.GELU())
        self.time_encoder = nn.ModuleList([
            TimeEncoderStage(c0, c1, self.time_strides[0]),
            TimeEncoderStage(c1, c2, self.time_strides[1]),
            TimeEncoderStage(c2, c3, self.time_strides[2]),
        ])
        self.time_decoder = nn.ModuleList([
            TimeDecoderStage(c3, c2, self.time_strides[2]),
            TimeDecoderStage(c2, c1, self.time_strides[1]),
            TimeDecoderStage(c1, c0, self.time_strides[0]),
        ])
        self.time_output_conv = nn.Conv1d(c0, num_stems * self.audio_channels, 7, padding=3, bias=True)

    # -- helpers -----------------------------------------------------------
    def clear_runtime_cache(self):
        self._stft_window_cache.clear()
        for module in self.modules():
            if module is self:
                continue
            clear = getattr(module, "clear_runtime_cache", None)
            if clear is not None:
                clear()

    def stft_window(self, device):
        key = (device.type, device.index)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(self.stft_win_length, device=device)
            self._stft_window_cache[key] = window
        return window

    def stft(self, raw_audio):
        """Return complex STFT (b, ch, f, t) and the band-split input (b, t, f*ch*2)."""
        b, c, length = raw_audio.shape
        window = self.stft_window(raw_audio.device)
        spec = torch.stft(raw_audio.reshape(b * c, length), window=window, return_complex=True, **self.stft_kwargs)
        spec = spec.reshape(b, c, spec.shape[1], spec.shape[2])
        # (b, ch, f, t, 2) -> (b, t, f, ch, 2): freq-major per band, real/imag last
        feats = torch.view_as_real(spec).permute(0, 3, 2, 1, 4).reshape(b, spec.shape[3], -1)
        return spec, feats

    def band_transformer_forward(self, feats, hidden):
        x = self.band_split(feats)
        bridge_index = 0
        for index, (time_transformer, freq_transformer) in enumerate(self.layers):
            b, n, f, d = x.shape
            x = time_transformer(x.permute(0, 2, 1, 3).reshape(b * f, n, d)).reshape(b, f, n, d).permute(0, 2, 1, 3)
            x = freq_transformer(x.reshape(b * n, f, d)).reshape(b, n, f, d)
            if index in self.bridge_layers:
                hidden, x = self.cross_bridges[bridge_index](hidden, x)
                bridge_index += 1
        return self.final_norm(x), hidden

    def forward(self, raw_audio, return_parts=False):
        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)
        batch, channels, length = raw_audio.shape
        if channels != self.audio_channels:
            raise ValueError(f"expected {self.audio_channels} channels, got {channels}")
        # global standardisation of the mixture, as in the traced graph
        mean = raw_audio.mean(dim=(1, 2), keepdim=True)
        scale = raw_audio.var(dim=(1, 2), unbiased=False, keepdim=True).sqrt()
        normalized = (raw_audio - mean) / (scale + self.standardize_eps)

        spec, feats = self.stft(raw_audio)

        hidden = self.time_input_conv(normalized)
        skips = [hidden]
        for stage in self.time_encoder:
            hidden = stage(hidden)
            skips.append(hidden)
        hidden = hidden.transpose(1, 2)                                        # (b, t, dim)

        x, hidden = self.band_transformer_forward(feats, hidden)

        mask = torch.stack([estimator(x) for estimator in self.mask_estimators], dim=1)
        b, stems, n, fc = mask.shape
        mask = mask.reshape(b, stems, n, fc // 2, 2).permute(0, 1, 3, 2, 4)     # (b, s, f*ch, t, 2)
        stft_repr = torch.view_as_real(spec).permute(0, 2, 1, 3, 4).reshape(b, fc // 2, n, 2)
        masked = torch.view_as_complex(stft_repr.unsqueeze(1).contiguous()) * torch.view_as_complex(mask.contiguous())

        hidden = hidden.transpose(1, 2)                                        # (b, dim, t)
        for stage, skip in zip(self.time_decoder, (skips[2], skips[1], skips[0])):
            hidden = stage(hidden, skip)
        residual = self.time_output_conv(hidden).reshape(b, stems, self.audio_channels, length)
        residual = residual * scale.reshape(b, 1, 1, 1) + mean.reshape(b, 1, 1, 1)

        freq_bins, frames = masked.shape[-2] // self.audio_channels, masked.shape[-1]
        # (b, s, f*ch, t) -> (b, s, ch, f, t) -> (b, s*ch, f, t); the flat index is f*ch + channel
        def to_bscft(tensor):
            return tensor.reshape(b, stems, freq_bins, self.audio_channels, frames).permute(0, 1, 3, 2, 4).reshape(
                b, stems * self.audio_channels, freq_bins, frames)
        if return_parts:
            return {"stft_real": to_bscft(masked.real), "stft_imag": to_bscft(masked.imag),
                    "time_output": residual}

        spectral = masked.reshape(b * stems, freq_bins, self.audio_channels, frames).permute(0, 2, 1, 3)
        spectral = torch.istft(spectral.reshape(b * stems * self.audio_channels, freq_bins, frames),
                               window=self.stft_window(raw_audio.device), length=length, **self.stft_kwargs)
        spectral = spectral.reshape(b, stems, self.audio_channels, length)
        if self.output_mode == "spectral":
            return spectral
        if self.output_mode == "time":
            return residual
        return spectral + residual
