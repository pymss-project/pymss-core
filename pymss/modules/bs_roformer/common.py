from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend
from rotary_embedding_torch import RotaryEmbedding
from typing import Tuple

from einops import rearrange


DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


def exists(val):
    return val is not None


def default(v, d):
    return v if exists(v) else d


def mask_to_complex_shape(mask, complex_dim=2):
    b, n, t, fc = mask.shape
    return mask.reshape(b, n, t, fc // complex_dim, complex_dim).permute(0, 1, 3, 2, 4)


def rotate_half(x):
    out = torch.empty_like(x)
    out[..., ::2] = -x[..., 1::2]
    out[..., 1::2] = x[..., ::2]
    return out


def apply_rotary_emb_fast(cos, sin, t):
    return (t * cos) + (rotate_half(t) * sin)


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, '_pymss_cos_sin_cache', None)
    if cache is None:
        cache = {}
        rotary_embed._pymss_cos_sin_cache = cache

    key = (seq_len, device.type, device.index, dtype)
    cached = cache.get(key)
    if cached is not None:
        return cached

    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0),
        cache_key=f'freqs:{seq_len}|offset:0'
    )

    freqs = rearrange(freqs, 'n d -> 1 n 1 d')
    freqs = freqs.to(device=device, dtype=dtype)
    cached = (freqs.cos(), freqs.sin())
    cache[key] = cached
    return cached


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    seq_len = q.shape[1]
    device, dtype = q.device, q.dtype
    cos, sin = cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def qkv_to_bnhd(qkv, heads):
    b, n, _ = qkv.shape
    qkv = qkv.view(b, n, 3, heads, -1)
    return qkv.unbind(dim=2)


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


TRAINING_LOSS_KWARGS = frozenset({
    'multi_stft_resolution_loss_weight',
    'multi_stft_resolutions_window_sizes',
    'multi_stft_hop_size',
    'multi_stft_normalized',
    'multi_stft_window_fn',
})

REMOVED_ROFORMER_KWARGS = frozenset({
    'linear_transformer_depth',
    'use_torch_checkpoint',
    'skip_connection',
    'sage_attention',
    'sage_attention_mode',
    'attention_layout',
    'dim_freqs_in',
})


def ignore_roformer_training_kwargs(kwargs):
    unexpected = set(kwargs) - TRAINING_LOSS_KWARGS - REMOVED_ROFORMER_KWARGS
    if unexpected:
        raise TypeError(f"unexpected RoFormer config keys: {sorted(unexpected)}")


def init_roformer_runtime(module, stereo, num_stems):
    module.stereo = stereo
    module.audio_channels = 2 if stereo else 1
    module.num_stems = num_stems


def init_roformer_layers(
        module,
        *,
        depth,
        time_transformer_depth,
        freq_transformer_depth,
        dim_head,
        transformer_kwargs,
):
    module.layers = ModuleList([])
    time_rotary_embed = RotaryEmbedding(dim=dim_head)
    freq_rotary_embed = RotaryEmbedding(dim=dim_head)

    for _ in range(depth):
        module.layers.append(nn.ModuleList([
            Transformer(
                depth=time_transformer_depth,
                rotary_embed=time_rotary_embed,
                **transformer_kwargs
            ),
            Transformer(
                depth=freq_transformer_depth,
                rotary_embed=freq_rotary_embed,
                **transformer_kwargs
            ),
        ]))


def init_roformer_stft(module, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn):
    module.stft_kwargs = dict(
        n_fft=stft_n_fft,
        hop_length=stft_hop_length,
        win_length=stft_win_length,
        normalized=stft_normalized,
    )
    module.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
    module._stft_window_cache = {}


def roformer_freqs_per_bands_with_complex(module, freqs_per_bands, freqs):
    assert len(freqs_per_bands) > 1
    assert sum(
        freqs_per_bands
    ) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'
    return tuple(2 * f * module.audio_channels for f in freqs_per_bands)


def init_roformer_band_modules(
        module,
        *,
        dim,
        freqs_per_bands_with_complex,
        num_stems,
        mask_estimator_cls,
        mask_estimator_depth,
        mlp_expansion_factor,
        mask_estimator_kwargs=None,
):
    module.band_split = BandSplit(dim=dim, dim_inputs=freqs_per_bands_with_complex)
    module.mask_estimators = nn.ModuleList([
        mask_estimator_cls(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex,
            depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
            **(mask_estimator_kwargs or {}),
        )
        for _ in range(num_stems)
    ])


class RoformerRuntimeMixin:
    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window

    def _warm_group_cache(self, tensor):
        self.band_split.warm_group_cache(tensor.device, tensor.dtype)
        for mask_estimator in self.mask_estimators:
            mask_estimator.warm_group_cache(tensor.device, tensor.dtype)


def forward_roformer_mask_core(module, stft_repr):
    b, fs, model_t, complex_dim = stft_repr.shape
    x = stft_repr.permute(0, 2, 1, 3).reshape(b, model_t, fs * complex_dim)
    x = module.band_split(x)

    for time_transformer, freq_transformer in module.layers:
        b, t, f, d = x.shape
        x = x.permute(0, 2, 1, 3).reshape(b * f, t, d)
        x = time_transformer(x)
        x = x.reshape(b, f, t, d).permute(0, 2, 1, 3)

        x = x.reshape(b * t, f, d)
        x = freq_transformer(x)
        x = x.reshape(b, t, f, d)

    x = module.final_norm(x)
    mask = torch.stack([fn(x) for fn in module.mask_estimators], dim=1)
    return mask_to_complex_shape(mask, complex_dim=2)


def forward_bandsplit_roformer(module, raw_audio):
    device = raw_audio.device
    x_is_mps = device.type == "mps"

    if raw_audio.ndim == 2:
        raw_audio = raw_audio.unsqueeze(1)

    batch, audio_channels, audio_length = raw_audio.shape
    assert (
        not module.stereo and audio_channels == 1
    ) or (
        module.stereo and audio_channels == 2
    ), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

    stft_audio = raw_audio.reshape(batch * audio_channels, audio_length)
    stft_window = module.stft_window(device)

    try:
        stft_repr = torch.stft(stft_audio, **module.stft_kwargs, window=stft_window, return_complex=True)
    except RuntimeError:
        stft_repr = torch.stft(
            stft_audio.cpu() if x_is_mps else stft_audio,
            **module.stft_kwargs,
            window=stft_window.cpu() if x_is_mps else stft_window,
            return_complex=True
        ).to(device)

    stft_repr = torch.view_as_real(stft_repr)
    stft_repr = stft_repr.reshape(batch, audio_channels, *stft_repr.shape[-3:])

    b, s, f, t, c = stft_repr.shape
    stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)

    module._warm_group_cache(stft_repr)
    mask = module._forward_mask_core(stft_repr)

    stft_repr = torch.view_as_complex(stft_repr.unsqueeze(1))
    mask = torch.view_as_complex(mask.contiguous())
    stft_repr = stft_repr * mask

    b, n, fs, t = stft_repr.shape
    stft_repr = stft_repr.reshape(b, n, f, s, t).permute(0, 1, 3, 2, 4).reshape(
        b * n * s,
        f,
        t
    )

    try:
        recon_audio = torch.istft(
            stft_repr,
            **module.stft_kwargs,
            window=stft_window,
            return_complex=False,
            length=audio_length
        )
    except RuntimeError:
        recon_audio = torch.istft(
            stft_repr.cpu() if x_is_mps else stft_repr,
            **module.stft_kwargs,
            window=stft_window.cpu() if x_is_mps else stft_window,
            return_complex=False,
            length=audio_length
        ).to(device)

    recon_audio = recon_audio.reshape(batch, len(module.mask_estimators), audio_channels, audio_length)

    if len(module.mask_estimators) == 1:
        return recon_audio[:, 0]

    return recon_audio


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim ** 0.5
        self.gamma = nn.Parameter(torch.ones(dim))
        self._gamma_dtype_cache = {}

    def forward(self, x):
        if not self.training and x.dtype in (torch.float16, torch.bfloat16):
            key = (x.device.type, x.device.index, x.dtype, self.gamma.data_ptr(), self.gamma._version)
            gamma = self._gamma_dtype_cache.get(key)
            if gamma is None:
                gamma = self.gamma.detach().to(device=x.device, dtype=x.dtype)
                self._gamma_dtype_cache.clear()
                self._gamma_dtype_cache[key] = gamma
            return F.rms_norm(x, (x.shape[-1],), gamma, eps=1e-12)
        return F.normalize(x, dim=-1) * self.scale * self.gamma


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
            shared_qkv_bias=None,
            shared_out_bias=None,
            rotary_embed=None,
            flash=True,
    ):
        super().__init__()
        self.heads = heads
        self.scale = dim_head ** -0.5
        dim_inner = heads * dim_head
        self.flash = flash
        self.dropout = dropout

        self.rotary_embed = rotary_embed
        self.attend = Attend(flash=flash, dropout=dropout)

        self.norm = RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, dim_inner * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias

        self.to_gates = nn.Linear(dim, heads)

        self.to_out = nn.Sequential(
            nn.Linear(dim_inner, dim, bias=(shared_out_bias is not None)),
            nn.Dropout(dropout)
        )
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def forward(self, x):
        x = self.norm(x)

        q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)

        if exists(self.rotary_embed):
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)

        if self.flash:
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
            shared_qkv_bias=None,
            shared_out_bias=None,
    ):
        super().__init__()
        self.layers = ModuleList([])

        for _ in range(depth):
            attn = Attention(
                dim=dim,
                dim_head=dim_head,
                heads=heads,
                dropout=attn_dropout,
                shared_qkv_bias=shared_qkv_bias,
                shared_out_bias=shared_out_bias,
                rotary_embed=rotary_embed,
                flash=flash_attn,
            )

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


class BandSplit(Module):
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

        outs = []
        for split_input, to_feature in zip(x.split(self.dim_inputs, dim=-1), self.to_features):
            outs.append(to_feature(split_input))

        return torch.stack(outs, dim=-2)


def MLP(
        dim_in,
        dim_out,
        dim_hidden=None,
        depth=1,
        activation=nn.Tanh,
        hidden_layers=None,
):
    dim_hidden = default(dim_hidden, dim_in)
    hidden_layers = default(hidden_layers, max(depth - 1, 0))

    net = []
    dims = (dim_in, *((dim_hidden,) * hidden_layers), dim_out)

    for ind, (layer_dim_in, layer_dim_out) in enumerate(zip(dims[:-1], dims[1:])):
        is_last = ind == (len(dims) - 2)

        net.append(nn.Linear(layer_dim_in, layer_dim_out))

        if is_last:
            continue

        net.append(activation())

    return nn.Sequential(*net)


class MaskEstimator(Module):
    def __init__(
            self,
            dim,
            dim_inputs: Tuple[int, ...],
            depth,
            mlp_expansion_factor=4,
            mlp_hidden_layers=None,
    ):
        super().__init__()
        self.dim_inputs = dim_inputs
        self._dim_groups = contiguous_dim_groups(dim_inputs)
        self._group_cache = {}
        self.use_grouped_forward = True
        self.to_freqs = ModuleList([])
        dim_hidden = dim * mlp_expansion_factor

        for dim_in in dim_inputs:
            mlp = nn.Sequential(
                MLP(dim, dim_in * 2, dim_hidden=dim_hidden, depth=depth, hidden_layers=mlp_hidden_layers),
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

        outs = []
        for band_features, mlp in zip(x.unbind(dim=-2), self.to_freqs):
            outs.append(mlp(band_features))

        return torch.cat(outs, dim=-1)

    def warm_group_cache(self, device, dtype):
        if not self._can_group_mlp():
            return
        for start, end, _ in self._dim_groups:
            self._get_group_params(start, end, device, dtype)
