import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from .attend import Attend


_CUDA_ATTENTION_BACKEND_ALIASES = {"auto": "auto", "torch": "default", "default": "default", "sdpa": "default", "flash": "flash",
    "flash_attention": "flash", "cudnn": "cudnn", "cudnn_attn": "cudnn", "cudnn_attention": "cudnn", "efficient": "efficient",
    "mem_efficient": "efficient", "memory_efficient": "efficient", "math": "math", "xformers": "xformers"}
_SDPA_BACKEND_ENUM_NAMES = {"flash": "FLASH_ATTENTION", "cudnn": "CUDNN_ATTENTION", "efficient": "EFFICIENT_ATTENTION", "math": "MATH"}
_MPS_BACKENDS = ("torch", "mlx", "mlx_attention", "mlx_transformer")


def normalize_cuda_attention_backend(backend):
    backend = str(backend or "cudnn").lower().replace("-", "_")
    if backend not in _CUDA_ATTENTION_BACKEND_ALIASES:
        raise ValueError("cuda_attention_backend must be one of: auto, default, flash, cudnn, efficient, math, xformers")
    return _CUDA_ATTENTION_BACKEND_ALIASES[backend]


def _sdpa_backend_enum(backend):
    enum_cls = getattr(getattr(torch.nn, "attention", None), "SDPBackend", None)
    enum_name = _SDPA_BACKEND_ENUM_NAMES.get(backend)
    return None if enum_cls is None or enum_name is None else getattr(enum_cls, enum_name, None)


def default_cuda_attention_backend():
    return "cudnn" if _sdpa_backend_enum("cudnn") is not None else "default"


def set_mps_attention_backend(module, backend=None, min_tokens=128, children=(), keep_mlx_transformer=False):
    # one switch shared by Attention / Transformer / Conformer; mlx_transformer maps to torch at leaf level
    backend = (backend or "torch").lower()
    if backend not in _MPS_BACKENDS:
        raise ValueError("mps_attention_backend must be 'torch', 'mlx', 'mlx_attention', or 'mlx_transformer'")
    module.mps_attention_backend = backend if keep_mlx_transformer else ("torch" if backend == "mlx_transformer" else backend)
    module.mps_mlx_min_tokens = 128 if min_tokens is None else int(min_tokens)
    for child in children:
        child.set_mps_attention_backend("torch" if backend == "mlx_transformer" else backend, module.mps_mlx_min_tokens)


def set_cuda_attention_backend(module, backend=None, children=()):
    module.cuda_attention_backend = normalize_cuda_attention_backend(backend)
    if hasattr(module, "_disabled_cuda_attention_backends"):
        module._disabled_cuda_attention_backends.clear()
    for child in children:
        child.set_cuda_attention_backend(module.cuda_attention_backend)


def _sdpa_with_backend(q, k, v, dropout_p, backend):
    kernel = getattr(getattr(torch.nn, "attention", None), "sdpa_kernel", None)
    enum = _sdpa_backend_enum(backend)
    if kernel is None or enum is None:
        raise RuntimeError(f"SDPA backend {backend!r} is not available in this PyTorch build")
    with kernel(enum):
        return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)


def _xformers_attention(q, k, v, dropout_p):
    import xformers.ops as xops
    return xops.memory_efficient_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), p=dropout_p).transpose(1, 2)


def apply_rotary_emb_fast(cos, sin, t):
    if t.is_cuda and t.dtype == torch.float16:
        rot = torch.complex(cos[..., ::2], sin[..., ::2])
        rotated = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2)) * rot
        return torch.view_as_real(rotated).reshape_as(t)
    cos, sin, t_even, t_odd = cos[..., ::2], sin[..., ::2], t[..., ::2], t[..., 1::2]
    out = torch.empty_like(t)
    out[..., ::2] = t_even * cos - t_odd * sin
    out[..., 1::2] = t_odd * cos + t_even * sin
    return out


def cached_rotary_cos_sin(rotary_embed, seq_len, device, dtype):
    cache = getattr(rotary_embed, "_pymss_cos_sin_cache", None)
    if cache is None:
        rotary_embed._pymss_cos_sin_cache = cache = {}
    key = (seq_len, device.type, device.index, dtype)
    if (cached := cache.get(key)) is not None:
        return cached
    freqs = rotary_embed.forward(
        lambda: rotary_embed.get_seq_pos(seq_len, device=device, dtype=dtype, offset=0), cache_key=f"freqs:{seq_len}|offset:0"
    )[None, :, None, :].to(device=device, dtype=dtype)
    cache[key] = cached = (freqs.cos(), freqs.sin())
    return cached


def rotate_qk_fast_bnhd(rotary_embed, q, k):
    cos, sin = cached_rotary_cos_sin(rotary_embed, q.shape[1], q.device, q.dtype)
    return apply_rotary_emb_fast(cos, sin, q), apply_rotary_emb_fast(cos, sin, k)


def qkv_to_bnhd(qkv, heads):
    b, n, _ = qkv.shape
    return qkv.view(b, n, 3, heads, -1).unbind(dim=2)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, theta=10000):
        super().__init__()
        freqs = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.freqs = nn.Parameter(freqs, requires_grad=False)
        self.cache = {}

    def get_seq_pos(self, seq_len, device, dtype, offset=0):
        return torch.arange(seq_len, device=device, dtype=dtype) + offset

    def forward(self, t, cache_key=None):
        if cache_key in self.cache:
            return self.cache[cache_key]
        t = t() if callable(t) else t
        freqs = (t.to(self.freqs.dtype)[:, None] * self.freqs[None]).repeat_interleave(2, -1)
        if cache_key is not None:
            self.cache[cache_key] = freqs
        return freqs


class RMSNorm(Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
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
    def __init__(self, dim, mult=4, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(RMSNorm(dim), nn.Linear(dim, int(dim * mult)), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(int(dim * mult), dim), nn.Dropout(dropout))

    def forward(self, x):
        return self.net(x)


class Attention(Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.0, shared_qkv_bias=None, shared_out_bias=None,
                 rotary_embed=None, flash=True):
        super().__init__()
        self.heads, self.flash, self.dropout, self.rotary_embed = heads, flash, dropout, rotary_embed
        self.mps_attention_backend, self.mps_mlx_min_tokens = "torch", 128
        self.cuda_attention_backend, self._disabled_cuda_attention_backends = default_cuda_attention_backend(), set()
        self.attend, self.norm = Attend(flash=False, dropout=dropout), RMSNorm(dim)
        self.to_qkv = nn.Linear(dim, heads * dim_head * 3, bias=(shared_qkv_bias is not None))
        if shared_qkv_bias is not None:
            self.to_qkv.bias = shared_qkv_bias
        self.to_gates = nn.Linear(dim, heads)
        self.to_out = nn.Sequential(nn.Linear(heads * dim_head, dim, bias=(shared_out_bias is not None)), nn.Dropout(dropout))
        if shared_out_bias is not None:
            self.to_out[0].bias = shared_out_bias

    def set_mps_attention_backend(self, backend=None, min_tokens=128):
        set_mps_attention_backend(self, backend, min_tokens)

    def set_cuda_attention_backend(self, backend=None):
        set_cuda_attention_backend(self, backend)

    def _use_mlx_attention_layer(self, x):
        return (self.flash and not self.training and self.mps_attention_backend == "mlx_attention" and x.device.type == "mps"
                and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps")) and x.shape[-2] >= self.mps_mlx_min_tokens)

    def _use_mlx_sdpa(self, q):
        return (self.flash and not self.training and self.mps_attention_backend == "mlx" and q.device.type == "mps"
                and q.dtype == torch.float16 and q.shape[-2] >= self.mps_mlx_min_tokens)

    def _attention(self, q, k, v):
        if self._use_mlx_sdpa(q):
            try:
                from .mlx_attention import mlx_bridge_sdpa
                return mlx_bridge_sdpa(q, k, v)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = "torch"
        if self.flash:
            return self._cuda_or_default_attention(q, k, v)
        return self.attend(q, k, v)

    def _cuda_or_default_attention(self, q, k, v):
        dropout_p = self.dropout if self.training else 0.0
        backend = self.cuda_attention_backend
        if not q.is_cuda or backend == "default":
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        if backend == "auto":
            for candidate in ("cudnn", "efficient"):
                if candidate in self._disabled_cuda_attention_backends:
                    continue
                try:
                    return _sdpa_with_backend(q, k, v, dropout_p, candidate)
                except torch.cuda.OutOfMemoryError:
                    raise
                except Exception as exc:
                    self._pymss_cuda_attention_backend_error = f"{candidate}: {exc!r}"
                    self._disabled_cuda_attention_backends.add(candidate)
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)
        try:
            if backend == "xformers":
                return _xformers_attention(q, k, v, dropout_p)
            return _sdpa_with_backend(q, k, v, dropout_p, backend)
        except torch.cuda.OutOfMemoryError:
            raise
        except Exception as exc:
            self._pymss_cuda_attention_backend_error = f"{backend}: {exc!r}"
            self.cuda_attention_backend = "default"
            return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p)

    def forward(self, x):
        if self._use_mlx_attention_layer(x):
            try:
                from .mlx_attention import mlx_bridge_attention
                return mlx_bridge_attention(self, x)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.mps_attention_backend = "torch"
        x = self.norm(x)
        q, k, v = qkv_to_bnhd(self.to_qkv(x), self.heads)
        if self.rotary_embed is not None:
            q, k = rotate_qk_fast_bnhd(self.rotary_embed, q, k)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        out = self._attention(q, k, v)
        return self.to_out((out.transpose(1, 2) * self.to_gates(x).unsqueeze(-1).sigmoid()).flatten(start_dim=-2))


class Transformer(Module):
    def __init__(self, *, dim, depth, dim_head=64, heads=8, attn_dropout=0.0, ff_dropout=0.0, ff_mult=4, norm_output=True,
                 rotary_embed=None, flash_attn=True, shared_qkv_bias=None, shared_out_bias=None):
        super().__init__()
        self.layers = ModuleList([ModuleList([
            Attention(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, shared_qkv_bias=shared_qkv_bias,
                      shared_out_bias=shared_out_bias, rotary_embed=rotary_embed, flash=flash_attn),
            FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)]) for _ in range(depth)])
        self.norm = RMSNorm(dim) if norm_output else nn.Identity()
        self.mps_attention_backend, self.mps_mlx_min_tokens, self.cuda_attention_backend = "torch", 128, default_cuda_attention_backend()

    def set_mps_attention_backend(self, backend=None, min_tokens=128):
        set_mps_attention_backend(self, backend, min_tokens, [attn for attn, _ in self.layers], keep_mlx_transformer=True)

    def set_cuda_attention_backend(self, backend=None):
        set_cuda_attention_backend(self, backend, [attn for attn, _ in self.layers])

    def _use_mlx_transformer(self, x):
        return (self.mps_attention_backend == "mlx_transformer" and not self.training and x.device.type == "mps"
                and (x.dtype == torch.float16 or torch.is_autocast_enabled("mps")) and x.shape[-2] >= self.mps_mlx_min_tokens)

    def forward(self, x):
        if self._use_mlx_transformer(x):
            try:
                from .mlx_attention import mlx_bridge_transformer
                return mlx_bridge_transformer(self, x)
            except Exception as exc:
                self._pymss_mlx_backend_error = repr(exc)
                self.set_mps_attention_backend("torch", self.mps_mlx_min_tokens)
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return self.norm(x)
