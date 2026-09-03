import torch
from torch import nn
from torch.nn.modules import activation

from .core.model.bsrnn.utils import band_widths_from_specs, check_no_gap, check_no_overlap, check_nonzero_bandwidth

def _resolve_channels(in_channels=None, in_channel=None):
    channels = in_channels if in_channels is not None else in_channel
    if channels is None: raise TypeError("in_channels is required")
    return channels

class BaseNormMLP(nn.Module):
    def __init__(self, emb_dim, mlp_dim, bandwidth, in_channels=None, in_channel=None, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True):
        super().__init__()
        self.hidden_activation_kwargs = hidden_activation_kwargs or {}
        self.norm = nn.LayerNorm(emb_dim)
        self.hidden = nn.Sequential(nn.Linear(emb_dim, mlp_dim), activation.__dict__[hidden_activation](**self.hidden_activation_kwargs))
        self.bandwidth = bandwidth
        self.in_channels = self.in_channel = _resolve_channels(in_channels, in_channel)
        self.complex_mask = complex_mask
        self.reim, self.glu_mult = 2 if complex_mask else 1, 2

class NormMLP(BaseNormMLP):
    def __init__(self, emb_dim, mlp_dim, bandwidth, in_channels=None, in_channel=None, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, use_combined=False, use_checkpoint=False):
        super().__init__(emb_dim, mlp_dim, bandwidth, in_channels, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask)
        self.output = nn.Sequential(nn.Linear(mlp_dim, self.bandwidth * self.in_channels * self.reim * 2), nn.GLU(dim=-1))
        self.use_checkpoint = use_checkpoint
        if use_combined: self.combined = nn.Sequential(self.norm, self.hidden, self.output)
    def reshape_output(self, mb):
        b, t = mb.shape[:2]
        mb = mb.reshape(b, t, self.in_channels, self.bandwidth, self.reim)
        if self.complex_mask: mb = torch.view_as_complex(mb.contiguous())
        return mb.permute(0, 2, 3, 1)
    def forward(self, qb):
        from torch.utils.checkpoint import checkpoint_sequential
        if hasattr(self, "combined"):
            mb = checkpoint_sequential(self.combined, 2, qb, use_reentrant=False) if self.use_checkpoint else self.combined(qb)
        else:
            mb = self.output(self.hidden(self.norm(qb)))
        return self.reshape_output(mb)

class MultAddNormMLP(NormMLP):
    def __init__(self, emb_dim, mlp_dim, bandwidth, in_channels=None, in_channel=None, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True): super().__init__(emb_dim, mlp_dim, bandwidth, in_channels, in_channel, hidden_activation, hidden_activation_kwargs, complex_mask); self.output2 = nn.Sequential(nn.Linear(mlp_dim, self.bandwidth * self.in_channels * self.reim * 2), nn.GLU(dim=-1))
    def forward(self, qb): qb = self.hidden(self.norm(qb)); return self.reshape_output(self.output(qb)), self.reshape_output(self.output2(qb))

class MaskEstimationModuleSuperBase(nn.Module):
    pass

class MaskEstimationModuleBase(MaskEstimationModuleSuperBase):
    def __init__(self, band_specs, emb_dim, mlp_dim, in_channels=None, in_channel=None, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, norm_mlp_cls=NormMLP, norm_mlp_kwargs=None):
        super().__init__()
        self.band_widths, self.n_bands = band_widths_from_specs(band_specs), len(band_specs)
        self.norm_mlp = nn.ModuleList([ norm_mlp_cls(bandwidth=bw, emb_dim=emb_dim, mlp_dim=mlp_dim, in_channels=_resolve_channels(in_channels, in_channel), hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs or {}, complex_mask=complex_mask, **(norm_mlp_kwargs or {})) for bw in self.band_widths])
    def compute_masks(self, q): return [nmlp(q[:, b, :, :]) for b, nmlp in enumerate(self.norm_mlp)]
    def compute_mask(self, q, b): return self.norm_mlp[b](q[:, b, :, :])

class OverlappingMaskEstimationModule(MaskEstimationModuleBase):
    def __init__(self, band_specs, freq_weights, n_freq, emb_dim, mlp_dim, in_channels=None, in_channel=None, cond_dim=0, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, norm_mlp_cls=NormMLP, norm_mlp_kwargs=None, use_freq_weights=True, register_all_freq_weights=True, allow_cond=True, output_dtype="mask", compute_all_masks=True):
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        if cond_dim > 0 and not allow_cond: raise NotImplementedError
        super().__init__(band_specs, emb_dim + cond_dim, mlp_dim, _resolve_channels(in_channels, in_channel), hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask, norm_mlp_cls=norm_mlp_cls, norm_mlp_kwargs=norm_mlp_kwargs)
        self.n_freq, self.band_specs, self.cond_dim, self.allow_cond = n_freq, band_specs, cond_dim, allow_cond
        self.in_channels = self.in_channel = _resolve_channels(in_channels, in_channel)
        self.output_dtype, self.compute_all_masks = output_dtype, compute_all_masks
        self.use_freq_weights = bool(freq_weights is not None and use_freq_weights)
        if freq_weights is not None and (register_all_freq_weights or use_freq_weights):
            for i, fw in enumerate(freq_weights): self.register_buffer(f"freq_weights/{i}", fw)
    def _append_cond(self, q, cond):
        if cond is None:
            if self.cond_dim <= 0: return q
            b, nb, t, _ = q.shape
            return torch.cat([q, torch.ones(b, nb, t, self.cond_dim, device=q.device, dtype=q.dtype)], dim=-1)
        if cond.ndim == 2:
            cond = cond[:, None, None, :].expand(-1, q.shape[1], q.shape[2], -1)
        elif cond.ndim != 3: raise ValueError(f"Invalid cond shape: {cond.shape}")
        else:
            assert cond.shape[1] == q.shape[2]
        return torch.cat([q, cond], dim=-1)
    def forward(self, q, cond=None):
        if not self.allow_cond and cond is not None: raise NotImplementedError
        q = self._append_cond(q, cond)
        b, n_bands, t, _ = q.shape
        mask_list = self.compute_masks(q) if self.compute_all_masks else None
        dtype = torch.complex64 if self.output_dtype == "complex64" else mask_list[0].dtype
        masks = torch.zeros(b, self.in_channels, self.n_freq, t, device=q.device, dtype=dtype)
        for im in range(n_bands):
            fstart, fend = self.band_specs[im]
            mask = mask_list[im] if mask_list is not None else self.compute_mask(q, im)
            if self.use_freq_weights: mask = mask * self.get_buffer(f'freq_weights/{im}')[:, None]
            masks[:, :, fstart:fend, :] += mask
        return masks

class MaskEstimationModule(OverlappingMaskEstimationModule):
    def __init__(self, band_specs, emb_dim, mlp_dim, in_channels=None, in_channel=None, hidden_activation="Tanh", hidden_activation_kwargs=None, complex_mask=True, **kwargs):
        check_nonzero_bandwidth(band_specs)
        check_no_gap(band_specs)
        check_no_overlap(band_specs)
        super().__init__(band_specs=band_specs, freq_weights=None, n_freq=0, emb_dim=emb_dim, mlp_dim=mlp_dim, in_channels=in_channels, in_channel=in_channel, hidden_activation=hidden_activation, hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask)
    def forward(self, q, cond=None): return torch.concat(self.compute_masks(q), dim=2)