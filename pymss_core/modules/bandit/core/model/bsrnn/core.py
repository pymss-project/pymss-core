from torch import nn

from . import BandsplitCoreBase
from .bandsplit import BandSplitModule
from .maskestim import MaskEstimationModule, OverlappingMaskEstimationModule
from .tfmodel import SeqBandModellingModule

__all__ = ("MultiSourceMultiMaskBandSplitCoreRNN",)


class MultiMaskBandSplitCoreBase(BandsplitCoreBase):
    def forward(self, x, cond=None, compute_residual=True):
        batch, in_chan, n_freq, n_time = x.shape
        x = x.reshape(-1, 1, n_freq, n_time)
        q = self.tf_model(self.band_split(x))
        out = {}
        for stem, mask_estimator in self.mask_estim.items():
            separated = self.mask(x, mask_estimator(q, cond=cond))
            out[stem] = separated.reshape(batch, in_chan, n_freq, n_time)
        return {"spectrogram": out}

    def instantiate_mask_estim(self, in_channel, stems, band_specs, emb_dim, mlp_dim, cond_dim, hidden_activation,
                               hidden_activation_kwargs=None, complex_mask=True, overlapping_band=False,
                               freq_weights=None, n_freq=None, use_freq_weights=True, mult_add_mask=False):
        if mult_add_mask:
            raise NotImplementedError("Bandit mult_add_mask is not supported by the inference-only wrapper")
        stems = [stem for stem in stems if stem != "mne:+"]
        kwargs = dict(emb_dim=emb_dim, mlp_dim=mlp_dim, in_channel=in_channel, hidden_activation=hidden_activation,
                      hidden_activation_kwargs=hidden_activation_kwargs or {}, complex_mask=complex_mask)
        if overlapping_band:
            assert freq_weights is not None
            assert n_freq is not None
            self.mask_estim = nn.ModuleDict({
                stem: OverlappingMaskEstimationModule(band_specs=band_specs, freq_weights=freq_weights, n_freq=n_freq,
                                                      use_freq_weights=use_freq_weights, **kwargs)
                for stem in stems
            })
        else:
            self.mask_estim = nn.ModuleDict({
                stem: MaskEstimationModule(band_specs=band_specs, cond_dim=cond_dim, **kwargs)
                for stem in stems
            })

    def instantiate_bandsplit(self, in_channel, band_specs, require_no_overlap=False, require_no_gap=True,
                              normalize_channel_independently=False, treat_channel_as_feature=True, emb_dim=128):
        self.band_split = BandSplitModule(in_channel=in_channel, band_specs=band_specs, emb_dim=emb_dim,
                                          require_no_overlap=require_no_overlap, require_no_gap=require_no_gap,
                                          normalize_channel_independently=normalize_channel_independently,
                                          treat_channel_as_feature=treat_channel_as_feature)


class MultiSourceMultiMaskBandSplitCoreRNN(MultiMaskBandSplitCoreBase):
    def __init__(self, in_channel, stems, band_specs, require_no_overlap=False, require_no_gap=True,
                 normalize_channel_independently=False, treat_channel_as_feature=True, n_sqm_modules=12, emb_dim=128,
                 rnn_dim=256, bidirectional=True, rnn_type="LSTM", mlp_dim=512, cond_dim=0, hidden_activation="Tanh",
                 hidden_activation_kwargs=None, complex_mask=True, overlapping_band=False, freq_weights=None,
                 n_freq=None, use_freq_weights=True, mult_add_mask=False):
        super().__init__()
        self.instantiate_bandsplit(in_channel=in_channel, band_specs=band_specs, require_no_overlap=require_no_overlap,
                                   require_no_gap=require_no_gap,
                                   normalize_channel_independently=normalize_channel_independently,
                                   treat_channel_as_feature=treat_channel_as_feature, emb_dim=emb_dim)
        self.tf_model = SeqBandModellingModule(n_modules=n_sqm_modules, emb_dim=emb_dim, rnn_dim=rnn_dim,
                                               bidirectional=bidirectional, rnn_type=rnn_type)
        self.instantiate_mask_estim(in_channel=in_channel, stems=stems, band_specs=band_specs, emb_dim=emb_dim,
                                    mlp_dim=mlp_dim, cond_dim=cond_dim, hidden_activation=hidden_activation,
                                    hidden_activation_kwargs=hidden_activation_kwargs, complex_mask=complex_mask,
                                    overlapping_band=overlapping_band, freq_weights=freq_weights, n_freq=n_freq,
                                    use_freq_weights=use_freq_weights, mult_add_mask=mult_add_mask)
