from __future__ import annotations

import torch

import pymss_core
from pymss_core import AttrDict, load_config, unwrap_state_dict
from pymss_core.modules._dsp import mel_filterbank
from pymss_core.modules.bs_roformer.common import SpectralContext, forward_roformer_mask_core, istft_roformer


def test_public_api_exports_core_functions():
    assert callable(pymss_core.get_model_from_config)
    assert callable(pymss_core.load_checkpoint)
    assert callable(pymss_core.load_model_weights)
    assert not hasattr(pymss_core, "demix")


def test_load_config_returns_attrdict(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        """
audio:
  chunk_size: 1024
training:
  instruments:
    - vocals
""".strip(),
        encoding="utf-8",
    )

    config = load_config(config_path)

    assert isinstance(config, AttrDict)
    assert config.audio.chunk_size == 1024
    assert config.training.instruments == ["vocals"]


def test_unwrap_state_dict_common_keys():
    state = {"weight": torch.ones(1)}

    assert unwrap_state_dict(state) is state
    assert unwrap_state_dict({"state": state}) is state
    assert unwrap_state_dict({"state_dict": state}) is state
    assert unwrap_state_dict({"model_state_dict": state}) is state


def test_model_internal_dsp_helpers_do_not_require_librosa():
    filters = mel_filterbank(sr=44100, n_fft=2048, n_mels=60)

    assert filters.shape == (60, 1025)


def test_roformer_istft_zero_dc(monkeypatch):
    class DummyModule:
        stft_kwargs = {}
        zero_dc = True

    captured = {}

    def fake_istft(stft_repr, **kwargs):
        captured["stft_repr"] = stft_repr.detach().clone()
        return torch.zeros(stft_repr.shape[0], kwargs["length"])

    monkeypatch.setattr(torch, "istft", fake_istft)

    context = SpectralContext(
        batch=1,
        channels=1,
        freq_bins=3,
        audio_length=4,
        stft_window=torch.ones(4),
        x_is_mps=False,
    )
    stft_repr = torch.ones(1, 1, 3, 2, dtype=torch.complex64)

    output = istft_roformer(DummyModule(), stft_repr, context, length=4)

    assert output.shape == (1, 1, 4)
    assert torch.equal(captured["stft_repr"][:, 0], torch.zeros_like(captured["stft_repr"][:, 0]))
    assert torch.equal(captured["stft_repr"][:, 1:], torch.ones_like(captured["stft_repr"][:, 1:]))


def test_roformer_mask_core_applies_configured_skip_connection():
    class AddModule(torch.nn.Module):
        def __init__(self, value):
            super().__init__()
            self.value = value
            self.inputs = []

        def forward(self, x):
            self.inputs.append(x.detach().clone())
            return x + self.value

    class DummyModule:
        final_norm = torch.nn.Identity()

        def __init__(self, skip_connection):
            self.skip_connection = skip_connection
            self.band_split = lambda x: x.reshape(1, 1, 1, 1)
            self.time_0 = AddModule(10)
            self.freq_0 = AddModule(100)
            self.time_1 = AddModule(1000)
            self.freq_1 = AddModule(10000)
            self.layers = [(self.time_0, self.freq_0), (self.time_1, self.freq_1)]
            self.final_x = None

        def _estimate_masks(self, x):
            self.final_x = x.detach().clone()
            return torch.ones(1, 1, 1, 2)

    module = DummyModule(skip_connection=True)

    output = forward_roformer_mask_core(module, torch.ones(1, 1, 1, 1))

    assert output.shape == (1, 1, 1, 1, 2)
    assert module.time_1.inputs[0].item() == 222
    assert module.final_x.item() == 11222

    module = DummyModule(skip_connection=False)

    output = forward_roformer_mask_core(module, torch.ones(1, 1, 1, 1))

    assert output.shape == (1, 1, 1, 1, 2)
    assert module.time_1.inputs[0].item() == 111
    assert module.final_x.item() == 11111


def test_roformer_constructors_preserve_skip_connection_flag():
    from pymss_core.modules.bs_roformer import BSRoformer, MelBandRoformer

    bs_roformer = BSRoformer(
        dim=4,
        depth=1,
        stereo=False,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        freqs_per_bands=(1,) * 9,
        heads=1,
        dim_head=4,
        stft_n_fft=16,
        stft_hop_length=4,
        stft_win_length=16,
        mask_estimator_depth=1,
        skip_connection=True,
    )
    mel_band_roformer = MelBandRoformer(
        dim=4,
        depth=1,
        stereo=False,
        num_stems=1,
        time_transformer_depth=1,
        freq_transformer_depth=1,
        heads=1,
        dim_head=4,
        mask_estimator_depth=1,
        skip_connection=True,
    )

    assert bs_roformer.skip_connection is True
    assert mel_band_roformer.skip_connection is True


def test_vr_network_structures_remain_importable():
    from pymss_core.modules.vocal_remover import CascadedASPPNet, CascadedNet, ModelParameters

    assert CascadedASPPNet is not None
    assert CascadedNet is not None
    assert ModelParameters is not None
