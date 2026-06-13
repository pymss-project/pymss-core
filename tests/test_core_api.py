from __future__ import annotations

import torch

import pymss_core
from pymss_core import AttrDict, load_config, unwrap_state_dict
from pymss_core.modules._dsp import mel_filterbank


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


def test_vr_network_structures_remain_importable():
    from pymss_core.modules.vocal_remover import CascadedASPPNet, CascadedNet, ModelParameters

    assert CascadedASPPNet is not None
    assert CascadedNet is not None
    assert ModelParameters is not None
