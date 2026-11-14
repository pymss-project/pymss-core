from copy import deepcopy

import pytest

from pymss_core import ModelTypeDetectionError, detect_model_type, get_model_from_config


@pytest.mark.parametrize(("config", "expected"), [
    ({"model": {"freqs_per_bands": [4, 5]}}, "bs_roformer"),
    ({"model": {"freqs_per_bands": [4, 5], "use_pope": True}}, "bs_roformer"),
    ({"model": {"num_bands": 60}}, "mel_band_roformer"),
    ({"model": {"freqs_per_bands": [4, 5], "time_conformer_depth": 2}}, "bs_conformer"),
    ({"model": {
        "freqs_per_bands": [4, 5], "bridge_layers": [0, 3, 6, 9],
        "time_channels": [64, 128, 192, 256], "time_strides": [8, 8, 16],
    }}, "ht_roformer"),
    ({"model": {
        "bridge_layers": [0, 3, 6, 9],
        "time_channels": [64, 128, 192, 256], "time_strides": [8, 8, 16],
    }}, "ht_roformer"),
    ({"model": {"num_bands": 60, "freq_conformer_depth": 2}}, "mel_band_conformer"),
    ({"model": "htdemucs", "htdemucs": {"channels": 48}}, "htdemucs"),
    ({"model": {"num_subbands": 4, "num_scales": 3, "num_blocks_per_scale": 2, "bottleneck_factor": 4}}, "mdx23c"),
    ({"model": {"dims": [4, 32, 64, 128], "num_dplayer": 6}}, "scnet"),
    ({"model": {"band_SR": [0.175, 0.392, 0.433], "band_stride": [1, 4, 16], "band_kernel": [3, 4, 16]}}, "scnet"),
    ({"model": {"sr": 44100, "win": 20, "feature_dim": 128, "layer": 6}}, "apollo"),
    ({"model": {"in_channel": 2, "stems": ["vocals"], "band_specs": "musdb:vocals"}}, "bandit"),
    ({"kwargs": {"in_channels": 2, "stems": ["vocals"]}}, "bandit_v2"),
])
def test_detects_supported_yaml_structures_without_mutation(config, expected):
    original = deepcopy(config)
    assert detect_model_type(config) == expected
    assert config == original


@pytest.mark.parametrize("path", [
    ("model_type",), ("model", "type"), ("model", "model_type"),
    ("model", "architecture"), ("training", "model_type"),
])
def test_explicit_architecture_metadata_takes_precedence_over_hints(path):
    config = {"model": {"freqs_per_bands": [4, 5]}}
    section = config
    for key in path[:-1]:
        section = section.setdefault(key, {})
    section[path[-1]] = " BS_ROFORMER_HYPERACE "
    assert detect_model_type(config) == "bs_roformer_hyperace"


@pytest.mark.parametrize("config", [
    {"model_type": "scnet", "model": {"type": "bs_roformer"}},
    {"model_type": "bs_roformer", "model": "htdemucs"},
    {"model": {"freqs_per_bands": [4, 5], "num_bands": 60}},
    {"model": {"freqs_per_bands": [4, 5]}, "kwargs": {"in_channels": 2, "stems": ["vocals"]}},
])
def test_conflicting_architectures_require_explicit_selection(config):
    with pytest.raises(ModelTypeDetectionError, match="Set model_type explicitly"):
        detect_model_type(config)


@pytest.mark.parametrize("config", [
    None, [], "bs_roformer", {}, {"model": {}}, {"model": {"dim": 8, "depth": 1}},
    {"model": {"use_pope": True}}, {"model": "unknown"},
    {"model_type": "unknown"}, {"model_type": ["bs_roformer"]},
    {"model": {"sr": 44100, "win": 20}}, {"kwargs": {"in_channels": 2}},
])
def test_unknown_or_incomplete_configuration_raises_runtime_error(config):
    with pytest.raises(ModelTypeDetectionError, match="Set model_type explicitly"):
        detect_model_type(config)


def test_auto_metadata_defers_to_structural_detection():
    assert detect_model_type({"model_type": "auto", "model": {"num_bands": 60}}) == "mel_band_roformer"


def test_explicit_vr_metadata_remains_available_to_catalog_filters():
    assert detect_model_type({"model_type": "vr"}) == "vr"


def test_detection_error_remains_a_runtime_error():
    assert issubclass(ModelTypeDetectionError, RuntimeError)


@pytest.mark.parametrize("contents", [None, b"model: [", b"\xff\xfe"])
def test_auto_factory_reports_config_read_errors(tmp_path, contents):
    path = tmp_path / "config.yaml"
    if contents is not None:
        path.write_bytes(contents)
    with pytest.raises(ModelTypeDetectionError, match="readable YAML") as error:
        get_model_from_config("auto", path)
    assert error.value.__cause__ is not None


def test_manual_factory_keeps_original_file_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        get_model_from_config("bs_roformer", tmp_path / "missing.yaml")
