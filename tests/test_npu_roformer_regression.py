from types import SimpleNamespace

import pytest
import torch

from pymss_core.modules.bs_roformer.bands import MaskEstimator
from pymss_core.modules.bs_roformer.common import RoformerRuntimeMixin
from _devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dim_inputs", [(2, 4, 2), (2, 2, 2)])
@pytest.mark.parametrize("hidden_layers", [0, 1, 2])
def test_packed_masks_match_separate_estimators(device, hidden_layers, dim_inputs):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    estimators = [MaskEstimator(4, dim_inputs, depth=2, mlp_hidden_layers=hidden_layers).eval().to(device) for _ in range(2)]
    x = torch.randn(1, 3, 3, 4, device=device)
    with torch.inference_mode():
        expected = torch.stack([torch.cat([mlp(band) for mlp, band in zip(est.to_freqs, x.unbind(-2))], -1) for est in estimators], 1)
        actual = MaskEstimator.forward_packed_estimators(estimators, x)
    assert actual is not None
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("packed", [False, True])
def test_noncontiguous_mask_groups_keep_band_order(device, packed):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    estimators = [MaskEstimator(4, (2, 4, 2), depth=2).eval().to(device) for _ in range(2 if packed else 1)]
    # The grouping plan supports heterogeneous layers, including bias choices.
    for estimator in estimators:
        estimator.to_freqs[1][0][0].bias = None
    x = torch.randn(1, 3, 3, 4, device=device)
    with torch.inference_mode():
        separate = [torch.cat([mlp(band) for mlp, band in zip(est.to_freqs, x.unbind(-2))], -1) for est in estimators]
        expected = torch.stack(separate, 1) if packed else separate[0]
        actual = MaskEstimator.forward_packed_estimators(estimators, x) if packed else estimators[0](x)
    torch.testing.assert_close(actual, expected)


def test_roformer_window_preserves_cpu_factory_contract():
    calls = []
    def window_fn(*, device):
        calls.append(device)
        return torch.ones(16, device=device)
    runtime = SimpleNamespace(_stft_window_cache={}, stft_window_fn=window_fn)
    with torch.device("meta"):
        window = RoformerRuntimeMixin.stft_window(runtime, torch.device("cpu"))
    assert window.device.type == "cpu"
    assert calls == [torch.device("cpu")]


def test_roformer_npu_window_is_created_on_cpu():
    calls = []
    target = SimpleNamespace(type="npu", index=0)
    class Window:
        device = torch.device("cpu")
        def to(self, device):
            self.device = device
            return self
    def window_fn(**kwargs):
        calls.append(kwargs)
        return Window()
    runtime = SimpleNamespace(_stft_window_cache={}, stft_window_fn=window_fn)
    window = RoformerRuntimeMixin.stft_window(runtime, target)
    assert calls == [{"device": "cpu", "dtype": torch.float32}]
    assert window.device is target
