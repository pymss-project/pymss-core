from __future__ import annotations

import copy

import pytest
import torch

from pymss_core import clear_model_runtime_caches
from pymss_core.modules.bandit.maskestim import MultAddNormMLP, NormMLP, OverlappingMaskEstimationModule
from pymss_core.modules.bandit_v2.bandit import Bandit


@pytest.fixture(autouse=True)
def single_threaded_cpu():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(previous)


@pytest.fixture(params=["cpu", "cuda", "dml"])
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
        return torch.device("cuda:0")
    if request.param == "dml":
        directml = pytest.importorskip("torch_directml")
        if not directml.is_available(): pytest.skip("DirectML unavailable")
        return directml.device(0)
    return torch.device("cpu")


@pytest.mark.parametrize("complex_mask", [False, True])
@pytest.mark.parametrize("kind", ["standard", "combined", "mult_add"])
def test_mask_mlp_preserves_channel_frequency_time_layout_and_gradients(kind, complex_mask):
    kwargs = dict(emb_dim=4, mlp_dim=8, bandwidth=3, in_channels=2, complex_mask=complex_mask)
    module = MultAddNormMLP(**kwargs) if kind == "mult_add" else NormMLP(**kwargs, use_combined=kind == "combined")
    inputs = torch.randn(2, 5, 4, requires_grad=True)
    output = module(inputs)
    masks = output if isinstance(output, tuple) else (output,)
    for mask in masks:
        assert mask.shape == (2, 2, 3, 5)
        assert mask.is_complex() is complex_mask
        assert mask.isfinite().all()
    sum(mask.abs().square().sum() for mask in masks).backward()
    assert inputs.grad is not None and inputs.grad.isfinite().all()
    for parameter in module.parameters():
        assert parameter.grad is not None and parameter.grad.isfinite().all()


def test_real_masks_merge_overlapping_bands_with_frequency_weights(device):
    weights = [torch.tensor([1.0, 1.0, 0.25]), torch.tensor([0.75, 1.0, 1.0])]
    module = OverlappingMaskEstimationModule(
        band_specs=[(0, 3), (2, 5)], freq_weights=weights, n_freq=5,
        emb_dim=4, mlp_dim=8, in_channels=2, complex_mask=False, output_dtype="complex64",
    ).eval().to(device)
    features = torch.randn(2, 2, 3, 4).to(device)
    with torch.no_grad():
        bands = [mask.cpu() for mask in module.compute_masks(features)]
        actual = module(features)
    expected = torch.zeros(2, 2, 5, 3, dtype=torch.complex64)
    expected[:, :, :3] += bands[0] * weights[0][:, None]
    expected[:, :, 2:] += bands[1] * weights[1][:, None]
    assert actual.device.type == ("cpu" if device.type == "privateuseone" else device.type)
    assert actual.dtype == torch.complex64
    torch.testing.assert_close(actual.cpu(), expected, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("complex_mask", [False, True])
def test_bandit_v2_real_and_complex_masks_preserve_stereo_tail_and_cpu_parity(device, complex_mask):
    torch.manual_seed(7)
    model = Bandit(
        in_channels=1, stems=["vocals", "other"], fs=8000, n_bands=8,
        n_fft=128, win_length=128, hop_length=32, n_sqm_modules=1,
        emb_dim=8, rnn_dim=8, mlp_dim=8, complex_mask=complex_mask,
    ).eval()
    inputs = torch.randn(2, 2, 257)
    weights = copy.deepcopy(model.state_dict())
    with torch.no_grad(): reference = model(inputs)
    clear_model_runtime_caches(model)
    model.to(device)
    try:
        with torch.no_grad(): actual = model(inputs.to(device))
        assert actual.shape == (2, 2, 2, 257)
        assert actual.device == device
        assert actual.isfinite().all()
        torch.testing.assert_close(actual.cpu(), reference, rtol=2e-4, atol=2e-5)
    finally:
        clear_model_runtime_caches(model)
        model.to("cpu")
    for name, expected in weights.items():
        torch.testing.assert_close(model.state_dict()[name], expected, rtol=0, atol=0)
