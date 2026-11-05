from types import SimpleNamespace

import pytest
import torch

from pymss_core.modules.bandit.core.model._spectral import _TorchSpectrogram
from pymss_core.modules.look2hear.apollo import Apollo
from pymss_core.modules.spectrogram import SubbandSTFT
from _devices import DEVICES


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("kind", ["bandit", "apollo", "subband"])
def test_existing_spectral_paths_preserve_input_precision(device, dtype, kind):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    torch.manual_seed(31)
    wave = torch.randn(2, 2, 1027, dtype=dtype, device=device)
    flat = wave.reshape(-1, wave.shape[-1])
    if kind == "bandit":
        model = _TorchSpectrogram(16, 16, 4, torch.hann_window, None, False, True, "reflect", True).to(device=device, dtype=dtype)
        actual = model(wave)
        expected = torch.stft(flat, 16, 4, window=model.window, return_complex=True)
        expected = expected.reshape(*wave.shape[:-1], *expected.shape[-2:])
    elif kind == "apollo":
        model = Apollo(sr=16000, win=20, feature_dim=8, layer=1).to(device=device, dtype=dtype)
        actual = model._stft(wave)
        expected = torch.stft(flat, model.win, model.stride, window=model._window(wave), return_complex=True)
    else:
        model = SubbandSTFT(SimpleNamespace(n_fft=16, hop_length=4, dim_f=9))
        actual = model(wave)
        spec = torch.stft(flat, 16, 4, window=model.window.to(device), return_complex=True)
        expected = torch.view_as_real(spec).permute(0, 3, 1, 2).reshape(2, 4, 9, -1)
    assert actual.dtype == expected.dtype
    tolerance = 1e-11 if dtype == torch.float64 else 1e-5
    torch.testing.assert_close(actual, expected, atol=tolerance, rtol=tolerance)
