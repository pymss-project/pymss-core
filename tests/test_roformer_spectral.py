from types import SimpleNamespace

import pytest
import torch

from pymss_core.modules.bs_roformer import common


@pytest.fixture(params=["cpu", "cuda", "dml"], scope="module")
def device(request):
    if request.param == "cuda":
        if not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
        return torch.device("cuda:0")
    if request.param == "dml":
        directml = pytest.importorskip("torch_directml")
        if not directml.is_available(): pytest.skip("DirectML unavailable")
        return directml.device(0)
    return torch.device("cpu")


def spectral_module(stereo=False):
    return SimpleNamespace(stereo=stereo, stft_kwargs=dict(n_fft=16, hop_length=4, win_length=16), stft_window=lambda device: torch.hann_window(16, device=device))


@pytest.mark.parametrize("operation", ["stft", "istft"])
def test_non_mps_spectral_errors_propagate_without_retry(device, operation, monkeypatch):
    module, audio = spectral_module(), torch.randn(1, 1, 64).to(device)
    spectrum, context = common.stft_roformer(module, audio)
    spectrum = common.real_to_complex(spectrum.unsqueeze(1))
    original = getattr(torch, operation)
    failure = RuntimeError("Spectral kernel failed")
    calls = []
    def fail_once(value, *args, **kwargs):
        calls.append(value.device)
        if len(calls) == 1: raise failure
        return original(value, *args, **kwargs)
    monkeypatch.setattr(torch, operation, fail_once)
    with pytest.raises(RuntimeError) as caught:
        if operation == "stft": common.stft_roformer(module, audio)
        else: common.istft_roformer(module, spectrum, context, 64)
    assert caught.value is failure
    assert len(calls) == 1


@pytest.mark.parametrize("stereo,stems", [(False, 1), (False, 2), (True, 1), (True, 2)])
def test_spectral_roundtrip_preserves_device_channel_stem_layout_and_length(device, stereo, stems):
    module = spectral_module(stereo)
    audio = torch.randn(2, 2 if stereo else 1, 65).to(device)
    spectrum, context = common.stft_roformer(module, audio)
    spectrum = common.real_to_complex(spectrum).unsqueeze(1).repeat(1, stems, 1, 1)
    output = common.istft_roformer(module, spectrum, context, context.audio_length)
    expected = audio if stems == 1 else audio.unsqueeze(1).repeat(1, stems, 1, 1)
    assert output.device == device
    assert output.shape == expected.shape
    torch.testing.assert_close(output.cpu(), expected.cpu(), rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("operation", ["stft", "istft"])
def test_mps_spectral_compatibility_retries_on_cpu_and_returns_to_original_device(operation, monkeypatch):
    module, audio = spectral_module(), torch.randn(1, 1, 64)
    expected_spectrum, context = common.stft_roformer(module, audio)
    spectrum = common.real_to_complex(expected_spectrum.unsqueeze(1))
    context = common.SpectralContext(context.batch, context.channels, context.freq_bins, context.audio_length, context.stft_window, True, torch.device("mps"))
    returned_to, cpu_calls = [], []
    original_to = torch.Tensor.to
    def record_to(tensor, *args, **kwargs):
        target = kwargs.get("device", args[0] if args else None)
        if isinstance(target, torch.device) and target.type == "mps":
            returned_to.append(target)
            return tensor
        return original_to(tensor, *args, **kwargs)
    monkeypatch.setattr(torch.Tensor, "to", record_to)
    original_kernel = getattr(torch, operation)
    def cpu_kernel(value, *args, **kwargs):
        assert value.device.type == kwargs["window"].device.type == "cpu"
        cpu_calls.append(value)
        return original_kernel(value, *args, **kwargs)
    def unsupported_kernel(*args, **kwargs): raise RuntimeError("MPS spectral kernel unavailable")
    if operation == "stft":
        monkeypatch.setattr(common, "stft_complex", unsupported_kernel)
        monkeypatch.setattr(torch, "stft", cpu_kernel)
        module.stft_window = lambda device: context.stft_window
        mps_audio = SimpleNamespace(device=context.device, ndim=audio.ndim, shape=audio.shape, reshape=audio.reshape)
        actual, actual_context = common.stft_roformer(module, mps_audio)
        expected = expected_spectrum
        assert actual_context.x_is_mps and actual_context.device == context.device
    else:
        calls = []
        def fail_first(value, *args, **kwargs):
            calls.append(value)
            if len(calls) == 1: return unsupported_kernel()
            return cpu_kernel(value, *args, **kwargs)
        monkeypatch.setattr(torch, "istft", fail_first)
        actual = common.istft_roformer(module, spectrum, context, 64)
        expected = audio
        assert len(calls) == 2
    torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-6)
    assert len(cpu_calls) == 1
    assert returned_to == [context.device]
