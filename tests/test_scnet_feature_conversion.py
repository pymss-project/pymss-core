import math

import pytest
import torch

from pymss_core.modules.scnet.separation import FeatureConversion


@pytest.fixture(params=["cpu", "cuda"])
def native_device(request):
    if request.param == "cuda" and not torch.cuda.is_available(): pytest.skip("CUDA unavailable")
    return torch.device(request.param)


@pytest.fixture(scope="module")
def dml_device():
    directml = pytest.importorskip("torch_directml")
    if not directml.is_available(): pytest.skip("DirectML adapter unavailable")
    return directml.device(0)


def _input(channels, length, offset, dtype, device):
    torch.manual_seed(23)
    base = torch.randn(2, channels, 3, length * 2 + 2 if offset else length, dtype=dtype).to(device)
    return base[..., 1:1 + length * 2:2] if offset else base


def _reference(value, inverse):
    value = value.float()
    if inverse:
        real, imag = value.chunk(2, dim=1)
        return torch.fft.irfft(torch.complex(real, imag), dim=3, norm="ortho")
    spectrum = torch.fft.rfft(value, dim=3, norm="ortho")
    return torch.cat((spectrum.real, spectrum.imag), dim=1)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.parametrize("offset", [False, True])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_native_conversion_preserves_transform_device_dtype_and_gradients(inverse, length, offset, dtype, native_device):
    channels = 4 if inverse else 2
    value = _input(channels, length, offset, dtype, native_device).detach().requires_grad_()
    reference_input = value.detach().clone().requires_grad_()
    if offset: assert not value.is_contiguous() and value.storage_offset() > 0
    actual = FeatureConversion(channels, inverse)(value)
    reference = _reference(reference_input, inverse)
    assert actual.device == value.device and actual.dtype == torch.float32
    assert actual.shape[-1] == (2 * (length - 1) if inverse else length // 2 + 1)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    weight = torch.linspace(0.25, 1.25, actual.numel(), device=actual.device).reshape_as(actual)
    (actual.square() * weight).sum().backward()
    (reference.square() * weight).sum().backward()
    assert value.grad.dtype == dtype
    torch.testing.assert_close(value.grad, reference_input.grad, rtol=2e-5, atol=2e-6)


def test_conversion_uses_orthonormal_real_then_imaginary_channel_layout():
    length = 8
    audio = torch.zeros(1, 2, 1, length)
    audio[:, 0, :, 0], audio[:, 1] = 1, 1
    packed = torch.zeros(1, 4, 1, length // 2 + 1)
    packed[:, 0], packed[:, 1, :, 0] = 1 / math.sqrt(length), math.sqrt(length)
    torch.testing.assert_close(FeatureConversion(2, False)(audio), packed, rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(FeatureConversion(4, True)(packed), audio, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("length", [7, 8])
def test_inverse_retains_native_even_length_inference(length, native_device):
    audio = _input(2, length, True, torch.float32, native_device)
    packed = FeatureConversion(2, False)(audio)
    actual = FeatureConversion(4, True)(packed)
    reference = torch.fft.irfft(torch.fft.rfft(audio, dim=3, norm="ortho"), dim=3, norm="ortho")
    assert actual.shape[-1] == 2 * (length // 2)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    if length % 2 == 0: torch.testing.assert_close(actual, audio, rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("inverse", [False, True])
@pytest.mark.parametrize("length", [7, 8])
@pytest.mark.parametrize("offset", [False, True])
def test_directml_conversion_preserves_values_and_returns_real_device_tensors(inverse, length, offset, dml_device):
    channels = 4 if inverse else 2
    cpu = _input(channels, length, offset, torch.float16, "cpu")
    gpu = _input(channels, length, offset, torch.float16, dml_device)
    if offset: assert not gpu.is_contiguous() and cpu.storage_offset() > 0
    torch.testing.assert_close(gpu.cpu(), cpu)
    with torch.no_grad(): actual = FeatureConversion(channels, inverse)(gpu)
    assert actual.device == dml_device and actual.dtype == torch.float32 and not actual.is_complex()
    assert torch.isfinite(actual.cpu()).all()
    torch.testing.assert_close(actual.cpu(), _reference(cpu, inverse), rtol=1e-5, atol=1e-6)
