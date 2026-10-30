import numpy as np
import pytest
import torch

from pymss_core.npu_fft import (
    _irfft_matmul,
    _rfft_matmul,
    complex_to_numpy_cpu,
    hann_window_npu,
    istft_npu,
    masked_spectrum,
    irfft_exact,
    numpy_complex_to_device,
    rfft_exact,
    stft_npu,
    vendor_fft_trusted,
)


@pytest.mark.parametrize("n_fft,hop,length", [(16, 4, 97), (16, 7, 97), (15, 4, 98), (512, 128, 1027), (1024, 320, 2051)])
@pytest.mark.parametrize("normalized", [False, True])
def test_centered_istft_preserves_unaligned_tail(n_fft, hop, length, normalized):
    torch.manual_seed(24)
    wave = torch.randn(2, length)
    window = torch.hann_window(n_fft)
    spectrum = torch.stft(wave, n_fft, hop, window=window, normalized=normalized, return_complex=True)
    expected = torch.istft(spectrum, n_fft, hop, window=window, normalized=normalized, length=length)
    actual = istft_npu(spectrum, n_fft, hop, window=window, normalized=normalized, length=length)
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("normalized", [False, True])
def test_stft_and_istft_match_torch_without_an_explicit_length(center, normalized):
    torch.manual_seed(19)
    wave = torch.randn(2, 97)
    window = torch.ones(16)
    expected = torch.stft(wave, 16, 4, window=window, center=center, normalized=normalized, return_complex=True)
    actual = stft_npu(wave, 16, 4, window=window, center=center, normalized=normalized)
    torch.testing.assert_close(actual, expected)
    restored = istft_npu(actual, 16, 4, window=window, center=center, normalized=normalized)
    reference = torch.istft(expected, 16, 4, window=window, center=center, normalized=normalized)
    torch.testing.assert_close(restored, reference)


def test_hann_window_is_built_on_cpu_even_with_a_different_default_device():
    with torch.device("meta"):
        window = hann_window_npu(16, device="cpu")
    assert window.device.type == "cpu"
    torch.testing.assert_close(window, torch.hann_window(16))


@pytest.mark.parametrize("win_length", [8, 12])
def test_default_rectangular_window_supports_short_window_lengths(win_length):
    wave = torch.randn(2, 97)
    expected = torch.stft(wave, 16, 4, win_length, window=torch.ones(win_length), return_complex=True)
    actual = stft_npu(wave, 16, 4, win_length)
    torch.testing.assert_close(actual, expected)


def test_stft_rejects_unsupported_real_pair_layout():
    with pytest.raises(ValueError, match="return_complex"):
        stft_npu(torch.randn(97), 16, window=torch.ones(16), return_complex=False)


def test_demucs_npu_hann_is_created_on_cpu(monkeypatch):
    from types import SimpleNamespace
    from pymss_core.modules import demucs_local

    created = []
    result = object()
    device = type("NpuDevice", (), {"type": "npu", "index": 0})()
    ref = SimpleNamespace(device=device, dtype=torch.float32)

    def factory(*args, **kwargs):
        window = torch.hann_window(*args, **kwargs)
        created.append(window.device.type)
        return SimpleNamespace(to=lambda target: result)

    monkeypatch.setattr(demucs_local, "torch", SimpleNamespace(hann_window=factory))
    monkeypatch.setattr(demucs_local, "_HANN_CACHE", {})
    with torch.device("meta"):
        assert demucs_local._hann(16, ref) is result
        assert demucs_local._hann(16, ref) is result
    assert created == ["cpu"]


class TestMaskedSpectrum:
    """BIREN has no complex64 * float32 pointwise kernel."""

    def test_complex_input_matches_reference_product(self):
        torch.manual_seed(0)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        got = masked_spectrum(mask, spec)
        assert got.dtype == torch.complex64
        assert torch.equal(got, mask.to(torch.complex64) * spec)

    def test_extra_plane_dimension_is_created_for_broadcast(self):
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        assert masked_spectrum(mask, spec).shape == (2, 5, 7)

    def test_real_input_takes_the_direct_path(self):
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7)
        assert torch.equal(masked_spectrum(mask, spec), mask * spec)

    def test_batched_mask_broadcasts_over_channels(self):
        mask = torch.rand(1, 5, 7)
        spec = torch.randn(3, 5, 7, dtype=torch.complex64)
        assert masked_spectrum(mask, spec).shape == (3, 5, 7)


class TestComplexHostCopy:
    """BIREN raises 'ErrorCode: 101, invalid device' for complex device->host copies."""

    def test_complex_to_numpy_cpu_round_trips_values(self):
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        got = complex_to_numpy_cpu(spec)
        assert got.dtype == np.complex64
        assert (got == spec.numpy()).all()

    def test_masked_spectrum_to_cpu_matches_reference(self):
        torch.manual_seed(3)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        got = masked_spectrum(mask, spec, to_cpu=True)
        want = (mask.to(torch.complex64) * spec).numpy()
        assert (got == want).all()

    def test_masked_spectrum_to_cpu_on_real_input(self):
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7)
        assert (masked_spectrum(mask, spec, to_cpu=True) == (mask * spec).numpy()).all()


class TestComplexHostUpload:
    """BIREN rejects complex host->device copies too ('Unsupported BR type')."""

    def test_round_trip_through_planes_is_bit_exact(self):
        rng = np.random.default_rng(5)
        array = (rng.standard_normal((2, 5, 7)) + 1j * rng.standard_normal((2, 5, 7))).astype(np.complex64)
        tensor = numpy_complex_to_device(array, "cpu")
        assert tensor.dtype == torch.complex64
        assert torch.equal(tensor, torch.from_numpy(array))

    def test_real_input_takes_the_direct_path(self):
        array = np.zeros((2, 3), dtype=np.float32)
        assert numpy_complex_to_device(array, "cpu").dtype == torch.float32


class TestExactDftFallback:
    """The vendor FFT silently returns zeros for some shapes (n=16, 9 bins)."""

    @pytest.mark.parametrize("n", [16, 32, 512, 1000, 1411])
    def test_matmul_rfft_matches_torch(self, n):
        x = torch.randn(3, n)
        got = _rfft_matmul(x, n)
        want = torch.fft.rfft(x, dim=-1)
        assert torch.allclose(got, want, atol=2e-3)

    @pytest.mark.parametrize("n", [16, 32, 512, 1000, 1411])
    def test_matmul_irfft_round_trips(self, n):
        x = torch.randn(3, n)
        spec = torch.fft.rfft(x, dim=-1)
        back = _irfft_matmul(spec, n)
        assert back.shape == x.shape
        assert torch.allclose(back, x, atol=2e-3)

    def test_power_of_two_sizes_use_the_vendor_kernel(self):
        # A clean container measures 39 ms for n=2048 with 862 frames against 60 ms
        # for the DFT matrix, so the vendor kernel is the fast path; non-power-of-two
        # sizes have no plan at all.
        assert vendor_fft_trusted(2048) is True
        assert vendor_fft_trusted(4096) is True
        assert vendor_fft_trusted(1024) is True
        assert vendor_fft_trusted(16) is True
        assert vendor_fft_trusted(1411) is False

    def test_a_size_that_misbehaved_is_remembered(self):
        from pymss_core import npu_fft

        npu_fft._VENDOR_FFT_BAD.add(4096)
        try:
            assert vendor_fft_trusted(4096) is False
        finally:
            npu_fft._VENDOR_FFT_BAD.discard(4096)

    def test_env_override_can_widen_the_allowlist(self, monkeypatch):
        monkeypatch.setenv("PYMSS_VENDOR_FFT_SIZES", "1024,2048")
        assert vendor_fft_trusted(1024) is True
        assert vendor_fft_trusted(4096) is False

    def test_stft_npu_is_exact_for_an_untrusted_size(self):
        x = torch.randn(1, 1, 64)
        got = stft_npu(x, 16, 4, window=torch.hann_window(16))
        want = torch.stft(x.reshape(-1, 64), 16, 4, window=torch.hann_window(16), return_complex=True)
        assert torch.allclose(got, want, atol=2e-3)


class TestDegenerateVendorResult:
    """A silent all-zero vendor answer must be detected, not trusted."""

    def test_zero_output_from_nonzero_input_is_degenerate(self):
        from pymss_core.npu_fft import _degenerate

        assert _degenerate(torch.zeros(2, 3, 8), torch.randn(2, 3, 5)) is True

    def test_zero_input_is_not_degenerate(self):
        from pymss_core.npu_fft import _degenerate

        assert _degenerate(torch.zeros(2, 3, 8), torch.zeros(2, 3, 5)) is False

    def test_nonzero_output_is_fine(self):
        from pymss_core.npu_fft import _degenerate

        assert _degenerate(torch.ones(2, 3), torch.ones(2, 3)) is False


class TestExactFftAnyDim:
    """Models call torch.fft.rfft with a dim and a norm; both must be honoured."""

    @pytest.mark.parametrize(
        ("dim", "n", "norm"),
        [(3, None, "ortho"), (-1, 64, "backward"), (2, 32, None), (1, 16, "forward")],
    )
    def test_rfft_matches_torch(self, dim, n, norm):
        x = torch.randn(2, 5, 7, 9)
        assert torch.allclose(rfft_exact(x, n=n, dim=dim, norm=norm), torch.fft.rfft(x, n=n, dim=dim, norm=norm), atol=1e-4)

    @pytest.mark.parametrize(
        ("dim", "n", "norm"),
        [(3, None, "ortho"), (-1, 64, "backward"), (2, 32, "forward")],
    )
    def test_irfft_matches_torch(self, dim, n, norm):
        x = torch.randn(2, 5, 7, 9)
        spec = torch.fft.rfft(x, n=n, dim=dim, norm=norm)
        assert torch.allclose(irfft_exact(spec, n=n, dim=dim, norm=norm), torch.fft.irfft(spec, n=n, dim=dim, norm=norm), atol=1e-4)

    def test_fft_shim_is_idempotent(self):
        from pymss_core.vendor_shims import install_fft_shim, reset_shims

        reset_shims()
        assert install_fft_shim() is True
        assert install_fft_shim() is False
        reset_shims()


class TestRadix2Fft:
    """The DFT matrix costs 2.3 s per 4096-point transform and can fail in SUDNN."""

    @pytest.mark.parametrize("n", [16, 256, 1024, 2048, 4096])
    def test_rfft_matches_torch(self, n):
        x = torch.randn(2, 5, n)
        assert torch.allclose(_rfft_matmul(x, n), torch.fft.rfft(x, dim=-1), atol=1e-3)

    @pytest.mark.parametrize("n", [16, 256, 1024, 2048, 4096])
    def test_irfft_round_trips(self, n):
        x = torch.randn(2, 5, n)
        spec = torch.fft.rfft(x, dim=-1)
        assert torch.allclose(_irfft_matmul(spec, n), x, atol=1e-4)

    @pytest.mark.parametrize("n", [1000, 1411])
    def test_non_power_of_two_still_uses_the_matrix(self, n):
        x = torch.randn(2, 3, n)
        assert torch.allclose(_rfft_matmul(x, n), torch.fft.rfft(x, dim=-1), atol=2e-3)

    def test_radix2_matches_the_matrix_path(self):
        from pymss_core.npu_fft import _rfft_matmul_dft

        x = torch.randn(2, 3, 256)
        assert torch.allclose(_rfft_matmul(x, 256), _rfft_matmul_dft(x, 256), atol=1e-6)
