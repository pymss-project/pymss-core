"""Tests for the dilated-conv CPU fallback."""
from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

import pymss_core.npu_conv as npu_conv
from pymss_core.npu_conv import replicate_pad_bins


def test_undilated_conv_never_falls_back():
    conv = torch.nn.Conv2d(4, 6, 3, padding=1)
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is False


def test_dilated_conv_is_native_on_supported_devices():
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    # Plain CPU tensors resolve to the CPU profile, which supports dilation.
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is False


def test_dilated_conv_falls_back_when_the_profile_blocks_it(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    assert npu_conv.needs_cpu_fallback(conv, torch.randn(1, 4, 8, 8)) is True
    conv1d = torch.nn.Conv1d(4, 6, 3, padding=2, dilation=2)
    assert npu_conv.needs_cpu_fallback(conv1d, torch.randn(1, 4, 16)) is True
    flat = torch.nn.Conv1d(4, 6, 3, padding=1)
    assert npu_conv.needs_cpu_fallback(flat, torch.randn(1, 4, 16)) is False


def test_cpu_fallback_matches_the_reference(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    torch.manual_seed(0)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2)).eval()
    x = torch.randn(2, 4, 21, 12)
    with torch.no_grad():
        expected = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        actual = npu_conv._conv_on_cpu(conv, x)
    torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)


def test_cpu_fallback_refuses_training(monkeypatch):
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2))
    with pytest.raises(RuntimeError, match="inference only"):
        npu_conv._conv_on_cpu(conv, torch.randn(1, 4, 8, 8, requires_grad=True))


def test_install_is_idempotent_and_keeps_the_original(monkeypatch):
    monkeypatch.setattr(npu_conv, "_installed", False)
    from torch.nn.modules.conv import _ConvNd

    original = _ConvNd.forward
    try:
        npu_conv.install_dilated_conv_fallback()
        patched = _ConvNd.forward
        npu_conv.install_dilated_conv_fallback()
        assert _ConvNd.forward is patched
        assert patched._pymss_conv_original_forward is original
    finally:
        _ConvNd.forward = original


def test_installed_fallback_runs_the_blocked_layer_on_cpu(monkeypatch):
    monkeypatch.setattr(npu_conv, "_installed", False)
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    from torch.nn.modules.conv import _ConvNd

    original = _ConvNd.forward
    torch.manual_seed(1)
    conv = torch.nn.Conv2d(4, 6, 3, padding=(4, 2), dilation=(4, 2)).eval()
    x = torch.randn(2, 4, 21, 12)
    try:
        with torch.no_grad():
            expected = F.conv2d(x, conv.weight, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)
        npu_conv.install_dilated_conv_fallback()
        with torch.no_grad():
            actual = conv(x)
        torch.testing.assert_close(actual, expected, rtol=1e-6, atol=1e-6)
    finally:
        _ConvNd.forward = original


def test_functional_conv_is_hooked_too(monkeypatch):
    """Some models bypass nn.Conv2d and call F.conv2d directly."""
    import torch.nn.functional as F

    monkeypatch.setattr(npu_conv, "_installed", False)
    monkeypatch.setattr(npu_conv, "dilated_conv_blocked", lambda device, dilation: True)
    from torch.nn.modules.conv import _ConvNd

    original_module_forward = _ConvNd.forward
    original_conv2d = F.conv2d
    torch.manual_seed(2)
    weight = torch.randn(6, 4, 3, 3)
    x = torch.randn(2, 4, 21, 12)
    try:
        npu_conv.install_dilated_conv_fallback()
        expected = torch.conv2d(x, weight, None, 1, (4, 2), (4, 2), 1)
        with_torch = torch.conv2d(x, weight, None, 1, (4, 2), (4, 2), 1) == expected
        with_f = F.conv2d(x, weight, None, 1, (4, 2), (4, 2), 1)
        assert with_torch.all() and torch.allclose(with_f, expected)
        # Undilated calls must still reach the original implementation.
        assert F.conv2d(x, weight, None, 1, 1, 1, 1).shape == (2, 6, 21, 12)
    finally:
        _ConvNd.forward = original_module_forward
        F.conv2d = original_conv2d


def test_blocked_predicate_keeps_the_dilations_the_kernel_supports(monkeypatch):
    import pymss_core.device as device_mod
    from pymss_core.device import BIREN, dilated_conv_blocked

    monkeypatch.setitem(device_mod._PROFILES, BIREN, device_mod._PROFILES[BIREN])
    # This asserts the SUDNN safety net, i.e. the eager kernels are off.
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", raising=False)
    device_mod.clear_device_cache()
    # (1,1) and (2,2) are native on BIREN; asymmetric and (4,4) are not.
    assert dilated_conv_blocked("supa", (1, 1)) is False
    assert dilated_conv_blocked("supa", (2, 2)) is False
    assert dilated_conv_blocked("supa", (4, 4)) is True
    assert dilated_conv_blocked("supa", (4, 2)) is True
    assert dilated_conv_blocked("supa", (2, 1)) is True
    # Vendors without the limitation never bounce.
    assert dilated_conv_blocked("cuda", (4, 2)) is False
    device_mod.clear_device_cache()


class TestReplicatePadBins:
    """F.pad(mode="replicate") maps to replication_pad2d, which SUPA fails on."""

    def test_matches_torch_replicate_pad(self):
        torch.manual_seed(0)
        x = torch.randn(2, 3, 5, 7)
        got = replicate_pad_bins(x, 9)
        want = F.pad(x, (0, 0, 0, 4), mode="replicate")
        assert got.shape == want.shape
        assert torch.equal(got, want)

    def test_no_padding_returns_input_unchanged(self):
        x = torch.randn(1, 2, 5, 3)
        assert replicate_pad_bins(x, 5) is x
        assert replicate_pad_bins(x, 4) is x

    def test_pads_the_requested_dimension(self):
        x = torch.arange(2 * 3 * 4).reshape(1, 2, 3, 4).float()
        got = replicate_pad_bins(x, 5, dim=3)
        assert got.shape == (1, 2, 3, 5)
        assert torch.equal(got[..., -1], x[..., -1])
