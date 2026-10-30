from __future__ import annotations

import os

import pytest
import torch

from pymss_core import DEVICE_ALIASES, DeviceProfile, device_profile
from pymss_core.device import (
    ALIAS_TO_VENDOR,
    ASCEND,
    BIREN,
    CPU,
    ENFLAME,
    HYGON,
    ILUVATAR,
    METAX,
    MLU,
    NVIDIA,
    ROCM,
    amp_supported,
    eager_conv_enabled,
    apply_vendor_env,
    dilated_conv_blocked,
    autocast_dtype,
    clear_device_cache,
    rnn_op_blocked,
    set_current_vendor_device,
    vendor_env_defaults,
    detect_vendor,
    fft_fallback_required,
    prepare_device_alias,
    profile_for_alias,
    requires_plugin,
    rnn_fallback_required,
)

VENDOR_DEVICE_NAMES = {
    "Hygon DCU K100_AI": HYGON,
    "Iluvatar BI-V150": ILUVATAR,
    "MetaX C500": METAX,
    "Enflame S60": ENFLAME,
    "Biren BR104": BIREN,
    "Cambricon MLU370": MLU,
    "NVIDIA A100-SXM4-40GB": NVIDIA,
}


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    clear_device_cache()
    for name in ("PYMSS_DEVICE_BACKEND", "PYMSS_FFT_FALLBACK", "PYMSS_RNN_FALLBACK", "PYMSS_AMP"):
        monkeypatch.delenv(name, raising=False)
    yield
    clear_device_cache()


def _fake_cuda(monkeypatch, name, available=True, count=1):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: available, raising=False)
    monkeypatch.setattr(torch.cuda, "device_count", lambda: count, raising=False)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda index=0: name, raising=False)


@pytest.mark.parametrize("device_name,vendor", sorted(VENDOR_DEVICE_NAMES.items()))
def test_detect_vendor_from_cuda_device_name(monkeypatch, device_name, vendor):
    _fake_cuda(monkeypatch, device_name)
    assert detect_vendor("cuda") == vendor


def test_detect_vendor_uses_private_device_types():
    assert detect_vendor("npu") == ASCEND
    assert detect_vendor("gcu") == ENFLAME
    assert detect_vendor("mlu") == MLU
    assert detect_vendor("supa") == BIREN
    assert detect_vendor("cpu") == CPU
    assert detect_vendor("mps") == "mps"


def test_rocm_falls_back_to_hip_version(monkeypatch):
    _fake_cuda(monkeypatch, "AMD Instinct MI300X")
    monkeypatch.setattr(torch.version, "hip", "6.2.41134", raising=False)
    assert detect_vendor("cuda") == ROCM


def test_unknown_cuda_device_is_treated_as_nvidia(monkeypatch):
    _fake_cuda(monkeypatch, "Generic Accelerator XYZ")
    monkeypatch.setattr(torch.version, "hip", None, raising=False)
    assert detect_vendor("cuda") == NVIDIA


def test_env_backend_forces_detection(monkeypatch):
    _fake_cuda(monkeypatch, "NVIDIA A100-SXM4-40GB")
    monkeypatch.setenv("PYMSS_DEVICE_BACKEND", "metax")
    assert detect_vendor("cuda") == METAX
    assert device_profile("cuda:0").vendor == METAX


def test_ascend_profile_ships_both_fallbacks():
    profile = device_profile("npu")
    assert profile.plugin == "torch_npu"
    assert profile.needs_fft_fallback and profile.needs_rnn_fallback
    assert not profile.supports_data_parallel


def test_domestic_cuda_impersonators_keep_native_operators(monkeypatch):
    _fake_cuda(monkeypatch, "Hygon DCU K100_AI")
    profile = device_profile("cuda:0")
    assert profile.vendor == HYGON
    assert profile.device_type == "cuda"
    assert not profile.needs_fft_fallback
    assert profile.is_accelerator


def test_env_flags_force_fallbacks(monkeypatch):
    _fake_cuda(monkeypatch, "Iluvatar BI-V150")
    assert not fft_fallback_required("cuda:0")
    monkeypatch.setenv("PYMSS_FFT_FALLBACK", "1")
    monkeypatch.setenv("PYMSS_RNN_FALLBACK", "1")
    clear_device_cache()
    assert fft_fallback_required("cuda:0")
    assert rnn_fallback_required("cuda:0")


def test_amp_flag_can_be_disabled(monkeypatch):
    monkeypatch.setenv("PYMSS_AMP", "0")
    clear_device_cache()
    assert not amp_supported("cuda")


def test_cpu_and_mps_are_not_accelerators():
    assert not device_profile("cpu").is_accelerator
    assert not device_profile("mps").is_accelerator


def test_alias_tables_are_consistent():
    assert set(ALIAS_TO_VENDOR) == set(DEVICE_ALIASES)
    for alias in DEVICE_ALIASES:
        profile = profile_for_alias(alias)
        assert isinstance(profile, DeviceProfile)


def test_private_device_aliases_require_their_plugin():
    assert requires_plugin("npu") == "torch_npu"
    assert requires_plugin("gcu") == "torch_gcu"
    assert requires_plugin("mlu") == "torch_mlu"
    assert requires_plugin("biren") == "torch_br"
    assert requires_plugin("hygon") is None


def test_prepare_device_alias_for_plugin_free_vendors():
    assert prepare_device_alias("cpu") == "cpu"
    assert prepare_device_alias("hygon") == "cuda"


def test_prepare_device_alias_reports_missing_plugin():
    with pytest.raises(ImportError):
        prepare_device_alias("gcu")


def test_profile_is_cached_until_cleared(monkeypatch):
    _fake_cuda(monkeypatch, "MetaX C500")
    first = device_profile("cuda")
    assert device_profile("cuda:1") is first
    clear_device_cache()
    _fake_cuda(monkeypatch, "Hygon DCU Z100L")
    assert device_profile("cuda").vendor == HYGON


def test_biren_profile_matches_the_measured_birensupa_stack():
    """BIREN106M + torch-br 1.7.0.20700: no complex stft, no correct fused RNN."""
    profile = device_profile("supa")
    assert profile.vendor == BIREN
    assert profile.device_type == "supa"
    assert profile.complex_fft is False
    # The fused _VF.lstm is silently wrong on this stack, so every cell bounces.
    assert profile.rnn is False
    assert profile.needs_rnn_fallback is True


def test_rnn_op_blocked_is_per_cell():
    # A device level rnn=False blocks every cell.
    assert rnn_op_blocked("supa", "gru") is True
    assert rnn_op_blocked("supa", "GRU") is True
    assert rnn_op_blocked("supa", "lstm") is True
    assert rnn_op_blocked("npu", "lstm") is True
    # Vendors with native recurrent kernels report nothing blocked.
    assert rnn_op_blocked("cuda", "gru") is False
    assert rnn_op_blocked("cuda", "lstm") is False


def test_rnn_blocked_selects_single_cells(monkeypatch):
    """rnn_blocked lets a vendor keep the working cells native."""
    from dataclasses import replace as dc_replace

    from pymss_core import device as device_mod

    partial = dc_replace(device_mod._PROFILES[BIREN], rnn=True, rnn_blocked=frozenset({"gru"}))
    monkeypatch.setitem(device_mod._PROFILES, BIREN, partial)
    device_mod.clear_device_cache()
    assert rnn_op_blocked("supa", "gru") is True
    assert rnn_op_blocked("supa", "lstm") is False
    device_mod.clear_device_cache()


def test_autocast_dtype_follows_the_profile():
    """BIREN br10x has no float16 matmul, so autocast must ask for bfloat16."""
    assert autocast_dtype("cuda") is torch.float16
    # bf16 is the reduced precision BIREN actually supports (no fp16 matmul).
    assert autocast_dtype("supa") is torch.bfloat16
    # amp=False devices fall back to plain float32.
    assert autocast_dtype("cpu") is torch.float32


def test_forced_vendor_does_not_reclassify_host_devices(monkeypatch):
    """PYMSS_DEVICE_BACKEND picks the accelerator, not CPU tensors."""
    monkeypatch.setenv("PYMSS_DEVICE_BACKEND", "biren")
    clear_device_cache()
    assert detect_vendor("supa") == BIREN
    assert detect_vendor("cpu") == CPU
    assert device_profile("cpu").rnn is True
    assert device_profile("cpu").needs_rnn_fallback is False
    clear_device_cache()


def test_biren_vendor_env_is_applied_before_the_plugin_import(monkeypatch):
    """The SUDNN heuristic search costs ~220 s per new conv shape; skip it."""
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", raising=False)
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", raising=False)
    monkeypatch.delenv("PYMSS_VENDOR_ENV", raising=False)
    clear_device_cache()

    expected = {"BRTB_ENABLE_FORCE_EAGER_CONV2D": "1", "BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE": "1"}
    assert vendor_env_defaults("biren") == expected
    assert apply_vendor_env("biren") == expected
    assert os.environ["BRTB_ENABLE_FORCE_EAGER_CONV2D"] == "1"


def test_vendor_env_does_not_override_user_choices(monkeypatch):
    monkeypatch.setenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", "0")
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", raising=False)
    monkeypatch.delenv("PYMSS_VENDOR_ENV", raising=False)
    assert apply_vendor_env("biren") == {"BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE": "1"}
    assert os.environ["BRTB_ENABLE_FORCE_EAGER_CONV2D"] == "0"


def test_vendor_env_can_be_disabled(monkeypatch):
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", raising=False)
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2DTRANSPOSE", raising=False)
    monkeypatch.setenv("PYMSS_VENDOR_ENV", "0")
    assert apply_vendor_env("biren") == {}
    assert "BRTB_ENABLE_FORCE_EAGER_CONV2D" not in os.environ


def test_vendor_env_is_empty_for_other_vendors(monkeypatch):
    monkeypatch.delenv("PYMSS_VENDOR_ENV", raising=False)
    assert vendor_env_defaults("nvidia") == {}
    assert apply_vendor_env("cuda") == {}


def test_current_vendor_device_is_set_for_privateuse1_backends(monkeypatch):
    """A device spec without an index uses the *current* device, which nothing sets."""
    monkeypatch.setenv("PYMSS_DEVICE_INDEX", "0")
    monkeypatch.delenv("PYMSS_VENDOR_ENV", raising=False)
    clear_device_cache()

    class FakeVendor:
        calls: list[int] = []

        @classmethod
        def set_device(cls, index):
            cls.calls.append(index)

    monkeypatch.setattr(torch, "fakevendor", FakeVendor, raising=False)
    assert set_current_vendor_device("fakevendor") == 0
    assert FakeVendor.calls == [0]
    assert set_current_vendor_device("fakevendor", index=2) == 2
    assert FakeVendor.calls == [0, 2]


def test_current_vendor_device_is_optional():
    assert set_current_vendor_device("cpu") is None


def test_dilated_convs_run_on_biren_when_eager_kernels_are_active(monkeypatch):
    """With eager conv kernels every asymmetric dilation measured passes."""
    monkeypatch.setenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    for dilation in ((1, 2), (2, 1), (1, 3), (3, 1), (4, 4)):
        assert dilated_conv_blocked("supa", dilation) is False


def test_dilated_convs_still_bounce_without_the_eager_kernels(monkeypatch):
    """The SUDNN plan path aborts the process; keep the safety net."""
    monkeypatch.delenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", raising=False)
    assert dilated_conv_blocked("supa", (1, 2)) is True
    assert dilated_conv_blocked("supa", (2, 2)) is False
    assert dilated_conv_blocked("supa", (4, 4)) is True


def test_eager_conv_enabled_parses_falsey_values(monkeypatch):
    monkeypatch.setenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", "0")
    assert eager_conv_enabled() is False
    monkeypatch.setenv("BRTB_ENABLE_FORCE_EAGER_CONV2D", "1")
    assert eager_conv_enabled() is True





def test_biren_uses_bfloat16_autocast(monkeypatch):
    """bf16 is the working reduced precision; only one op needs a shim."""
    monkeypatch.delenv("PYMSS_AMP", raising=False)
    clear_device_cache()
    profile = device_profile("supa")
    assert profile.amp is True
    assert profile.amp_dtype is torch.bfloat16
    assert profile.interpolate_shim is True


def test_prepare_device_installs_the_rnn_fallback(monkeypatch):
    """A profile that says rnn=False must be made usable by prepare_device_alias."""
    import pymss_core.device as dev
    import pymss_core.npu_rnn as rnn

    fake = dev.DeviceProfile(
        dev.BIREN, "cpu", plugin=None, complex_fft=False, rnn=False, amp=True,
        amp_dtype=torch.bfloat16, channels_last=False, multi_device=False,
    )
    calls = []
    monkeypatch.setattr(dev, "profile_for_alias", lambda alias: fake)
    monkeypatch.setattr(dev, "set_current_vendor_device", lambda device_type: None)
    monkeypatch.setattr(dev, "device_profile", lambda device=None: fake)
    # Spy instead of installing: patching nn.LSTM globally would leak into other tests.
    monkeypatch.setattr(rnn, "install_rnn_cpu_fallback", lambda: calls.append("rnn"))

    dev.prepare_device_alias("biren")

    assert calls == ["rnn"]


class TestRotaryPathChoice:
    """The rotated layout is permuted; scores must still match the reference."""

    @staticmethod
    def _reference(cos, sin, t):
        rot = torch.complex(cos, sin)
        rotated = torch.view_as_complex(t.reshape(*t.shape[:-1], -1, 2).contiguous()) * rot
        return torch.view_as_real(rotated).reshape(t.shape)

    def test_permuted_layout_matches_interleaved(self):
        import torch

        from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast

        torch.manual_seed(0)
        t = torch.randn(2, 3, 4, 8)
        cos = torch.randn(1, 3, 1, 4)
        sin = torch.randn(1, 3, 1, 4)
        got = apply_rotary_emb_fast(cos, sin, t)
        want = self._reference(cos, sin, t)
        permuted = torch.cat((want[..., ::2], want[..., 1::2]), dim=-1)
        assert torch.allclose(got, permuted, atol=1e-5)

    def test_attention_scores_are_unchanged(self):
        import torch

        from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast

        torch.manual_seed(0)
        q = torch.randn(2, 3, 4, 8)
        k = torch.randn(2, 3, 4, 8)
        cos = torch.randn(1, 3, 1, 4)
        sin = torch.randn(1, 3, 1, 4)
        reference = self._reference(cos, sin, q) @ self._reference(cos, sin, k).transpose(-1, -2)
        permuted = apply_rotary_emb_fast(cos, sin, q) @ apply_rotary_emb_fast(cos, sin, k).transpose(-1, -2)
        assert torch.allclose(reference, permuted, atol=1e-5)

    def test_handles_non_contiguous_input(self):
        import torch

        from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast

        torch.manual_seed(0)
        base = torch.randn(2, 4, 3, 8)
        t = base.transpose(1, 2)
        cos = torch.randn(1, 3, 1, 4)
        sin = torch.randn(1, 3, 1, 4)
        got = apply_rotary_emb_fast(cos, sin, t)
        want = self._reference(cos, sin, t)
        permuted = torch.cat((want[..., ::2], want[..., 1::2]), dim=-1)
        assert torch.allclose(got, permuted, atol=1e-5)
