from types import SimpleNamespace
import sys

import pytest
import torch
import torch.nn.functional as F

from pymss_core import device
from pymss_core import vendor_shims as shims


@pytest.fixture(autouse=True)
def reset_state():
    device.clear_device_cache()
    shims.reset_shims()
    yield
    shims.reset_shims()
    device.clear_device_cache()


def test_musa_aliases_share_a_conservative_profile():
    profile = device.profile_for_alias('musa')
    assert profile is not None
    assert device.profile_for_alias('moore') is profile
    assert (profile.vendor, profile.device_type, profile.plugin) == ('moore', 'musa', 'torch_musa')
    assert device.device_profile('musa:0').vendor == 'moore'
    assert not profile.amp and not profile.multi_device
    assert profile.fft_dtype_shim and not profile.fft_shim
    assert not profile.scatter_add and not profile.rope_shim
    assert profile.rope_kernel_shim


def test_moore_profile_matches_the_measured_stack():
    """Pins the flags measured on MTT S70 + torch_musa 2.7.1 (rationale in device.py).

    Native on that stack: complex STFT/ISTFT, rfft/irfft, complex-by-real
    multiply/divide, complex cat/fill/copy, ``_thnn_fused_gru_cell`` and
    ``_native_multi_head_attention``.  Only the gaps are overridden.
    """
    profile = device.profile_for_alias('musa')
    assert profile.complex_fft and not profile.fft_shim
    assert profile.complex_mul and profile.complex_reshape and not profile.complex_cat
    assert profile.rnn and not profile.rnn_blocked and not profile.sdpa_shim
    # fp16 muFFT is a hardware gate below capability (2, 2); complex
    # scatter_add_/index_add_ are missing in every released torch_musa.
    assert profile.fft_dtype_shim and not profile.scatter_add
    # fp16 autocast aborts in muDNN's fused conv; no MCCL on these desktop cards.
    assert not profile.amp and not profile.multi_device
    # Acceleration only: the reference rotary path is correct without it.
    assert profile.rope_kernel_shim and not profile.rope_shim

def test_prepare_musa_imports_plugin_and_selects_current_device(monkeypatch):
    selected = []
    module = SimpleNamespace(is_available=lambda: True, device_count=lambda: 1, set_device=selected.append)
    monkeypatch.setattr(torch, 'musa', module, raising=False)
    monkeypatch.setitem(sys.modules, 'torch_musa', module)
    monkeypatch.delenv('PYMSS_DEVICE_INDEX', raising=False)
    assert device.prepare_device_alias('moore') == 'musa'
    assert selected == [0]
    assert 'fft_dtype' in shims.installed_shims()


def test_unavailable_musa_does_not_install_global_shims(monkeypatch):
    module = SimpleNamespace(is_available=lambda: False)
    monkeypatch.setattr(torch, 'musa', module, raising=False)
    monkeypatch.setitem(sys.modules, 'torch_musa', module)
    assert device.prepare_device_alias('musa') == 'musa'
    assert not shims.installed_shims()


@pytest.mark.parametrize('available', [False, True])
def test_prepare_checks_availability_on_plugin_without_torch_namespace(monkeypatch, available):
    module = SimpleNamespace(is_available=lambda: available)
    monkeypatch.delattr(torch, 'xla', raising=False)
    monkeypatch.setitem(sys.modules, 'torch_gcu', module)
    assert device.prepare_device_alias('enflame') == 'xla'
    assert bool(shims.installed_shims()) is available


def test_unknown_alias_preserves_the_public_extension_contract():
    assert device.prepare_device_alias('custom_backend') == 'custom_backend'


def test_fft_dtype_shim_only_promotes_registered_backends(monkeypatch):
    calls = []

    class Operand:
        def __init__(self, kind, dtype=torch.float16):
            self.device = SimpleNamespace(type=kind)
            self.dtype = dtype

        def float(self):
            return Operand(self.device.type, torch.float32)

        def is_complex(self):
            return False

        def is_floating_point(self):
            return True

        def to(self, dtype):
            return Operand(self.device.type, dtype)

    def native(operand, *args, **kwargs):
        calls.append((operand.device.type, operand.dtype))
        return operand

    monkeypatch.setattr(torch.fft, 'rfft', native)
    shims.install_fft_dtype_shim('musa')
    wrapper = torch.fft.rfft
    wrapper(Operand('cuda'))
    wrapper(Operand('musa'))
    assert calls == [('cuda', torch.float16), ('musa', torch.float32)]
    shims.install_fft_dtype_shim('cuda')
    assert torch.fft.rfft is wrapper
    wrapper(Operand('cuda'))
    assert calls[-1] == ('cuda', torch.float32)


@pytest.mark.parametrize('installer', [
    shims.install_complex_mul_shim,
    shims.install_window_shim,
    shims.install_glu_shim,
    shims.install_complex_reshape_shim,
    shims.install_group_norm_shim,
    shims.install_interpolate_shim,
    shims.install_conv1d_shim,
    shims.install_complex_cat_shim,
    shims.install_fft_shim,
    shims.install_sdpa_shim,
    shims.install_cpu_shim,
    shims.install_std_shim,
    shims.install_fft_dtype_shim,
    shims.install_embedding_shim,
    shims.install_matmul_guard_shim,
    shims.install_conv_transpose1d_shim,
    shims.install_layernorm_module_shim,
    shims.install_layer_norm_shim,
    shims.install_linear_dtype_shim,
    shims.install_fold_shim,
    shims.install_complex_view_shim,
    lambda: shims.install_as_strided_shim('supa'),
    lambda: shims.install_unfold_shim('supa'),
    shims.install_rope_kernel_shim,
])
def test_reset_restores_all_patched_targets(installer):
    missing = object()
    targets = [(torch.Tensor, n) for n in ('__mul__', '__rmul__', 'reshape', 'view', 'std', 'cpu', 'as_strided', 'unfold')]
    targets += [(torch, n) for n in ('mul', 'hann_window', 'hamming_window', 'blackman_window', 'bartlett_window',
                                   'cat', 'matmul', 'as_strided', 'view_as_complex', 'stft', 'istft', 'scaled_dot_product_attention')]
    targets += [(F, n) for n in ('glu', 'group_norm', 'interpolate', 'conv1d', 'scaled_dot_product_attention',
                                'embedding', 'conv_transpose1d', 'layer_norm', 'linear', 'fold', 'col2im')]
    targets += [(torch.nn.LayerNorm, 'forward')]
    targets += [(torch.fft, n) for n in ('rfft', 'irfft', 'fft', 'ifft', 'hfft', 'ihfft')]
    originals = [getattr(obj, name, missing) for obj, name in targets]
    for _ in range(2):
        installer()
        shims.reset_shims()
        assert not shims.installed_shims()
        for (obj, name), original in zip(targets, originals):
            assert getattr(obj, name, missing) is original, name


def test_reset_preserves_a_replacement_installed_by_another_library(monkeypatch):
    original = F.interpolate
    shims.install_interpolate_shim()

    def foreign(*args, **kwargs):
        return original(*args, **kwargs)

    foreign._pymss_original = original
    monkeypatch.setattr(F, 'interpolate', foreign)
    shims.reset_shims()
    assert F.interpolate is foreign
