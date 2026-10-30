import importlib

import pytest
import torch
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, PackedSequence

import pymss_core.npu_rnn as npu_rnn
from _devices import DEVICES


@pytest.fixture
def patched_runtime(monkeypatch):
    for cls in (nn.LSTM, nn.GRU, nn.RNN):
        monkeypatch.setattr(cls, "forward", cls.forward)
    monkeypatch.setattr(npu_rnn, "_installed", False)
    yield npu_rnn


@pytest.mark.parametrize("kind", [nn.LSTM, nn.GRU, nn.RNN])
@pytest.mark.parametrize("subclass", [False, True])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("device", DEVICES)
def test_install_preserves_native_forward_and_gradients(patched_runtime, kind, subclass, packed, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    cls = type("Child", (kind,), {}) if subclass else kind
    model = cls(3, 4, batch_first=True, bidirectional=True).to(device)
    wave = torch.randn(2, 5, 3, device=device, requires_grad=True)
    input = pack_padded_sequence(wave, [5, 3], batch_first=True, enforce_sorted=False) if packed else wave
    hidden = torch.zeros(2, 2, 4, device=device)
    hx = (hidden, hidden.clone()) if kind is nn.LSTM else hidden
    expected, _ = model(input, hx)
    patched_runtime.install_npu_rnn_fallback()
    actual, _ = model(input, hx)
    expected_data = expected.data if packed else expected
    actual_data = actual.data if packed else actual
    torch.testing.assert_close(actual_data, expected_data)
    actual_data.sum().backward()
    assert wave.grad is not None and torch.isfinite(wave.grad).all()


@pytest.mark.parametrize("kind", [nn.LSTM, nn.GRU, nn.RNN])
@pytest.mark.parametrize("packed", [False, True])
def test_cpu_fallback_preserves_packed_and_dense_contract(patched_runtime, monkeypatch, kind, packed):
    model = kind(3, 4, batch_first=True, bidirectional=True).eval()
    wave = torch.randn(2, 5, 3)
    input = pack_padded_sequence(wave, [3, 5], batch_first=True, enforce_sorted=False) if packed else wave
    hidden = torch.randn(2, 2, 4)
    hx = (hidden, hidden.clone()) if kind is nn.LSTM else hidden
    with torch.inference_mode():
        expected, expected_hx = model(input, hx)
    patched_runtime.install_npu_rnn_fallback()
    monkeypatch.setattr(patched_runtime, "_needs_cpu_fallback", lambda value: True)
    with torch.inference_mode():
        actual, actual_hx = model(input, hx)
    if packed:
        assert isinstance(actual, PackedSequence)
        torch.testing.assert_close(actual.data, expected.data)
        torch.testing.assert_close(actual.batch_sizes, expected.batch_sizes)
        torch.testing.assert_close(actual.sorted_indices, expected.sorted_indices)
    else:
        torch.testing.assert_close(actual, expected)
    torch.testing.assert_close(actual_hx, expected_hx)


def test_fallback_does_not_repeat_subclass_forward_or_hooks(patched_runtime, monkeypatch):
    class Child(nn.LSTM):
        def forward(self, input, hx=None):
            output, hidden = super().forward(input, hx)
            return output * 2, hidden

    model = Child(3, 4, batch_first=True).eval()
    wave = torch.randn(2, 5, 3)
    with torch.inference_mode():
        expected, _ = model(wave)
    hooks = []
    handle = model.register_forward_hook(lambda *args: hooks.append(1))
    patched_runtime.install_npu_rnn_fallback()
    monkeypatch.setattr(patched_runtime, "_needs_cpu_fallback", lambda value: True)
    with torch.inference_mode():
        actual, _ = model(wave)
    handle.remove()
    torch.testing.assert_close(actual, expected)
    assert hooks == [1]


def test_reinstall_after_reload_preserves_original_forward(patched_runtime):
    model = nn.LSTM(3, 4)
    wave = torch.randn(5, 2, 3)
    expected, _ = model(wave)
    patched_runtime.install_npu_rnn_fallback()
    first = nn.LSTM.forward
    patched_runtime.install_npu_rnn_fallback()
    assert nn.LSTM.forward is first
    importlib.reload(patched_runtime)
    patched_runtime.install_npu_rnn_fallback()
    actual, _ = model(wave)
    torch.testing.assert_close(actual, expected)


def test_npu_fallback_rejects_training_instead_of_detaching_gradients(patched_runtime, monkeypatch):
    patched_runtime.install_npu_rnn_fallback()
    monkeypatch.setattr(patched_runtime, "_needs_cpu_fallback", lambda value: True)
    model = nn.GRU(3, 4)
    with pytest.raises(RuntimeError, match="inference"):
        model(torch.randn(5, 2, 3, requires_grad=True))


@pytest.mark.parametrize("kind", [nn.LSTM, nn.GRU, nn.RNN])
@pytest.mark.parametrize("packed", [False, True])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("low_precision_weights", [False, True])
def test_cpu_fallback_handles_amp_inputs(patched_runtime, monkeypatch, kind, packed, dtype, low_precision_weights):
    model = kind(3, 4, batch_first=True).eval()
    if low_precision_weights:
        model = model.to(dtype=dtype)
    reference = kind(3, 4, batch_first=True).eval()
    reference.load_state_dict({name: value.float() for name, value in model.state_dict().items()})
    wave = torch.randn(2, 5, 3).to(dtype=dtype)
    input = pack_padded_sequence(wave, [3, 5], batch_first=True, enforce_sorted=False) if packed else wave
    reference_input = input.to(dtype=torch.float32)
    with torch.inference_mode():
        expected, expected_hidden = reference(reference_input)
    original_dtype = model.weight_ih_l0.dtype
    patched_runtime.install_npu_rnn_fallback()
    monkeypatch.setattr(patched_runtime, "_needs_cpu_fallback", lambda value: True)
    with torch.inference_mode():
        actual, actual_hidden = model(input)
    actual_data = actual.data if packed else actual
    expected_data = expected.data if packed else expected
    assert actual_data.dtype == dtype
    torch.testing.assert_close(actual_data, expected_data.to(dtype=dtype))
    if kind is nn.LSTM:
        torch.testing.assert_close(actual_hidden, tuple(value.to(dtype=dtype) for value in expected_hidden))
    else:
        torch.testing.assert_close(actual_hidden, expected_hidden.to(dtype=dtype))
    assert model.weight_ih_l0.dtype == original_dtype


def test_blocked_cell_falls_back_while_native_cell_stays_on_device(patched_runtime, monkeypatch):
    """BIREN ships native LSTM but no fused GRU cell: only the GRU bounces."""
    monkeypatch.setattr(npu_rnn, "_needs_cpu_fallback", lambda value: False)
    monkeypatch.setattr(npu_rnn, "rnn_op_blocked", lambda device, op: op == "gru")
    calls = []
    original = npu_rnn._NpuRnnCpuFallback.forward

    def spy(self, input, hx=None):
        calls.append(type(self._rnn).__name__)
        return original(self, input, hx)

    monkeypatch.setattr(npu_rnn._NpuRnnCpuFallback, "forward", spy)
    wave = torch.randn(2, 5, 3)
    torch.manual_seed(0)
    lstm = nn.LSTM(3, 4, batch_first=True)
    gru = nn.GRU(3, 4, batch_first=True)
    with torch.no_grad():
        expected_lstm, _ = lstm(wave)
        expected_gru, _ = gru(wave)
    patched_runtime.install_npu_rnn_fallback()
    with torch.no_grad():
        actual_lstm, _ = lstm(wave)
        actual_gru, _ = gru(wave)
    torch.testing.assert_close(actual_lstm, expected_lstm)
    torch.testing.assert_close(actual_gru, expected_gru)
    assert calls == ["GRU"]
