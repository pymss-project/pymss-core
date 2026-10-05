from __future__ import annotations
import copy
import pytest
import torch
from parity import cases, _roformer_kwargs
from pymss_core import clear_model_runtime_caches, is_directml_device
from pymss_core import dml_backend


def test_device_detection_uses_torch_backend_and_not_gpu_vendor():
    assert is_directml_device(torch.device("privateuseone:0"))
    assert is_directml_device("privateuseone:2")
    assert not is_directml_device(torch.device("cpu"))
    assert not is_directml_device(torch.zeros(1))
    assert not is_directml_device("cuda:0")


def test_inference_checkpoint_preserves_cpu_views_without_transfer(monkeypatch):
    def unexpected_transfer(value): raise AssertionError("CPU checkpoints must not transfer tensors")
    value = torch.arange(24, dtype=torch.float64).reshape(2, 3, 4)[:, 1:, ::2]
    expected, version = value.clone(), value._version
    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_transfer)
    with torch.no_grad(): actual = dml_backend.inference_checkpoint(value)
    assert actual is value and actual._version == version and not actual.is_contiguous()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_inference_checkpoint_preserves_training_gradients(monkeypatch):
    def unexpected_transfer(value): raise AssertionError("Training checkpoints must not transfer tensors")
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_transfer)
    source = torch.tensor([1., 2., 3., 4.], dtype=torch.float64, requires_grad=True)
    value = source.square()[::2]
    actual = dml_backend.inference_checkpoint(value)
    assert actual is value and actual.grad_fn is value.grad_fn
    actual.sum().backward()
    torch.testing.assert_close(source.grad, torch.tensor([2., 0., 6., 0.], dtype=torch.float64), rtol=0, atol=0)


def test_inference_checkpoint_reads_only_one_scalar_and_preserves_output(monkeypatch):
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value = source[:, 1:, ::2]
    expected, version, transfers = value.clone(), value._version, []
    native_cpu = torch.Tensor.cpu
    def capture_transfer(tensor): transfers.append((tensor.ndim, tensor.numel(), tensor.requires_grad)); return native_cpu(tensor)
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    monkeypatch.setattr(torch.Tensor, "cpu", capture_transfer)
    with torch.no_grad(): actual = dml_backend.inference_checkpoint(value)
    assert transfers == [(0, 1, False)]
    assert actual is value and actual.dtype == torch.float32 and actual._version == version
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("value", [torch.empty(0), torch.ones(2, dtype=torch.complex64), torch.ones(2, dtype=torch.int64), None])
def test_inference_checkpoint_skips_empty_and_non_real_values(value, monkeypatch):
    def unexpected_transfer(value): raise AssertionError("Unsupported checkpoints must not transfer tensors")
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    monkeypatch.setattr(torch.Tensor, "cpu", unexpected_transfer)
    with torch.no_grad(): assert dml_backend.inference_checkpoint(value) is value


def _strided_attention_inputs(dtype, layout, queries=13, keys=17):
    leading = {"batched": ((2, 3), (2, 3), (2, 3)), "broadcast": ((2, 1), (1, 3), (2, 3)), "value_broadcast": ((2, 1), (1, 1), (1, 3)), "matrix": ((), (), ())}[layout]
    return tuple(torch.randn(*shape, length, width * 2, dtype=dtype)[..., 1::2] for shape, length, width in zip(leading, (queries, keys, keys), (8, 8, 6)))


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("layout,queries", [("batched", 13), ("broadcast", 13), ("value_broadcast", 13), ("matrix", 43)])
@pytest.mark.parametrize("is_causal,scale", [(False, None), (True, None), (False, 0.17), (True, 0.17)])
def test_chunked_attention_matches_native_with_strided_broadcast_rectangular_inputs(dtype, layout, queries, is_causal, scale, monkeypatch):
    torch.manual_seed(7)
    q, k, v = _strided_attention_inputs(dtype, layout, queries)
    assert all(not value.is_contiguous() for value in (q, k, v))
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 2560)
    with torch.no_grad():
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=is_causal, scale=scale)
        actual = dml_backend._chunked_attention(q, k, v, is_causal=is_causal, scale=scale)
    assert actual.device == q.device and actual.dtype == dtype and actual.shape == expected.shape
    tolerance = 2e-6 if dtype == torch.float32 else 1e-12
    torch.testing.assert_close(actual, expected, rtol=tolerance, atol=tolerance)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("queries,keys", [(0, 17), (13, 0), (0, 0)])
def test_chunked_attention_preserves_empty_query_and_key_shapes(dtype, queries, keys, monkeypatch):
    q, k, v = _strided_attention_inputs(dtype, "broadcast", queries, keys)
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 2560)
    with torch.no_grad():
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True)
        actual = dml_backend._chunked_attention(q, k, v, is_causal=True)
    assert actual.shape == (2, 3, queries, 6) and actual.device == q.device and actual.dtype == dtype
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_scaled_attention_preserves_training_gradients_without_chunking(monkeypatch):
    def unexpected_chunking(*args, **kwargs): raise AssertionError("Gradient-enabled attention must use the dense path")
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 1)
    monkeypatch.setattr(dml_backend, "_chunked_attention", unexpected_chunking)
    torch.manual_seed(11)
    inputs = tuple(torch.randn(2, length, width, dtype=torch.float64, requires_grad=True) for length, width in ((7, 4), (5, 4), (5, 6)))
    actual = dml_backend.scaled_dot_product_attention(*inputs, is_causal=True, scale=0.17)
    expected = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=True, scale=0.17)
    actual_gradients = torch.autograd.grad(actual.square().sum(), inputs)
    expected_gradients = torch.autograd.grad(expected.square().sum(), inputs)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    torch.testing.assert_close(actual_gradients, expected_gradients, rtol=1e-12, atol=1e-12)


def test_scaled_attention_preserves_dropout_rng_without_chunking(monkeypatch):
    def unexpected_chunking(*args, **kwargs): raise AssertionError("Attention dropout must use the dense path")
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 1)
    monkeypatch.setattr(dml_backend, "_chunked_attention", unexpected_chunking)
    q, k = torch.zeros(2, 1, 4, dtype=torch.float64), torch.zeros(2, 5, 4, dtype=torch.float64)
    v = torch.eye(5, dtype=torch.float64).expand(2, -1, -1)
    with torch.no_grad():
        torch.manual_seed(42)
        expected = torch.nn.functional.dropout(torch.full((2, 1, 5), 0.2, dtype=torch.float64), p=0.25, training=True)
        torch.manual_seed(42)
        actual = dml_backend.scaled_dot_product_attention(q, k, v, dropout_p=0.25)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_scaled_attention_keeps_native_cpu_backend_without_chunking(monkeypatch):
    def unexpected_chunking(*args, **kwargs): raise AssertionError("Native CPU attention must retain its backend")
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 1)
    monkeypatch.setattr(dml_backend, "_chunked_attention", unexpected_chunking)
    q, k, v = _strided_attention_inputs(torch.float32, "broadcast")
    with torch.no_grad():
        actual = dml_backend.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.17)
        expected = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.17)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("training", [False, True])
def test_directml_attend_route_preserves_scale_and_training_dropout(training, monkeypatch):
    from pymss_core.modules.bs_roformer import attend as attend_module
    module = attend_module.Attend(flash=False, dropout=0.25, scale=0.17).train(training)
    torch.manual_seed(7)
    inputs = _strided_attention_inputs(torch.float64, "batched")
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 2560)
    monkeypatch.setattr(attend_module, "is_directml_device", lambda value: False)
    with torch.no_grad():
        torch.manual_seed(42)
        reference = module(*inputs)
    monkeypatch.setattr(attend_module, "is_directml_device", lambda value: True)
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    with torch.no_grad():
        torch.manual_seed(42)
        actual = module(*inputs)
    torch.testing.assert_close(actual, reference, rtol=1e-12, atol=1e-12)


def test_native_complex_helpers_preserve_double_precision():
    real, imag = torch.randn(8, dtype=torch.float64), torch.randn(8, dtype=torch.float64)
    spectrum = dml_backend.complex_from_parts(real, imag)
    assert spectrum.dtype == torch.complex128
    torch.testing.assert_close(dml_backend.real_to_complex(torch.view_as_real(spectrum)), spectrum)


def test_spectral_context_keeps_its_existing_tuple_contract():
    from pymss_core.modules.bs_roformer.common import SpectralContext
    window = torch.ones(16)
    context = SpectralContext(1, 2, 9, 64, window, False, torch.device("privateuseone:0"))
    assert tuple(context) == (1, 2, 9, 64, window, False)
    assert context.device == torch.device("privateuseone:0")


def test_bandit_nonoverlapping_masks_construct_and_run():
    from pymss_core.modules.bandit.maskestim import MaskEstimationModule
    module = MaskEstimationModule(band_specs=[(0, 2), (2, 4)], emb_dim=4, mlp_dim=4, in_channel=1)
    assert module(torch.randn(1, 2, 3, 4)).shape == (1, 1, 4, 3)


@pytest.mark.parametrize("kind,batch_first,bidirectional,layers,projection,bias", [
    ("LSTM", False, False, 1, 0, True),
    ("LSTM", True, True, 2, 0, False),
    ("LSTM", False, True, 2, 2, True),
    ("GRU", True, True, 2, 0, True),
    ("RNN", False, False, 2, 0, False),
])
def test_recurrent_real_gate_equations_match_native_cpu(kind, batch_first, bidirectional, layers, projection, bias, monkeypatch):
    kwargs = dict(input_size=3, hidden_size=4, num_layers=layers, batch_first=batch_first, bidirectional=bidirectional, bias=bias)
    if projection: kwargs["proj_size"] = projection
    module = getattr(torch.nn, kind)(**kwargs).double().eval()
    audio = torch.randn((2, 5, 3) if batch_first else (5, 2, 3), dtype=torch.float64)
    hidden = torch.randn(layers * (2 if bidirectional else 1), 2, projection or 4, dtype=torch.float64)
    state = (hidden, torch.randn(layers * (2 if bidirectional else 1), 2, 4, dtype=torch.float64)) if kind == "LSTM" else hidden
    weights = copy.deepcopy(module.state_dict())
    with torch.no_grad(): expected = module(audio, state)
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    with torch.no_grad(): actual = dml_backend.run_rnn(module, audio, state)
    torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
    for key, value in weights.items(): torch.testing.assert_close(module.state_dict()[key], value)


def test_directml_recurrent_adapter_rejects_training_and_packed_inputs(monkeypatch):
    module = torch.nn.LSTM(3, 4).eval()
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda value: True)
    packed = torch.nn.utils.rnn.pack_sequence([torch.randn(3, 3)])
    with pytest.raises(ValueError, match="dense batched"): dml_backend.run_rnn(module, packed)
    module.train()
    with pytest.raises(ValueError, match="eval mode"): dml_backend.run_rnn(module, torch.randn(3, 1, 3))


@pytest.mark.parametrize("dim,keepdim", [(-1, True), ((1, 2), True), ((1, 2), False)])
def test_real_standard_deviation_preserves_unbiased_normalization(dim, keepdim, monkeypatch):
    value = torch.randn(2, 3, 9, dtype=torch.float64)
    expected = value.std(dim=dim, keepdim=keepdim)
    monkeypatch.setattr(dml_backend, "is_directml_device", lambda tensor: True)
    torch.testing.assert_close(dml_backend.real_std(value, dim, keepdim), expected, rtol=1e-12, atol=1e-12)


@pytest.fixture(scope="module")
def dml_device():
    directml = pytest.importorskip("torch_directml")
    if directml.device_count() == 0: pytest.skip("DirectML adapter unavailable")
    return directml.device(0)


def test_directml_inference_checkpoint_preserves_device_values_and_view(dml_device):
    source = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
    value = source.to(dml_device)[:, 1:, ::2]
    version = value._version
    with torch.no_grad(): actual = dml_backend.inference_checkpoint(value)
    assert actual is value and actual.device == dml_device and actual.dtype == torch.float32
    assert actual._version == version and not actual.is_contiguous()
    torch.testing.assert_close(actual.cpu(), source[:, 1:, ::2], rtol=0, atol=0)


def test_directml_roformer_chunked_attention_matches_cpu_network(dml_device, monkeypatch):
    from pymss_core.modules.bs_roformer import BSRoformer
    torch.manual_seed(7)
    model = BSRoformer(**_roformer_kwargs(freqs_per_bands=(1,) * 9)).eval()
    audio = torch.randn(1, 1, 64)
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 2560)
    with torch.no_grad(): reference = model(audio)
    clear_model_runtime_caches(model)
    model.to(dml_device)
    try:
        with torch.no_grad(): actual = model(audio.to(dml_device))
        assert actual.device == dml_device and actual.dtype == torch.float32 and actual.shape == reference.shape
        assert torch.isfinite(actual.cpu()).all()
        torch.testing.assert_close(actual.cpu(), reference, rtol=5e-3, atol=5e-3)
    finally:
        clear_model_runtime_caches(model)
        model.to("cpu")


@pytest.mark.parametrize("layout,queries,keys,is_causal,scale", [
    ("batched", 17, 13, True, None),
    ("broadcast", 13, 17, True, 0.17),
    ("value_broadcast", 17, 13, False, 0.17),
    ("matrix", 43, 17, False, None),
])
def test_directml_chunked_attention_preserves_all_strided_broadcast_query_rows(layout, queries, keys, is_causal, scale, dml_device, monkeypatch):
    torch.manual_seed(7)
    inputs = _strided_attention_inputs(torch.float32, layout, queries, keys)
    assert all(value.storage_offset() > 0 and not value.is_contiguous() for value in inputs)
    # Exercise device-side strided slices and verify their exact input values.
    gpu_inputs = tuple(torch.stack((torch.zeros_like(value), value), dim=-1).flatten(-2).to(dml_device)[..., 1::2] for value in inputs)
    assert all(not value.is_contiguous() for value in gpu_inputs)
    for gpu_input, cpu_input in zip(gpu_inputs, inputs):
        assert gpu_input.shape == cpu_input.shape
        torch.testing.assert_close(gpu_input.cpu(), cpu_input, rtol=0, atol=0)
    monkeypatch.setattr(dml_backend, "_ATTENTION_SCORE_BUDGET_BYTES", 2560)
    with torch.no_grad():
        expected = torch.nn.functional.scaled_dot_product_attention(*inputs, is_causal=is_causal, scale=scale)
        actual = dml_backend.scaled_dot_product_attention(*gpu_inputs, is_causal=is_causal, scale=scale)
    assert actual.device == dml_device and actual.dtype == torch.float32 and actual.shape == expected.shape
    torch.testing.assert_close(actual.cpu(), expected, rtol=5e-6, atol=5e-6)


def test_directml_packed_mask_estimators_preserve_all_stems_and_frequency_offsets(dml_device):
    from pymss_core.modules.bs_roformer.bands import MaskEstimator
    torch.manual_seed(7)
    estimators = torch.nn.ModuleList([MaskEstimator(16, (4, 8, 4, 12), 2, 4) for _ in range(2)]).eval()
    base = torch.randn(2, 34, 4, 32)
    inputs = base[:, 1::2, :, 1::2]
    assert inputs.storage_offset() > 0 and not inputs.is_contiguous()
    with torch.no_grad():
        reference = torch.stack([estimator(inputs) for estimator in estimators], dim=1)
        cpu_packed = MaskEstimator.forward_packed_estimators(estimators, inputs)
    assert cpu_packed is not None
    torch.testing.assert_close(cpu_packed, reference, rtol=2e-6, atol=2e-6)
    clear_model_runtime_caches(estimators)
    estimators.to(dml_device)
    try:
        gpu_inputs = base.to(dml_device)[:, 1::2, :, 1::2]
        assert gpu_inputs.shape == inputs.shape
        torch.testing.assert_close(gpu_inputs.cpu(), inputs, rtol=0, atol=0)
        with torch.no_grad(): actual = MaskEstimator.forward_packed_estimators(estimators, gpu_inputs)
        assert actual is not None and actual.device == dml_device and actual.shape == (2, 2, 17, 28)
        torch.testing.assert_close(actual.cpu(), reference, rtol=5e-6, atol=5e-6)
    finally:
        clear_model_runtime_caches(estimators)
        estimators.to("cpu")


def test_directml_mask_time_chunks_match_original_core_with_tail_and_sliced_inputs(dml_device):
    from pymss_core.modules.bs_roformer import BSRoformer
    torch.manual_seed(7)
    model = BSRoformer(**_roformer_kwargs(dim=16, heads=2, dim_head=8, stereo=True, num_stems=2,
        freqs_per_bands=(1, 2, 1, 3), stft_n_fft=12, stft_hop_length=3, stft_win_length=12, mask_estimator_depth=2)).eval()
    base = torch.randn(2, 262, 4, 32)
    inputs = base[:, 1::2, :, 1::2]
    assert inputs.storage_offset() > 0 and not inputs.is_contiguous()
    with torch.no_grad(): reference = model._estimate_masks_core(inputs)
    clear_model_runtime_caches(model)
    model.to(dml_device)
    try:
        gpu_inputs = base.to(dml_device)[:, 1::2, :, 1::2]
        assert gpu_inputs.shape == inputs.shape and gpu_inputs.shape[1] == 131
        torch.testing.assert_close(gpu_inputs.cpu(), inputs, rtol=0, atol=0)
        with torch.no_grad(): original_core = model._estimate_masks_core(gpu_inputs)
        torch.testing.assert_close(original_core.cpu(), reference, rtol=5e-6, atol=5e-6)
        with torch.no_grad(): actual = model._estimate_masks(gpu_inputs)
        assert actual.device == dml_device and actual.dtype == reference.dtype and actual.shape == (2, 2, 131, 28)
        torch.testing.assert_close(actual.cpu(), reference, rtol=5e-6, atol=5e-6)
    finally:
        clear_model_runtime_caches(model)
        model.to("cpu")


@pytest.mark.parametrize("layout", ["qkv", "strided"])
def test_directml_rotary_embedding_preserves_all_even_odd_values_for_sliced_inputs(layout, dml_device):
    from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast, qkv_to_bnhd
    torch.manual_seed(7)
    if layout == "qkv":
        base = torch.randn(2, 17, 72)
        inputs = qkv_to_bnhd(base, 3)[1]
        gpu_inputs = qkv_to_bnhd(base.to(dml_device), 3)[1]
    else:
        base = torch.randn(2, 17, 3, 16)
        inputs, gpu_inputs = base[..., 1::2], base.to(dml_device)[..., 1::2]
    assert inputs.storage_offset() > 0 and not inputs.is_contiguous()
    assert gpu_inputs.shape == inputs.shape
    torch.testing.assert_close(gpu_inputs.cpu(), inputs, rtol=0, atol=0)
    angles = torch.randn(1, 17, 1, 4)
    cos, sin = angles.cos(), angles.sin()
    even, odd = inputs[..., ::2], inputs[..., 1::2]
    reference = torch.stack((even * cos - odd * sin, odd * cos + even * sin), dim=-1).flatten(-2)
    with torch.no_grad(): actual = apply_rotary_emb_fast(cos.to(dml_device), sin.to(dml_device), gpu_inputs)
    assert actual.device == dml_device and actual.dtype == inputs.dtype and actual.shape == inputs.shape
    torch.testing.assert_close(actual.cpu(), reference, rtol=5e-6, atol=5e-6)


def test_directml_spectral_boundaries_keep_complex_tensors_on_cpu(dml_device, monkeypatch):
    audio = torch.randn(1, 64).to(dml_device)
    native_stft, native_istft = torch.stft, torch.istft
    def cpu_stft(value, *args, **kwargs):
        assert value.device.type == "cpu" and value.dtype == torch.float32
        assert kwargs["window"].device.type == "cpu"
        return native_stft(value, *args, **kwargs)
    def cpu_istft(value, *args, **kwargs):
        assert value.device.type == "cpu" and value.dtype == torch.complex64
        assert kwargs["window"].device.type == "cpu"
        return native_istft(value, *args, **kwargs)
    monkeypatch.setattr(torch, "stft", cpu_stft)
    monkeypatch.setattr(torch, "istft", cpu_istft)
    window = torch.hann_window(16).to(dml_device)
    spectrum = dml_backend.stft_complex(audio, n_fft=16, hop_length=4, window=window, return_complex=True)
    real = dml_backend.spectrum_to_real(spectrum, audio)
    assert spectrum.device.type == "cpu" and spectrum.is_complex()
    assert real.device == dml_device and real.dtype == torch.float32 and not real.is_complex()
    restored = dml_backend.istft_complex(dml_backend.real_to_complex(real), n_fft=16, hop_length=4, window=window, length=64, output_device=dml_device)
    assert restored.device == dml_device
    torch.testing.assert_close(restored.cpu(), audio.cpu(), atol=1e-5, rtol=1e-5)


def test_directml_zero_padding_preserves_noncontiguous_frequency_slice(dml_device):
    tensor = torch.randn(1, 4, 20, 8)
    view = tensor.to(dml_device)[:, :, 7:]
    for _ in range(3):
        padded = dml_backend.pad(view, (0, 0, 0, 0))
        assert padded is not view
        torch.testing.assert_close(padded.cpu(), tensor[:, :, 7:])
        padded.add_(1)
        torch.testing.assert_close(view.cpu(), tensor[:, :, 7:])


@pytest.mark.parametrize("size,noncontiguous", [
    ((2, 3), False), ((5, 8), False), ((7, 4), False), ((3, 5), False),
    ((2, 3), True), ((5, 8), True), ((3, 5), True),
])
def test_directml_hyperace_resize_matches_pytorch_half_pixel(size, noncontiguous, dml_device):
    from pymss_core.modules.bs_roformer.hyperace_segm import _interp
    source = torch.arange(60, dtype=torch.float32).reshape(1, 2, 3, 10)
    cpu = source[:, :, :, 1::2] if noncontiguous else source[:, :, :, :5].contiguous()
    gpu = source.to(dml_device)[:, :, :, 1::2] if noncontiguous else cpu.to(dml_device)
    reference = torch.nn.functional.interpolate(cpu, size=size, mode="bilinear", align_corners=False)
    actual = _interp(gpu, size)
    assert actual.device == dml_device
    torch.testing.assert_close(actual.cpu(), reference, atol=1e-5, rtol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_hyperace_native_resize_preserves_dtype_and_normal_precision_policy(dtype):
    from pymss_core.modules.bs_roformer.hyperace_segm import _interp
    value = torch.randn(2, 3, 5, 7, dtype=dtype)
    reference = torch.nn.functional.interpolate(value, size=(8, 4), mode="bilinear", align_corners=False)
    actual = _interp(value, (8, 4))
    assert actual.dtype == dtype
    torch.testing.assert_close(actual, reference, rtol=0, atol=0)


@pytest.mark.parametrize("mode,selected", [("full", None), ("segm_only", None), ("full", [1]), ("segm_only", [1])])
def test_directml_hyperace_segment_modes_preserve_stereo_stems_and_cpu_parity(mode, selected, dml_device):
    from pymss_core.modules.bs_roformer import BSRoformerHyperACE
    torch.manual_seed(7)
    model = BSRoformerHyperACE(
        dim=16, depth=1, heads=2, dim_head=8, flash_attn=False, stereo=True, num_stems=2,
        freqs_per_bands=(2,) * 61 + (7,), stft_n_fft=256, stft_hop_length=64,
        stft_win_length=256, mask_estimator_depth=1, skip_connection=True,
    ).eval()
    model.set_mask_mode(mode)
    if selected is not None: model._pymss_source_indices = selected
    value = torch.randn(1, 2, 8192)
    weights = copy.deepcopy(model.state_dict())
    with torch.no_grad(): reference = model(value)
    clear_model_runtime_caches(model)
    model.to(dml_device)
    try:
        with torch.no_grad(): actual = model(value.to(dml_device))
        assert actual.shape == ((1, 2, 8192) if selected else (1, 2, 2, 8192))
        assert actual.device == dml_device
        torch.testing.assert_close(actual.cpu(), reference, rtol=1e-3, atol=1e-3)
        clear_model_runtime_caches(model)
        with torch.no_grad(): repeated = model(value.to(dml_device))
        torch.testing.assert_close(repeated.cpu(), actual.cpu(), rtol=1e-4, atol=1e-4)
    finally:
        clear_model_runtime_caches(model)
        model.to("cpu")
    for key, expected in weights.items(): torch.testing.assert_close(model.state_dict()[key], expected, rtol=0, atol=0)


def _model_cases():
    result = {name: (builder, shape) for name, (builder, shape, _) in cases().items()}
    from pymss_core.modules.bs_roformer import BSRoformer
    from pymss_core.modules.vocal_remover import CascadedNet
    result["bs_pope"] = (lambda: BSRoformer(**_roformer_kwargs(freqs_per_bands=(1,) * 9, use_pope=True)), (1, 1, 64))
    result["vr"] = (lambda: CascadedNet(n_fft=256, nn_arch_size=56817, nout=4, nout_lstm=16), (1, 2, 129, 64))
    return result


MODEL_CASES = _model_cases()


@pytest.mark.parametrize("name", MODEL_CASES)
def test_directml_model_real_network_matches_cpu(name, dml_device, monkeypatch):
    builder, shape = MODEL_CASES[name]
    torch.manual_seed(7)
    model = builder().eval()
    audio = torch.randn(shape)
    with torch.no_grad(): reference = model(audio)
    model.to(dml_device)
    neural_devices = []
    def neural_hook(module, inputs, output):
        if inputs and isinstance(inputs[0], torch.Tensor): neural_devices.append(inputs[0].device)
    hooks = [module.register_forward_hook(neural_hook) for module in model.modules() if isinstance(module, (torch.nn.Linear, torch.nn.Conv1d, torch.nn.Conv2d))]
    native_complex, native_view_complex, native_stft = torch.complex, torch.view_as_complex, torch.stft
    def cpu_complex(real, imag):
        assert real.device.type == imag.device.type == "cpu"
        return native_complex(real, imag)
    def cpu_view_complex(real):
        assert real.device.type == "cpu"
        return native_view_complex(real)
    def cpu_stft(real, *args, **kwargs):
        assert real.device.type == "cpu"
        return native_stft(real, *args, **kwargs)
    monkeypatch.setattr(torch, "complex", cpu_complex)
    monkeypatch.setattr(torch, "view_as_complex", cpu_view_complex)
    monkeypatch.setattr(torch, "stft", cpu_stft)
    try:
        with torch.no_grad(): actual = model(audio.to(dml_device))
        assert actual.device == dml_device and actual.shape == reference.shape
        assert neural_devices and all(device == dml_device for device in neural_devices)
        assert torch.isfinite(actual.cpu()).all()
        torch.testing.assert_close(actual.cpu(), reference, rtol=5e-3, atol=5e-3)
        clear_model_runtime_caches(model)
        with torch.no_grad(): repeated = model(audio.to(dml_device))
        torch.testing.assert_close(repeated.cpu(), actual.cpu(), rtol=1e-4, atol=1e-4)
    finally:
        for hook in hooks: hook.remove()
        clear_model_runtime_caches(model)
        model.to("cpu")
