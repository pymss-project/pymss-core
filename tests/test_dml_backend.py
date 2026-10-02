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
