"""DirectML real-tensor inference boundaries; complex DSP stays on the CPU."""
from contextlib import nullcontext
from math import prod
import torch
from torch.nn import functional as F


def is_directml_device(value):
    device = getattr(value, "device", value)
    return getattr(device, "type", None) == "privateuseone" or isinstance(device, str) and device.split(":", 1)[0] == "privateuseone"


def _cpu_spectral_kwargs(kwargs):
    kwargs = dict(kwargs)
    if kwargs.get("window") is not None: kwargs["window"] = kwargs["window"].to(device="cpu", dtype=torch.float32)
    return kwargs


def stft_complex(audio, *args, **kwargs):
    if is_directml_device(audio): audio, kwargs = audio.to(device="cpu", dtype=torch.float32), _cpu_spectral_kwargs(kwargs)
    elif kwargs.get("window") is not None and kwargs["window"].device != audio.device: kwargs = {**kwargs, "window": kwargs["window"].to(audio.device)}
    return torch.stft(audio, *args, **kwargs)


def istft_complex(spectrum, *args, output_device=None, **kwargs):
    if is_directml_device(output_device): spectrum, kwargs = spectrum.to(device="cpu", dtype=torch.complex64), _cpu_spectral_kwargs(kwargs)
    elif kwargs.get("window") is not None and kwargs["window"].device != spectrum.device: kwargs = {**kwargs, "window": kwargs["window"].to(spectrum.device)}
    audio = torch.istft(spectrum, *args, **kwargs)
    return audio.to(device=output_device, dtype=torch.float32) if is_directml_device(output_device) else audio


def real_to_model(value, reference):
    device = getattr(reference, "device", reference)
    return value.to(device=device, dtype=torch.float32) if is_directml_device(device) else value


def spectrum_to_real(spectrum, reference):
    return real_to_model(torch.view_as_real(spectrum), reference)


def real_to_complex(value):
    if is_directml_device(value): value = value.to(device="cpu", dtype=torch.float32).contiguous()
    return torch.view_as_complex(value)


def complex_from_parts(real, imag):
    if is_directml_device(real): real, imag = real.cpu().float(), imag.cpu().float()
    return torch.complex(real, imag)


def multiply_spectrum(spectrum, mask):
    if is_directml_device(mask) and spectrum.device.type == "cpu": mask = mask.cpu()
    return spectrum * mask


def rfft_real(value, *args, **kwargs):
    source = value.to(device="cpu", dtype=torch.float32) if is_directml_device(value) else value
    return spectrum_to_real(torch.fft.rfft(source, *args, **kwargs), value)


def irfft_real(value, *args, **kwargs):
    audio = torch.fft.irfft(real_to_complex(value), *args, **kwargs)
    return real_to_model(audio, value)


def autocast_disabled(value):
    return nullcontext() if is_directml_device(value) else torch.autocast(device_type=value.device.type, enabled=False)


def pad(tensor, padding, mode="constant", value=None):
    # DirectML's zero-padding kernel can read invalid storage offsets for views.
    if is_directml_device(tensor) and not any(padding): return tensor.clone()
    return F.pad(tensor, padding, mode=mode, value=value)


def glu(value, dim=-1):
    if not is_directml_device(value): return F.glu(value, dim=dim)
    linear, gate = value.chunk(2, dim=dim)
    return linear * gate.sigmoid()


class DirectMLGLU(torch.nn.GLU):
    def forward(self, value): return glu(value, self.dim)


def real_std(value, dim, keepdim=True):
    """Unbiased real standard deviation for model normalization."""
    if not is_directml_device(value): return value.std(dim=dim, keepdim=keepdim)
    axes = (dim,) if isinstance(dim, int) else dim
    count = prod(value.shape[axis] for axis in axes)
    difference = value - value.mean(dim=dim, keepdim=True)
    return (difference.square().sum(dim=dim, keepdim=keepdim) / (count - 1)).sqrt()


def bilinear_resize(value, size):
    """NCHW resize with PyTorch's align_corners=False half-pixel coordinates."""
    if not is_directml_device(value): return F.interpolate(value, size=size, mode="bilinear", align_corners=False)
    if value.ndim != 4 or not value.is_floating_point(): raise ValueError("DirectML bilinear resize requires a real NCHW tensor")
    height, width = int(size[0]), int(size[1])
    if height <= 0 or width <= 0: raise ValueError("Bilinear output dimensions must be positive")
    if (height, width) == value.shape[-2:]: return value.clone()
    def positions(source, target):
        coordinates = (torch.arange(target, dtype=torch.float32, device="cpu") + 0.5) * (source / target) - 0.5
        lower = coordinates.floor()
        weight = (coordinates - lower).to(device=value.device, dtype=value.dtype)
        return lower.clamp(0, source - 1).long().to(value.device), (lower + 1).clamp(0, source - 1).long().to(value.device), weight
    y0, y1, wy = positions(value.shape[2], height)
    x0, x1, wx = positions(value.shape[3], width)
    wy, wx = wy.reshape(1, 1, height, 1), wx.reshape(1, 1, 1, width)
    top, bottom = value.index_select(2, y0), value.index_select(2, y1)
    top = top.index_select(3, x0) * (1 - wx) + top.index_select(3, x1) * wx
    bottom = bottom.index_select(3, x0) * (1 - wx) + bottom.index_select(3, x1) * wx
    return top * (1 - wy) + bottom * wy


def scaled_dot_product_attention(q, k, v, *, dropout_p=0.0, is_causal=False, scale=None):
    if not is_directml_device(q): return F.scaled_dot_product_attention(q, k, v, dropout_p=dropout_p, is_causal=is_causal, scale=scale)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5 if scale is None else scale)
    if is_causal:
        mask = torch.ones(q.shape[-2], k.shape[-2], dtype=torch.bool, device=q.device).tril()
        scores = scores.masked_fill(~mask, float("-inf"))
    weights = F.dropout(scores.softmax(dim=-1), dropout_p, training=True) if dropout_p else scores.softmax(dim=-1)
    return torch.matmul(weights, v)


def run_rnn(module, value, state=None):
    """Keep recurrent gates on DirectML without changing registered weights."""
    if not is_directml_device(next(module.parameters())): return module(value, state) if state is not None else module(value)
    if not isinstance(value, torch.Tensor) or value.ndim != 3: raise ValueError("DirectML recurrent inference requires a dense batched tensor")
    if module.training: raise ValueError("DirectML recurrent inference requires eval mode")
    if not isinstance(module, (torch.nn.LSTM, torch.nn.GRU, torch.nn.RNN)): raise TypeError("Unsupported DirectML recurrent module")
    sequence = value.transpose(0, 1) if module.batch_first else value
    directions, batch = 2 if module.bidirectional else 1, sequence.shape[1]
    lstm = isinstance(module, torch.nn.LSTM)
    hidden_size = getattr(module, "proj_size", 0) or module.hidden_size
    if state is None:
        hidden = value.new_zeros(module.num_layers * directions, batch, hidden_size)
        cell = value.new_zeros(module.num_layers * directions, batch, module.hidden_size) if lstm else None
    else: hidden, cell = state if lstm else (state, None)
    final_hidden, final_cell = [], []
    for layer in range(module.num_layers):
        layer_outputs = []
        for direction in range(directions):
            suffix = f"_l{layer}" + ("_reverse" if direction else "")
            bias_ih = getattr(module, f"bias_ih{suffix}") if module.bias else None
            bias_hh = getattr(module, f"bias_hh{suffix}") if module.bias else None
            input_gates = F.linear(sequence, getattr(module, f"weight_ih{suffix}"), bias_ih)
            h = hidden[layer * directions + direction]
            c = cell[layer * directions + direction] if lstm else None
            outputs = []
            steps = range(sequence.shape[0] - 1, -1, -1) if direction else range(sequence.shape[0])
            for index in steps:
                recurrent = F.linear(h, getattr(module, f"weight_hh{suffix}"), bias_hh)
                if lstm:
                    i, f, g, o = (input_gates[index] + recurrent).chunk(4, dim=-1)
                    c = f.sigmoid() * c + i.sigmoid() * g.tanh()
                    h = o.sigmoid() * c.tanh()
                    if module.proj_size: h = F.linear(h, getattr(module, f"weight_hr{suffix}"))
                elif isinstance(module, torch.nn.GRU):
                    ir, iz, inn = input_gates[index].chunk(3, dim=-1)
                    hr, hz, hn = recurrent.chunk(3, dim=-1)
                    reset, update = (ir + hr).sigmoid(), (iz + hz).sigmoid()
                    h = (1 - update) * (inn + reset * hn).tanh() + update * h
                else:
                    h = input_gates[index] + recurrent
                    h = h.relu() if module.nonlinearity == "relu" else h.tanh()
                outputs.append(h)
            if direction: outputs.reverse()
            layer_outputs.append(torch.stack(outputs))
            final_hidden.append(h)
            if lstm: final_cell.append(c)
        sequence = torch.cat(layer_outputs, dim=-1)
    output = sequence.transpose(0, 1) if module.batch_first else sequence
    state = torch.stack(final_hidden)
    return output, (state, torch.stack(final_cell)) if lstm else state
