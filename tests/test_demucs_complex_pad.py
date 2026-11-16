"""Complex padding for the demucs ports.

`F.pad` on a complex tensor has no kernel on some backends (Moore Threads MUSA
raises "SetMUTensorDType Unsupported tensor dtype: ComplexFloat"), so both demucs
decoders pad the real/imaginary planes instead.
"""
import torch
import torch.nn.functional as F

from pymss_core.modules.demucs_local import pad_complex


def test_complex_pad_matches_the_real_part_reference():
    torch.manual_seed(0)
    z = torch.randn(2, 3, 5, 7, dtype=torch.complex64)
    pad = (2, 2, 0, 1)
    got = pad_complex(z, pad)
    want = torch.complex(F.pad(z.real, pad), F.pad(z.imag, pad))
    assert got.shape == want.shape
    assert torch.equal(got, want)


def test_real_tensors_stay_on_the_plain_path():
    torch.manual_seed(1)
    z = torch.randn(2, 3, 5, 7)
    assert torch.equal(pad_complex(z, (1, 1, 0, 1)), F.pad(z, (1, 1, 0, 1)))