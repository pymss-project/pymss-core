"""Explicit low-precision networks consume fp32 spectra without autocast."""
import pytest
import torch

from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer


@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
def test_roformer_network_dtype_with_fp32_audio(dtype):
    torch.manual_seed(0)
    model = BSRoformer(
        dim=16, depth=1, stereo=True, num_stems=1,
        time_transformer_depth=1, freq_transformer_depth=1,
        heads=2, dim_head=8, freqs_per_bands=(3, 3, 3),
        stft_n_fft=16, stft_win_length=16, stft_hop_length=4,
    ).eval().to(dtype=dtype)
    audio = torch.randn(1, 2, 64)
    seen = []
    handle = model.band_split.register_forward_pre_hook(
        lambda module, inputs: seen.append(inputs[0].dtype)
    )
    try:
        with torch.inference_mode():
            out = model(audio)
    finally:
        handle.remove()
    assert seen == [dtype]
    assert out.shape == audio.shape
    assert out.dtype == torch.float32
    assert torch.isfinite(out).all()


def test_rotary_bf16_keeps_inverse_frequencies_and_phase_in_fp32():
    from pymss_core.modules.bs_roformer.common import RotaryEmbedding
    from pymss_core.modules.bs_roformer.transformer import cached_rotary_cos_sin

    rotary = RotaryEmbedding(64)
    original = rotary.freqs.detach().clone()
    cached_rotary_cos_sin(rotary, 862, torch.device('cpu'), torch.float32)
    rotary.to(dtype=torch.bfloat16)
    assert rotary.freqs.dtype == torch.float32
    assert torch.equal(rotary.freqs, original)
    assert not rotary.cache
    assert not rotary._pymss_cos_sin_cache
    cos, sin = cached_rotary_cos_sin(rotary, 862, torch.device('cpu'), torch.bfloat16)
    # Positions above 256 cannot all be represented in bf16. Compute their phase
    # and trig in fp32 before rounding the final tables to the activation dtype.
    phase = torch.arange(862, dtype=torch.float32)[:, None] * original[None]
    assert torch.equal(cos[0, :, 0], phase.cos().to(torch.bfloat16))
    assert torch.equal(sin[0, :, 0], phase.sin().to(torch.bfloat16))
    assert all(table.dtype == torch.float32 for table in rotary.cache.values())
