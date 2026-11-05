"""Vendor RoPE integration: weight-permutation algebra and gating.

The vendor op rotates split-half pairs (i, i+d/2); BS-Roformer channels are
interleaved (2i, 2i+1).  prepare_model_for_vendor_rope folds that difference
into to_qkv row order.  These tests pin the algebra on CPU (no vendor build
needed): permuted projection + split-half rotation == original projection +
apply_rotary_emb_fast, plus the gating that keeps unprepared models untouched.
"""
import pytest
import torch

from pymss_core import vendor_shims
from pymss_core.device import BIREN, clear_device_cache, device_profile
from pymss_core.modules.bs_roformer.bs_roformer import BSRoformer
from pymss_core.modules.bs_roformer.transformer import apply_rotary_emb_fast
from pymss_core.vendor_shims import (
    _rope_chunked,
    _rope_row_index,
    _vendor_rope_rotate,
    prepare_model_for_vendor_rope,
)


class _FakeRotary:
    """Minimal rotary embedding for cached_rotary_cos_sin (theta 10000)."""

    def __init__(self, seq, dim_head):
        inv = 1.0 / (10000 ** (torch.arange(0, dim_head, 2).float() / dim_head))
        self.angles = torch.arange(seq).float()[:, None] * inv[None, :]

    def get_seq_pos(self, seq_len, device=None, dtype=None, offset=0):
        return self.angles

    def forward(self, fn, cache_key=None):
        return fn().repeat_interleave(2, dim=-1)


def _split_half_rotate(x, cos_half, sin_half):
    half = x.shape[-1] // 2
    left, right = x[..., :half], x[..., half:]
    return torch.cat((left * cos_half - right * sin_half,
                      right * cos_half + left * sin_half), dim=-1)


def test_row_index_permutation_equivalence():
    """permuted to_qkv + split-half rotation == original projection + rotary."""
    torch.manual_seed(0)
    heads, dim_head, dim, tokens = 8, 64, 384, 6
    weight = torch.randn(3 * heads * dim_head, dim) * 0.05
    bias = torch.randn(3 * heads * dim_head) * 0.05
    x = torch.randn(tokens, dim)
    index = _rope_row_index(heads, dim_head)

    original = torch.nn.functional.linear(x, weight, bias)
    permuted = torch.nn.functional.linear(x, weight[index], bias[index])

    c, s = torch.tensor(0.9), torch.tensor(0.1)
    # Pairing lives INSIDE each head's dim_head channels, not across the fused
    # qkv vector: view as (tokens, 3, heads, dim_head) and compare per segment.
    o_view = original.view(tokens, 3, heads, dim_head)
    p_view = permuted.view(tokens, 3, heads, dim_head)
    for segment in (0, 1):  # q and k
        interleaved = o_view[:, segment]
        halves = p_view[:, segment]
        rotary = torch.cat((interleaved[..., ::2] * c - interleaved[..., 1::2] * s,
                            interleaved[..., 1::2] * c + interleaved[..., ::2] * s), dim=-1)
        expected = _split_half_rotate(halves, c, s)
        assert torch.allclose(rotary, expected, atol=1e-6), f"segment {segment}"
    # The value segment must be untouched by the permutation.
    assert torch.equal(o_view[:, 2], p_view[:, 2])


def test_row_index_shape_and_segments():
    for heads, dim_head in ((8, 64), (2, 16)):
        index = _rope_row_index(heads, dim_head)
        size = 3 * heads * dim_head
        assert index.shape == (size,)
        assert sorted(index.tolist()) == list(range(size)), "must be a permutation"
        evens = list(range(0, dim_head, 2)) + list(range(1, dim_head, 2))
        for head in range(heads):
            base_q = head * dim_head
            qblock = index[base_q:base_q + dim_head].tolist()
            assert [v - base_q for v in qblock] == evens, "q segment must be [evens|odds] per head"
            base_k = heads * dim_head + head * dim_head
            kblock = index[base_k:base_k + dim_head].tolist()
            assert [v - base_k for v in kblock] == evens
        assert index[2 * heads * dim_head:].tolist() == list(range(2 * heads * dim_head, size)), \
            "value segment must stay untouched"


def test_prepare_model_permutes_and_marks():
    torch.manual_seed(0)
    model = BSRoformer(dim=16, depth=1, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval()
    attention = next(m for n, m in model.named_modules() if hasattr(m, "to_qkv"))
    raw_weight = attention.to_qkv.weight.data.clone()
    index = _rope_row_index(attention.heads, attention.to_qkv.out_features // (3 * attention.heads))

    prepared = prepare_model_for_vendor_rope(model)
    assert prepared, "expected at least one to_qkv"
    assert all(name.endswith("to_qkv") for name in prepared)
    assert torch.equal(attention.to_qkv.weight.data, raw_weight[index])
    assert getattr(attention.to_qkv, "_pymss_rope_permuted", False)
    assert getattr(attention.rotary_embed, "_pymss_vendor_rope", False)

    # Idempotent: a second call must not permute again.
    once = attention.to_qkv.weight.data.clone()
    prepare_model_for_vendor_rope(model)
    assert torch.equal(attention.to_qkv.weight.data, once)


def test_prepare_model_load_state_dict_repermutes():
    torch.manual_seed(0)
    model = BSRoformer(dim=16, depth=1, stereo=True, num_stems=1,
                       time_transformer_depth=1, freq_transformer_depth=1).eval()
    attention = next(m for n, m in model.named_modules() if hasattr(m, "to_qkv"))
    raw = {key: value.clone() for key, value in model.state_dict().items()}
    index = _rope_row_index(attention.heads, attention.to_qkv.out_features // (3 * attention.heads))

    prepare_model_for_vendor_rope(model)
    model.load_state_dict(raw)  # fresh raw checkpoint load must be re-permuted
    assert torch.equal(attention.to_qkv.weight.data, raw["{}.to_qkv.weight".format(
        next(n for n, m in model.named_modules() if m is attention))][index])


@pytest.mark.parametrize("env,expected", [("manual", False), ("vendor", True)])
def test_rope_backend_env_override(env, expected, monkeypatch):
    monkeypatch.setenv("PYMSS_ROPE_BACKEND", env)
    clear_device_cache()
    assert device_profile("supa").rope_shim is expected
    clear_device_cache()


def test_biren_profile_enables_rope_by_default(monkeypatch):
    monkeypatch.delenv("PYMSS_ROPE_BACKEND", raising=False)
    clear_device_cache()
    profile = device_profile("supa")
    assert profile.vendor == BIREN
    assert profile.rope_shim is True
    clear_device_cache()


def test_install_rope_shim_noop_without_vendor_op(monkeypatch):
    """Without torch_br the installer must refuse and leave the rotary alone."""
    from pymss_core.modules.bs_roformer import transformer
    if getattr(torch.ops.supa, "supa_rope_apply", None) is not None:
        pytest.skip("vendor op present; noop path untestable here")
    before = transformer.rotate_qk_fast_bnhd
    had_rope = "rope" in vendor_shims._installed
    vendor_shims._installed.discard("rope")
    try:
        assert vendor_shims.install_rope_shim() is False
        assert transformer.rotate_qk_fast_bnhd is before
    finally:
        if had_rope:
            vendor_shims._installed.add("rope")


def test_vendor_rope_rotate_matches_reference(monkeypatch):
    """With the op mocked as exact fp32 split-half rotation, the shim path
    reproduces apply_rotary_emb_fast on the interleaved preimage."""
    torch.manual_seed(0)
    b, s, h, d = 3, 5, 2, 8
    rotary = _FakeRotary(s, d)

    def mock_apply(x, cos, sin):
        half = x.shape[-1] // 2
        return _split_half_rotate(x.float(), cos[..., :half], sin[..., :half]).to(x.dtype)

    monkeypatch.setattr(vendor_shims, "_supa_rope_apply", mock_apply)
    q_inter = torch.randn(b, s, h, d, dtype=torch.float32)
    k_inter = torch.randn(b, s, h, d, dtype=torch.float32)
    unpack = lambda t: torch.cat((t[..., ::2], t[..., 1::2]), dim=-1)

    q_out, k_out = _vendor_rope_rotate(rotary, unpack(q_inter), unpack(k_inter))
    # _vendor_rope_rotate returns (B, S, H, D)-shaped views (memory ordered for
    # the downstream transpose); compare values directly.
    angles = rotary.angles
    expected_q = apply_rotary_emb_fast(angles.cos()[None, :, None, :],
                                       angles.sin()[None, :, None, :], q_inter)
    assert torch.allclose(q_out, expected_q, atol=1e-5)
    expected_k = apply_rotary_emb_fast(angles.cos()[None, :, None, :],
                                       angles.sin()[None, :, None, :], k_inter)
    assert torch.allclose(k_out, expected_k, atol=1e-5)
    assert q_out.shape == q_inter.shape


def test_feedforward_fast_path_matches_eager(monkeypatch):
    """The fused-FFN fast path must reproduce eager Linear->GELU->Linear."""
    torch.manual_seed(0)
    import torch.nn as nn
    from pymss_core.modules.bs_roformer.transformer import FeedForward, RMSNorm

    def mock_fwd(x, up_weight, down_weight, up_bias, down_bias, act_mode=None):
        assert act_mode == 'gelu', 'activation must be requested explicitly'
        hidden = torch.nn.functional.linear(x.float(), up_weight.float(), up_bias.float())
        activated = torch.nn.functional.gelu(hidden)
        return [torch.nn.functional.linear(activated, down_weight.float(), down_bias.float()).to(x.dtype)]

    monkeypatch.setattr(vendor_shims, '_supa_fused_ffn_fwd', mock_fwd)
    module = FeedForward(dim=32, mult=4)
    assert vendor_shims._ffn_fast_path_ok(module)
    x = torch.randn(2, 5, 32)
    reference = module(x)
    fast = vendor_shims._vendor_fused_ffn(module, module.net[0](x))
    assert torch.allclose(reference, fast, atol=1e-4), 'fused FFN mismatch'


def test_feedforward_structure_gate_rejects_variants():
    import torch.nn as nn
    from pymss_core.modules.bs_roformer.transformer import FeedForward

    module = FeedForward(dim=32, mult=4)
    assert vendor_shims._ffn_fast_path_ok(module)
    module.net[2] = nn.ReLU()          # wrong activation
    assert not vendor_shims._ffn_fast_path_ok(module)
    module2 = FeedForward(dim=32, mult=4)
    module2.net[1].bias = None          # bias-free projection
    assert not vendor_shims._ffn_fast_path_ok(module2)
    module3 = FeedForward(dim=32, mult=4, dropout=0.1)
    assert vendor_shims._ffn_fast_path_ok(module3)   # dropout p irrelevant in eval (caller gates training)


def test_vendor_rope_rotate_bhqd_matches_reference(monkeypatch):
    """The regrouped (B,H,S,D) rotate path equals the original route."""
    torch.manual_seed(0)
    b, s, h, d = 3, 5, 2, 8
    rotary = _FakeRotary(s, d)

    def mock_apply(x, cos, sin):
        half = x.shape[-1] // 2
        return _split_half_rotate(x.float(), cos[..., :half], sin[..., :half]).to(x.dtype)

    monkeypatch.setattr(vendor_shims, "_supa_rope_apply", mock_apply)
    q_inter = torch.randn(b, s, h, d)
    # The fast path consumes weights already permuted to [evens | odds]: unpack
    # the interleaved preimage first, then regroup to (B, H, S, D).
    q_halves = torch.cat((q_inter[..., ::2], q_inter[..., 1::2]), dim=-1)
    q_bhqd = q_halves.transpose(1, 2).contiguous()
    q_out, _ = vendor_shims._vendor_rope_rotate_bhqd(rotary, q_bhqd, q_bhqd.clone())
    angles = rotary.angles
    expected = apply_rotary_emb_fast(angles.cos()[None, :, None, :],
                                     angles.sin()[None, :, None, :], q_inter).transpose(1, 2)
    assert torch.allclose(q_out, expected, atol=1e-5)
    assert q_out.shape == (b, h, s, d)


def test_rope_chunked_splits_within_launch_cap(monkeypatch):
    calls = []

    def mock_apply(x, cos, sin):
        calls.append(x.shape[0])
        return x * cos

    monkeypatch.setattr(vendor_shims, "_supa_rope_apply", mock_apply)
    monkeypatch.setattr(vendor_shims, "_ROPE_MAX_BATCH", 4)
    x = torch.arange(10 * 3 * 2, dtype=torch.float32).reshape(10, 3, 2)
    cos = torch.ones(1, 3, 2)
    out = _rope_chunked(x, cos, cos, 10)
    assert out.shape == x.shape
    assert max(calls) <= 4
    assert calls[0] == 4 and calls[-1] == 2
    assert torch.equal(out, x), "cos=1 mock must be the identity"
