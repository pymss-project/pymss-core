"""Contract tests for the explicit Biren attention adapter, without GPU hardware."""
import pytest
import torch
import torch.nn.functional as F

from pymss_core.vendor_shims import sudnn_sdpa


def test_sudnn_adapter_preserves_head_order_for_strided_operands(monkeypatch):
    calls = []
    def operator(q, k, v, mask, dropout, causal, scale, algorithm, slopes, distribute):
        calls.append((q.shape, q.is_contiguous(), k.is_contiguous(), v.is_contiguous()))
        assert (mask, dropout, causal, algorithm, slopes, distribute) == (None, 0.0, False, 'FMHA', None, None)
        return [F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=scale).to(q.dtype)]
    monkeypatch.setattr(torch.ops.supa, 'sudnn_scaled_dot_product_attention_fwd', operator, raising=False)
    torch.manual_seed(0)
    q, k, v = [torch.randn(2, 5, 3, 8).to(torch.bfloat16).transpose(1, 2) for _ in range(3)]
    expected = F.scaled_dot_product_attention(q.float(), k.float(), v.float(), scale=0.3).to(q.dtype)
    out = sudnn_sdpa(q, k, v, scale=0.3)
    assert calls == [(torch.Size([6, 5, 8]), True, True, True)]
    assert out.shape == q.shape and out.dtype == q.dtype
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


def test_sudnn_errors_are_not_silently_replaced_by_manual_attention(monkeypatch):
    def operator(*args):
        raise RuntimeError('FMHA diagnostic failure')
    monkeypatch.setattr(torch.ops.supa, 'sudnn_scaled_dot_product_attention_fwd', operator, raising=False)
    q = torch.zeros(2, 3, 4, 8, dtype=torch.bfloat16)
    with pytest.raises(RuntimeError, match='FMHA diagnostic failure'):
        sudnn_sdpa(q, q, q)


def test_sudnn_rejects_unvalidated_dtype():
    q = torch.zeros(2, 3, 4, 8)
    with pytest.raises(ValueError, match='bf16'):
        sudnn_sdpa(q, q, q)


def test_sudnn_rejects_mismatched_attention_shapes():
    q = torch.zeros(2, 3, 4, 8, dtype=torch.bfloat16)
    k = torch.zeros(2, 3, 5, 8, dtype=torch.bfloat16)
    with pytest.raises(ValueError, match='matching self-attention shapes'):
        sudnn_sdpa(q, k, k)
