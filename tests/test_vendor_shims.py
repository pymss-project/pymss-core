"""The shims that route around missing vendor kernels."""
from types import SimpleNamespace

import pytest
import torch

from pymss_core.vendor_shims import (
    complex_planes_product,
    install_vendor_shims,
    installed_shims,
    reset_shims,
)


class TestComplexPlanesProduct:
    def test_real_times_complex_matches_reference(self):
        torch.manual_seed(0)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        got = complex_planes_product(mask, spec)
        assert got.dtype == torch.complex64
        assert torch.equal(got, mask.to(torch.complex64) * spec)

    def test_complex_times_real_is_commutative(self):
        torch.manual_seed(1)
        mask = torch.rand(2, 5, 7)
        spec = torch.randn(2, 5, 7, dtype=torch.complex64)
        assert torch.equal(complex_planes_product(spec, mask), complex_planes_product(mask, spec))

    def test_complex_times_complex_matches_reference(self):
        torch.manual_seed(2)
        a = torch.randn(2, 4, 6, dtype=torch.complex64)
        b = torch.randn(2, 4, 6, dtype=torch.complex64)
        assert torch.allclose(complex_planes_product(a, b), a * b, atol=1e-5)

    def test_broadcast_mask_over_channels(self):
        mask = torch.rand(1, 5, 7)
        spec = torch.randn(3, 5, 7, dtype=torch.complex64)
        assert complex_planes_product(mask, spec).shape == (3, 5, 7)


class TestShimInstallation:
    def test_cpu_installs_nothing(self):
        reset_shims()
        assert install_vendor_shims("cpu") == []
        assert installed_shims() == frozenset()

    def test_profile_flags_choose_the_shims(self):
        reset_shims()
        installed = install_vendor_shims(
            "supa", complex_mul=False, window_on_device=False, glu=True,
            complex_reshape=False, group_norm=True, interpolate=True,
        )
        assert set(installed) == {
            "complex_mul", "window", "glu", "complex_reshape", "group_norm", "interpolate",
            "fold", "complex_view", "linear_dtype", "embedding", "matmul_guard", "layer_norm", "layernorm_module", "conv_transpose1d",
        }
        assert install_vendor_shims("supa", complex_mul=False, window_on_device=False, glu=True) == []
        reset_shims()

    def test_window_shim_builds_on_host_for_device_targets(self):
        reset_shims()
        install_vendor_shims("supa", complex_mul=False, window_on_device=False, glu=False, group_norm=False, interpolate=False)
        # "meta" keeps this test device-independent while exercising the shim path.
        window = torch.hann_window(8, device="meta")
        assert window.device.type == "meta"
        reset_shims()


class TestGluShim:
    def test_equivalent_matches_the_fused_kernel(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import glu_equivalent

        torch.manual_seed(4)
        x = torch.randn(2, 8, 5)
        assert torch.allclose(glu_equivalent(x, 1), F.glu(x, 1), atol=1e-6)

    def test_shim_leaves_cpu_calls_alone(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import install_glu_shim, reset_shims

        reset_shims()
        original = F.glu
        install_glu_shim()
        x = torch.randn(2, 8, 5)
        assert torch.allclose(F.glu(x, 1), original(x, 1), atol=1e-6)
        reset_shims()


class TestScalarOperands:
    def test_complex_times_python_float(self):
        from pymss_core.vendor_shims import complex_planes_product

        spec = torch.randn(2, 9, 17, dtype=torch.complex64)
        assert torch.allclose(complex_planes_product(spec, 0.5), spec * 0.5, atol=1e-6)

    def test_complex_times_python_complex(self):
        from pymss_core.vendor_shims import complex_planes_product

        spec = torch.randn(2, 9, 17, dtype=torch.complex64)
        assert torch.allclose(complex_planes_product(spec, 1 - 2j), spec * (1 - 2j), atol=1e-5)

    def test_float_tensor_times_scalar_is_untouched(self):
        from pymss_core.vendor_shims import _needs_plane_product

        assert _needs_plane_product(torch.randn(3), 2.0) is False


class TestScatterAddEquivalent:
    def test_matches_scatter_add_in_five_dimensions(self):
        from pymss_core.vendor_shims import scatter_add_equivalent

        torch.manual_seed(0)
        src = torch.randn(2, 3, 13, 5, 2)
        index = torch.tensor([0, 0, 1, 2, 2, 3, 4, 4, 4, 5, 6, 6, 1])
        want = torch.zeros(2, 3, 9, 5, 2)
        want.scatter_add_(2, index[None, None, :, None, None].expand(2, 3, -1, 5, 2), src)
        got = scatter_add_equivalent(src, index, 2, 9)
        assert got.shape == want.shape
        assert torch.allclose(got, want, atol=1e-6)

    def test_last_dimension(self):
        from pymss_core.vendor_shims import scatter_add_equivalent

        torch.manual_seed(1)
        src = torch.randn(4, 6)
        index = torch.tensor([0, 1, 1, 3, 5, 5])
        want = torch.zeros(4, 7)
        want.scatter_add_(1, index[None, :].expand(4, 6), src)
        assert torch.allclose(scatter_add_equivalent(src, index, 1, 7), want, atol=1e-6)
    def test_blocked_devices_fold_row_wise(self, monkeypatch):
        """A profile that blocks scatter_add must not build the one-hot matrix.

        The one-hot matmul is ``size x index.numel() x features``: 24.24 GiB for a
        10 s mel-band clip, which OOMs the 7 GiB MUSA cards.  ``scatter_add=False``
        therefore has to route through the row-wise fold.
        """
        from pymss_core import vendor_shims

        folds = []
        real_fold = vendor_shims._fold_rows

        def spy(flat, index_1d, size):
            folds.append((tuple(flat.shape), int(size)))
            return real_fold(flat, index_1d, size)

        monkeypatch.setattr(vendor_shims, "_fold_rows", spy)
        monkeypatch.setattr(vendor_shims, "_scatter_add_blocked", lambda device: True)

        torch.manual_seed(2)
        src = torch.randn(2, 3, 4, 5)
        index = torch.tensor([0, 0, 1, 2])
        want = torch.zeros(2, 3, 6, 5)
        want.scatter_add_(2, index[None, None, :, None].expand(2, 3, -1, 5), src)
        got = vendor_shims.scatter_add_equivalent(src, index, 2, 6)
        assert folds == [((6, 4, 5), 6)]
        assert torch.allclose(got, want, atol=1e-6)

    def test_unblocked_devices_keep_the_one_hot_matmul(self, monkeypatch):
        from pymss_core import vendor_shims

        monkeypatch.setattr(
            vendor_shims, "_fold_rows",
            lambda *args, **kwargs: pytest.fail("row-wise fold asked for on a native device"),
        )
        monkeypatch.setattr(vendor_shims, "_scatter_add_blocked", lambda device: False)
        torch.manual_seed(3)
        src = torch.randn(2, 3, 4, 5)
        index = torch.tensor([0, 0, 1, 2])
        want = torch.zeros(2, 3, 6, 5)
        want.scatter_add_(2, index[None, None, :, None].expand(2, 3, -1, 5), src)
        assert torch.allclose(vendor_shims.scatter_add_equivalent(src, index, 2, 6), want, atol=1e-6)

    def test_complex_sources_fold_on_the_real_planes(self):
        """MUSA has no complex ``index_add_``, so complex folds run re/im plane-wise."""
        from pymss_core.vendor_shims import _fold_rows

        torch.manual_seed(4)
        src = torch.randn(2, 4, 3, dtype=torch.complex64)
        index = torch.tensor([0, 0, 1, 2])
        want = torch.zeros(2, 5, 3, dtype=torch.complex64)
        for row in range(2):
            want[row].index_add_(0, index, src[row])
        got = _fold_rows(src, index, 5)
        assert got.dtype == torch.complex64
        assert torch.allclose(got, want, atol=1e-6)


class TestPlaneProductPredicate:
    def test_scalar_on_either_side_requests_the_plane_kernel(self):
        from pymss_core.vendor_shims import _needs_plane_product

        spec = torch.randn(2, 18, 2049, dtype=torch.complex64)
        assert _needs_plane_product(spec, 0.015625) is True
        assert _needs_plane_product(0.015625, spec) is True
        assert _needs_plane_product(spec, 1 - 2j) is True

    def test_real_operands_stay_native(self):
        from pymss_core.vendor_shims import _needs_plane_product

        assert _needs_plane_product(torch.randn(2, 3), 0.5) is False
        assert _needs_plane_product(torch.randn(2, 3), torch.randn(2, 3)) is False

    def test_scalar_normalisation_does_not_lose_the_tensor(self):
        from pymss_core.vendor_shims import complex_planes_product

        spec = torch.randn(2, 18, 2049, dtype=torch.complex64)
        scale = (1.0 / 4096) ** 0.5
        assert torch.allclose(complex_planes_product(0.015625, spec), spec * scale, atol=1e-6)


class TestComplexReshape:
    def test_matches_native_reshape(self):
        from pymss_core.vendor_shims import complex_reshape

        spec = torch.view_as_complex(torch.randn(2, 6, 5, 2))
        assert torch.equal(complex_reshape(spec, 2, 30), spec.reshape(2, 30))
        assert torch.equal(complex_reshape(spec, -1, 5), spec.reshape(-1, 5))

    def test_flatten_is_exact(self):
        from pymss_core.vendor_shims import complex_reshape

        spec = torch.view_as_complex(torch.randn(12, 2))
        assert torch.equal(complex_reshape(spec, 12), spec.reshape(12))

    def test_installation_is_idempotent(self):
        from pymss_core.vendor_shims import install_complex_reshape_shim, reset_shims

        reset_shims()
        assert install_complex_reshape_shim() is True
        assert install_complex_reshape_shim() is False
        reset_shims()


class TestGroupNormEquivalent:
    def test_matches_the_fused_kernel(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import group_norm_equivalent

        for shape, groups in (((2, 32, 16, 16), 8), ((2, 8, 5), 4), ((1, 3, 7, 9), 3)):
            torch.manual_seed(0)
            x = torch.randn(*shape)
            weight, bias = torch.randn(shape[1]), torch.randn(shape[1])
            want = F.group_norm(x, groups, weight, bias, 1e-5)
            got = group_norm_equivalent(x, groups, weight, bias, 1e-5)
            assert torch.allclose(got, want, atol=1e-5)

    def test_without_affine(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import group_norm_equivalent

        x = torch.randn(2, 6, 4, 4)
        assert torch.allclose(group_norm_equivalent(x, 3), F.group_norm(x, 3), atol=1e-5)


class TestInterpolateShim:
    def test_bf16_bilinear_keeps_its_dtype_and_shape(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import install_interpolate_shim, reset_shims

        reset_shims()
        install_interpolate_shim()
        x = torch.randn(2, 4, 8, 8).bfloat16()
        got = F.interpolate(x, scale_factor=2, mode="bilinear")
        assert got.dtype == torch.bfloat16
        assert got.shape == (2, 4, 16, 16)
        assert torch.allclose(got.float(), F.interpolate(x.float(), scale_factor=2, mode="bilinear"), atol=1e-2)
        reset_shims()

    def test_nearest_is_left_alone(self):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import install_interpolate_shim, reset_shims

        reset_shims()
        install_interpolate_shim()
        x = torch.randn(1, 2, 4, 4).bfloat16()
        assert F.interpolate(x, scale_factor=2, mode="nearest").dtype == torch.bfloat16
        reset_shims()


class TestGroupedConv1d:
    """BIREN rejects 1 < groups < in_channels with '801, operation not supported'."""

    @pytest.mark.parametrize(("groups", "cin", "cout"), [(4, 8, 16), (8, 32, 16), (2, 6, 4)])
    def test_split_matches_the_grouped_convolution(self, groups, cin, cout):
        import torch.nn.functional as F

        from pymss_core.vendor_shims import conv1d_grouped_equivalent

        torch.manual_seed(0)
        x = torch.randn(2, cin, 21)
        w = torch.randn(cout, cin // groups, 3)
        b = torch.randn(cout)
        ref = F.conv1d(x, w, b, 1, 1, 1, groups)
        got = conv1d_grouped_equivalent(x, w, b, 1, 1, 1, groups)
        torch.testing.assert_close(got, ref, rtol=1e-5, atol=1e-6)

    def test_biren_profile_asks_for_it(self):
        from pymss_core.device import clear_device_cache, device_profile

        clear_device_cache()
        assert device_profile("supa").grouped_conv1d is True


class TestComplexCat:
    """torch.cat of complex tensors dies inside Concat.cpp (SUDNN, ErrorCode 11)."""

    def test_concatenates_through_the_planes(self):
        from pymss_core.vendor_shims import install_complex_cat_shim, reset_shims

        reset_shims()
        a = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        b = torch.complex(torch.randn(2, 4), torch.randn(2, 4))
        want = torch.cat([a, b], dim=1)
        install_complex_cat_shim()
        got = torch.cat([a, b], dim=1)
        assert torch.equal(got, want)
        assert got.dtype == torch.complex64
        reset_shims()

    def test_mixed_dtype_list_is_left_to_torch(self):
        """torch promotes a mixed list itself; the shim must not interfere."""
        from pymss_core.vendor_shims import install_complex_cat_shim, reset_shims

        reset_shims()
        a = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        b = torch.randn(2, 3)
        want = torch.cat([a, b], dim=1)
        install_complex_cat_shim()
        got = torch.cat([a, b], dim=1)
        assert got.dtype == want.dtype == torch.complex64
        assert torch.equal(got, want)
        reset_shims()

    def test_real_cat_is_unaffected(self):
        from pymss_core.vendor_shims import install_complex_cat_shim, reset_shims

        reset_shims()
        a, b = torch.randn(2, 3), torch.randn(2, 3)
        want = torch.cat([a, b], dim=1)
        install_complex_cat_shim()
        assert torch.equal(torch.cat([a, b], dim=1), want)
        reset_shims()


class TestStdEquivalent:
    """std.cpp answers 'SUPA Kernel Launch Error' for legacy_v3_demucs' shapes."""

    @pytest.mark.parametrize(
        ("dim", "keepdim", "unbiased"),
        [(-1, True, True), ((1, 2, 3), True, False), (1, False, True), (None, False, False)],
    )
    def test_matches_torch_std(self, dim, keepdim, unbiased):
        from pymss_core.vendor_shims import std_equivalent

        torch.manual_seed(0)
        x = torch.randn(2, 3, 4, 5)
        want = x.std(dim=dim, keepdim=keepdim, unbiased=unbiased)
        got = std_equivalent(x, dim=dim, keepdim=keepdim, unbiased=unbiased)
        assert torch.allclose(got, want, atol=1e-6)


class TestCpuCopy:
    """A non-contiguous device tensor cannot be copied to the host directly."""

    def test_contiguous_copy_matches(self):
        from pymss_core.vendor_shims import install_cpu_shim, reset_shims

        reset_shims()
        base = torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)
        view = base[:, :, ::2]
        assert not view.is_contiguous()
        want = view.clone()
        install_cpu_shim()
        got = view.cpu()
        assert torch.equal(got, want)
        reset_shims()

    def test_contiguous_tensor_is_untouched(self):
        from pymss_core.vendor_shims import install_cpu_shim, reset_shims

        reset_shims()
        x = torch.randn(3, 4)
        install_cpu_shim()
        assert torch.equal(x.cpu(), x.clone())
        reset_shims()


class TestCpuCopyComplex:
    """Complex host copies fail with 'ErrorCode: 101, invalid device'."""

    def test_complex_copy_preserves_values(self):
        from pymss_core.vendor_shims import install_cpu_shim, reset_shims

        reset_shims()
        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        assert torch.equal(x.cpu(), x)
        reset_shims()

    def test_non_contiguous_complex_copy(self):
        from pymss_core.vendor_shims import install_cpu_shim, reset_shims

        reset_shims()
        base = torch.complex(torch.randn(4, 6), torch.randn(4, 6))
        view = base[:, ::2]
        assert not view.is_contiguous()
        install_cpu_shim()
        assert torch.equal(view.cpu(), view.clone())
        reset_shims()


class TestComplexToHost:
    """torch.Tensor has no .view_as_real method; the plane copy is a helper."""

    def test_round_trip_keeps_values(self):
        from pymss_core.vendor_shims import complex_to_host

        x = torch.complex(torch.randn(2, 3), torch.randn(2, 3))
        out = complex_to_host(x, lambda t: t)
        assert out.dtype == torch.complex64
        assert torch.equal(out, x)

    def test_handles_strided_input(self):
        from pymss_core.vendor_shims import complex_to_host

        base = torch.complex(torch.randn(4, 6), torch.randn(4, 6))
        view = base[:, ::2]
        out = complex_to_host(view, lambda t: t)
        assert torch.equal(out, view.clone())


class TestSdpaEquivalent:
    """The fused kernel cannot be planned for a production roformer's batch."""

    def test_matches_torch_math_backend(self):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 6, 8) for _ in range(3))
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        assert torch.allclose(sdpa_equivalent(q, k, v), want, atol=1e-5)

    def test_scale_and_causal(self):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 5, 4) for _ in range(3))
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v, is_causal=True, scale=0.3)
        assert torch.allclose(sdpa_equivalent(q, k, v, is_causal=True, scale=0.3), want, atol=1e-5)

    def test_additive_mask(self):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(1, 2, 3, 4) for _ in range(3))
        mask = torch.zeros(3, 3).masked_fill(torch.ones(3, 3, dtype=torch.bool).tril().logical_not(), float("-inf"))
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        assert torch.allclose(sdpa_equivalent(q, k, v, mask), want, atol=1e-5)


class TestSdpaChunking:
    """Some folded batch sizes cannot be planned, so the batch is walked in blocks."""

    def test_chunked_matches_unchunked(self, monkeypatch):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(4, 3, 5, 8) for _ in range(3))
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "0")
        whole = sdpa_equivalent(q, k, v)
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "5")
        piecewise = sdpa_equivalent(q, k, v)
        assert torch.allclose(whole, piecewise, atol=1e-6)

    def test_chunked_causal_matches_unchunked(self, monkeypatch):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 2, 4, 8) for _ in range(3))
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "0")
        whole = sdpa_equivalent(q, k, v, is_causal=True)
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "3")
        piecewise = sdpa_equivalent(q, k, v, is_causal=True)
        assert torch.allclose(whole, piecewise, atol=1e-6)


class TestSdpaTransposedKey:
    """A strided k^T runs the GEMM at a third of the speed, so it is materialised."""

    def test_unit_scale_path_works(self, monkeypatch):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 3, 4, 8) for _ in range(3))
        want = torch.nn.functional.scaled_dot_product_attention(q, k, v, scale=1.0)
        assert torch.allclose(sdpa_equivalent(q, k, v, scale=1.0), want, atol=1e-5)

    def test_chunked_matches_unchunked_with_contiguous_key(self, monkeypatch):
        from pymss_core.vendor_shims import sdpa_equivalent

        torch.manual_seed(0)
        q, k, v = (torch.randn(6, 3, 4, 8) for _ in range(3))
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "0")
        whole = sdpa_equivalent(q, k, v)
        monkeypatch.setenv("PYMSS_SDPA_BATCH_CHUNK", "4")
        piecewise = sdpa_equivalent(q, k, v)
        assert torch.allclose(whole, piecewise, atol=1e-6)


class _FakeTensor:
    """Duck-typed tensor: the shim only reads ``dtype``, ``device`` and ``shape``."""

    def __init__(self, kind, dtype, shape):
        self.dtype = dtype
        self.shape = shape
        self.device = SimpleNamespace(type=kind, index=0)


class TestRopeKernelTable:
    """The fused kernel wants the angles interleaved, not in halves."""

    def test_even_lanes_carry_the_reference_angles(self):
        from pymss_core.modules.bs_roformer.common import RotaryEmbedding
        from pymss_core.modules.bs_roformer.transformer import cached_rotary_cos_sin
        from pymss_core.vendor_shims import _fused_rope_table

        rotary, device, seq_len, dim = RotaryEmbedding(8), torch.device("cpu"), 4, 8
        table = _fused_rope_table(rotary, seq_len, device, torch.float32)
        cos, _ = cached_rotary_cos_sin(rotary, seq_len, device, torch.float32)
        assert table.shape == (seq_len, dim)
        # One angle per axis element, duplicated: widening in halves instead
        # would rotate the wrong pairs.
        assert torch.allclose(table[..., ::2].cos(), cos[0, :, 0, :])
        assert torch.allclose(table[..., 1::2].cos(), cos[0, :, 0, :])


class TestRopeKernelShim:
    """``aten::_fused_rope_forward`` replaces the reference rotation on MUSA only.

    Measured on MTT S70 + torch_musa 2.7.1: 372 us -> 49 us for the q/k pair of a
    roformer at ``(1, 517, 8, 64)``, with attention scores matching to 3.6e-07.
    """

    def test_missing_vendor_kernel_installs_nothing(self, monkeypatch):
        from pymss_core import vendor_shims

        reset_shims()
        monkeypatch.setattr(vendor_shims, "_fused_rope_op", lambda: None)
        assert vendor_shims.install_rope_kernel_shim("musa") is False
        assert installed_shims() == frozenset()

    def test_routes_musa_fp32_through_the_kernel_and_restores(self, monkeypatch):
        from pymss_core import vendor_shims
        from pymss_core.modules.bs_roformer import transformer as transformer_module

        fused_calls, fallbacks = [], []

        def fused(tensor, table, interleaved, conjugate):
            fused_calls.append((tensor.device.type, tuple(table.shape), interleaved, conjugate))
            return tensor

        def original(rotary_embed, q, k):
            fallbacks.append(q.device.type)
            return q, k

        monkeypatch.setattr(vendor_shims, "_fused_rope_op", lambda: fused)
        # The table needs a real device, so it is stubbed out here; its own
        # correctness is covered by TestRopeKernelTable.
        monkeypatch.setattr(vendor_shims, "_fused_rope_table", lambda *args: torch.ones(4, 8))
        monkeypatch.setattr(transformer_module, "rotate_qk_fast_bnhd", original)
        reset_shims()
        try:
            assert vendor_shims.install_rope_kernel_shim("musa") is True
            assert "rope_kernel" in installed_shims()
            rotate = transformer_module.rotate_qk_fast_bnhd
            rotary = SimpleNamespace(dim=8)

            rotate(rotary, _FakeTensor("musa", torch.float32, (1, 4, 2, 8)), _FakeTensor("musa", torch.float32, (1, 4, 2, 8)))
            assert fused_calls == [("musa", (4, 8), True, True), ("musa", (4, 8), True, True)]

            # Half precision keeps the reference implementation even on MUSA,
            # and so does every other device type.
            rotate(rotary, _FakeTensor("musa", torch.float16, (1, 4, 2, 8)), _FakeTensor("musa", torch.float16, (1, 4, 2, 8)))
            rotate(rotary, torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2, 8))
            assert fallbacks == ["musa", "cpu"]
            assert len(fused_calls) == 2
        finally:
            reset_shims()
        assert transformer_module.rotate_qk_fast_bnhd is original
        assert installed_shims() == frozenset()
