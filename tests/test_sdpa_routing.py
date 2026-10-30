"""Exercise the installed SDPA routing closure without requiring a vendor device."""
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F

import pymss_core.vendor_shims as shims


def install_router(monkeypatch, backend):
    monkeypatch.setenv('PYMSS_SDPA_BACKEND', backend)
    monkeypatch.setattr(shims, '_installed', set())
    calls = []
    def record(route):
        def invoke(*args, **kwargs):
            calls.append(route)
            return route
        return invoke
    # Track the globals before install so monkeypatch restores them even though
    # the installer itself assigns them directly.
    monkeypatch.setattr(F, 'scaled_dot_product_attention', record('native'))
    monkeypatch.setattr(torch, 'scaled_dot_product_attention', record('native'), raising=False)
    monkeypatch.setattr(shims, 'sdpa_equivalent', record('manual'))
    monkeypatch.setattr(shims, 'sudnn_sdpa', record('sudnn'))
    assert shims.install_sdpa_shim()
    return F.scaled_dot_product_attention, calls


def query_for(device):
    return SimpleNamespace(device=SimpleNamespace(type=device), is_cuda=device == 'cuda')


@pytest.mark.parametrize('backend', ['manual', 'sudnn'])
@pytest.mark.parametrize('device', ['cpu', 'meta', 'cuda', 'mps', 'npu', 'gcu', 'mlu', 'supa'])
def test_installed_router_scopes_vendor_backend_to_supa(monkeypatch, backend, device):
    function, calls = install_router(monkeypatch, backend)
    q = query_for(device)
    expected = 'native' if device in ('cpu', 'meta', 'cuda') else (
        'sudnn' if backend == 'sudnn' and device == 'supa' else 'manual')
    assert function(q, q, q) == expected
    assert calls == [expected]


@pytest.mark.parametrize('options', [
    {'attn_mask': object()}, {'is_causal': True}, {'dropout_p': 0.1}, {'enable_gqa': True},
])
def test_sudnn_router_falls_back_to_math_for_unsupported_semantics(monkeypatch, options):
    function, calls = install_router(monkeypatch, 'sudnn')
    q = query_for('supa')
    result = function(q, q, q, **options)
    assert calls == ['manual']
    assert result == 'manual'


def test_invalid_backend_fails_during_install(monkeypatch):
    with pytest.raises(ValueError, match='manual or sudnn'):
        install_router(monkeypatch, 'unknown')


def test_removed_vendor_chunk_warns_instead_of_being_silently_ignored(monkeypatch):
    monkeypatch.setenv('PYMSS_SDPA_VENDOR_CHUNK', '1024')
    with pytest.warns(FutureWarning, match='obsolete'):
        install_router(monkeypatch, 'manual')


def test_active_backend_is_observable_and_installation_is_idempotent(monkeypatch, caplog):
    with caplog.at_level('INFO', logger='pymss_core.vendor_shims'):
        function, calls = install_router(monkeypatch, 'manual')
    assert function._pymss_sdpa_backend == 'manual'
    assert 'Installed vendor SDPA backend: manual' in caplog.text
    monkeypatch.setenv('PYMSS_SDPA_BACKEND', 'sudnn')
    assert not shims.install_sdpa_shim()
    assert F.scaled_dot_product_attention is function
    assert function._pymss_sdpa_backend == 'manual'
