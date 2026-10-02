import copy

import pytest
import torch

from pymss_core import clear_model_runtime_caches
from pymss_core.modules.bs_roformer.common import RotaryEmbedding


def test_runtime_cache_cleanup_releases_cached_tensors_without_changing_weights_or_user_data():
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), RotaryEmbedding(4))
    model[0].register_buffer("persistent_data", torch.ones(4))
    weights = copy.deepcopy(model.state_dict())
    packed_cache = model[0]._packed_cache = {"weights": torch.ones(4)}
    model[0]._pymss_cos_sin_cache = {"position": torch.ones(4)}
    model[0]._pymss_group_cache_warm_key = ("cpu", torch.float32)
    model[0].cache = {"user": "keep"}
    model[1].cache["frequency"] = torch.ones(4)
    for _ in range(2):
        clear_model_runtime_caches(model)
        assert model[0]._packed_cache is packed_cache and not packed_cache
        assert not model[0]._pymss_cos_sin_cache and not model[1].cache
        assert model[0].cache == {"user": "keep"}
        assert model[0]._pymss_group_cache_warm_key is None
        for key, value in weights.items(): torch.testing.assert_close(model.state_dict()[key], value, rtol=0, atol=0)


def test_runtime_cache_cleanup_visits_shared_modules_once_and_preserves_hook_errors():
    class CachedModule(torch.nn.Module):
        def __init__(self): super().__init__(); self.calls = 0; self.failure = None
        def clear_runtime_cache(self):
            self.calls += 1
            if self.failure is not None: raise self.failure
    child = CachedModule()
    model = torch.nn.ModuleList([child, child])
    clear_model_runtime_caches(model)
    assert child.calls == 1
    child.failure = RuntimeError("Runtime cache cleanup failed")
    with pytest.raises(RuntimeError) as caught: clear_model_runtime_caches(model)
    assert caught.value is child.failure
    assert child.calls == 2
