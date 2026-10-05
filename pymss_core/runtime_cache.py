"""Lifecycle cleanup for derived model tensors, independent of inference backend."""


def clear_model_runtime_caches(model):
    """Release known inference caches and module-owned hooks without changing weights."""
    cache_names = ("_stft_window_cache", "_pymss_cos_sin_cache", "_gamma_dtype_cache", "_group_cache", "_layer_group_cache", "_index_cache", "_packed_layer_group_cache", "_apollo_inference_cache", "_rotary_freq_cache", "_packed_cache")
    first_error = None
    for module in model.modules():
        clear = getattr(module, "clear_runtime_cache", None)
        if callable(clear):
            try: clear()
            except Exception as error:
                if first_error is None: first_error = error
        for name in cache_names:
            cache = getattr(module, name, None)
            if isinstance(cache, dict): cache.clear()
        if hasattr(module, "_pymss_group_cache_warm_key"): module._pymss_group_cache_warm_key = None
    if first_error is not None: raise first_error
