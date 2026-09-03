import torch

from .bandit.tfmodel import ResidualRNN, Transpose
from .mlx_backend import check_dtype, generic_activation, gelu, glu, layer_norm, group_norm, istft, linear, mx_dtype, param, periodic_hann_window, relu, rnn_forward, stft, to_mx, to_torch

torch_to_mlx_input = to_mx


def _spectral_stft(stft_module, raw_audio, dtype):
    n_fft, win_length, hop = int(stft_module.n_fft), int(stft_module.win_length), int(stft_module.hop_length)
    window = periodic_hann_window(win_length, dtype)
    if win_length < n_fft:
        import mlx.core as mx

        left = (n_fft - win_length) // 2
        window = mx.pad(window, [(left, n_fft - win_length - left)])
    elif win_length > n_fft:
        raise ValueError("MLX Bandit STFT does not support win_length > n_fft")
    spec = stft(raw_audio, n_fft, hop, window, dtype, center=stft_module.center, pad_mode=stft_module.pad_mode,
                normalized=stft_module.normalized)
    return spec, {"n_fft": n_fft, "hop": hop, "window": window, "normalized": stft_module.normalized,
                  "center": stft_module.center, "dtype": dtype}


def _spectral_istft(istft_module, spec, context, length):
    return istft(spec, context["window"], context["hop"], length, context["dtype"], n_fft=context["n_fft"],
                 center=context["center"], normalized=context["normalized"])


_activation = generic_activation


def _norm_fc(module, xb, dtype):
    if hasattr(module, "combined"):
        xb = layer_norm(module.combined[0], xb, dtype)
        return linear(xb, param(module.combined[1], "weight", module.combined[1].weight, dtype),
                      param(module.combined[1], "bias", module.combined[1].bias, dtype))
    batch, n_time, in_channels, ribw = xb.shape
    xb = layer_norm(module.norm, xb.reshape(batch, n_time, in_channels * ribw), dtype)
    w = param(module.fc, "weight", module.fc.weight, dtype)
    b = param(module.fc, "bias", module.fc.bias, dtype)
    if module.treat_channel_as_feature:
        return linear(xb, w, b)
    return linear(xb.reshape(batch, n_time, in_channels, ribw), w, b).reshape(batch, n_time, -1)


def _band_split(module, x, dtype):
    import mlx.core as mx

    batch, in_channels, _, n_time = x.shape
    xr = mx.stack((x.real, x.imag), axis=-1)
    if module.complex_order == "reim_freq":
        xr = xr.transpose(0, 3, 1, 4, 2)
    elif module.complex_order == "freq_reim":
        xr = xr.transpose(0, 3, 1, 2, 4)
    else:
        raise ValueError(f"unsupported complex_order: {module.complex_order}")
    outs = []
    for i, nfm in enumerate(module.norm_fc_modules):
        fstart, fend = module.band_specs[i]
        xb = (xr[..., fstart:fend] if module.complex_order == "reim_freq" else xr[:, :, :, fstart:fend]).reshape(
            batch, n_time, in_channels, -1)
        outs.append(_norm_fc(nfm, xb.reshape(batch, n_time, -1) if module.flatten_input else xb, dtype))
    return mx.stack(outs, axis=1)


def _residual_rnn(module, z, dtype):
    import mlx.core as mx

    z0 = z
    if module.use_layer_norm:
        z = layer_norm(module.norm, z, dtype)
    else:
        z = group_norm(module.norm, z.transpose(0, 3, 1, 2), dtype).transpose(0, 2, 3, 1)
    batch, n_uncrossed, n_across, emb_dim = z.shape
    if module.use_batch_trick:
        z = rnn_forward(module.rnn, z.reshape(batch * n_uncrossed, n_across, emb_dim), dtype)
        z = z.reshape(batch, n_uncrossed, n_across, -1)
    else:
        z = mx.stack([rnn_forward(module.rnn, z[:, i], dtype) for i in range(n_uncrossed)], axis=1)
    return linear(z, param(module.fc, "weight", module.fc.weight, dtype), param(module.fc, "bias", module.fc.bias, dtype)) + z0


def _tf_model(module, z, dtype):
    if module.parallel_mode:
        for sbm_t, sbm_f in module.seqband:
            zt = _residual_rnn(sbm_t, z, dtype)
            zf = _residual_rnn(sbm_f, z.transpose(0, 2, 1, 3), dtype)
            z = zt + zf.transpose(0, 2, 1, 3)
        return z
    if isinstance(module.seqband, torch.nn.Sequential):
        for layer in module.seqband:
            if isinstance(layer, ResidualRNN):
                z = _residual_rnn(layer, z, dtype)
            elif isinstance(layer, Transpose):
                z = z.swapaxes(layer.dim0, layer.dim1)
            else:
                raise TypeError(f"unsupported Bandit TF layer for MLX full backend: {type(layer).__name__}")
        return z
    for sbm in module.seqband:
        z = _residual_rnn(sbm, z, dtype)
        z = z.swapaxes(1, 2)
    return z


def _norm_mlp(module, qb, dtype):
    x = layer_norm(module.norm, qb, dtype)
    x = linear(x, param(module.hidden[0], "weight", module.hidden[0].weight, dtype),
               param(module.hidden[0], "bias", module.hidden[0].bias, dtype))
    x = _activation(module.hidden[1], x)
    output = module.output[0]
    x = glu(linear(x, param(output, "weight", output.weight, dtype), param(output, "bias", output.bias, dtype)), axis=-1)
    batch, n_time, _ = x.shape
    if module.complex_mask:
        x = x.reshape(batch, n_time, module.in_channels, module.bandwidth, 2)
        x = x[..., 0] + (1j * x[..., 1])
    else:
        x = x.reshape(batch, n_time, module.in_channels, module.bandwidth)
    return x.transpose(0, 2, 3, 1)


def _append_cond(module, q, cond):
    import mlx.core as mx

    if cond is not None:
        batch, n_bands, n_time, _ = q.shape
        if cond.ndim == 2:
            cond = mx.broadcast_to(cond[:, None, None, :], (batch, n_bands, n_time, cond.shape[-1]))
        elif cond.ndim != 3:
            raise ValueError(f"Invalid cond shape: {cond.shape}")
        return mx.concatenate((q, cond), axis=-1)
    if module.cond_dim <= 0:
        return q
    batch, n_bands, n_time, _ = q.shape
    return mx.concatenate((q, mx.ones((batch, n_bands, n_time, module.cond_dim), dtype=q.dtype)), axis=-1)


def _mask_estimator(module, q, dtype, cond=None):
    import mlx.core as mx

    q = _append_cond(module, q, cond)
    if getattr(module, "n_freq", 0) <= 0:
        return mx.concatenate([_norm_mlp(nmlp, q[:, b], dtype) for b, nmlp in enumerate(module.norm_mlp)], axis=2)
    batch, _, n_time, _ = q.shape
    mask_real = mx.zeros((batch, module.in_channels, module.n_freq, n_time), dtype=mx.float32)
    mask_imag = mx.zeros_like(mask_real)
    for band_index, nmlp in enumerate(module.norm_mlp):
        fstart, fend = module.band_specs[band_index]
        mask = _norm_mlp(nmlp, q[:, band_index], dtype)
        if module.use_freq_weights:
            fw = to_mx(module.get_buffer(f"freq_weights/{band_index}"), dtype)
            mask = mask * fw.reshape(1, 1, -1, 1)
        padding = [(0, 0), (0, 0), (fstart, module.n_freq - fend), (0, 0)]
        mask_real = mask_real + mx.pad(mask.real.astype(mask_real.dtype), padding)
        mask_imag = mask_imag + mx.pad(mask.imag.astype(mask_imag.dtype), padding)
    return mask_real + (1j * mask_imag)


def _bsrnn_core(module, x, dtype):
    batch, in_chan, n_freq, n_time = x.shape
    x = x.reshape(-1, 1, n_freq, n_time)
    q = _tf_model(module.tf_model, _band_split(module.band_split, x, dtype), dtype)
    return [_mask_estimator(mask_estimator, q, dtype) * x for mask_estimator in module.mask_estim.values()]


def mlx_forward_bandit_mx(module, raw_audio, dtype=torch.float16):
    check_dtype(dtype, "Bandit")
    dtype = mx_dtype(dtype)
    init_shape = raw_audio.shape
    mono = raw_audio.reshape(-1, 1, raw_audio.shape[-1]).astype(dtype)
    x, context = _spectral_stft(module.stft, mono, dtype)
    length = mono.shape[-1]
    if hasattr(module, "bsrnn"):
        specs = _bsrnn_core(module.bsrnn, x, dtype)
    else:
        q = _tf_model(module.tf_model, _band_split(module.band_split, x, dtype), dtype)
        specs = [_mask_estimator(mask_estimator, q, dtype) * x for mask_estimator in module.mask_estim.values()]
    estimates = [_spectral_istft(module.istft, spec, context, length) for spec in specs]
    estimates = [estimate.reshape(-1, init_shape[1], init_shape[2]) for estimate in estimates]
    import mlx.core as mx

    return mx.stack(estimates, axis=1)


def mlx_forward_bandit(module, raw_audio, dtype=torch.float16):
    return to_torch(mlx_forward_bandit_mx(module, to_mx(raw_audio, dtype=dtype), dtype), raw_audio)
