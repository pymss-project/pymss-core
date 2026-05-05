from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype.typing import Tuple, Optional, Callable
from beartype import beartype

from rotary_embedding_torch import RotaryEmbedding

from einops import rearrange, reduce, repeat
from librosa import filters

from .common import (
    BandSplit,
    MaskEstimator,
    Transformer,
    default,
    exists,
    forward_roformer_mask_core,
    set_rmsnorm_fp32,
)


# main class

class MelBandRoformer(Module):

    @beartype
    def __init__(
            self,
            dim,
            *,
            depth,
            stereo=False,
            num_stems=1,
            time_transformer_depth=2,
            freq_transformer_depth=2,
            linear_transformer_depth=0,
            num_bands=60,
            dim_head=64,
            heads=8,
            attn_dropout=0.1,
            ff_dropout=0.1,
            flash_attn=True,
            dim_freqs_in=1025,
            sample_rate=44100,  # needed for mel filter bank from librosa
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=1,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            match_input_audio_length=False,  # if True, pad output tensor to match length of input tensor
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            sage_attention_mode='none',
            attention_layout='bhnd',
    ):
        super().__init__()

        self.stereo = stereo
        self.audio_channels = 2 if stereo else 1
        self.num_stems = num_stems
        self.use_torch_checkpoint = use_torch_checkpoint
        self.skip_connection = skip_connection
        self.inference_layer_skip = None
        self.inference_time_layer_skip = None
        self.inference_freq_layer_skip = None
        self.inference_grouped_band_ops = True
        self.inference_rmsnorm_fp32 = True
        self._rmsnorm_fp32_state = None

        self.layers = ModuleList([])
        valid_sage_modes = {'none', 'time', 'freq', 'all'}
        if sage_attention_mode not in valid_sage_modes:
            raise ValueError(f"sage_attention_mode must be one of {sorted(valid_sage_modes)}")
        valid_attention_layouts = {'bhnd', 'bnhd'}
        if attention_layout not in valid_attention_layouts:
            raise ValueError(f"attention_layout must be one of {sorted(valid_attention_layouts)}")

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            sage_attention=sage_attention,
            attention_layout=attention_layout,
        )

        time_rotary_embed = RotaryEmbedding(dim=dim_head)
        freq_rotary_embed = RotaryEmbedding(dim=dim_head)

        for _ in range(depth):
            tran_modules = []
            if linear_transformer_depth > 0:
                tran_modules.append(Transformer(depth=linear_transformer_depth, linear_attn=True, **transformer_kwargs))
            tran_modules.append(
                Transformer(
                    depth=time_transformer_depth,
                    rotary_embed=time_rotary_embed,
                    sage_mode=sage_attention_mode in ('time', 'all'),
                    **transformer_kwargs
                )
            )
            tran_modules.append(
                Transformer(
                    depth=freq_transformer_depth,
                    rotary_embed=freq_rotary_embed,
                    sage_mode=sage_attention_mode in ('freq', 'all'),
                    **transformer_kwargs
                )
            )
            self.layers.append(nn.ModuleList(tran_modules))

        self.final_norm = nn.Identity()

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self._stft_window_cache = {}

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_n_fft), return_complex=True).shape[1]

        # create mel filter bank
        # with librosa.filters.mel as in section 2 of paper

        mel_filter_bank_numpy = filters.mel(sr=sample_rate, n_fft=stft_n_fft, n_mels=num_bands)

        mel_filter_bank = torch.from_numpy(mel_filter_bank_numpy)

        # for some reason, it doesn't include the first freq? just force a value for now

        mel_filter_bank[0][0] = 1.

        # In some systems/envs we get 0.0 instead of ~1.9e-18 in the last position,
        # so let's force a positive value

        mel_filter_bank[-1, -1] = 1.

        # binary as in paper (then estimated masks are averaged for overlapping regions)

        freqs_per_band = mel_filter_bank > 0
        assert freqs_per_band.any(dim=0).all(), 'all frequencies need to be covered by all bands for now'

        repeated_freq_indices = repeat(torch.arange(freqs), 'f -> b f', b=num_bands)
        freq_indices = repeated_freq_indices[freqs_per_band]

        if stereo:
            freq_indices = repeat(freq_indices, 'f -> f s', s=2)
            freq_indices = freq_indices * 2 + torch.arange(2)
            freq_indices = rearrange(freq_indices, 'f s -> (f s)')

        self.register_buffer('freq_indices', freq_indices, persistent=False)
        self.register_buffer('freqs_per_band', freqs_per_band, persistent=False)

        num_freqs_per_band = reduce(freqs_per_band, 'b f -> b', 'sum')
        num_bands_per_freq = reduce(freqs_per_band, 'b f -> f', 'sum')

        self.register_buffer('num_freqs_per_band', num_freqs_per_band, persistent=False)
        self.register_buffer('num_bands_per_freq', num_bands_per_freq, persistent=False)
        self.register_buffer(
            'num_bands_per_channel_freq',
            num_bands_per_freq.repeat_interleave(self.audio_channels).view(1, 1, -1, 1),
            persistent=False
        )

        # band split and mask estimator

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in num_freqs_per_band.tolist())

        self.band_split = BandSplit(
            dim=dim,
            dim_inputs=freqs_per_bands_with_complex
        )

        self.mask_estimators = nn.ModuleList([])

        for _ in range(num_stems):
            mask_estimator = MaskEstimator(
                dim=dim,
                dim_inputs=freqs_per_bands_with_complex,
                depth=mask_estimator_depth,
                mlp_expansion_factor=mlp_expansion_factor,
                mlp_hidden_layers=mask_estimator_depth,
            )

            self.mask_estimators.append(mask_estimator)

        # for the multi-resolution stft loss

        self.multi_stft_resolution_loss_weight = multi_stft_resolution_loss_weight
        self.multi_stft_resolutions_window_sizes = multi_stft_resolutions_window_sizes
        self.multi_stft_n_fft = stft_n_fft
        self.multi_stft_window_fn = multi_stft_window_fn

        self.multi_stft_kwargs = dict(
            hop_length=multi_stft_hop_size,
            normalized=multi_stft_normalized
        )

        self.match_input_audio_length = match_input_audio_length

    def stft_window(self, device):
        key = (device.type, device.index, torch.float32)
        window = self._stft_window_cache.get(key)
        if window is None or window.device != device:
            window = self.stft_window_fn(device=device)
            self._stft_window_cache[key] = window
        return window

    def _prepare_inference_core_options(self):
        rmsnorm_fp32 = bool(self.inference_rmsnorm_fp32 if not self.training else True)
        if self._rmsnorm_fp32_state is not rmsnorm_fp32:
            set_rmsnorm_fp32(self, rmsnorm_fp32)
            self._rmsnorm_fp32_state = rmsnorm_fp32

        grouped_band_ops = self.inference_grouped_band_ops if not self.training else True
        self.band_split.use_grouped_forward = bool(grouped_band_ops)
        for mask_estimator in self.mask_estimators:
            mask_estimator.use_grouped_forward = bool(grouped_band_ops)

    def _forward_mask_core(self, selected_stft_repr, full_t):
        return forward_roformer_mask_core(
            self,
            selected_stft_repr,
            use_checkpoint=self.training and self.use_torch_checkpoint,
        )

    def _compiled_mask_core(self, selected_stft_repr, full_t):
        mode = self.__dict__.get('_pymss_torch_compile_mode', 'default')
        cache = self.__dict__.setdefault('_pymss_compiled_mask_cores', {})
        key = (
            tuple(selected_stft_repr.shape),
            int(full_t),
            selected_stft_repr.device.type,
            selected_stft_repr.device.index,
            selected_stft_repr.dtype,
            mode,
        )
        compiled = cache.get(key)
        if compiled is None:
            self._prepare_inference_core_options()
            if self.band_split.use_grouped_forward:
                self.band_split.warm_group_cache(selected_stft_repr.device, selected_stft_repr.dtype)
                for mask_estimator in self.mask_estimators:
                    mask_estimator.warm_group_cache(selected_stft_repr.device, selected_stft_repr.dtype)
            compiled = torch.compile(self._forward_mask_core, mode=mode, fullgraph=False)
            cache[key] = compiled
        return compiled(selected_stft_repr, full_t)

    def _forward_mask_core_maybe_compiled(self, selected_stft_repr, full_t):
        if (
            not self.training
            and self.__dict__.get('_pymss_torch_compile_enabled', False)
            and self.__dict__.get('_pymss_torch_compile_scope') == 'core'
            and self.__dict__.get('_pymss_compile_core_this_call', True)
        ):
            return self._compiled_mask_core(selected_stft_repr, full_t)
        return self._forward_mask_core(selected_stft_repr, full_t)

    def forward(
            self,
            raw_audio,
            target=None,
            return_loss_breakdown=False
    ):
        """
        einops

        b - batch
        f - freq
        t - time
        s - audio channel (1 for mono, 2 for stereo)
        n - number of 'stems'
        c - complex (2)
        d - feature dimension
        """

        device = raw_audio.device

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, channels, raw_audio_length = raw_audio.shape

        istft_length = raw_audio_length if self.match_input_audio_length else None

        assert (not self.stereo and channels == 1) or (
                    self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        stft_audio = raw_audio.reshape(batch * channels, raw_audio_length)

        stft_window = self.stft_window(device)

        stft_repr = torch.stft(stft_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = stft_repr.reshape(batch, channels, *stft_repr.shape[-3:])

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        b, s, f, t, c = stft_repr.shape
        stft_freq_bins = f
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)

        # index out all frequencies for all frequency ranges across bands ascending in one go

        batch_arange = torch.arange(batch, device=device)[..., None]

        # account for stereo

        x = stft_repr[batch_arange, self.freq_indices]

        num_stems = len(self.mask_estimators)
        self._prepare_inference_core_options()
        masks = self._forward_mask_core_maybe_compiled(x, stft_repr.shape[-2])

        # modulate frequency representation

        stft_repr = stft_repr.unsqueeze(1)

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        masks = torch.view_as_complex(masks.contiguous())

        masks = masks.type(stft_repr.dtype)

        # need to average the estimated mask for the overlapped frequencies

        scatter_indices = self.freq_indices.view(1, 1, -1, 1).expand(batch, num_stems, -1, stft_repr.shape[-1])

        masks_summed = stft_repr.new_zeros(batch, num_stems, stft_repr.shape[2], stft_repr.shape[-1])
        masks_summed.scatter_add_(2, scatter_indices, masks)

        denom = self.num_bands_per_channel_freq

        masks_averaged = masks_summed / denom.clamp(min=1e-8)

        # modulate stft repr with estimated mask

        stft_repr = stft_repr * masks_averaged

        # istft

        b, n, fs, t = stft_repr.shape
        stft_repr = stft_repr.reshape(b, n, stft_freq_bins, s, t).permute(0, 1, 3, 2, 4).reshape(
            b * n * s,
            stft_freq_bins,
            t
        )

        recon_audio = torch.istft(stft_repr, **self.stft_kwargs, window=stft_window, return_complex=False,
                                  length=istft_length)

        recon_audio = recon_audio.reshape(batch, num_stems, channels, recon_audio.shape[-1])

        if num_stems == 1:
            recon_audio = recon_audio[:, 0]

        # if a target is passed in, calculate loss for learning

        if not exists(target):
            return recon_audio

        if self.num_stems > 1:
            assert target.ndim == 4 and target.shape[1] == self.num_stems

        if target.ndim == 2:
            target = rearrange(target, '... t -> ... 1 t')

        target = target[..., :recon_audio.shape[-1]]  # protect against lost length on istft

        loss = F.l1_loss(recon_audio, target)

        multi_stft_resolution_loss = 0.

        for window_size in self.multi_stft_resolutions_window_sizes:
            res_stft_kwargs = dict(
                n_fft=max(window_size, self.multi_stft_n_fft),  # not sure what n_fft is across multi resolution stft
                win_length=window_size,
                return_complex=True,
                window=self.multi_stft_window_fn(window_size, device=device),
                **self.multi_stft_kwargs,
            )

            recon_Y = torch.stft(rearrange(recon_audio, '... s t -> (... s) t'), **res_stft_kwargs)
            target_Y = torch.stft(rearrange(target, '... s t -> (... s) t'), **res_stft_kwargs)

            multi_stft_resolution_loss = multi_stft_resolution_loss + F.l1_loss(recon_Y, target_Y)

        weighted_multi_resolution_loss = multi_stft_resolution_loss * self.multi_stft_resolution_loss_weight

        total_loss = loss + weighted_multi_resolution_loss

        if not return_loss_breakdown:
            return total_loss

        return total_loss, (loss, multi_stft_resolution_loss)
