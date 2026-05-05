from functools import partial

import torch
from torch import nn
from torch.nn import Module, ModuleList
import torch.nn.functional as F

from beartype.typing import Tuple, Optional, Callable
from beartype import beartype
from rotary_embedding_torch import RotaryEmbedding
from einops import rearrange

from .common import (
    BandSplit,
    MaskEstimator,
    RMSNorm,
    Transformer,
    default,
    exists,
    forward_roformer_mask_core,
    set_rmsnorm_fp32,
)


# main class



DEFAULT_FREQS_PER_BANDS = (
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
    2, 2, 2, 2,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    12, 12, 12, 12, 12, 12, 12, 12,
    24, 24, 24, 24, 24, 24, 24, 24,
    48, 48, 48, 48, 48, 48, 48, 48,
    128, 129,
)


class BSRoformer(Module):

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
            freqs_per_bands: Tuple[int, ...] = DEFAULT_FREQS_PER_BANDS,
            # in the paper, they divide into ~60 bands, test with 1 for starters
            dim_head=64,
            heads=8,
            attn_dropout=0.,
            ff_dropout=0.,
            flash_attn=True,
            dim_freqs_in=1025,
            stft_n_fft=2048,
            stft_hop_length=512,
            # 10ms at 44100Hz, from sections 4.1, 4.4 in the paper - @faroit recommends // 2 or // 4 for better reconstruction
            stft_win_length=2048,
            stft_normalized=False,
            stft_window_fn: Optional[Callable] = None,
            mask_estimator_depth=2,
            multi_stft_resolution_loss_weight=1.,
            multi_stft_resolutions_window_sizes: Tuple[int, ...] = (4096, 2048, 1024, 512, 256),
            multi_stft_hop_size=147,
            multi_stft_normalized=False,
            multi_stft_window_fn: Callable = torch.hann_window,
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            sage_attention_mode='none',
            attention_layout='bhnd',
            use_shared_bias=False,
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

        shared_qkv_bias = None
        shared_out_bias = None
        if use_shared_bias:
            dim_inner = heads * dim_head
            self.linear_62_bias_0 = nn.Parameter(torch.ones(dim_inner * 3))
            self.linear_64_bias_0 = nn.Parameter(torch.ones(dim))
            shared_qkv_bias = self.linear_62_bias_0
            shared_out_bias = self.linear_64_bias_0

        transformer_kwargs = dict(
            dim=dim,
            heads=heads,
            dim_head=dim_head,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            flash_attn=flash_attn,
            norm_output=False,
            sage_attention=sage_attention,
            attention_layout=attention_layout,
            shared_qkv_bias=shared_qkv_bias,
            shared_out_bias=shared_out_bias,
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

        self.final_norm = RMSNorm(dim)

        self.stft_kwargs = dict(
            n_fft=stft_n_fft,
            hop_length=stft_hop_length,
            win_length=stft_win_length,
            normalized=stft_normalized
        )

        self.stft_window_fn = partial(default(stft_window_fn, torch.hann_window), stft_win_length)
        self._stft_window_cache = {}

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]

        assert len(freqs_per_bands) > 1
        assert sum(
            freqs_per_bands) == freqs, f'the number of freqs in the bands must equal {freqs} based on the STFT settings, but got {sum(freqs_per_bands)}'

        freqs_per_bands_with_complex = tuple(2 * f * self.audio_channels for f in freqs_per_bands)

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

    def _forward_mask_core(self, stft_repr):
        return forward_roformer_mask_core(
            self,
            stft_repr,
            use_checkpoint=self.training and self.use_torch_checkpoint,
        )

    def _compiled_mask_core(self, stft_repr):
        mode = self.__dict__.get('_pymss_torch_compile_mode', 'default')
        cache = self.__dict__.setdefault('_pymss_compiled_mask_cores', {})
        key = (
            tuple(stft_repr.shape),
            stft_repr.device.type,
            stft_repr.device.index,
            stft_repr.dtype,
            mode,
        )
        compiled = cache.get(key)
        if compiled is None:
            self._prepare_inference_core_options()
            if self.band_split.use_grouped_forward:
                self.band_split.warm_group_cache(stft_repr.device, stft_repr.dtype)
                for mask_estimator in self.mask_estimators:
                    mask_estimator.warm_group_cache(stft_repr.device, stft_repr.dtype)
            compiled = torch.compile(self._forward_mask_core, mode=mode, fullgraph=False)
            cache[key] = compiled
        return compiled(stft_repr)

    def _forward_mask_core_maybe_compiled(self, stft_repr):
        if (
            not self.training
            and self.__dict__.get('_pymss_torch_compile_enabled', False)
            and self.__dict__.get('_pymss_torch_compile_scope') == 'core'
            and self.__dict__.get('_pymss_compile_core_this_call', True)
        ):
            return self._compiled_mask_core(stft_repr)
        return self._forward_mask_core(stft_repr)

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

        # defining whether model is loaded on MPS (MacOS GPU accelerator)
        x_is_mps = True if device.type == "mps" else False

        if raw_audio.ndim == 2:
            raw_audio = raw_audio.unsqueeze(1)

        batch, audio_channels, audio_length = raw_audio.shape
        channels = raw_audio.shape[1]
        assert (not self.stereo and channels == 1) or (self.stereo and channels == 2), 'stereo needs to be set to True if passing in audio signal that is stereo (channel dimension of 2). also need to be False if mono (channel dimension of 1)'

        # to stft

        stft_audio = raw_audio.reshape(batch * audio_channels, audio_length)

        stft_window = self.stft_window(device)

        # RuntimeError: FFT operations are only supported on MacOS 14+
        # Since it's tedious to define whether we're on correct MacOS version - simple try-catch is used
        try:
            stft_repr = torch.stft(stft_audio, **self.stft_kwargs, window=stft_window, return_complex=True)
        except:
            stft_repr = torch.stft(
                stft_audio.cpu() if x_is_mps else stft_audio,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=True
            ).to(device)

        stft_repr = torch.view_as_real(stft_repr)

        stft_repr = stft_repr.reshape(batch, audio_channels, *stft_repr.shape[-3:])

        # merge stereo / mono into the frequency, with frequency leading dimension, for band splitting
        b, s, f, t, c = stft_repr.shape
        stft_freq_bins = f
        stft_repr = stft_repr.permute(0, 2, 1, 3, 4).reshape(b, f * s, t, c)

        num_stems = len(self.mask_estimators)
        self._prepare_inference_core_options()
        mask = self._forward_mask_core_maybe_compiled(stft_repr)

        # modulate frequency representation

        stft_repr = stft_repr.unsqueeze(1)

        # complex number multiplication

        stft_repr = torch.view_as_complex(stft_repr)
        mask = torch.view_as_complex(mask.contiguous())

        stft_repr = stft_repr * mask

        # istft

        b, n, fs, t = stft_repr.shape
        stft_repr = stft_repr.reshape(b, n, stft_freq_bins, s, t).permute(0, 1, 3, 2, 4).reshape(
            b * n * s,
            stft_freq_bins,
            t
        )

        # same as torch.stft() fix for MacOS MPS above
        try:
            recon_audio = torch.istft(
                stft_repr,
                **self.stft_kwargs,
                window=stft_window,
                return_complex=False,
                length=audio_length
            )
        except:
            recon_audio = torch.istft(
                stft_repr.cpu() if x_is_mps else stft_repr,
                **self.stft_kwargs,
                window=stft_window.cpu() if x_is_mps else stft_window,
                return_complex=False,
                length=audio_length
            ).to(device)

        recon_audio = recon_audio.reshape(batch, num_stems, audio_channels, audio_length)

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
