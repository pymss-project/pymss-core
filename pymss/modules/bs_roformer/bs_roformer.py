import torch
from torch import nn
from torch.nn import Module

from beartype.typing import Tuple, Optional, Callable
from beartype import beartype

from .common import (
    DEFAULT_FREQS_PER_BANDS,
    MaskEstimator,
    RMSNorm,
    RoformerRuntimeMixin,
    forward_bandsplit_roformer,
    forward_roformer_mask_core,
    ignore_roformer_training_kwargs,
    init_roformer_band_modules,
    init_roformer_layers,
    init_roformer_runtime,
    init_roformer_stft,
    roformer_freqs_per_bands_with_complex,
    validate_roformer_attention_options,
)


class BSRoformer(RoformerRuntimeMixin, Module):

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
            mlp_expansion_factor=4,
            use_torch_checkpoint=False,
            skip_connection=False,
            sage_attention=False,
            sage_attention_mode='none',
            attention_layout='bhnd',
            use_shared_bias=False,
            **kwargs,
    ):
        super().__init__()
        ignore_roformer_training_kwargs(kwargs)
        init_roformer_runtime(self, stereo, num_stems, use_torch_checkpoint, skip_connection)
        validate_roformer_attention_options(sage_attention_mode, attention_layout)

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

        init_roformer_layers(
            self,
            dim=dim,
            depth=depth,
            time_transformer_depth=time_transformer_depth,
            freq_transformer_depth=freq_transformer_depth,
            linear_transformer_depth=linear_transformer_depth,
            dim_head=dim_head,
            sage_attention_mode=sage_attention_mode,
            transformer_kwargs=transformer_kwargs,
        )

        self.final_norm = RMSNorm(dim)
        init_roformer_stft(self, stft_n_fft, stft_hop_length, stft_win_length, stft_normalized, stft_window_fn)

        freqs = torch.stft(torch.randn(1, 4096), **self.stft_kwargs, window=torch.ones(stft_win_length), return_complex=True).shape[1]
        freqs_per_bands_with_complex = roformer_freqs_per_bands_with_complex(self, freqs_per_bands, freqs)
        init_roformer_band_modules(
            self,
            dim=dim,
            freqs_per_bands_with_complex=freqs_per_bands_with_complex,
            num_stems=num_stems,
            mask_estimator_cls=MaskEstimator,
            mask_estimator_depth=mask_estimator_depth,
            mlp_expansion_factor=mlp_expansion_factor,
        )

    def _forward_mask_core(self, stft_repr):
        return forward_roformer_mask_core(
            self,
            stft_repr,
            use_checkpoint=self.training and self.use_torch_checkpoint,
        )

    def forward(self, raw_audio):
        return forward_bandsplit_roformer(self, raw_audio)
