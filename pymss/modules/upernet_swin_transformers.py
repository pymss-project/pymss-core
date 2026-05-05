import torch.nn as nn
from transformers import UperNetForSemanticSegmentation

from .spectrogram import SubbandSTFT, forward_subband_mask_model, get_activation


class Swin_UperNet_Model(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        act = get_activation(config.model.act)

        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels
        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.swin_upernet_model = UperNetForSemanticSegmentation.from_pretrained("openmmlab/upernet-swin-large")

        self.swin_upernet_model.auxiliary_head.classifier = nn.Conv2d(256, c, kernel_size=(1, 1), stride=(1, 1))
        self.swin_upernet_model.decode_head.classifier = nn.Conv2d(512, c, kernel_size=(1, 1), stride=(1, 1))
        self.swin_upernet_model.backbone.embeddings.patch_embeddings.projection = nn.Conv2d(c, 192, kernel_size=(4, 4), stride=(4, 4))

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = SubbandSTFT(config.audio)

    def _forward_core(self, x):
        return self.swin_upernet_model(x).logits

    def forward(self, x):
        return forward_subband_mask_model(self, x, self._forward_core)
