from functools import partial
import torch
import torch.nn as nn
from transformers import UperNetForSemanticSegmentation

from .spectrogram import SubbandSTFT, cac_to_cws, cws_to_cac, get_activation

def get_norm(norm_type):
    def norm(c, norm_type):
        if norm_type == 'BatchNorm':
            return nn.BatchNorm2d(c)
        elif norm_type == 'InstanceNorm':
            return nn.InstanceNorm2d(c, affine=True)
        elif 'GroupNorm' in norm_type:
            g = int(norm_type.replace('GroupNorm', ''))
            return nn.GroupNorm(num_groups=g, num_channels=c)
        else:
            return nn.Identity()

    return partial(norm, norm_type=norm_type)


class Upscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class Downscale(nn.Module):
    def __init__(self, in_c, out_c, scale, norm, act):
        super().__init__()
        self.conv = nn.Sequential(
            norm(in_c),
            act,
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=scale, stride=scale, bias=False)
        )

    def forward(self, x):
        return self.conv(x)


class TFC_TDF(nn.Module):
    def __init__(self, in_c, c, l, f, bn, norm, act):
        super().__init__()

        self.blocks = nn.ModuleList()
        for i in range(l):
            block = nn.Module()

            block.tfc1 = nn.Sequential(
                norm(in_c),
                act,
                nn.Conv2d(in_c, c, 3, 1, 1, bias=False),
            )
            block.tdf = nn.Sequential(
                norm(c),
                act,
                nn.Linear(f, f // bn, bias=False),
                norm(c),
                act,
                nn.Linear(f // bn, f, bias=False),
            )
            block.tfc2 = nn.Sequential(
                norm(c),
                act,
                nn.Conv2d(c, c, 3, 1, 1, bias=False),
            )
            block.shortcut = nn.Conv2d(in_c, c, 1, 1, 0, bias=False)

            self.blocks.append(block)
            in_c = c

    def forward(self, x):
        for block in self.blocks:
            s = block.shortcut(x)
            x = block.tfc1(x)
            x = x + block.tdf(x)
            x = block.tfc2(x)
            x = x + s
        return x


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

    def forward(self, x):

        x = self.stft(x)

        mix = x = cac_to_cws(x, self.num_subbands)

        first_conv_out = x = self.first_conv(x)

        x = x.transpose(-1, -2)

        x = self.swin_upernet_model(x).logits

        x = x.transpose(-1, -2)

        x = x * first_conv_out  # reduce artifacts

        x = self.final_conv(torch.cat([mix, x], 1))

        x = cws_to_cac(x, self.num_subbands)

        if self.num_target_instruments > 1:
            b, c, f, t = x.shape
            x = x.reshape(b, self.num_target_instruments, -1, f, t)

        x = self.stft.inverse(x)
        return x
