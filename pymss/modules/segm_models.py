import torch.nn as nn
import segmentation_models_pytorch as smp

from .spectrogram import SubbandSTFT, forward_subband_mask_model, get_activation

def get_decoder(config, c):
    decoder = None
    decoder_options = dict()
    if config.model.decoder_type == 'unet':
        try:
            decoder_options = dict(config.decoder_unet)
        except:
            pass
        decoder = smp.Unet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'fpn':
        try:
            decoder_options = dict(config.decoder_fpn)
        except:
            pass
        decoder = smp.FPN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'unet++':
        try:
            decoder_options = dict(config.decoder_unet_plus_plus)
        except:
            pass
        decoder = smp.UnetPlusPlus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'manet':
        try:
            decoder_options = dict(config.decoder_manet)
        except:
            pass
        decoder = smp.MAnet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'linknet':
        try:
            decoder_options = dict(config.decoder_linknet)
        except:
            pass
        decoder = smp.Linknet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pspnet':
        try:
            decoder_options = dict(config.decoder_pspnet)
        except:
            pass
        decoder = smp.PSPNet(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'pan':
        try:
            decoder_options = dict(config.decoder_pan)
        except:
            pass
        decoder = smp.PAN(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3':
        try:
            decoder_options = dict(config.decoder_deeplabv3)
        except:
            pass
        decoder = smp.DeepLabV3(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    elif config.model.decoder_type == 'deeplabv3plus':
        try:
            decoder_options = dict(config.decoder_deeplabv3plus)
        except:
            pass
        decoder = smp.DeepLabV3Plus(
            encoder_name=config.model.encoder_name,
            encoder_weights="imagenet",
            in_channels=c,
            classes=c,
            **decoder_options,
        )
    return decoder


class Segm_Models_Net(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        act = get_activation(config.model.act)

        self.num_target_instruments = 1 if config.training.target_instrument else len(config.training.instruments)
        self.num_subbands = config.model.num_subbands

        dim_c = self.num_subbands * config.audio.num_channels * 2
        c = config.model.num_channels
        self.first_conv = nn.Conv2d(dim_c, c, 1, 1, 0, bias=False)

        self.unet_model = get_decoder(config, c)

        self.final_conv = nn.Sequential(
            nn.Conv2d(c + dim_c, c, 1, 1, 0, bias=False),
            act,
            nn.Conv2d(c, self.num_target_instruments * dim_c, 1, 1, 0, bias=False)
        )

        self.stft = SubbandSTFT(config.audio)

    def _forward_core(self, x):
        return self.unet_model(x)

    def forward(self, x):
        return forward_subband_mask_model(self, x, self._forward_core)
