import torch
from torch import nn
import torch.nn.functional as F

from . import layers


class BaseASPPNet(nn.Module):
    def __init__(self, nn_architecture, nin, ch, dilations=(4, 8, 16)):
        super(BaseASPPNet, self).__init__()
        self.nn_architecture = nn_architecture
        self.enc1 = layers.Encoder(nin, ch, 3, 2, 1)
        self.enc2 = layers.Encoder(ch, ch * 2, 3, 2, 1)
        self.enc3 = layers.Encoder(ch * 2, ch * 4, 3, 2, 1)
        self.enc4 = layers.Encoder(ch * 4, ch * 8, 3, 2, 1)
        if self.nn_architecture == 129605:
            self.enc5 = layers.Encoder(ch * 8, ch * 16, 3, 2, 1)
            self.aspp = layers.ASPPModule(nn_architecture, ch * 16, ch * 32, dilations)
            self.dec5 = layers.Decoder(ch * (16 + 32), ch * 16, 3, 1, 1)
        else:
            self.aspp = layers.ASPPModule(nn_architecture, ch * 8, ch * 16, dilations)
        self.dec4 = layers.Decoder(ch * (8 + 16), ch * 8, 3, 1, 1)
        self.dec3 = layers.Decoder(ch * (4 + 8), ch * 4, 3, 1, 1)
        self.dec2 = layers.Decoder(ch * (2 + 4), ch * 2, 3, 1, 1)
        self.dec1 = layers.Decoder(ch * (1 + 2), ch, 3, 1, 1)

    def forward(self, input_tensor):
        hidden_state, e1 = self.enc1(input_tensor)
        hidden_state, e2 = self.enc2(hidden_state)
        hidden_state, e3 = self.enc3(hidden_state)
        hidden_state, e4 = self.enc4(hidden_state)
        if self.nn_architecture == 129605:
            hidden_state, e5 = self.enc5(hidden_state)
            hidden_state = self.dec5(self.aspp(hidden_state), e5)
        else:
            hidden_state = self.aspp(hidden_state)
        hidden_state = self.dec4(hidden_state, e4)
        hidden_state = self.dec3(hidden_state, e3)
        hidden_state = self.dec2(hidden_state, e2)
        return self.dec1(hidden_state, e1)


def determine_model_capacity(n_fft_bins, nn_architecture):
    ch = {31191: 16, 33966: 16, 123821: 32, 123812: 32, 537238: 64, 537227: 64}[nn_architecture]
    caps = [(2, ch), (2, ch), (ch + 2, ch // 2, 1, 1, 0), (ch // 2, ch), (2 * ch + 2, ch, 1, 1, 0), (ch, 2 * ch),
            (2 * ch, 2, 1), (ch, 2, 1), (ch, 2, 1)]
    return CascadedASPPNet(n_fft_bins, caps, nn_architecture)


class CascadedASPPNet(nn.Module):
    def __init__(self, n_fft, model_capacity_data, nn_architecture):
        super(CascadedASPPNet, self).__init__()
        self.stg1_low_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[0])
        self.stg1_high_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[1])
        self.stg2_bridge = layers.Conv2DBNActiv(*model_capacity_data[2])
        self.stg2_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[3])
        self.stg3_bridge = layers.Conv2DBNActiv(*model_capacity_data[4])
        self.stg3_full_band_net = BaseASPPNet(nn_architecture, *model_capacity_data[5])
        self.out = nn.Conv2d(*model_capacity_data[6], bias=False)
        self.aux1_out = nn.Conv2d(*model_capacity_data[7], bias=False)
        self.aux2_out = nn.Conv2d(*model_capacity_data[8], bias=False)
        self.max_bin = n_fft // 2
        self.output_bin = n_fft // 2 + 1
        self.offset = 128

    def forward(self, input_tensor):
        mix = input_tensor.detach()
        input_tensor = input_tensor.clone()[:, :, : self.max_bin]
        bandwidth = input_tensor.size()[2] // 2
        aux1 = torch.cat([self.stg1_low_band_net(input_tensor[:, :, :bandwidth]),
                          self.stg1_high_band_net(input_tensor[:, :, bandwidth:])], dim=2)
        aux2 = self.stg2_full_band_net(self.stg2_bridge(torch.cat([input_tensor, aux1], dim=1)))
        hidden_state = self.stg3_full_band_net(self.stg3_bridge(torch.cat([input_tensor, aux1, aux2], dim=1)))
        mask = torch.sigmoid(self.out(hidden_state))
        mask = F.pad(mask, (0, 0, 0, self.output_bin - mask.size()[2]), mode="replicate")
        if self.training:
            pad = lambda t: F.pad(t, (0, 0, 0, self.output_bin - t.size()[2]), mode="replicate")
            return mask * mix, pad(torch.sigmoid(self.aux1_out(aux1))) * mix, pad(torch.sigmoid(self.aux2_out(aux2))) * mix
        return mask

    def predict_mask(self, input_tensor):
        mask = self.forward(input_tensor)
        return mask[:, :, :, self.offset:-self.offset] if self.offset > 0 else mask
