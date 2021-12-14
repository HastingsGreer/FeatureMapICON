from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

def pad_or_crop(x, shape, dimension):
    y = x[:, : shape[1]]
    if x.size()[1] < shape[1]:
        if dimension == 3:
            y = F.pad(y, (0, 0, 0, 0, 0, 0, shape[1] - x.size()[1], 0))
        else:
            y = F.pad(y, (0, 0, 0, 0, shape[1] - x.size()[1], 0))
    assert y.size()[1] == shape[1]

    return y
class Residual(nn.Module):
    def __init__(self, features):
        super(Residual, self).__init__()
        self.bn1 = nn.BatchNorm2d(num_features=features)
        self.bn2 = nn.BatchNorm2d(num_features=features)

        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)

    def forward(self, x):
        y = F.relu(self.bn1(x))
        y = self.conv1(y)
        y = F.relu(self.bn2(y))
        y = self.conv2(y)
        return y + x
class NormScaleFeature(nn.Module):

    def __init__(self, init_value=1):
         super().__init__()
         self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, input):
        magnitudes = .000001 + torch.sqrt(torch.sum(input**2, axis=1, keepdims=True))
        output = self.scale * input / magnitudes
        return output



class UNet2(nn.Module):
    def __init__(self, num_layers, channels, dimension, input_channels=1, half_res_output=False, normalize_output=True):
        super(UNet2, self).__init__()
        self.dimension = dimension
        self.half_res_output = half_res_output
        self.normalize_output = normalize_output
        if dimension == 2:
            self.BatchNorm = nn.BatchNorm2d
            self.Conv = nn.Conv2d
            self.ConvTranspose = nn.ConvTranspose2d
            self.avg_pool = F.avg_pool2d
            self.interpolate_mode = "bilinear"
        else:
            self.BatchNorm = nn.BatchNorm3d
            self.Conv = nn.Conv3d
            self.ConvTranspose = nn.ConvTranspose3d
            self.avg_pool = F.avg_pool3d
            self.interpolate_mode = "trilinear"
        self.num_layers = num_layers
        down_channels = np.array(channels[0])
        up_channels_out = np.array(channels[1])
        up_channels_in = down_channels[1:] + np.concatenate([up_channels_out[1:], [0]])
        self.downConvs = nn.ModuleList([])
        self.upConvs = nn.ModuleList([])
        self.residues = nn.ModuleList([])
        self.batchNorms = nn.ModuleList(
            [
                self.BatchNorm(num_features=up_channels_out[_])
                for _ in range(self.num_layers)
            ]
        )
        for depth in range(self.num_layers):
            self.downConvs.append(
                self.Conv(
                    down_channels[depth],
                    down_channels[depth + 1],
                    kernel_size=3,
                    padding=1,
                    stride=2,
                )
            )
            self.upConvs.append(
                self.ConvTranspose(
                    up_channels_in[depth],
                    up_channels_out[depth],
                    kernel_size=4,
                    padding=1,
                    stride=2,
                )
            )
            self.residues.append(
                Residual(up_channels_out[depth])
            )
        if half_res_output:
            self.lastConv = self.Conv(128 + 64, 128, kernel_size=3, padding=1)
        else:
            self.lastConv = self.Conv(up_channels_out[0] + down_channels[0], up_channels_out[0], kernel_size=3, padding=1)
        if self.normalize_output:
            self.outNorm = NormScaleFeature(12)
        #torch.nn.init.zeros_(self.lastConv.weight)
        #torch.nn.init.zeros_(self.lastConv.bias)

    def forward(self, x):

        skips = []
        for depth in range(self.num_layers):
            skips.append(x)
            y = self.downConvs[depth](F.leaky_relu(x))
            x = y + pad_or_crop(
                self.avg_pool(x, 2, ceil_mode=True), y.size(), self.dimension
            )
            y = F.layer_norm

        if self.half_res_output:
            depths = list(reversed(range(self.num_layers)))[:-1]
        else:
            depths = list(reversed(range(self.num_layers)))
        for depth in depths:
            y = self.upConvs[depth](F.leaky_relu(x))
            #x = y + F.interpolate(
            #    pad_or_crop(x, y.size(), self.dimension),
            #    scale_factor=2,
            #    mode=self.interpolate_mode,
            #    align_corners=False,
            #)
            y = self.residues[depth](y)
            # x = self.batchNorms[depth](x)
            x = y

            x = x[:, :, : skips[depth].size()[2], : skips[depth].size()[3]]
            x = torch.cat([x, skips[depth]], 1)
        x = self.lastConv(x)
        if self.normalize_output:
            x = self.outNorm(x)
        return x

def tallerUNet64(dimension=2, normalize_output=True):
    return UNet2(
        7,
        [[3, 64, 64, 128, 256, 512, 512, 512], [64, 64, 128, 256, 256, 512, 512]],
        dimension, normalize_output=normalize_output
    )
def tallerUNet128(dimension=2, normalize_output=True):
    return UNet2(
        7,
        [[3, 64, 64, 128, 256, 512, 512, 512], [128, 128, 128, 256, 256, 512, 512]],
        dimension, normalize_output=normalize_output
    )