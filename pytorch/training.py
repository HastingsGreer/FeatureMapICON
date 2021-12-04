try:
    import pytorch.videoReplayFast as videoReplayFast
except:
    import videoReplayFast
import random
import footsteps
from pykeops.torch import LazyTensor
gen = videoReplayFast.threadedProvide()


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



class UNet2(nn.Module):
    def __init__(self, num_layers, channels, dimension, input_channels=1):
        super(UNet2, self).__init__()
        self.dimension = dimension
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
        self.lastConv = self.Conv(67, 64, kernel_size=3, padding=1)
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

        for depth in reversed(range(self.num_layers)):
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
        return x
def tallerUNet2(dimension=2):
    return UNet2(
        7,
        [[3, 16, 32, 64, 256, 512, 512, 512], [64, 64, 64, 128, 256, 512, 512]],
        dimension,
    )
def tallerUNet2(dimension=2):
    return UNet2(
        4,
        [[3, 64, 64, 128, 256], [64, 64, 128, 256]],
        dimension,
    )
def tallerUNet2(dimension=2):
    return UNet2(
        7,
        [[3, 64, 64, 128, 256, 512, 512, 512], [64, 64, 128, 256, 256, 512, 512]],
        dimension,
    )

# In[4]:


def warping(net, tensor):
    identity = torch.Tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]]]).cuda()
    mask = torch.Tensor([[[1., 1, 1], [1, 1, 1], [0, 0, 0]]]).cuda()
    noise = torch.randn((tensor.shape[0], 3, 3)).cuda()

    forward = identity + .05 * noise * mask

    backward = torch.inverse(forward)
    
    if random.random() < .5:
        forward, backward = backward, forward

    forward_grid = F.affine_grid(forward[:, :2], tensor[:, :3].shape)
    
    
    warped_input = F.grid_sample(tensor, forward_grid)
    
    warped_output = net(warped_input)
    
    backward_grid = F.affine_grid(backward[:, :2], warped_output.shape)
    
    unwarped_output = F.grid_sample(warped_output, backward_grid)

    
    return unwarped_output
    
    
    
class FMAPModelWarping(nn.Module):

    def __init__(self, net, feature_length=128):
        super().__init__()
        self.feature_length = feature_length

        self.net = net

    def forward(self, input_a, input_b):
        feats_a_h = warping(self.net, input_a.cuda())#[:, :, 20:-20, 20:-20]
        feats_b_h = warping(self.net, input_b.cuda())#[:, :, 20:-20, 20:-20]
        
        feats_a_v = warping(self.net, input_a.cuda())#[:, :, 20:-20, 20:-20]
        feats_b_v = warping(self.net, input_b.cuda())#[:, :, 20:-20, 20:-20]

        feats_a_h, feats_b_h = (
            tn.reshape([feats_a_h.shape[0], feats_a_h.shape[1], -1])
             .transpose(1, 2).contiguous() for tn in (feats_a_h, feats_b_h)
        )
        
        feats_a_v, feats_b_v = (
            tn.reshape([feats_a_v.shape[0], feats_a_v.shape[1], -1])
             .transpose(1, 2).contiguous() for tn in (feats_a_v, feats_b_v)
        )

        l_feats_a_h = LazyTensor(feats_a_h[:, :, None, :])
        l_feats_b_h = LazyTensor(feats_b_h[:, None, :, :])
        
        l_feats_a_v = LazyTensor(feats_a_v[:, :, None, :])
        l_feats_b_v = LazyTensor(feats_b_v[:, None, :, :])

        M_unn_h = (l_feats_a_h * l_feats_b_h).sum(3)
        M_unn_v = (l_feats_a_v * l_feats_b_v).sum(3)
        
        with torch.no_grad():
            vm = M_unn_v.max(1)
            hm = M_unn_h.max(2)
        
        M_v = (M_unn_v - vm[:, None]).exp()
        
        #print(M_v.max(1))
        
        
        
        M_h = (M_unn_h - hm[:, :, None]).exp()
        
        

        vs = 1 / (M_v.sum(1) + .0001)
        hs = 1 / (M_h.sum(2) + .0001)

        #(M * vs[:, None]).sum(1) == 1
        #(M * hs[:,:, None]).sum(2) == 1
        
        res = ((M_v * LazyTensor(vs[:, None])) * (M_h * LazyTensor(hs[:, :, None])))
        
        
        loss = torch.log(res.sum(1) + .0001).mean()
        #loss = torch.clip(res.sum(1), 0.0, 0.6).mean()
        loss

        return loss

if __name__ == "__main__":
    feature_net = tallerUNet2().cuda()


    #feature_net.load_state_dict(torch.load("results/deeeep_warp/network00006.trch"))
    loss_model_2 = FMAPModelWarping(feature_net, 64)

    optimizer = torch.optim.RMSprop(feature_net.parameters(), lr=.00001)
    #optimizer = torch.optim.Adam(feature_net.parameters(), lr=.00001)
    feature_net.train()
    feature_net.cuda()
    losses = []

    for i in range(1000000):
        torch.save(feature_net.state_dict(), footsteps.output_dir + f"network{i:05}.trch")
        torch.save(optimizer.state_dict(), footsteps.output_dir + f"opt{i:05}.trch")
        torch.save(losses, footsteps.output_dir + f"loss{i:05}.trch")
        
        for j in range(1000):
            q = next(gen) / 255
            loss = -loss_model_2(q[:, :3], q[:, 3:])
            loss.backward()
            optimizer.step()
            print(loss)
            losses.append(loss.item())
        


