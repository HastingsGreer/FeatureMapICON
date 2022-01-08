import platform 
import pykeops
import os
import random
import footsteps
from pykeops.torch import LazyTensor

try:
    cache_path=f"/playpen-raid1/tgreer/keops_cache/{''.join(''.join(platform.node().split('-')).split('.'))}"
    os.makedirs(cache_path, exist_ok=True)
    pykeops.set_bin_folder(cache_path)
    print(f"keops path is {cache_path}")
except Exception as e:
    print("setting path failed"  + str(e))
    print(f"default keops path")

from torch import nn
import torch
import torch.nn.functional as F
import numpy as np

from fmapicon.models import tallerUNet64
import fmapicon.models as models

def multi_roll_d2(input, offsets):
    index = torch.arange(0, input.shape[2], dtype=torch.long, device=input.device)
    index_shifted = (
        offsets[:, None, None, None] + index[None, None, :, None]
    ) % 120 + torch.zeros(1, input.shape[1], 1, input.shape[3], dtype=torch.long, device=input.device)
    return torch.gather(input, 2, index_shifted )

def multi_roll_d3(input, offsets):
    index = torch.arange(0, input.shape[3], dtype=torch.long, device=input.device)
    index_shifted = (
        offsets[:, None, None, None] + index[None, None, None, :]
    ) % 120 + torch.zeros(1, input.shape[1], input.shape[2], 1, dtype=torch.long, device=input.device)
    return torch.gather(input, 3, index_shifted)

def warping(net, tensor):
    identity = torch.Tensor([[[1., 0, 0], [0, 1, 0], [0, 0, 1]]]).cuda()
    mask = torch.Tensor([[[1., 1, 1], [1, 1, 1], [0, 0, 0]]]).cuda()
    noise = torch.randn((tensor.shape[0], 3, 3)).cuda()

    forward = identity + .05 * noise * mask

    backward = torch.inverse(forward)
    
    if random.random() < .5:
        forward, backward = backward, forward

    forward_grid = F.affine_grid(forward[:, :2], tensor[:, :3].shape)
   
    #Add some random rolling
    u_roll = torch.randint(0, 120, (64,), device=tensor.device)
    v_roll = torch.randint(0, 120, (64,), device=tensor.device)

    #tensor = torch.roll(tensor, (u_roll, v_roll), (2, 3))
    tensor = multi_roll_d2(tensor, u_roll)
    tensor = multi_roll_d3(tensor, v_roll)
    
    warped_input = F.grid_sample(tensor, forward_grid)
    
    warped_output = net(warped_input)
    
    backward_grid = F.affine_grid(backward[:, :2], warped_output.shape)
    
    unwarped_output = F.grid_sample(warped_output, backward_grid)
    #magnitudes = .000001 + torch.sqrt(torch.sum(unwarped_output**2, axis=1, keepdims=True))
    #unwarped_output = 12 * unwarped_output / magnitudes

    #unwarped_output = torch.roll(unwarped_output, (-u_roll, -v_roll), (2, 3))

    unwarped_output = multi_roll_d3(unwarped_output, -v_roll)
    unwarped_output = multi_roll_d2(unwarped_output, -u_roll)
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

        M_unnormalized_h = (l_feats_a_h * l_feats_b_h).sum(3)
        M_unnormalized_v = (l_feats_a_v * l_feats_b_v).sum(3)
        
        with torch.no_grad():
            vmax = M_unnormalized_v.max(1)
            hmax = M_unnormalized_h.max(2)
        
        M_v = (M_unnormalized_v - vmax[:, None]).exp()
        
        #print(M_v.max(1))
        
        
        
        M_h = (M_unnormalized_h - hmax[:, :, None]).exp()
        
        

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

    import fmapicon.threaded_video_dataset as threaded_video_dataset
    gen = threaded_video_dataset.threadedProvide()
    feature_net = models.shortUNet64(normalize_pixels=True).cuda()


    #stat = torch.load("results/rolling_shortnet_30maxgap/network00121.trch")
    #feature_net.load_state_dict(stat)

    loss_model_2 = FMAPModelWarping(feature_net, 64)

    optimizer = torch.optim.RMSprop(feature_net.parameters(), lr=.001)
    #optimizer = torch.optim.Adam(feature_net.parameters(), lr=.0001)
    #optimizer.load_state_dict(torch.load("results/rolling_shortnet_30maxgap/opt00121.trch"))

    feature_net.train()
    feature_net.cuda()
    losses = []

    for i in range(1000000):
        torch.save(feature_net.state_dict(), footsteps.output_dir + f"network{i:05}.trch")
        torch.save(optimizer.state_dict(), footsteps.output_dir + f"opt{i:05}.trch")
        torch.save(losses, footsteps.output_dir + f"loss{i:05}.trch")
        
        for j in range(100):
            q = next(gen)[:] / 255
            loss = -loss_model_2(q[:, :3], q[:, 3:])
            loss.backward()
            parameters = feature_net.parameters()
            parameters = [p for p in parameters if p.grad is not None]
            total_norm = torch.norm(torch.stack([torch.norm(p.grad.detach(), 2) for p in parameters]), 2)
            
            torch.nn.utils.clip_grad_norm_(feature_net.parameters(), 20, 2)

            optimizer.step()
            print(f"Loss: {loss.item()}, scale: {feature_net.outNorm.scale.data.item()}, grad norm: {total_norm.item()}")
            losses.append({"loss":loss.item(), "grad_norm":total_norm.item()})
        
