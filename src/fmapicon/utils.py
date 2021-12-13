import torch
import numpy as np
import fmapicon.models

import matplotlib.pyplot as plt
import torch.nn as nn
import torch.nn.functional as F
l = []

def execute_model(A, B, model):
    N = A.shape[0]
    SIDE_LENGTH = A.shape[2]
    
    
    A = torch.tensor(A).cuda().permute(0, 3, 1, 2).float()
    B = torch.tensor(B).cuda().permute(0, 3, 1, 2).float()
    feats_a = model(A)
    feats_b = model(B)

    # to (N, C, H*W)
    feats_a = feats_a.reshape(feats_a.shape[0],
        feats_a.shape[1], feats_a.shape[2] * feats_a.shape[3])
    feats_b = feats_b.reshape(feats_b.shape[0],
        feats_b.shape[1], feats_b.shape[2] * feats_b.shape[3])

    feats_a = feats_a.permute(0, 2, 1)

    #return feats_a
    cc = torch.bmm(feats_a, feats_b)

    cc = nn.functional.softmax(cc, dim=-1)


    cc = cc.reshape([N] + [SIDE_LENGTH] * 4)
    #cc = np.array(cc.cpu().detach())
    
    cc2 = cc.reshape([N, SIDE_LENGTH, SIDE_LENGTH, SIDE_LENGTH**2])
    
    index_grid = torch.argmax(cc2, axis=-1).cpu().detach()[:, :, :, None].numpy()
    
    


    grid = np.concatenate([index_grid % SIDE_LENGTH, index_grid / SIDE_LENGTH], axis=-1)
    
    return cc, grid

class FmapICONSegmentationModel:
    def __init__(self, weights_path):
        self.inner_model = fmapicon.training.tallerUNet2().cuda()
        self.inner_model.load_state_dict(torch.load(weights_path))
        
    def __call__(self, initial_frame, initial_mask, prev_frame, prev_mask, current_frame):
        crop = (initial_frame.shape[1] - initial_frame.shape[0]) // 2
        B = initial_frame[None, ::4, crop:-crop:4] / 255.   
        A = current_frame[None, ::4, crop:-crop:4] / 255.
        
        cc, grid = execute_model(A, B, self.inner_model)
        m = initial_mask[2::4, 2+crop:-crop:4]

        shitty_res = m[grid[0, :, :, 1].astype(int), grid[0, :, :, 0].astype(int)]

        up_res = np.repeat(np.repeat(shitty_res, 4, axis=0),4, axis=1)
        
        out_mask = initial_mask.copy()
        
        out_mask[:, crop:-crop] = up_res
        """
        plt.imshow(A[0])
        plt.show()
        plt.imshow(B[0])
        plt.show()
        plt.imshow(shitty_res)
        plt.show()
        plt.imshow(out_mask)
        plt.show()
        #l.append(up_res)
        
        plt.imshow(grid[0, :, :, 0])
        plt.show()
        
        l.append((cc, A, B, initial_mask))
        """
        

        return out_mask
