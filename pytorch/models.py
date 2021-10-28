import torch
import torch.nn as nn

LOSS_UPPER_BOUND = .6
LOSS_LOWER_BOUND = 0.

class PatchwiseDenseModel(nn.Module):

    def __init__(self, feature_length=128):
        super().__init__()
        self.feature_length = feature_length

        self.net = nn.Sequential(

            nn.ZeroPad2d(30),

            nn.Conv2d(3, 64, 11, stride=1, padding='valid', bias=False),
            nn.BatchNorm2d(64, affine=False),
            nn.ReLU(inplace=False),

            nn.Conv2d(64, 128, 11, stride=1, padding='valid', bias=False),
            nn.BatchNorm2d(128, affine=False),
            nn.ReLU(inplace=False),

            nn.Conv2d(128, 256, 11, stride=1, padding='valid', bias=False),
            nn.BatchNorm2d(256, affine=False),
            nn.ReLU(inplace=False),

            nn.Conv2d(256, 512, 1, stride=1, padding='valid', bias=False),
            nn.BatchNorm2d(512, affine=False),
            nn.ReLU(inplace=False),

            nn.Conv2d(512, feature_length, 1, stride=1, padding='valid', bias=False),
            # nn.BatchNorm2d(64, affine=False),
            # nn.ReLU(inplace=False),
        )

    def forward(self, input_a, input_b):

        feats_a = self.net(input_a)
        feats_b = self.net(input_b)

        # to (N, C, H*W)
        feats_a = feats_a.reshape(feats_a.shape[0],
            feats_a.shape[1], feats_a.shape[2] * feats_a.shape[3])
        feats_b = feats_b.reshape(feats_b.shape[0],
            feats_b.shape[1], feats_b.shape[2] * feats_b.shape[3])
        
        # testing data
        # import numpy as np
        # F_A = np.load("../tensorflow/test_data/F_A.npy")
        # F_B = np.load("../tensorflow/test_data/F_B.npy")
        # feats_a = torch.Tensor(F_A).permute(0, 2, 1)
        # feats_b = torch.Tensor(F_B).permute(0, 2, 1)

        # permute for batch matrix multiplication (bmm)
        # (N, C, H*W) -> (N, H*W, C)
        feats_a = feats_a.permute(0, 2, 1)

        cc = torch.bmm(feats_a, feats_b)

        cc_a = nn.functional.softmax(cc, dim=-1)
        cc_b = nn.functional.softmax(cc, dim=-2)

        return cc_a, cc_b


# Load a tensorflow weight file into a compatible pytorch model
def load_tensorflow_weight_file(model, path):
    import tensorflow as tf
    import numpy as np
    tf_weights = np.load(path)
    for layer in model.modules():
        if isinstance(layer, nn.Conv2d):
            layer.weight.data = torch.from_numpy(tf_weights[layer.weight.data.shape[0]][layer.weight.data.shape[1]][layer.weight.data.shape[2]][layer.weight.data.shape[3]])
            layer.bias.data = torch.from_numpy(tf_weights[layer.bias.data.shape[0]][layer.bias.data.shape[1]][layer.bias.data.shape[2]][layer.bias.data.shape[3]])
        elif isinstance(layer, nn.BatchNorm2d):
            layer.weight.data = torch.from_numpy(tf_weights[layer.weight.data.shape[0]][layer.weight.data.shape[1]])
            layer.bias.data = torch.from_numpy(tf_weights[layer.bias.data.shape[0]][layer.bias.data.shape[1]])
            layer.running_mean.data = torch.from_numpy(tf_weights[layer.running_mean.data.shape[0]][layer.running_mean.data.shape[1]])
            layer.running_var.data = torch.from_numpy(tf_weights[layer.running_var.data.shape[0]][layer.running_var.data.shape[1]])
        elif isinstance(layer, nn.Linear):
            layer.weight.data = torch.from_numpy(tf_weights[layer.weight.data.shape[0]][layer.weight.data.shape[1]])
            layer.bias.data = torch.from_numpy(tf_weights[layer.bias.data.shape[0]][layer.bias.data.shape[1]])
    return model

if __name__ == '__main__':
    model = PatchwiseDenseModel().cuda()

    input_a = torch.rand((4, 3, 60, 60)).cuda()
    input_b = torch.rand((4, 3, 60, 60)).cuda()

    # return the forward & backward likelihood
    fwd_lkhd, bwd_lkhd = model(input_a, input_b)

    # TODO: implement loss class
    loss = (fwd_lkhd * bwd_lkhd).sum(dim=-1)
    loss = torch.clip(loss, min=LOSS_LOWER_BOUND,
        max=LOSS_UPPER_BOUND)
    loss = loss.mean()
