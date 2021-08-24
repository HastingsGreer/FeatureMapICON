import os
import scipy.ndimage
import scipy.ndimage.measurements
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import torchvision
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.autograd import Function
from torch.nn import Module
import footsteps


def show(x):
    while len(x.shape) > 2:
        x = x[0]
    plt.imshow(x.detach().cpu())


def get_dataset(split):
    ds = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(
            "./files/",
            transform=torchvision.transforms.ToTensor(),
            download=True,
            train=(split == "train"),
        ),
        batch_size=500,
    )
    images = []
    for _, batch in enumerate(ds):
        label = np.array(batch[1])
        batch_nines = label == 5
        images.append(np.array(batch[0])[batch_nines])
    images = np.concatenate(images)

    ds = torch.utils.data.TensorDataset(torch.Tensor(images))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=128,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


d1_mnist, d2_mnist = get_dataset("train")
d1_mnist_test, d2_mnist_test = get_dataset("test")


N = 28
BATCH_SIZE = 32


def get_dataset_triangles(split):
    x, y = np.mgrid[0 : 1 : N * 1j, 0 : 1 : N * 1j]
    x = np.reshape(x, (1, N, N))
    y = np.reshape(y, (1, N, N))
    cx = np.random.random((6000, 1, 1)) * 0.2 + 0.4
    cy = np.random.random((6000, 1, 1)) * 0.2 + 0.4
    r = np.random.random((6000, 1, 1)) * 0.2 + 0.2
    theta = np.random.random((6000, 1, 1)) * np.pi * 2
    isTriangle = np.random.random((6000, 1, 1)) > 0.5

    isHollow = np.random.random((6000, 1, 1)) > 0.5

    triangles = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r * np.cos(np.pi / 3) / np.cos(
        (np.arctan2(x - cx, y - cy) + theta) % (2 * np.pi / 3) - np.pi / 3
    )

    triangles = np.tanh(-40 * triangles)

    circles = np.tanh(-40 * (np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r))

    images = isTriangle * triangles + (1 - isTriangle) * circles

    hollow = 1 - images ** 2

    filled = (images + 1) / 2

    images = isHollow * hollow + (1 - isHollow) * filled

    ds = torch.utils.data.TensorDataset(torch.Tensor(np.expand_dims(images, 1)))
    d1, d2 = (
        torch.utils.data.DataLoader(
            ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
        )
        for _ in (1, 1)
    )
    return d1, d2


d1_triangles, d2_triangles = get_dataset_triangles("train")
d1_triangles_test, d2_triangles_test = get_dataset_triangles("test")


class RegisNetNoPad(nn.Module):
    def __init__(self):
        super(RegisNetNoPad, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(11, 10, kernel_size=5)
        self.conv3 = nn.Conv2d(21, 10, kernel_size=5)
        self.conv4 = nn.Conv2d(31, 10, kernel_size=5)
        self.conv5 = nn.Conv2d(41, 10, kernel_size=5)
        self.conv6 = nn.Conv2d(51, 64, kernel_size=5)

    def forward(self, x):
        x = torch.nn.functional.pad(x, [12] * 4)
        x = torch.cat([x[:, :, 2:-2, 2:-2], F.relu(self.conv1(x))], 1)
        x = torch.cat([x[:, :, 2:-2, 2:-2], F.relu(self.conv2(x))], 1)
        x = torch.cat([x[:, :, 2:-2, 2:-2], F.relu(self.conv3(x))], 1)
        x = torch.cat([x[:, :, 2:-2, 2:-2], F.relu(self.conv4(x))], 1)
        x = torch.cat([x[:, :, 2:-2, 2:-2], F.relu(self.conv5(x))], 1)

        out = self.conv6(x)

        ##normalize
        # out_norms = torch.sqrt(torch.sum(out**2, 1, keepdim=True))

        # out = out / (out_norms + .0001)

        return out * 10


def train(net, d1, d2):
    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    net.train()
    net.cuda()
    loss_history = []
    print("[", end="")
    for epoch in range(400):
        print("-", end="")
        if (epoch + 1) % 50 == 0:
            print("]", end="\n[")
        for A, B in list(zip(d1, d2)):
            loss_ = pass_(A, B, net, optimizer)
            if loss_ is not None:
                loss = loss_
        loss_history.append([loss])
        print(loss)
    print("]")
    return loss_history


def pass_(A, B, net, optimizer):

    if A[0].size()[0] == BATCH_SIZE:
        image_A = A[0].cuda()
        image_B = B[0].cuda()
        optimizer.zero_grad()

        nA = net(image_A)[::, ::].reshape(-1, BATCH_SIZE, N * N)
        nB = net(image_B)[::, ::].reshape(-1, BATCH_SIZE, N * N)

        cc = torch.einsum("icn,ick->ink", nA, nB)

        cc_A = torch.softmax(cc, axis=1)
        cc_B = torch.softmax(cc, axis=2)
        loss = cc_A * cc_B
        loss = torch.clamp(loss, max=0.3)
        loss = -torch.sum(loss) / BATCH_SIZE / (N * N)

        loss.backward()
        optimizer.step()
        return loss.detach()


def do_many_visualizations(prefix, A, B, net):
    prefix = footsteps.output_dir + prefix + "/"
    os.mkdir(prefix)
    plt.subplot(1, 2, 1)

    show(B)
    plt.subplot(1, 2, 2)
    show(A)
    plt.savefig(prefix + "A,B.png")
    plt.clf()
    net.cpu()

    for i in range(30):
        plt.subplot(5, 6, i + 1)
        plt.xticks([])
        plt.yticks([])
        show(net(A)[0, i])
        # plt.colorbar()

    nA = net(A).reshape(-1, 64, N * N)
    nB = net(B).reshape(-1, 64, N * N)

    cc = torch.einsum("icn,ick->ink", nA, nB)

    cc_A = torch.softmax(cc, axis=1)
    cc_B = torch.softmax(cc, axis=2)
    loss = cc_A * cc_B

    show(loss)
    plt.colorbar()
    plt.savefig(prefix + "loss_full.png")
    plt.clf()

    i, j = 10, 12

    show(cc.reshape([N] * 4)[i, j])
    plt.colorbar()
    y, x = scipy.ndimage.measurements.center_of_mass(
        cc_A.reshape([N] * 4)[:, :, i, j].detach().numpy()
    )

    plt.scatter(x, y)

    plt.savefig(prefix + "One_pixel_regis.png")
    plt.clf()

    reshaped = cc_A.reshape([N] * 4).detach().numpy()

    grid = np.array(
        [
            [
                scipy.ndimage.measurements.center_of_mass(reshaped[i, j].transpose())
                for i in range(N)
            ]
            for j in range(N)
        ]
    )
    # grid[:, :, 0] = scipy.ndimage.gaussian_filter(grid[:, :, 0], 1)
    # grid[:, :, 1] = scipy.ndimage.gaussian_filter(grid[:, :, 1], 1)

    grid = grid[3:-3, 3:-3]

    plt.plot(grid[:, :, 0], grid[:, :, 1])
    plt.plot(grid[:, :, 0].transpose(), grid[:, :, 1].transpose())
    plt.ylim(N, 0)
    plt.savefig(prefix + "/grid.png")
    plt.clf()
    show(B)
    plt.scatter(grid[:, :, 0], grid[:, :, 1], c="red", s=100)
    plt.scatter(
        grid[:, :, 0], grid[:, :, 1], c=np.array(A[0, 0, 3:-3, 3:-3]).transpose(), s=90
    )
    plt.ylim(N, 0)
    plt.savefig(prefix + "/dots.png")
    plt.clf()

    show(torch.sum(loss, axis=1).reshape(N, N))
    plt.colorbar()

    plt.savefig(prefix + "/loss_image.png")
    plt.clf()


net = RegisNetNoPad()

A = list(d1_triangles)[0][0][:1]
B = list(d1_triangles)[1][0][:1]

do_many_visualizations("before_train", A, B, net)


l = train(net, d1_triangles, d2_triangles)

do_many_visualizations("after_train", A, B, net)

torch.save(net.state_dict(), footsteps.output_dir + "tri_cir_hol.pth")
