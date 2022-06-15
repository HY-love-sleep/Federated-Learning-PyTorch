# dlg 结合 Fmnist
import os
import copy
import time
import pickle
import numpy as np

import torch
from tensorboardX import SummaryWriter
import visualization_utils

from options import args_parser
from update import LocalUpdate, test_inference
from models import MLP, CNNMnist, CNNFashion_Mnist, CNNCifar
from utils import get_dataset, average_weights, exp_details

import argparse
import numpy as np
from pprint import pprint

import visualization_utils

from PIL import Image
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad
import torchvision
from torchvision import models, datasets, transforms

from utils import label_to_onehot, cross_entropy_for_onehot
print(torch.__version__, torchvision.__version__)
if __name__ == "__main__" :

    args = args_parser()

    device = "cpu"
    # fmnist数据地址
    data_dir = '../../data/fmnist/'
    # 数据格式修改
    # apply_transform = transforms.Compose([
    #             transforms.ToTensor(),
    #             transforms.Normalize((0.1307,), (0.3081,))])

    dst = datasets.FashionMNIST(data_dir, train=True, download=True)


    # 取其中一张图片进行dlg攻击, 这里随便取了idx = 100的图片
    img_index = 1000
    tp = transforms.ToTensor()
    tt = transforms.ToPILImage()
    gt_data = tp(dst[img_index][0]).to(device)

    gt_data = gt_data.view(1, *gt_data.size())
    gt_label = torch.Tensor([dst[img_index][1]]).long().to(device)
    gt_label = gt_label.view(1, )
    gt_onehot_label = label_to_onehot(gt_label)

    plt.imshow(tt(gt_data[0].cpu()))

    from fl_dlg.Version import LeNet, weights_init
    net = LeNet().to(device)

    torch.manual_seed(1234)

    net.apply(weights_init)
    criterion = cross_entropy_for_onehot

    # 计算原始梯度
    pred = net(gt_data)
    y = criterion(pred, gt_onehot_label)
    dy_dx = torch.autograd.grad(y, net.parameters())

    original_dy_dx = list((_.detach().clone() for _ in dy_dx))

    # 生成随机初始化图像和标签
    dummy_data = torch.randn(gt_data.size()).to(device).requires_grad_(True)
    dummy_label = torch.randn(gt_onehot_label.size()).to(device).requires_grad_(True)

    plt.imshow(tt(dummy_data[0].cpu()))

    optimizer = torch.optim.LBFGS([dummy_data, dummy_label])

    # 进行30轮次的梯度逼近
    history = []
    for iters in range(30):
        def closure():
            optimizer.zero_grad()

            dummy_pred = net(dummy_data)
            dummy_onehot_label = F.softmax(dummy_label, dim=-1)
            dummy_loss = criterion(dummy_pred, dummy_onehot_label)
            dummy_dy_dx = torch.autograd.grad(dummy_loss, net.parameters(), create_graph=True)

            grad_diff = 0
            for gx, gy in zip(dummy_dy_dx, original_dy_dx):
                grad_diff += ((gx - gy) ** 2).sum()
            grad_diff.backward()

            return grad_diff


        optimizer.step(closure)

        current_loss = closure()
        print(iters, "%.4f" % current_loss.item())
        history.append(tt(dummy_data[0].cpu()))

    # 画图像
    plt.figure(figsize=(12, 8))
    for i in range(30):
        plt.subplot(3, 10, i + 1)
        plt.imshow(history[i])
        plt.title("iter=%d" % i)
        plt.axis('off')

    plt.show()