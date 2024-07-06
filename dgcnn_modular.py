#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: model.py
@Time: 2018/10/13 6:35 PM
https://github.com/WangYueFt/dgcnn/blob/master/pytorch/model.py
"""

# # changed by lsh

from torch_cluster import knn
from dgcnn import get_graph_feature
import torch
import torch.nn as nn
import numpy as np

class DGCNN_MODULAR(nn.Module):
    def __init__(self, num_neighs, in_features_dim, nn_depth, bb_size, latent_dim):
        super(DGCNN_MODULAR, self).__init__()

        self.num_neighs = num_neighs
        self.input_features = in_features_dim * 2
        self.depth = nn_depth
        self.latent_dim = latent_dim

        self.convs = []
        for i in range(self.depth):
            in_features = self.input_features if i == 0 else bb_size * (2 ** (i+1)) * 2
            out_features = bb_size * 4 if i == 0 else in_features
            self.convs.append(nn.Sequential(
                nn.Conv2d(in_features, out_features, kernel_size=1, bias=False),
                # nn.BatchNorm2d(out_features), nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm2d(out_features), nn.ReLU(),
            )
        )
        last_in_dim = bb_size * 2 * sum([2 ** i for i in range(1, self.depth + 1, 1)])
        self.convs.append(
            nn.Sequential(
                nn.Conv1d(last_in_dim, self.latent_dim, kernel_size=1, bias=False),
                # nn.BatchNorm1d(self.latent_dim), nn.LeakyReLU(negative_slope=0.2),
                nn.BatchNorm1d(self.latent_dim), nn.ReLU(),
            )
        )
        self.convs = nn.ModuleList(self.convs)

    # # INPUT x:B, D, N
    def forward(self, x):

        B, D, N = x.shape[0:3]
        batch_x = torch.tensor(np.array([i*np.ones([N]) for i in range(B)]), dtype=torch.int64, device=x.device)
        batch_y = batch_x.view(B*N)
        batch_x = batch_y

        start_neighs = knn(x.transpose(1, 2).contiguous().view(B*N, D), x.transpose(1, 2).contiguous().view(B*N, D), self.num_neighs, batch_x, batch_y)   # including itself
        start_neighs = start_neighs[1].view(B, N, self.num_neighs)
        start_neighs = torch.tensor(np.array([start_neighs[i].cpu().numpy()-i*N for i in range(B)]), device=start_neighs.device)

        x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs, only_intrinsic=False)
        # x = get_graph_feature(x, k=self.num_neighs, idx=start_neighs, only_intrinsic=True)

        outs = [x]
        for conv in self.convs[:-1]:
            if(len(outs) > 1):
                x = get_graph_feature(outs[-1], k=self.num_neighs, idx=start_neighs, only_intrinsic=False)  # same neighbors
                # x = get_graph_feature(outs[-1], k=self.num_neighs, idx=start_neighs, only_intrinsic=True)
            x = conv(x)
            outs.append(x.max(dim=-1, keepdim=False)[0])

        x = torch.cat(outs[1:], dim=1)
        features = self.convs[-1](x)
        return features.transpose(1, 2)    # , start_neighs  # B, N, D