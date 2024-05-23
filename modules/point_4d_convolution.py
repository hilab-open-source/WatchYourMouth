# Code adapted from PSTTransformer by Hehe Fan
# GitHub: https://github.com/hehefan/PST-Transformer/blob/main/modules/point_4d_convolution.py
# Accessed on May 23, 2024

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch import Tensor

import math
import os
import sys
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

import pointnet2_utils
from typing import List


class P4DConv(nn.Module):
    def __init__(self,
                 in_planes: int,
                 mlp_planes: List[int],
                 mlp_batch_norm: List[bool],
                 mlp_activation: List[bool],
                 spatial_kernel_size: [float, int],
                 spatial_stride: int,
                 temporal_kernel_size: int,
                 temporal_stride: int = 1,
                 temporal_padding: [int, int] = [0, 0],
                 temporal_padding_mode: str = 'replicate',
                 operator: str = 'addition',
                 spatial_pooling: str = 'max',
                 temporal_pooling: str = 'sum',
                 bias: bool = False):

        super().__init__()

        self.in_planes = in_planes
        self.mlp_planes = mlp_planes
        self.mlp_batch_norm = mlp_batch_norm
        self.mlp_activation = mlp_activation

        self.r, self.k = spatial_kernel_size
        self.spatial_stride = spatial_stride

        self.temporal_kernel_size = temporal_kernel_size
        self.temporal_stride = temporal_stride
        self.temporal_padding = temporal_padding
        self.temporal_padding_mode = temporal_padding_mode

        self.operator = operator
        self.spatial_pooling = spatial_pooling
        self.temporal_pooling = temporal_pooling

        conv_d = [nn.Conv2d(in_channels=4, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
        if mlp_batch_norm[0]:
            conv_d.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
        if mlp_activation[0]:
            conv_d.append(nn.ReLU(inplace=True))
        self.conv_d = nn.Sequential(*conv_d)

        if in_planes != 0:
            conv_f = [nn.Conv2d(in_channels=in_planes, out_channels=mlp_planes[0], kernel_size=1, stride=1, padding=0, bias=bias)]
            if mlp_batch_norm[0]:
                conv_f.append(nn.BatchNorm2d(num_features=mlp_planes[0]))
            if mlp_activation[0]:
                conv_f.append(nn.ReLU(inplace=True))
            self.conv_f = nn.Sequential(*conv_f)

        mlp = []
        for i in range(1, len(mlp_planes)):
            if mlp_planes[i] != 0:
                mlp.append(nn.Conv2d(in_channels=mlp_planes[i-1], out_channels=mlp_planes[i], kernel_size=1, stride=1, padding=0, bias=bias))
            if mlp_batch_norm[i]:
                mlp.append(nn.BatchNorm2d(num_features=mlp_planes[i]))
            if mlp_activation[i]:
                mlp.append(nn.ReLU(inplace=True))
        self.mlp = nn.Sequential(*mlp)


    def forward(self, xyzs: torch.Tensor, features: torch.Tensor = None) -> (torch.Tensor, torch.Tensor):
        """
        Args:
            xyzs: torch.Tensor
                 (B, T, N, 3) tensor of sequence of the xyz coordinates
            features: torch.Tensor
                 (B, T, C, N) tensor of sequence of the features
        """
        # start = time.time()
        device = xyzs.get_device()

        nframes = xyzs.size(1)
        npoints = xyzs.size(2)
        assert (self.temporal_kernel_size % 2 == 1), "P4DConv: Temporal kernel size should be odd!"
        assert ((nframes + sum(self.temporal_padding) - self.temporal_kernel_size) % self.temporal_stride == 0), "P4DConv: Temporal length error!"
        # print('Here is the shape of input: ', xyzs.size())
        xyzs = torch.split(tensor=xyzs, split_size_or_sections=1, dim=1)
        xyzs = [torch.squeeze(input=xyz, dim=1).contiguous() for xyz in xyzs]
        #### Now the parameter temporal_padding_model='replicate'
        if self.temporal_padding_mode == 'zeros':
            xyz_padding = torch.zeros(xyzs[0].size(), dtype=torch.float32, device=device)
            for i in range(self.temporal_padding[0]):
                xyzs = [xyz_padding] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyz_padding]
        else:
            # print(self.temporal_padding[0])
            for i in range(self.temporal_padding[0]):
                xyzs = [xyzs[0]] + xyzs
            for i in range(self.temporal_padding[1]):
                xyzs = xyzs + [xyzs[-1]]

        if self.in_planes != 0:
            features = torch.split(tensor=features, split_size_or_sections=1, dim=1)
            features = [torch.squeeze(input=feature, dim=1).contiguous() for feature in features]

            if self.temporal_padding_mode == 'zeros':
                feature_padding = torch.zeros(features[0].size(), dtype=torch.float32, device=device)
                for i in range(self.temporal_padding[0]):
                    features = [feature_padding] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [feature_padding]
            else:
                for i in range(self.temporal_padding[0]):
                    features = [features[0]] + features
                for i in range(self.temporal_padding[1]):
                    features = features + [features[-1]]

        new_xyzs = []
        new_features = []
        for t in range(self.temporal_kernel_size//1, len(xyzs)-self.temporal_kernel_size//1, self.temporal_stride):                # temporal anchor frames
            # spatial anchor point subsampling by FPS
            # print('Here are some infomations: ', self.temporal_kernel_size//1, len(xyzs), self.temporal_kernel_size//1, len(xyzs)-self.temporal_kernel_size//1, self.temporal_stride)
            anchor_idx = pointnet2_utils.furthest_point_sample(xyzs[t], npoints//self.spatial_stride)                               # (B, N//self.spatial_stride)
            anchor_xyz_flipped = pointnet2_utils.gather_operation(xyzs[t].transpose(1, 2).contiguous(), anchor_idx)                 # (B, 3, N//self.spatial_stride)
            anchor_xyz_expanded = torch.unsqueeze(anchor_xyz_flipped, 3)                                                            # (B, 3, N//spatial_stride, 1)
            anchor_xyz = anchor_xyz_flipped.transpose(1, 2).contiguous()                                                            # (B, N//spatial_stride, 3)
            new_feature = []
            for i in range(t-self.temporal_kernel_size//1, t+self.temporal_kernel_size//1+1):
                neighbor_xyz = xyzs[i]
                idx = pointnet2_utils.ball_query(self.r, self.k, neighbor_xyz, anchor_xyz)

                neighbor_xyz_flipped = neighbor_xyz.transpose(1, 2).contiguous()                                                    # (B, 3, N)
                neighbor_xyz_grouped = pointnet2_utils.grouping_operation(neighbor_xyz_flipped, idx)                                # (B, 3, N//spatial_stride, k)

                xyz_displacement = neighbor_xyz_grouped - anchor_xyz_expanded                                                       # (B, 3, N//spatial_stride, k)
                t_displacement = torch.ones((xyz_displacement.size()[0], 1, xyz_displacement.size()[2], xyz_displacement.size()[3]), dtype=torch.float32, device=device) * (i-t)
                displacement = torch.cat(tensors=(xyz_displacement, t_displacement), dim=1, out=None)                               # (B, 4, N//spatial_stride, k)
                displacement = self.conv_d(displacement)
                if self.in_planes != 0:
                    features_flipped = features[i].transpose(1, 2).contiguous()
                    neighbor_feature_grouped = pointnet2_utils.grouping_operation(features_flipped, idx)                                 # (B, in_planes, N//spatial_stride, k)
                    feature = self.conv_f(neighbor_feature_grouped)
                    if self.operator == '+':
                        feature = feature + displacement
                    else:
                        feature = feature * displacement
                else:
                    feature = displacement
                feature = self.mlp(feature)
                if self.spatial_pooling == 'max':
                    feature = torch.max(input=feature, dim=-1, keepdim=False)[0]                                                        # (B, out_planes, n)
                elif self.spatial_pooling == 'sum':
                    feature = torch.sum(input=feature, dim=-1, keepdim=False)
                else:
                    feature = torch.mean(input=feature, dim=-1, keepdim=False)
                new_feature.append(feature)
            new_feature = torch.stack(tensors=new_feature, dim=1)
            if self.temporal_pooling == 'max':
                new_feature = torch.max(input=new_feature, dim=1, keepdim=False)[0]
            elif self.temporal_pooling == 'sum':
                new_feature = torch.sum(input=new_feature, dim=1, keepdim=False)
            else:
                new_feature = torch.mean(input=new_feature, dim=1, keepdim=False)
            new_xyzs.append(anchor_xyz)
            new_features.append(new_feature)
        new_xyzs = torch.stack(tensors=new_xyzs, dim=1)
        new_features = torch.stack(tensors=new_features, dim=1)
        # print(time.time()-start)
        return new_xyzs, new_features
