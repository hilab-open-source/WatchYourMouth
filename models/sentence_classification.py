# Portions of this code are adapted from PSTTransformer by Hehe Fan
# GitHub: https://github.com/hehefan/PST-Transformer/blob/main/models/sequence_classification.py
# Accessed on May 23, 2024

import torch
import torch.nn.functional as F
from torch import nn
import numpy as np
import sys 
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)
sys.path.append(os.path.join(ROOT_DIR, 'modules'))

from point_4d_convolution import P4DConv
from transformer import Transformer
from TNet import STN3d

class DepthSpeechRecognition(nn.Module):
    def __init__(self, channel, in_planes, radius, nsamples, spatial_stride,
                 temporal_kernel_size, temporal_stride, 
                 dim, depth, heads, dim_head, dropout1,
                 mlp_dim, num_points, dropout2): 
        super().__init__()
        self.channel = channel

        self.Tnet = STN3d(channel=channel)

        self.tube_embedding = P4DConv(in_planes=in_planes, mlp_planes=[dim], mlp_batch_norm=[False], mlp_activation=[False],
                                  spatial_kernel_size=[radius, nsamples], spatial_stride=spatial_stride,
                                  temporal_kernel_size=temporal_kernel_size, temporal_stride=temporal_stride,
                                  temporal_padding=[temporal_kernel_size, temporal_kernel_size],
                                  operator='+', spatial_pooling='max', temporal_pooling='max')

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout=dropout1)

        self.gru1 = nn.GRU(dim*64, 256*2, 1, bidirectional=True, batch_first=True)
        self.drp1 = nn.Dropout(0.5)
        self.gru2 = nn.GRU(256*4, 128*2, 1, bidirectional=True, batch_first=True)
        self.drp2 = nn.Dropout(0.5)
        self.pred = nn.Linear(128*4, 28)

    def forward(self, input):
        # device = input.get_device()
        if self.channel == 3:
            input = input[:, :, :, :3]
            feature = None
            # ---------------------- Trans with xyz and normals ----------------------
            trans = self.Tnet(input)
            input = torch.einsum("B D H W, B D W J -> B D H J", [input, trans])
            # ------------------------------------------------------------------------
        elif self.channel == 6:
            trans = self.Tnet(input)
            input = torch.einsum("B D H W, B D W J -> B D H J", [input, trans])
            feature = input[:, :, :, 3:]
            input = input[:, :, :, :3]


        # ---------------------- Point Clouds 4D Convolution ---------------------
        xyzs, features = self.tube_embedding(xyzs = input, features = feature)     #### [B, L, n, 3], [B, L, C, n] 
        features = features.permute(0, 1, 3, 2)
        # ------------------------------------------------------------------------


        # ---------------------- Feature Transformer -----------------------------
        output = self.transformer(xyzs, features)
        output = output.view(output.shape[0], output.shape[1], -1)
        # ------------------------------------------------------------------------

        # ---------------------- GRU ---------------------------------------------
        self.gru1.flatten_parameters()
        output, _ = self.gru1(output)
        output = self.drp1(output)
        self.gru2.flatten_parameters()
        output, _ = self.gru2(output)
        output = self.drp2(output)
        # ------------------------------------------------------------------------

        output = self.pred(output)

        return output

