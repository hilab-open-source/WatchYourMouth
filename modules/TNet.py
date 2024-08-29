import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable
import torch.nn.functional as F

class STN3d(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.channel = channel
        self.conv1 = torch.nn.Conv3d(channel, 64, (1, 1, 1))
        self.conv2 = torch.nn.Conv3d(64, 128, (1, 1, 1))
        self.conv3 = torch.nn.Conv3d(128, 1024, (1, 1, 1))
        self.fc1 = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, channel**2)

        self.bn1 = nn.BatchNorm3d(64)
        self.bn2 = nn.BatchNorm3d(128)
        self.bn3 = nn.BatchNorm3d(1024)
        self.bn4 = nn.BatchNorm1d(512)
        self.bn5 = nn.BatchNorm1d(256)

    def forward(self, x):
        x = x.permute(0, 3, 1, 2)
        x = torch.unsqueeze(x, -1)
        B, D, nframes, _, _ = x.size()

        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = torch.max(x, 3, keepdim=True)[0]
        x = x.view(B, 1024, -1)

        x = F.relu(self.bn4(self.fc1(x.transpose(2, 1)).transpose(2, 1)))
        x = F.relu(self.bn5(self.fc2(x.transpose(2, 1)).transpose(2, 1)))
        x = self.fc3(x.transpose(2, 1))

        # arr = np.array([1, 0, 0, 0, 1, 0, 0, 0, 1]*nframes)
        arr = np.array([1, 0, 0, 0, 0, 0,
                        0, 1, 0, 0, 0, 0,
                        0, 0, 1, 0, 0, 0,
                        0, 0, 0, 1, 0, 0,
                        0, 0, 0, 0, 1, 0,
                        0, 0, 0, 0, 0, 1]*nframes)
        # arr3 = np.array(np.diag(np.full(self.channel, 1)).flatten().tolist()*nframes)
        # arr = np.array(np.eye(self.channel).flatten().tolist()*nframes)
        # print(np.array_equal(arr, arr3))
        iden = Variable(torch.from_numpy(arr.astype(np.float32))).view(nframes, self.channel**2).repeat(
            B, 1, 1)
        if x.is_cuda:
            iden = iden.cuda()
        x = x + iden
        x = x.view(B, nframes, self.channel, self.channel)
        return x
