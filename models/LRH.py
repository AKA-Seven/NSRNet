import torch
from torch import nn
import math
from util import initialize_weights

class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(in_channels, in_channels // reduction)
        self.fc2 = nn.Linear(in_channels // reduction, in_channels)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channel, _, _ = x.size()
        y = x.mean([2, 3], keepdim=True)  # Global Average Pooling
        y = y.view(batch, channel)
        y = self.fc1(y)
        y = self.fc2(y)
        y = self.sigmoid(y).view(batch, channel, 1, 1)
        return x * y  # Scale the input by the attention weights

class ResidualDenseBlock(nn.Module):
    def __init__(self, input, output, bias=True):
        super(ResidualDenseBlock, self).__init__()
        self.conv1 = nn.Conv2d(input, 32, 3, 1, 1, bias=bias)
        self.conv2 = nn.Conv2d(input + 32, 32, 3, 1, 1, bias=bias)
        self.conv3 = nn.Conv2d(input + 2 * 32, 32, 3, 1, 1, bias=bias)
        self.conv4 = nn.Conv2d(input + 3 * 32, 32, 3, 1, 1, bias=bias)
        self.conv5 = nn.Conv2d(input + 4 * 32, output, 3, 1, 1, bias=bias)
        self.lrelu = nn.LeakyReLU(inplace=True)
        self.se = SEBlock(output)
        initialize_weights([self.conv5], 0.)

    def forward(self, x):
        x1 = self.lrelu(self.conv1(x))
        x2 = self.lrelu(self.conv2(torch.cat((x, x1), 1)))
        x3 = self.lrelu(self.conv3(torch.cat((x, x1, x2), 1)))
        x4 = self.lrelu(self.conv4(torch.cat((x, x1, x2, x3), 1)))
        x5 = self.conv5(torch.cat((x, x1, x2, x3, x4), 1))
        return self.se(x5)

class WLBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.split_len1 = 12
        self.split_len2 = 12   
        self.net = ResidualDenseBlock(self.split_len1, self.split_len2)
        self.attention = SEBlock(self.split_len2)

    def forward(self, x):
        out = self.net(x)
        out = self.attention(out)
        return out

class LiftingStep(nn.Module):
    def __init__(self, clamp=2.0):    
        super(LiftingStep, self).__init__()
        self.r = WLBlock()
        self.y = WLBlock()
        self.f = WLBlock()
        self.clamp = clamp

    def e(self, s):
        return torch.exp(self.clamp * 2 * (torch.sigmoid(s) - 0.5))

    def forward(self, x1, x2):
        t2 = self.f(x2)
        y1 = x1 + t2
        s1, t1 = self.r(y1), self.y(y1)
        y2 = self.e(s1) * x2 + t1
        return y1, y2
    def inverse(self, x1, x2):
        s1, t1 = self.r(x1), self.y(x1)
        y2 = (x2 - t1) / self.e(s1)
        t2 = self.f(y2)
        y1 = (x1 - t2)
        return y1, y2

class LRH_f(nn.Module):
    def __init__(self, num_step):
        super(LRH_f, self).__init__()
        self.net = nn.Sequential(*[LiftingStep() for _ in range(num_step)])

    def forward(self, x1, x2, rev=False):
        for layer in self.net:
            x1, x2 = layer(x1, x2)
        return x1, x2

class LRH_r(nn.Module):
    def __init__(self, num_step):
        super(LRH_r, self).__init__()
        self.net = nn.Sequential(*[LiftingStep() for _ in range(num_step)])

    def forward(self, x1, x2, rev=True):
        for layer in reversed(self.net):
            x1, x2 = layer.inverse(x1, x2)
        return x1, x2
