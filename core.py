import functools
import gym
import numpy as np
import torch
import torch.nn as nn


class ReplayBuffer:

    def __init__(self, max_size, obs_shape, act_dim):
        self.state = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.nex_state = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        self.action = np.zeros(max_size, dtype=np.uint8)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.uint8)

        self.max_size = max_size
        self.cur_size = 0
        self.ptr = 0

    def put_batch(self, s, a, n_s, r, d):
        bs = len(r)
        if self.ptr + bs > self.max_size:
            self.state[ptr:] = s[:self.max_size - ptr]
            self.action[ptr:] = a[:self.max_size - ptr]
            self.nex_state[ptr:] = n_s[:self.max_size - ptr]
            self.reward[ptr:] = r[:self.max_size - ptr]
            self.done[ptr:] = d[:self.max_size - ptr]

            remaining = bs - (self.max_size - ptr)
            self.state[:remaining] = s[self.max_size - ptr:]
            self.action[:remaining] = a[self.max_size - ptr:]
            self.nex_state[:remaining] = n_s[self.max_size - ptr:]
            self.reward[:remaining] = r[self.max_size - ptr:]
            self.done[:remaining] = d[self.max_size - ptr:]
        else:
            self.state[ptr:ptr + bs] = s
            self.action[ptr:ptr + bs] = a
            self.nex_state[ptr:ptr + bs] = n_s
            self.reward[ptr:ptr + bs] = r
            self.done[ptr:ptr + bs] = d
        self.ptr = (self.ptr + bs) % self.max_size
        self.cur_size = min(self.cur_size + bs, self.max_size)

    def put(self, s, a, n_s, r, d):
        self.state[ptr] = s
        self.action[ptr] = a
        self.nex_state[ptr] = n_s
        self.reward[ptr] = r
        self.done[ptr] = d
        self.ptr = (self.ptr + 1) % self.max_size
        self.cur_size = min(self.cur_size + 1, self.max_size)

    def get(self, bs):
        if self.cur_size < bs:
            return None
        else:
            idx = np.random.choice(self.cur_size, bs, replace=False)
            return (self.state[idx], self.action[idx], self.nex_state[idx],
                    self.reward[idx], self.done[idx])


CARDINALITY = 32
DEPTH = 4
BASEWIDTH = 64


class ResNextBottleNeckC(nn.Module):

    def __init__(self, in_channels, out_channels, stride, act_fn, batchnorm):
        super().__init__()

        C = CARDINALITY  #How many groups a feature map was splitted into

        #"""We note that the input/output width of the template is fixed as
        #256-d (Fig. 3), We note that the input/output width of the template
        #is fixed as 256-d (Fig. 3), and all widths are dou- bled each time
        #when the feature map is subsampled (see Table 1)."""
        D = int(DEPTH * out_channels /
                BASEWIDTH)  #number of channels per group
        self.split_transforms = nn.Sequential(
            nn.Conv2d(in_channels, C * D, kernel_size=1, groups=C, bias=False),
            nn.BatchNorm2d(C * D) if batchnorm else nn.Identity(),
            act_fn(),
            nn.Conv2d(C * D,
                      C * D,
                      kernel_size=3,
                      stride=stride,
                      groups=C,
                      padding=1,
                      bias=False),
            nn.BatchNorm2d(C * D) if batchnorm else nn.Identity(),
            act_fn(),
            nn.Conv2d(C * D, out_channels * 4, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * 4) if batchnorm else nn.Identity(),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * 4:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels,
                          out_channels * 4,
                          stride=stride,
                          kernel_size=1,
                          bias=False), nn.BatchNorm2d(out_channels * 4))

        self.output_act_fn = act_fn()

    def forward(self, x):
        return self.output_act_fn(self.split_transforms(x) + self.shortcut(x))


class AtariDQN(nn.Module):

    def __init__(self,
                 num_blocks,
                 act_dim,
                 act_fn,
                 batchnorm=True,
                 dueling=False):
        super().__init__()
        self.in_channels = 64

        block = functools.partial(ResNextBottleNeckC,
                                  act_fn=act_fn,
                                  batchnorm=batchnorm)

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 64, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 128, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 128, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, act_dim)

        self._dueling = dueling
        if dueling:
            self.v = nn.Linear(512, 1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        out = self.fc(x)
        if self._dueling:
            return out + self.v(x)
        else:
            return out

    def _make_layer(self, block, num_block, out_channels, stride):
        """Building resnext block
        Args:
            block: block type(default resnext bottleneck c)
            num_block: number of blocks per layer
            out_channels: output channels per block
            stride: block stride
        Returns:
            a resnext layer
        """
        strides = [stride] + [1] * (num_block - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * 4

        return nn.Sequential(*layers)
