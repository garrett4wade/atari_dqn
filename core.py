import functools
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:

    def __init__(self, max_size, obs_shape):
        if len(obs_shape) > 1:
            self.state = np.zeros((max_size, *obs_shape), dtype=np.uint8)
            self.nex_state = np.zeros((max_size, *obs_shape), dtype=np.uint8)
        else:
            self.state = np.zeros((max_size, *obs_shape), dtype=np.float32)
            self.nex_state = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.action = np.zeros(max_size, dtype=np.int64)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=np.uint8)

        self.max_size = max_size
        self.cur_size = 0
        self.ptr = 0

    def put_batch(self, s, a, n_s, r, d):
        bs = len(r)
        ptr = self.ptr
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
        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.nex_state[self.ptr] = n_s
        self.reward[self.ptr] = r
        self.done[self.ptr] = d
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
            nn.Conv2d(4, 64, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64), nn.ReLU(inplace=True))

        self.conv2 = self._make_layer(block, num_blocks[0], 64, 1)
        self.conv3 = self._make_layer(block, num_blocks[1], 64, 2)
        self.conv4 = self._make_layer(block, num_blocks[2], 128, 2)
        self.conv5 = self._make_layer(block, num_blocks[3], 128, 2)
        self.avg = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = PopArtValueHead(512, act_dim, dueling=dueling)

    def forward(self, x):
        assert x.dtype == torch.float32, x.dtype
        assert (0 <= x).all() and (x <= 1).all()
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.avg(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

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


class MLPDQN(nn.Module):

    def __init__(self,
                 obs_dim,
                 act_dim,
                 act_fn,
                 hidden_dim=128,
                 dueling=False):
        super().__init__()
        self.base = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim),
            act_fn(),
            nn.Linear(hidden_dim, hidden_dim),
            act_fn(),
        )
        self.fc = PopArtValueHead(hidden_dim, act_dim, dueling=dueling)

    def forward(self, x):
        return self.fc(self.base(x))


class RunningMeanStd(nn.Module):

    def __init__(self, input_shape, beta=0.99999, epsilon=1e-5):
        super().__init__()
        self.__beta = beta
        self.__eps = epsilon
        self.__input_shape = input_shape

        self.__mean = nn.Parameter(torch.zeros(input_shape),
                                   requires_grad=False)
        self.__mean_sq = nn.Parameter(torch.zeros(input_shape),
                                      requires_grad=False)
        self.__debiasing_term = nn.Parameter(torch.zeros(1),
                                             requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        self.__mean.zero_()
        self.__mean_sq.zero_()
        self.__debiasing_term.zero_()

    def forward(self, *args, **kwargs):
        # we don't implement the forward function because its meaning
        # is somewhat ambiguous
        raise NotImplementedError

    def __check(self, x):
        assert isinstance(x, torch.Tensor)
        trailing_shape = x.shape[-len(self.__input_shape):]
        assert trailing_shape == self.__input_shape, (
            'Trailing shape of input tensor'
            f'{x.shape} does not equal to configured input shape {self.__input_shape}'
        )

    @torch.no_grad()
    def update(self, x):
        self.__check(x)
        norm_dims = tuple(range(len(x.shape) - len(self.__input_shape)))

        batch_mean = x.mean(dim=norm_dims)
        batch_sq_mean = x.square().mean(dim=norm_dims)

        self.__mean.mul_(self.__beta).add_(batch_mean * (1.0 - self.__beta))
        self.__mean_sq.mul_(self.__beta).add_(batch_sq_mean *
                                              (1.0 - self.__beta))
        self.__debiasing_term.mul_(self.__beta).add_(1.0 * (1.0 - self.__beta))

    @torch.no_grad()
    def mean_std(self):
        debiased_mean = self.__mean / self.__debiasing_term.clamp(
            min=self.__eps)
        debiased_mean_sq = self.__mean_sq / self.__debiasing_term.clamp(
            min=self.__eps)
        debiased_var = (debiased_mean_sq - debiased_mean**2).clamp(min=1e-2)
        return debiased_mean, debiased_var.sqrt()

    @torch.no_grad()
    def normalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return (x - mean) / std

    @torch.no_grad()
    def denormalize(self, x):
        self.__check(x)
        mean, std = self.mean_std()
        return x * std + mean


class PopArtValueHead(nn.Module):

    def __init__(self,
                 input_dim,
                 act_dim,
                 beta=0.999,
                 epsilon=1e-5,
                 dueling=False):
        super().__init__()
        self.__rms = RunningMeanStd((1, ), beta, epsilon)

        self.__weight = nn.Parameter(torch.zeros(act_dim, input_dim))
        nn.init.orthogonal_(self.__weight)
        self.__bias = nn.Parameter(torch.zeros(act_dim))

        self.__dueling = dueling
        if dueling:
            self.__v_weight = nn.Parameter(torch.zeros(1, input_dim))
            self.__v_bias = nn.Parameter(torch.zeros(1))

    def forward(self, feature):
        if not self.__dueling:
            return F.linear(feature, self.__weight, self.__bias)
        else:
            adv = F.linear(feature, self.__weight, self.__bias)
            v = F.linear(feature, self.__v_weight, self.__v_bias)
            return adv + v

    @torch.no_grad()
    def update(self, x):
        old_mean, old_std = self.__rms.mean_std()
        self.__rms.update(x)
        new_mean, new_std = self.__rms.mean_std()

        self.__weight.data[:] = self.__weight * (old_std /
                                                 new_std).unsqueeze(-1)
        self.__bias.data[:] = (old_std * self.__bias + old_mean -
                               new_mean) / new_std
        if self.__dueling:
            self.__v_weight.data[:] = self.__v_weight * (old_std /
                                                         new_std).unsqueeze(-1)
            self.__v_bias.data[:] = (old_std * self.__v_bias + old_mean -
                                     new_mean) / new_std

    @torch.no_grad()
    def normalize(self, x):
        return self.__rms.normalize(x)

    @torch.no_grad()
    def denormalize(self, x):
        return self.__rms.denormalize(x)

    @torch.no_grad()
    def mean_std(self):
        return self.__rms.mean_std()
