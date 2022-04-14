import functools
import gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class ReplayBuffer:

    def __init__(self, max_size, obs_shape):
        self.state = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.nex_state = np.zeros((max_size, *obs_shape), dtype=np.float32)
        self.action = np.zeros(max_size, dtype=np.int64)
        self.reward = np.zeros(max_size, dtype=np.float32)
        self.done = np.zeros(max_size, dtype=bool)

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


class AtariDQN(nn.Module):

    def __init__(self, num_actions, linear, dueling):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4)
        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)
        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        self.fc1 = nn.Linear(in_features=7 * 7 * 64, out_features=512)
        self.fc2 = nn.Linear(in_features=512, out_features=num_actions)

        self.linear = linear
        self.dueling = dueling
        if dueling:
            self.fc_v = nn.Linear(512, 1)

        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def forward(self, x):
        # assert (x >= 0).all() and (x <= 1).all()
        # assert x.dtype == torch.float32
        act_fn = F.relu if not self.linear else lambda x: x
        x = act_fn(self.conv1(x))
        x = act_fn(self.conv2(x))
        x = act_fn(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = act_fn(self.fc1(x))
        if not self.dueling:
            return self.fc2(x)
        else:
            v, adv = self.fc_v(x), self.fc2(x)
            return v + adv - adv.mean(-1, keepdim=True)


class DQN(nn.Module):

    def __init__(self, obs_dim, act_dim, linear, dueling, hidden_dim=512):
        super().__init__()

        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.linear = linear
        self.dueling = dueling
        if dueling:
            self.V = nn.Linear(hidden_dim, 1)
            self.A = nn.Linear(hidden_dim, act_dim)
        else:
            self.Q = nn.Linear(hidden_dim, act_dim)

        self.device = torch.device(
            "cuda:0") if torch.cuda.is_available() else torch.device("cpu")
        self.to(self.device)

    def forward(self, state):
        act_fn = (lambda x: x) if self.linear else F.relu
        flat1 = act_fn(self.fc1(state))
        if self.dueling:
            V = self.V(flat1)
            A = self.A(flat1)

            return V + A - A.mean(-1, keepdim=True)
        else:
            return self.Q(flat1)
