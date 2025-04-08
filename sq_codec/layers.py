import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.parametrizations import weight_norm

import sq_codec.base.nn as xnn


# ! use weight_norm or not
# ! use weight_init or not
def nn_wrapper(nn_class, norm_weight=True, init_weight=True):
    def nn_builder(*args, **kwargs):
        nn_instance = nn_class(*args, **kwargs)
        if init_weight:
            nn.init.trunc_normal_(nn_instance.weight, std=.02)
            nn.init.constant_(nn_instance.bias, 0)
        if norm_weight:
            nn_instance = weight_norm(nn_instance)
        return nn_instance

    return nn_builder


Conv1d = nn_wrapper(nn.Conv1d, norm_weight=True, init_weight=True)
Linear = nn_wrapper(nn.Linear, norm_weight=True, init_weight=True)


# Scripting this brings model speed up 1.4x
@torch.jit.script
def snake(x, alpha):
    # torch.clamp_(alpha, 0.05, 50.)
    x = x + (alpha + xnn.EPS).reciprocal() * torch.sin(alpha * x).pow(2)
    return x


class Snake1d(nn.Module):
    def __init__(self, channels, data_format="channels_first"):
        super().__init__()
        if data_format == "channels_first":
            self.alpha = nn.Parameter(torch.ones(1, channels, 1))
        elif data_format == "channels_last":
            self.alpha = nn.Parameter(torch.ones(1, 1, channels))
        else:
            raise NotImplementedError

    def forward(self, x):
        return snake(x, self.alpha)


@torch.jit.script
def channel_norm(x, weight, bias, eps):
    u = x.mean(1, keepdim=True)
    s = (x - u).pow(2).mean(1, keepdim=True)
    x = (x - u) / torch.sqrt(s + eps)
    x = weight * x + bias
    return x


class ChannelNorm(nn.Module):
    """ ChannelNorm that supports two data formats: channels_last (default) or channels_first.
    Channels_last corresponds to inputs with shape (batch_size, ..., channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, ...).
    """

    def __init__(self, n_channels, eps=xnn.EPS, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        self.weight = nn.Parameter(torch.ones(n_channels))
        self.bias = nn.Parameter(torch.zeros(n_channels))
        self.eps = torch.tensor(eps)

    def forward(self, x):

        if self.data_format == "channels_first":
            extend_dims = ((1,) * len(x.shape[2:]))
            return channel_norm(x, self.weight.view(-1, *extend_dims), self.bias.view(-1, *extend_dims), self.eps)

        elif self.data_format == "channels_last":
            return F.layer_norm(x, (self.n_channels,), self.weight, self.bias, self.eps.item())

        else:
            raise NotImplementedError

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, {self.data_format})"


class GRN(nn.Module):
    """ GRN (Global Response Normalization) layer
    Which supports two data formats: channels_last (default) or channels_first.
    Channels_last corresponds to inputs with shape (batch_size, Sequence, channels)
    while channels_first corresponds to inputs with shape (batch_size, channels, Sequence).
    """

    def __init__(self, n_channels, eps=xnn.EPS, data_format="channels_last"):
        super().__init__()
        self.n_channels = n_channels
        self.data_format = data_format
        if data_format == "channels_last":
            self.gamma = nn.Parameter(torch.zeros(1, n_channels))
            self.beta = nn.Parameter(torch.zeros(1, n_channels))
            self.channel_dim = -1
        elif data_format == "channels_first":
            self.gamma = nn.Parameter(torch.zeros(n_channels, 1))
            self.beta = nn.Parameter(torch.zeros(n_channels, 1))
            self.channel_dim = 1
        else:
            raise ValueError(f"Unsupported data_format: {data_format}")
        self.eps = torch.tensor(eps)

    def forward(self, x):
        g_x = torch.norm(x, p=2, dim=[1, 2], keepdim=True)
        n_x = g_x / (g_x.mean(dim=self.channel_dim, keepdim=True) + self.eps)
        return self.gamma * (x * n_x) + self.beta + x

    def __repr__(self):
        return f"{self.__class__.__name__}(n_channels={self.n_channels}, {self.data_format})"
