from functools import partial

import torch
from torch import nn
from ..layers import ChannelNorm, Conv1d, Linear, GRN, Snake1d
from sq_codec.base.nn.layers import Residual


class TrendConv(nn.Module):
    def __init__(self, output_dim, kernels=(7, 7, 7), dilations=(1, 3, 9)):
        super().__init__()
        assert dilations[0] == 1
        target_dims = [output_dim // len(dilations) - i for i, dil in enumerate(dilations)]
        target_dims[0] += output_dim - sum(target_dims)

        self.conv = nn.ModuleList([
            nn.Sequential(
                nn.MaxPool1d(kernel_size=dilation, stride=1, padding=dilation // 2) if dilation > 1 else nn.Identity(),
                nn.AvgPool1d(kernel_size=dilation, stride=1, padding=dilation // 2) if dilation > 1 else nn.Identity(),
                Conv1d(1, target_dim, kernel_size=kernel, dilation=dilation,
                       padding=(kernel - 1) * dilation // 2)
            )

            for (target_dim, kernel, dilation) in zip(target_dims, kernels, dilations)
        ])

    def forward(self, x):
        out = torch.cat([conv(x) for conv in self.conv], dim=1)
        return out


class SuperConvUnit(nn.Module):
    def __init__(self, dim, snake_act=True, norm=False):
        super().__init__()
        assert snake_act, "SuperUnit only supports snake_act=True"
        assert norm == False, "SuperUnit only supports norm=False"
        self.block = nn.Sequential(
            # Snake1d(dim),
            TrendConv(dim, dim, kernels=(7, 7, 7, 7), dilations=(1, 3, 5, 9)),
            ChannelNorm(dim, data_format="channels_first"),
            Conv1d(dim, dim * 4, kernel_size=1),
            Snake1d(dim * 4),
            Conv1d(dim * 4, dim, kernel_size=1),
        )

    def forward(self, x):
        return self.block(x)


ResidualSuperConvUnit = lambda *args, drop_rate=0., **kwargs: (
    Residual(SuperConvUnit(*args, **kwargs), drop_prob=drop_rate))
