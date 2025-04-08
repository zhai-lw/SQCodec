from torch import nn, Tensor


class Permute(nn.Module):
    def __init__(self, *dims):
        super().__init__()
        self.dims = dims

    def forward(self, x: Tensor):
        return x.permute(*self.dims)


class Squeeze(nn.Module):
    def __init__(self, dim=None):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.squeeze(dim=self.dim)


class Unsqueeze(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor):
        return x.unsqueeze(dim=self.dim)


class Take(nn.Module):
    def __init__(self, dim: int = 1, position: int = 0):
        super().__init__()
        self.dim = dim
        self.position = position

    def forward(self, x: Tensor) -> Tensor:
        return x.unbind(dim=self.dim)[self.position]


class Residual(nn.Module):
    def __init__(self, module: nn.Module, drop_prob: float = 0., scale_by_keep: bool = True):
        super().__init__()
        assert 0 <= drop_prob < 1
        self.module = module
        self.drop_prob = drop_prob
        self.scale_by_keep = scale_by_keep

    def drop_path(self, x_side: Tensor):
        if self.drop_prob == 0. or not self.training:
            return x_side
        keep_prob = 1 - self.drop_prob
        shape = (x_side.shape[0],) + (1,) * (x_side.ndim - 1)
        keep_mask = x_side.new_empty(shape).bernoulli_(keep_prob)
        if self.scale_by_keep:
            keep_mask.div_(keep_prob)
        return x_side * keep_mask

    def forward(self, x: Tensor):
        x_side = self.module(x)
        x_side = self.drop_path(x_side)
        return x + x_side
