import torch
from torch import nn, Tensor


def get_eps(data_type):
    return torch.finfo(data_type).eps


def get_eps_from_data(tensor_data: torch.Tensor):
    return torch.finfo(tensor_data.dtype).eps


EPS = get_eps(torch.float32)


class FuncLayer(nn.Module):
    def __init__(self, lambda_fun):
        super().__init__()
        self.lambda_func = lambda_fun

    def forward(self, x: Tensor):
        return self.lambda_func(x)


def seed_everything(seed: int):
    import random
    random.seed(seed)
    import numpy
    numpy.random.seed(seed)
    import torch
    torch.manual_seed(seed)


def t2n(tensor_data: torch.Tensor):
    return tensor_data.detach().cpu().numpy()


def get_lr(optimizer: torch.optim.Optimizer):
    return optimizer.param_groups[0]['lr']


def get_lrs(optimizer: torch.optim.Optimizer):
    return [param_group['lr'] for param_group in optimizer.param_groups]


class FreeCacheContext:
    def __enter__(self):
        torch.cuda.empty_cache()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        torch.cuda.empty_cache()
