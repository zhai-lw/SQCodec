from typing import Any

import torch
from torch import nn
import vector_quantize_pytorch
from vector_quantize_pytorch.finite_scalar_quantization import round_ste as fsq_round_ste

from sq_codec.base import utils


class PerplexityCalculator:
    class CodebookRecoder:
        def __init__(self, size: int):
            self.size = size
            self.recoder: torch.Tensor = torch.zeros(self.size)
            self.expectation = self.calc_harmonic(self.size) * self.size

        def reset(self):
            self.recoder.zero_()

        def update(self, indices: torch.Tensor):
            idx, counts = indices.unique(return_counts=True)
            self.recoder.scatter_add_(0, idx, counts.float())

        @property
        def est_probs(self):
            return self.recoder * (self.size / (self.recoder.sum() + 1e-8))

        @property
        def usage_probs(self):
            return (self.recoder > 0).float().mean()

        @property
        def avg_probs(self):
            est_probs = self.est_probs
            est_probs[est_probs > 1] = 1.
            return est_probs.mean()

        @staticmethod
        def calc_harmonic(n, estimate=True):
            if estimate and n > 1e4:
                import numpy
                import math
                return numpy.euler_gamma + math.log(n) + 0.5 / n - 1. / (12 * n ** 2) + 1. / (120 * n ** 4)
            else:
                return sum(1.0 / d for d in range(1, n + 1))

    def __init__(self, vq=None, codebook_size=None, codebook_num=1, ):
        if vq is None:
            self.codebook_num, self.codebook_size = codebook_num, codebook_size
        elif isinstance(vq, vector_quantize_pytorch.VectorQuantize):
            self.codebook_num, self.codebook_size = 1, vq.codebook_size
        elif isinstance(vq, vector_quantize_pytorch.ResidualVQ):
            self.codebook_num, self.codebook_size, _ = vq.codebooks.shape
        else:
            self.codebook_size, self.codebook_num = vq.codebook_size, vq.num_codebooks
        self.recorders = [self.CodebookRecoder(self.codebook_size) for _ in range(self.codebook_num)]

    def reset(self):
        utils.iter.consume(recorder.reset() for recorder in self.recorders)

    def update(self, indices):
        indices = indices.view(-1, self.codebook_num).permute(1, 0).type(torch.int64).cpu()
        for i, recorder in enumerate(self.recorders):
            recorder.update(indices[i])
        return self.avg_probs

    @property
    def est_probs(self):
        return torch.stack([recorder.est_probs for recorder in self.recorders], dim=0)

    @property
    def usage_probs(self):
        return torch.stack([recorder.usage_probs for recorder in self.recorders], dim=0)

    @property
    def avg_probs(self):
        return torch.stack([recorder.avg_probs for recorder in self.recorders], dim=0)

    @property
    def perplexity(self):
        return torch.exp(-torch.sum(self.avg_probs * torch.log(self.avg_probs + 1e-10)))


class VQEmbed(nn.Module):
    def __init__(self, vq, feature_dim, codebook_dim):
        super().__init__()
        self.vq = vq
        self.get_pc = lambda: PerplexityCalculator(self.vq)
        if feature_dim == codebook_dim:
            self.project_in = nn.Identity()
            self.project_out = nn.Identity()
        else:
            self.project_in = nn.Linear(feature_dim, codebook_dim)
            self.project_out = nn.Linear(codebook_dim, feature_dim)

    def to_indices(self, features):
        raise NotImplementedError

    def to_features(self, indices):
        # latents = self.vq.get_output_from_indices(indices)
        latents = self.vq.indices_to_codes(indices)
        return self.project_out(latents)

    def forward(self, x):
        latents = self.project_in(x)
        q_latents, indices, *ret = self.vq(latents)
        q_features = self.project_out(q_latents)
        vq_loss = ret[0].sum() if len(ret) > 0 else torch.Tensor([0.]).to(dtype=x.dtype, device=x.device)
        return q_features, indices, vq_loss


class DacVQ(nn.Module):
    def __init__(self, feature_dim, codebook_dim, codebook_size, codebook_num):
        super().__init__()
        from dac.nn import quantize as dac_quantize
        self.vq = dac_quantize.ResidualVectorQuantize(
            input_dim=feature_dim, codebook_dim=codebook_dim,
            n_codebooks=codebook_num, codebook_size=codebook_size,
        )

    def features_to_indices(self, features):
        raise NotImplementedError

    def indices_to_features(self, indices):
        raise NotImplementedError

    def forward(self, x):
        x = x.permute(0, 2, 1)
        q_feature, indices, latents, commitment_loss, codebook_loss = self.vq(x)
        return q_feature.permute(0, 2, 1), indices.permute(0, 2, 1), 0.25 * commitment_loss + 1 * codebook_loss


class TAAE_FSQ(vector_quantize_pytorch.FSQ):
    def quantize(self, z):
        """ Hybrid Quantize"""
        bounded_z = self.bound(z)
        quantized = fsq_round_ste(bounded_z)
        if self.training:
            mask = torch.rand_like(quantized) > 0.5  # ! 0.5
            quantized[mask] = bounded_z[mask] + (torch.rand_like(bounded_z[mask]) - 0.5)
        half_width = self._levels // 2  # Renormalize to [-1, 1].
        return quantized / half_width


def build_vq(name="vq", feature_dim: int = 256, codebook_dim: int | Any = 8, codebook_size: int = 8096,
             codebook_num: int = 1):
    """
    return a VQ module which takes (B, T, feature_dim) input and output (B, T)or (B, T, codebook_num)
    """
    match name:
        case "dac_rvq":
            return DacVQ(feature_dim=feature_dim, codebook_dim=codebook_dim,
                         codebook_size=codebook_size, codebook_num=codebook_num)
        case "rvq":
            vq = vector_quantize_pytorch.ResidualVQ(
                dim=codebook_dim,
                codebook_size=codebook_size, num_quantizers=codebook_num,
                kmeans_init=True, kmeans_iters=10
            )
        case "vq":
            assert codebook_num == 1
            vq = vector_quantize_pytorch.VectorQuantize(
                dim=codebook_dim,
                codebook_size=codebook_size, use_cosine_sim=True,
            )
        case "lfq":
            assert codebook_num == 1
            vq = vector_quantize_pytorch.LFQ(
                codebook_size=codebook_size,  # codebook size, must be a power of 2
                dim=codebook_dim,  # this is the input feature dimension, defaults to log2(codebook_size) if not defined
                # entropy_loss_weight=0.1,  # how much weight to place on entropy loss
                # diversity_gamma=1.
            )
        case "fsq":
            assert codebook_num == 1
            if codebook_dim == 3:
                fsq_levels = [8, 6, 5]
            elif codebook_dim == 4:
                fsq_levels = [8, 5, 5, 5]
            elif codebook_dim == 5:
                fsq_levels = [7, 5, 5, 5, 5]
            else:
                fsq_levels = codebook_dim
            codebook_dim = len(fsq_levels)
            vq = vector_quantize_pytorch.FSQ(
                levels=fsq_levels
            )
        case "taae_fsq":
            assert codebook_num == 1
            levels = codebook_dim
            codebook_dim = len(levels)
            vq = TAAE_FSQ(
                levels=levels
            )
        case "sim_vq":
            assert codebook_num == 1
            vq = vector_quantize_pytorch.SimVQ(
                dim=codebook_dim,
                codebook_size=codebook_size
            )
            vq.num_codebooks = 1
        case _:
            raise ValueError(f"Unknown vq name: {name}")

    return VQEmbed(vq, feature_dim, codebook_dim)


if __name__ == "__main__":
    import torch

    # vq_model = build_vq(name="dac_rvq", feature_dim=256, codebook_dim=8, codebook_size=1024, codebook_num=3)
    # vq_model = build_vq(name="dac_rvq", feature_dim=256, codebook_dim=8, codebook_size=1024, codebook_num=1)
    # vq_model = build_vq(name="rvq", feature_dim=256, codebook_dim=8, codebook_size=1024, codebook_num=3)
    # vq_model = build_vq(name="vq", feature_dim=256, codebook_dim=8, codebook_size=1024)
    # vq_model = build_vq(name="lfq", feature_dim=256, codebook_dim=10, codebook_size=1024)
    vq_model = build_vq(name="fsq", feature_dim=256, codebook_dim=[8, 5, 5, 5], codebook_size=1024)

    q_x, ids, nn_loss = vq_model(torch.randn(5, 46, 256))
    pc = vq_model.get_pc()
    pc.update(ids)
    print(q_x.shape, ids.shape, nn_loss)
    print(q_x.min(), q_x.max())
    print(ids.min(), ids.max())
