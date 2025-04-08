import torch
from torch import nn

from sq_codec.layers import Conv1d


class LocalTrans(nn.Module):
    def __init__(
            self,
            dim=512,
            depth=6,
            causal=True,
            local_attn_window_size=512,
            dim_head=64,
            heads=8,
            ff_mult=4,
            attn_dropout=0.,
            ff_dropout=0.,
            use_dynamic_pos_bias=False,
            qk_rmsnorm=False,
    ):

        from local_attention.transformer import DynamicPositionBias, LocalMHA, FeedForward
        super().__init__()

        self.layers = nn.ModuleList([])

        self.window_size = local_attn_window_size
        self.use_rotary_pos_emb = not use_dynamic_pos_bias
        self.dynamic_pos_bias = None if self.use_rotary_pos_emb else DynamicPositionBias(dim=dim // 2, heads=heads)

        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                LocalMHA(dim=dim, dim_head=dim_head, heads=heads, dropout=attn_dropout, causal=causal,
                         window_size=self.window_size, use_xpos=False, xpos_scale_base=None,
                         use_rotary_pos_emb=self.use_rotary_pos_emb, prenorm=True,
                         qk_rmsnorm=qk_rmsnorm, exact_windowsize=False,
                         ),
                FeedForward(dim=dim, mult=ff_mult, dropout=ff_dropout)
            ]))

    def forward(self, x, mask=None):
        attn_bias = None if self.use_rotary_pos_emb else self.dynamic_pos_bias(self.window_size, self.window_size * 2)
        for attn, ff in self.layers:
            x = attn(x, mask=mask, attn_bias=attn_bias) + x
            x = ff(x) + x

        return x

    @classmethod
    def builder(cls, feature_dim=128, depth=2, local_window_size=200, use_dynamic_pos_bias=False):
        return cls(dim=feature_dim, depth=depth, dim_head=feature_dim // 4, heads=6, ff_mult=4,
                   causal=True, local_attn_window_size=local_window_size, use_dynamic_pos_bias=use_dynamic_pos_bias, )


class LocalEncoder(nn.Module):
    def __init__(self, feature_dim=128,
                 depth=2, local_window_size=200, use_dynamic_pos_bias=False):
        super().__init__()
        self.local_trans = LocalTrans.builder(
            feature_dim=feature_dim, depth=depth,
            local_window_size=local_window_size, use_dynamic_pos_bias=use_dynamic_pos_bias,
        )

    def forward(self, feature):
        """
        Args:
            feature: (B, C, T)
        Returns:
            local_feature: (B, T, C)
        """
        feature = feature.permute(0, 2, 1)
        feature = self.local_trans(feature)
        return feature


class LocalDecoder(nn.Module):
    def __init__(self, feature_dim=128,
                 depth=2, local_window_size=200, use_dynamic_pos_bias=False):
        super().__init__()
        self.local_trans = LocalTrans.builder(
            feature_dim=feature_dim, depth=depth,
            local_window_size=local_window_size, use_dynamic_pos_bias=use_dynamic_pos_bias,
        )

    def forward(self, feature):
        """
        Args:
            feature: (B, T, C)
        Returns:
            feature: (B, C, T)
        """
        feature = self.local_trans(feature)
        return feature.permute(0, 2, 1)


class UpTransV1(nn.Module):
    def __init__(self, feature_dim=128, window_size=200, compress_rate=2, depth=2, **kwargs):
        super().__init__()
        assert window_size % compress_rate == 0
        self.feature_dim = feature_dim
        self.compress_rate = compress_rate
        self.trans = LocalTrans.builder(feature_dim, local_window_size=window_size, depth=depth, **kwargs)
        self.compressed_tokens = nn.ParameterList([nn.Parameter(torch.randn(1, 1, feature_dim))
                                                   for _ in range(compress_rate - 1)])

    def forward(self, x):
        compressed_tokens = [compressed_token.expand(x.shape) for compressed_token in self.compressed_tokens]
        x = torch.stack([x, ] + compressed_tokens, dim=2).reshape(x.shape[0], -1, self.feature_dim)
        x = self.trans(x)
        return x


class UpTransV2(nn.Module):
    def __init__(self, feature_dim=128, window_size=200, compress_rate=2, depth=2, **kwargs):
        super().__init__()
        assert window_size % compress_rate == 0
        self.feature_dim = feature_dim
        self.compress_rate = compress_rate
        self.trans = LocalTrans.builder(feature_dim, local_window_size=window_size, depth=depth, **kwargs)
        self.up_layer = nn.Upsample(scale_factor=self.compress_rate, mode='linear', align_corners=False)

    def forward(self, x):
        x = self.up_layer(x.permute(0, 2, 1)).permute(0, 2, 1)
        x = self.trans(x)
        return x


class DownTrans(nn.Module):
    def __init__(self, feature_dim=128, window_size=200, compress_rate=2, depth=2, **kwargs):
        super().__init__()
        assert window_size % compress_rate == 0
        self.feature_dim = feature_dim
        self.compress_rate = compress_rate
        self.trans = LocalTrans.builder(feature_dim, local_window_size=window_size, depth=depth, **kwargs)
        self.down_layer = Conv1d(feature_dim, feature_dim, kernel_size=compress_rate, stride=compress_rate)

    def forward(self, x):
        x = self.trans(x)
        # x = x[:, ::self.compress_rate, :]  # v1
        x = self.down_layer(x.permute(0, 2, 1)).permute(0, 2, 1)  # v2
        return x


class CompressedLocalEncoderWithCache(nn.Module):
    def __init__(self, feature_dim=128, local_window_size=200, compress_rate=2, cache_size=3, depth=4, **kwargs):
        super().__init__()
        self.local_window_size = local_window_size
        self.cache_size = cache_size
        self.compress_rate = compress_rate
        self.trans_window_size = local_window_size + cache_size

        self.cache_token = nn.Parameter(torch.randn(1, self.cache_size * self.compress_rate, feature_dim))

        self.down_trans = DownTrans(feature_dim, window_size=self.trans_window_size * compress_rate,
                                    compress_rate=compress_rate, depth=2, **kwargs)

        self.local_trans = LocalTrans.builder(
            feature_dim, local_window_size=self.trans_window_size, depth=depth - 2, **kwargs)

    def forward(self, feature):
        feature = feature.permute(0, 2, 1)
        split_feature = torch.split(feature, self.local_window_size * self.compress_rate, dim=1)
        cache_token = self.cache_token.expand(feature.shape[0], -1, -1)
        feature = torch.cat([f for fs in split_feature for f in (cache_token, fs,)], dim=1)
        # assert feature[:, self.down_trans_window_size: 2*self.down_trans_window_size, :].equal(
        #     feature.reshape(B, -1, self.down_trans_window_size, C)[:, 1, :, :])
        feature = self.down_trans(feature)
        feature = self.local_trans(feature)
        return feature


class CompressedLocalDecoderWithCache(nn.Module):
    def __init__(self, feature_dim=128, local_window_size=200, compress_rate=2, cache_size=3, depth=4, **kwargs):
        super().__init__()
        self.local_window_size = local_window_size
        self.cache_size = cache_size
        self.compress_rate = compress_rate
        self.trans_window_size = local_window_size + cache_size

        self.up_trans = UpTransV2(feature_dim=feature_dim, window_size=self.trans_window_size * self.compress_rate,
                                  compress_rate=self.compress_rate, depth=2, **kwargs)

        self.local_trans = LocalTrans.builder(
            feature_dim=feature_dim, local_window_size=self.trans_window_size, depth=depth - 2, **kwargs)

    def forward(self, feature):
        b, t, c = feature.shape
        feature = self.local_trans(feature)
        feature = self.up_trans(feature)

        feature = (feature.reshape(b, -1, self.trans_window_size * self.compress_rate, c)
                   [:, :, (self.cache_size * self.compress_rate):, :]
                   .reshape(b, -1, c))
        return feature.permute(0, 2, 1)


def main():
    import torch

    encode_model = CompressedLocalEncoderWithCache(feature_dim=33, local_window_size=300,
                                                   compress_rate=3, cache_size=6, ).cuda()
    decode_model = CompressedLocalDecoderWithCache(feature_dim=33, local_window_size=300,
                                                   compress_rate=3, cache_size=6, ).cuda()
    # encode_model = LocalEncoder(feature_dim=33, depth=2, local_window_size=200).cuda()
    # decode_model = LocalDecoder(feature_dim=33, depth=2, local_window_size=200).cuda()

    x = torch.randn(3, 33, 900).cuda()

    y = encode_model(x)  # (3,

    x_ = decode_model(y)  # (3,

    print(y.shape)


if __name__ == '__main__':
    main()
