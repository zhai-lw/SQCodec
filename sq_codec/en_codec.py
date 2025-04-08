from functools import cached_property

import torch
from pydantic import computed_field

import sq_codec.codec as base_codec


class ModelConfig(base_codec.ModelConfig):
    en_coder_depth: int = 2,
    en_coder_window_size: int = 500,
    en_coder_dynamic_pos: bool = False,
    en_coder_compress_rate: int = 1,
    en_coder_cache_size: int = 0,

    @computed_field
    @property
    def hop_length(self) -> int:
        return super().hop_length * self.en_coder_compress_rate


class EnCodec(base_codec.Codec):
    def __init__(self, mc: ModelConfig):
        super().__init__(mc)
        if mc.en_coder_compress_rate == 1 and mc.en_coder_cache_size == 0:
            from .local_trans import LocalEncoder, LocalDecoder
            self.en_encoder = LocalEncoder(feature_dim=mc.feature_dim, depth=mc.en_coder_depth,
                                           local_window_size=mc.en_coder_window_size,
                                           use_dynamic_pos_bias=mc.en_coder_dynamic_pos)
            self.en_decoder = LocalDecoder(feature_dim=mc.feature_dim, depth=mc.en_coder_depth,
                                           local_window_size=mc.en_coder_window_size,
                                           use_dynamic_pos_bias=mc.en_coder_dynamic_pos)
        else:
            from .local_trans import CompressedLocalEncoderWithCache, CompressedLocalDecoderWithCache
            self.en_encoder = CompressedLocalEncoderWithCache(feature_dim=mc.feature_dim, depth=mc.en_coder_depth,
                                                              local_window_size=mc.en_coder_window_size,
                                                              cache_size=mc.en_coder_cache_size,
                                                              compress_rate=mc.en_coder_compress_rate,
                                                              use_dynamic_pos_bias=mc.en_coder_dynamic_pos)
            self.en_decoder = CompressedLocalDecoderWithCache(feature_dim=mc.feature_dim, depth=mc.en_coder_depth,
                                                              local_window_size=mc.en_coder_window_size,
                                                              cache_size=mc.en_coder_cache_size,
                                                              compress_rate=mc.en_coder_compress_rate,
                                                              use_dynamic_pos_bias=mc.en_coder_dynamic_pos)

    @property
    def trainable_modules(self):
        return super().trainable_modules | {
            "en_encoder": self.en_encoder,
            "en_decoder": self.en_decoder,
        }

    @property
    def fill_length(self):
        return self.mc.hop_length * self.mc.en_coder_window_size

    def forward(self, audio_data: torch.Tensor):
        audio_data, audio_length = self.preprocess(audio_data)

        feature = self.encoder(audio_data.unsqueeze(1))
        trans_feature = self.en_encoder(feature)

        q_trans_feature, indices, commit_loss = self.quantizer(trans_feature)

        q_feature = self.en_decoder(q_trans_feature)
        y = self.decoder(q_feature).squeeze(1)

        return {
            'generated_audio': y[..., :audio_length],
            'embedded_audio': q_feature,
            'embedded_indices': indices,
            'commit_loss': commit_loss,
            'hidden_feature': dict(encoded_feature=feature, encoded_trans_feature=trans_feature,
                                   quantized_trans_feature=q_trans_feature, quantized_feature=q_feature, )
        }
