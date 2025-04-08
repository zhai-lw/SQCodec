import math
from functools import reduce, cached_property
from typing import Self

import torch
from pydantic import BaseModel, model_validator, Field, computed_field
from torch import nn

import sq_codec.base.nn as xnn
from sq_codec import modules
from sq_codec.quantize import build_vq


class ModelConfig(BaseModel):
    # ! lr: float = 1e-4,
    # weight_decay: float = 1e-5,
    # optimizer_name: str = 'AdamW',
    feature_dim: int = 256,
    compress_rates: tuple[int, ...] = (2, 2, 2, 2, 2),
    encoder_dims: tuple[int, ...] = (16, 32, 32, 64, 128, 256),
    encoder_depths: tuple[int, ...] = (1, 1, 2, 1, 1, 1),
    decode_rates: tuple[int, ...] = (2, 2, 2, 2, 2),
    decoder_dims: tuple[int, ...] = (384, 192, 96, 48, 48, 24),
    decoder_depths: tuple[int, ...] = (1, 1, 1, 2, 1, 1),
    base_unit: str = 'normal',
    use_norm: bool = True,
    use_snake_act: bool = True,
    decoder_last_layer: str = None,
    vq_config: dict = Field(default_factory=dict),

    @computed_field
    @property
    def hop_length(self) -> int:
        return reduce(lambda x, y: x * y, self.compress_rates)

    @model_validator(mode='after')
    def check_config(self) -> Self:
        assert len(self.compress_rates) + 1 == len(self.encoder_dims) == len(self.encoder_depths)
        assert len(self.decode_rates) + 1 == len(self.decoder_dims) == len(self.decoder_depths)
        return self


class Codec(xnn.Module):
    def __init__(self, mc: ModelConfig):
        super().__init__()
        assert mc.base_unit == 'normal'
        self.mc = mc
        self.encoder = modules.Encoder(
            feature_dim=mc.feature_dim,
            dims=mc.encoder_dims,
            strides=mc.compress_rates,
            depths=mc.encoder_depths,
            drop_path_rate=0.0,
            use_norm=mc.use_norm,
            use_snake_act=mc.use_snake_act,
        )

        self.quantizer = build_vq(feature_dim=mc.feature_dim, **mc.vq_config)

        self.decoder = modules.Decoder(
            feature_dim=mc.feature_dim,
            dims=mc.decoder_dims,
            strides=mc.decode_rates,
            depths=mc.decoder_depths,
            drop_path_rate=0.0,
            use_norm=mc.use_norm,
            use_snake_act=mc.use_snake_act,
            decoder_last_layer=mc.decoder_last_layer,
        )

    @property
    def trainable_modules(self) -> dict[str, torch.nn.Module]:
        return {
            "encoder": self.encoder,
            "quantizer": self.quantizer,
            "decoder": self.decoder,
        }

    @property
    def fill_length(self):
        return self.mc.hop_length

    def preprocess(self, audio_data):
        length = audio_data.shape[-1]
        right_pad = math.ceil(length / self.fill_length) * self.fill_length - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data, length

    def forward(
            self,
            audio_data: torch.Tensor,
    ):
        """
        Parameters
        ----------
        audio_data : Tensor[B x T]
            Audio data to encode
        """
        audio_data, audio_length = self.preprocess(audio_data)

        feature = self.encoder(audio_data.unsqueeze(1)).permute(0, 2, 1)

        q_feature, indices, network_loss = self.quantizer(feature)

        y = self.decoder(q_feature.permute(0, 2, 1)).squeeze(1)

        return {
            'generated_audio': y[..., :audio_length],
            'embedded_audio': q_feature,
            'embedded_indices': indices,
            'network_loss': [('nn', network_loss, 10.0), ],
            'hidden_feature': dict(encoded_feature=feature, quantized_feature=q_feature)
        }

    def compress(self, audio_data: torch.Tensor):
        feature = self.encoder(audio_data).permute(0, 2, 1)
        q_feature, indices, network_loss = self.quantizer(feature)
        return indices, q_feature

    def decompress(self, indices: torch.Tensor = None, q_feature: torch.Tensor = None):
        if q_feature is None:
            q_feature = self.quantizer.to_features(indices)
        y = self.decoder(q_feature.permute(0, 2, 1))
        return y

    @torch.no_grad()
    def extract_unit(self, audio_data: torch.Tensor, process_window: int = 5 * 16000):
        """
        Parameters
        ----------
        audio_data : Tensor[1 x T]
            Audio data to encode
        process_window : int
        """
        assert len(audio_data) == 1, 'Only support batch size 1'
        audio_data, audio_length = self.preprocess(audio_data)
        process_window = process_window // self.fill_length * self.fill_length

        chunk_audio = ChunkData(chunk_len=process_window, prefix_len=self.fill_length, original_data=audio_data[0])
        chunk_indices, chunk_q_feature = [], []
        for x in chunk_audio.chunk_data:
            indices, q_feature = self.compress(x[None, None, :])
            chunk_indices.append(indices[0])
            chunk_q_feature.append(q_feature[0])

        codec_chunk_len, codec_prefix_len = process_window // self.hop_length, self.fill_length // self.hop_length

        return (ChunkData(chunk_len=codec_chunk_len, prefix_len=codec_prefix_len, chunk_data=chunk_indices),
                ChunkData(chunk_len=codec_chunk_len, prefix_len=codec_prefix_len, chunk_data=chunk_q_feature))

    @torch.no_grad()
    def decode_unit(self, chunk_indices=None, chunk_q_feature=None, audio_length: int = None):
        if chunk_q_feature is None:
            chunk_audio = [self.decompress(indices=x[None, :])[0, 0] for x in chunk_indices.chunk_data]
        else:
            chunk_audio = [self.decompress(q_feature=x[None, :, :])[0, 0] for x in chunk_q_feature.chunk_data]
        chunk_audio = ChunkData(chunk_len=len(chunk_audio[0]), prefix_len=self.fill_length, chunk_data=chunk_audio)
        return chunk_audio.data[None, :]


class ChunkData:
    def __init__(self, chunk_len: int, prefix_len: int, original_data=None, chunk_data=None, ):
        assert chunk_len > prefix_len
        self.chunk_len = chunk_len
        self.prefix_len = prefix_len
        self._original_data = original_data
        self._chunk_data = chunk_data

    @property
    def data(self):
        if self._original_data is not None:
            return self._original_data
        else:
            original_data = [self._chunk_data[0], ]
            for x in self._chunk_data[1:]:
                original_data.append(x[self.prefix_len:])
            return torch.cat(original_data, dim=0)

    @property
    def chunk_data(self):
        if self._chunk_data is not None:
            return self._chunk_data
        else:
            chunk_data = []
            for i in range(0, len(self._original_data), self.chunk_len):
                if i == 0:
                    chunk_data.append(self._original_data[:self.chunk_len])
                else:
                    chunk_data.append(self._original_data[i - self.prefix_len:i + self.chunk_len])
            return chunk_data
