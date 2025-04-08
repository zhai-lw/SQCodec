import logging
from pathlib import Path

import requests
import torch
from pydantic import computed_field

from .en_codec import ModelConfig
from .en_codec import EnCodec
from sq_codec.base.config import FileConfig

CONFIG_DIR = Path(__file__).parent / "configs"

log = logging.getLogger("SQCodec")


class SQCodecConfig(FileConfig):
    config_file: Path

    model_name: str = "debug"
    sample_rate: int = 16000
    model_version: str = "v0"
    model_dir: Path = Path.home() / ".cache" / "sq_codec"
    weight_url: str = None

    network_config: ModelConfig = None

    @computed_field
    @property
    def model_tag(self) -> str:
        return f"{self.model_name}.{self.model_version}"

    @property
    def model_path(self) -> Path:
        return self.model_dir / self.model_tag


class SQCodec:

    def __init__(self, config: SQCodecConfig):
        self.config = config
        self.network = EnCodec(config.network_config)

    def download_weights(self):
        self.config.model_path.mkdir(parents=True, exist_ok=True)
        for module_name in self.network.trainable_modules:
            weight_url = self.config.weight_url.format(module_name)
            weight_path = self.config.model_path / f"{module_name}.pt"
            if weight_path.exists():
                log.info(f"{module_name}({weight_path}) already exists, skip download")
            else:
                log.warning(f"Downloading {module_name}({weight_url}) to {weight_path}")
                response = requests.get(weight_url)
                response.raise_for_status()
                weight_path.write_bytes(response.content)

    def load_pretrained(self):
        self.download_weights()
        self.network.load_model(model_path=self.config.model_path)

    def encode_audio(self, audio_data: torch.Tensor) -> (torch.Tensor, torch.Tensor):
        audio_data, audio_length = self.network.preprocess(audio_data)
        feature = self.network.encoder(audio_data.unsqueeze(1))
        trans_feature = self.network.en_encoder(feature)

        q_trans_feature, indices, _ = self.network.quantizer(trans_feature)
        return q_trans_feature, indices

    def decode_audio(self, audio_feature: torch.Tensor = None, indices: torch.Tensor = None) -> torch.Tensor:
        if audio_feature is None:
            audio_feature = self.network.quantizer.to_features(indices)
        q_feature = self.network.en_decoder(audio_feature)
        audio_data = self.network.decoder(q_feature).squeeze(1)
        return audio_data


def list_models() -> list[str]:
    # return [str(config_path.relative_to(CONFIG_DIR))[:-5] for config_path in CONFIG_DIR.rglob("*.toml")]
    return ['0k75bps', '1k5bps', '3kbps', '6kbps', '12kbps', '12kbps_24khz', '24kbps_24khz', ]


def get_model(config_name, **kwargs) -> SQCodec:
    codec_config = SQCodecConfig(config_file=CONFIG_DIR / f"{config_name}.toml", **kwargs)
    sq_codec = SQCodec(codec_config)
    sq_codec.load_pretrained()
    return sq_codec
