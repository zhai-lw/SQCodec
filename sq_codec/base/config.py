from pathlib import Path

# requirements: pip install pydantic-settings
from pydantic_settings import BaseSettings, SettingsConfigDict, PydanticBaseSettingsSource, TomlConfigSettingsSource


class FileConfig(BaseSettings):
    model_config = SettingsConfigDict()

    config_file: Path | None

    def __init__(self, config_file: Path = None, **kwargs):
        self.model_config['toml_file'] = config_file
        super().__init__(config_file=config_file, **kwargs)

    @classmethod
    def settings_customise_sources(
            cls,
            settings_cls: type[BaseSettings],
            init_settings: PydanticBaseSettingsSource,
            env_settings: PydanticBaseSettingsSource,
            dotenv_settings: PydanticBaseSettingsSource,
            file_secret_settings: PydanticBaseSettingsSource,
    ) -> tuple[PydanticBaseSettingsSource, ...]:
        return (
            init_settings,
            env_settings,
            dotenv_settings,
            file_secret_settings,
            TomlConfigSettingsSource(settings_cls),
        )
