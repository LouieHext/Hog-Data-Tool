import sys
from functools import cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

from hog_data_tool.hog_data.definitions import WeightUnit


def get_env_file() -> str | tuple[str, ...]:
    """Get the environment file(s) to use based on the context."""
    if "pytest" in sys.modules:
        return ".env.shared"
    return (".env.shared", ".env.local")


class EnvSettings(BaseSettings):
    """Base settings class for environment configuration."""

    model_config = SettingsConfigDict(
        title="Environment setting configuration",
        env_file=get_env_file(),
        strict=False,
        frozen=True,
        extra="ignore",
    )


class EnvConfig(EnvSettings):
    """Configuration from the env for project runs."""

    input_data_path: Path
    output_data_path: Path
    weight_unit: WeightUnit


@cache
def get_env_config() -> EnvConfig:
    """Get the environment configuration singleton."""
    return EnvConfig()  # pyright: ignore[reportCallIssue]

@cache
def get_weight_unit() -> WeightUnit:
    config = get_env_config()
    return config.weight_unit