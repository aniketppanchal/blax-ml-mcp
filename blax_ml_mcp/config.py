import re
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    bl_api_key: str = Field(
        validation_alias="BL_API_KEY",
    )
    bl_workspace: str = Field(
        validation_alias="BL_WORKSPACE",
    )
    bl_region: Literal["us-pdx-1", "us-was-1", "eu-lon-1", "eu-fra-1"] = Field(
        default="us-pdx-1",
        validation_alias="BL_REGION",
    )

    model_trainer_sandbox_name: str = "blax-ml-model-trainer"
    model_trainer_sandbox_image: str = "blax-ml-model-trainer:latest"
    model_trainer_sandbox_memory: Literal[2048, 4096, 8192] = 4096

    model_server_sandbox_image: str = "blax-ml-model-server:latest"
    model_server_sandbox_memory: Literal[2048, 4096, 8192] = 4096
    model_server_sandbox_ttl: str = "1d"

    csv_download_timeout_seconds: int = 300
    csv_profile_timeout_seconds: int = 60000
    model_training_timeout_seconds: int = 300000

    @field_validator("model_server_sandbox_ttl")
    def validate_model_server_sandbox_ttl(cls, v: str) -> str:
        v = v.lower()
        if not re.fullmatch(r"\d+[hdw]", v):
            raise ValueError("Invalid TTL format. Must match <number>[h|d|w]")
        return v

    model_config = SettingsConfigDict(
        env_prefix="BLAX_ML_MCP_",
        case_sensitive=False,
    )


settings = Settings()
