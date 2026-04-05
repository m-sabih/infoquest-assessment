from functools import lru_cache

from pydantic import Field, PostgresDsn
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = Field(default="InfoQuest Expert Search API", validation_alias="APP_NAME")
    database_url: PostgresDsn = Field(validation_alias="DATABASE_URL")


@lru_cache
def get_settings() -> Settings:
    return Settings()
