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

    openrouter_api_key: str = Field(default="", validation_alias="OPENROUTER_API_KEY")
    openrouter_base_url: str = Field(
        default="https://openrouter.ai/api/v1",
        validation_alias="OPENROUTER_BASE_URL",
    )
    embedding_model: str = Field(
        default="openai/text-embedding-3-small",
        validation_alias="EMBEDDING_MODEL",
    )
    chat_model: str = Field(
        default="openai/gpt-5.4",
        validation_alias="CHAT_MODEL",
    )
    chat_checkpoint_path: str = Field(
        default="./data/langgraph/checkpoints.sqlite",
        validation_alias="CHAT_CHECKPOINT_PATH",
    )

    chroma_path: str = Field(default="./data/chroma", validation_alias="CHROMA_PATH")
    collection_name: str = Field(default="candidates", validation_alias="COLLECTION_NAME")

    candidate_page_size: int = Field(default=200, ge=1, le=20_000, validation_alias="CANDIDATE_PAGE_SIZE")
    embedding_batch_size: int = Field(default=24, ge=1, le=1024, validation_alias="EMBEDDING_BATCH_SIZE")
    max_profile_chars: int = Field(default=24_000, ge=1000, validation_alias="MAX_PROFILE_CHARS")

    # Retrieve rerank_multiplier × top_k candidates from the vector store, then
    # let the LLM score them so only genuinely relevant profiles are returned.
    rerank_multiplier: int = Field(default=3, ge=1, le=10, validation_alias="RERANK_MULTIPLIER")
    # Minimum LLM relevance score (0–10) a candidate must receive to be included
    # in the final result set.  Candidates below this threshold are dropped.
    min_rerank_score: float = Field(default=4.0, ge=0.0, le=10.0, validation_alias="MIN_RERANK_SCORE")


@lru_cache
def get_settings() -> Settings:
    return Settings()
