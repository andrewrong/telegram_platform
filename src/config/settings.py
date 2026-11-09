"""Application settings using pydantic-settings."""

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Application settings loaded from environment variables.

    Attributes:
        telegram_bot_token: Telegram Bot API token.
        asr_service_url: URL of the ASR service.
        allowed_group_ids: List of group IDs to monitor for voice messages.
                           If empty, all voice messages will be processed.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
    )

    telegram_bot_token: str
    asr_service_url: str = "http://127.0.0.1:28111"
    typeless_api_key: str | None = None
    ocr_service_url: str = "http://127.0.0.1:18100"
    allowed_group_ids: list[int] = []

    # LLM Configuration
    llm_model: str = "gemini-3.6-flash"
    llm_api_key: str | None = None
    llm_temperature: float = 0.7
    llm_max_tokens: int = 4096

    # Ingredient Analysis Configuration
    # Channel/Group where ingredient analysis is enabled
    ingredient_analysis_channel_id: int | None = None
    # Specific topic ID within the channel (for forum channels)
    ingredient_analysis_topic_id: int | None = None
    ingredient_ocr_engine: Literal["auto", "paddle", "glm", "qwen"] = "glm"

    # OCR Configuration
    # Maximum concurrent OCR requests (default: 4)
    ocr_max_concurrent: int = 4
    ocr_request_timeout_seconds: float = Field(default=900.0, gt=0)
    ocr_max_file_size_mb: int = Field(default=20, gt=0)
    ocr_default_engine: Literal["auto", "paddle", "glm", "qwen"] = "auto"
    ocr_channel_id: int | None = None
    ocr_topic_id: int | None = None

    # Reliable Telegram update processing
    update_store_path: str = "/app/data/inbox.db"
    update_worker_count: int = Field(default=2, ge=1, le=8)


def get_settings() -> Settings:
    """Get application settings instance.

    Returns:
        Settings instance.
    """
    return Settings()
