"""Tests for application configuration defaults."""

from src.config.settings import Settings


def test_default_llm_model_is_latest_gemini_flash() -> None:
    """Use the current stable Gemini Flash model by default."""
    settings = Settings(telegram_bot_token="test-token", _env_file=None)

    assert settings.llm_model == "gemini-3.6-flash"


def test_ocr_frontend_destination_is_optional() -> None:
    """Keep global OCR routing until a dedicated destination is configured."""
    settings = Settings(telegram_bot_token="test-token", _env_file=None)

    assert settings.ocr_channel_id is None
    assert settings.ocr_topic_id is None


def test_ingredient_images_default_to_glm_ocr() -> None:
    """Use the engine proven reliable for full-frame ingredient photos."""
    settings = Settings(telegram_bot_token="test-token", _env_file=None)

    assert settings.ingredient_ocr_engine == "glm"


def test_typeless_v1_defaults_to_local_service_without_auth() -> None:
    """Keep local development simple while using the stable Typeless base URL."""
    settings = Settings(telegram_bot_token="test-token", _env_file=None)

    assert settings.asr_service_url == "http://127.0.0.1:28111"
    assert settings.typeless_api_key is None
