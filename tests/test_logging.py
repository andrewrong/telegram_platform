"""Tests for secure application logging."""

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.main import (
    LOG_BACKUP_COUNT,
    LOG_MAX_BYTES,
    SensitiveDataFilter,
    create_rotating_file_handler,
)


def make_record(message: str, args: tuple[object, ...] = ()) -> logging.LogRecord:
    """Create a log record for filter tests."""
    return logging.LogRecord(
        name="httpx",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg=message,
        args=args,
        exc_info=None,
    )


def test_redacts_telegram_token_from_formatted_url() -> None:
    """Remove a Telegram token supplied through logging arguments."""
    record = make_record(
        "HTTP Request: %s",
        ("https://api.telegram.org/bot123456:secret-token/getUpdates",),
    )

    assert SensitiveDataFilter().filter(record) is True
    assert record.getMessage() == (
        "HTTP Request: https://api.telegram.org/bot<redacted>/getUpdates"
    )


def test_redacts_telegram_token_from_file_download_url() -> None:
    """Remove a Bot API token embedded in Telegram's file-download URL."""
    record = make_record(
        "Got file path: %s",
        ("https://api.telegram.org/file/bot123456:secret-token/voice/file.oga",),
    )

    assert SensitiveDataFilter().filter(record) is True
    assert record.getMessage() == (
        "Got file path: https://api.telegram.org/file/bot<redacted>/voice/file.oga"
    )


def test_preserves_safe_log_message() -> None:
    """Leave messages without credentials unchanged."""
    record = make_record("Application started")

    SensitiveDataFilter().filter(record)

    assert record.getMessage() == "Application started"


def test_creates_bounded_rotating_file_handler(tmp_path: Path) -> None:
    """Bound the log file size and retained backup count."""
    handler = create_rotating_file_handler(tmp_path / "bot.log")
    try:
        assert isinstance(handler, RotatingFileHandler)
        assert handler.maxBytes == LOG_MAX_BYTES == 20 * 1024 * 1024
        assert handler.backupCount == LOG_BACKUP_COUNT == 5
        assert any(isinstance(item, SensitiveDataFilter) for item in handler.filters)
    finally:
        handler.close()
