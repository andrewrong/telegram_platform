"""Tests for secure application logging."""

import ast
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


FORBIDDEN_LOG_NAMES = {
    "chat_id",
    "user_id",
    "username",
    "chat_title",
    "file_id",
    "allowed_group_ids",
    "topic_id",
    "media_group_id",
    "channel_id",
}
LOG_METHODS = {"debug", "info", "warning", "error", "exception"}


def _forbidden_in(node: ast.AST) -> list[str]:
    """Names of identifiers or attribute chains that a logging argument mentions."""
    found: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Name) and child.id in FORBIDDEN_LOG_NAMES:
            found.append(child.id)
        elif isinstance(child, ast.Attribute):
            if child.attr in FORBIDDEN_LOG_NAMES:
                found.append(child.attr)
            owner = child.value
            if child.attr == "id" and (
                (isinstance(owner, ast.Attribute) and owner.attr in {"chat", "effective_chat"})
                or (isinstance(owner, ast.Name) and owner.id in {"user", "chat"})
                or (isinstance(owner, ast.Attribute) and owner.attr in {"user", "effective_user"})
            ):
                found.append(f"{ast.unparse(owner)}.id")
    return found


def test_bot_package_logs_no_chat_or_user_identifiers() -> None:
    """No logger call under src/bot/ mentions chat, user, file, topic or group ids."""
    root = Path(__file__).resolve().parents[1] / "src" / "bot"
    violations: list[str] = []
    for path in sorted(root.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr in LOG_METHODS
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "logger"
            ):
                continue
            for argument in [*node.args, *(keyword.value for keyword in node.keywords)]:
                violations.extend(
                    f"{path.relative_to(root)}:{node.lineno}: {name}"
                    for name in _forbidden_in(argument)
                )
    assert violations == []
