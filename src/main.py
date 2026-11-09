"""Main entry point for the Telegram Platform bot."""

import logging
import re
import sys
from logging.handlers import RotatingFileHandler
from pathlib import Path

from src.bot.core import TelegramBot
from src.config.settings import get_settings

LOG_MAX_BYTES = 20 * 1024 * 1024
LOG_BACKUP_COUNT = 5
TELEGRAM_API_TOKEN_PATTERN = re.compile(r"(https://api\.telegram\.org/(?:file/)?bot)[^/\s\"']+")
TELEGRAM_BARE_TOKEN_PATTERN = re.compile(r"(?<!\w)\d{6,}:[A-Za-z0-9_-]{20,}")


def redact_sensitive_data(text: str) -> str:
    """Replace Telegram Bot API credentials in log text."""
    redacted = TELEGRAM_API_TOKEN_PATTERN.sub(r"\1<redacted>", text)
    return TELEGRAM_BARE_TOKEN_PATTERN.sub("<telegram-bot-token-redacted>", redacted)


class SensitiveDataFilter(logging.Filter):
    """Redact credentials before a log record reaches any handler."""

    def filter(self, record: logging.LogRecord) -> bool:
        """Render and sanitize the record message."""
        message = record.getMessage()
        redacted = redact_sensitive_data(message)
        if redacted != message:
            record.msg = redacted
            record.args = ()
        return True


class SensitiveDataFormatter(logging.Formatter):
    """Redact credentials from the fully formatted message and traceback."""

    def format(self, record: logging.LogRecord) -> str:
        """Format and sanitize one complete log entry."""
        return redact_sensitive_data(super().format(record))


def create_rotating_file_handler(log_file: Path) -> RotatingFileHandler:
    """Create the bounded, credential-safe application file handler."""
    handler = RotatingFileHandler(
        log_file,
        maxBytes=LOG_MAX_BYTES,
        backupCount=LOG_BACKUP_COUNT,
        encoding="utf-8",
    )
    handler.addFilter(SensitiveDataFilter())
    return handler


def setup_logging() -> None:
    """Configure logging for the application.

    Logs are written to both console and a log file in the project directory.
    Log file location: logs/bot.log
    """
    log_dir = Path(__file__).parent.parent / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / "bot.log"

    formatter = SensitiveDataFormatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    sensitive_filter = SensitiveDataFilter()

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(sensitive_filter)

    # File handler
    file_handler = create_rotating_file_handler(log_file)
    file_handler.setFormatter(formatter)

    # Root logger configuration
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    for existing_handler in root_logger.handlers[:]:
        root_logger.removeHandler(existing_handler)
        existing_handler.close()
    root_logger.addHandler(console_handler)
    root_logger.addHandler(file_handler)

    # httpx includes the full Telegram Bot API URL at INFO, which contains the token.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)

    # Log the log file location
    startup_logger = logging.getLogger(__name__)
    startup_logger.info(f"Logging to file: {log_file.absolute()}")


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, non-zero for error).
    """
    setup_logging()
    logger = logging.getLogger(__name__)

    try:
        settings = get_settings()
        logger.info("Starting Telegram Platform Bot...")

        bot = TelegramBot(settings)
        bot.run()

        return 0
    except Exception as e:
        logger.error(f"Failed to start bot: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
