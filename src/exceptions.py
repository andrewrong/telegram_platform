"""Custom exceptions for the Telegram Platform."""


class TelegramPlatformError(Exception):
    """Base exception for all platform errors."""

    pass


class ASRServiceError(TelegramPlatformError):
    """Exception raised when ASR service fails."""

    pass


class AudioDownloadError(TelegramPlatformError):
    """Exception raised when audio download fails."""

    pass


class ConfigError(TelegramPlatformError):
    """Exception raised when configuration is invalid."""

    pass


class OCRServiceError(TelegramPlatformError):
    """Exception raised when OCR service fails."""

    pass


class LLMServiceError(TelegramPlatformError):
    """Exception raised when LLM service fails."""

    pass
