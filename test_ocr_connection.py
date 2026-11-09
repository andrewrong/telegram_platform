"""Verify connectivity to the configured OCR service."""

import asyncio
import logging
import sys

from src.config.settings import Settings
from src.exceptions import OCRServiceError
from src.services.ocr import OCRService

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


async def main() -> int:
    """Check the OCR endpoint and report every configured model."""
    settings = Settings(telegram_bot_token="unused-by-ocr-health-check")
    service = OCRService(
        base_url=settings.ocr_service_url,
        max_concurrent_requests=settings.ocr_max_concurrent,
        timeout_seconds=settings.ocr_request_timeout_seconds,
    )
    try:
        health = await service.health()
    except OCRServiceError as error:
        logger.error("OCR health check failed at %s: %s", settings.ocr_service_url, error)
        return 1
    finally:
        await service.close()

    logger.info("OCR status=%s ollama=%s", health.status, health.ollama)
    for model in health.models:
        logger.info("engine=%s model=%s available=%s", model.engine, model.name, model.available)
    return 0 if health.status == "ok" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
