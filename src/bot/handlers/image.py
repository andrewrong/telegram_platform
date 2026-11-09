"""Backward-compatible image-only wrapper around the document OCR handler."""

from telegram import Update
from telegram.ext import ContextTypes

from src.bot.handlers.document import DocumentOCRHandler
from src.services.ocr import OCRService


class ImageHandler(DocumentOCRHandler):
    """Preserve the former ImageHandler import and optional chat filtering."""

    def __init__(
        self,
        ocr_service: OCRService,
        allowed_group_ids: list[int] | None = None,
    ) -> None:
        """Initialize the compatibility handler."""
        super().__init__(ocr_service)
        self.allowed_group_ids = set(allowed_group_ids or [])

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Process an image when it belongs to an allowed chat."""
        message = update.effective_message
        if message is None:
            return
        if self.allowed_group_ids and message.chat_id not in self.allowed_group_ids:
            return
        await super().handle(update, context)
