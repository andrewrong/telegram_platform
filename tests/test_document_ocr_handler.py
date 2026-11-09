"""Tests for Telegram image and PDF OCR handling."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Document, Message, PhotoSize, Update, User
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from src.bot.handlers.document import DocumentOCRHandler, parse_ocr_options
from src.models.schemas import OCRHealthResponse, OCRModelStatus, OCRProcessResult
from src.services.ocr import OCRService
from src.utils.retry import is_retryable_error


@pytest.fixture
def ocr_service() -> MagicMock:
    """Create a mocked OCR service."""
    service = MagicMock(spec=OCRService)
    service.process_image = AsyncMock()
    service.process_pdf = AsyncMock()
    service.health = AsyncMock()
    return service


@pytest.fixture
def update() -> MagicMock:
    """Create a Telegram update with a writable message mock."""
    mocked_update = MagicMock(spec=Update)
    mocked_update.effective_user = MagicMock(spec=User, id=123, username="tester")
    message = MagicMock(spec=Message)
    message.caption = None
    message.photo = None
    message.document = None
    message.reply_text = AsyncMock()
    message.reply_document = AsyncMock()
    mocked_update.effective_message = message
    return mocked_update


@pytest.fixture
def context() -> MagicMock:
    """Create a Telegram callback context mock."""
    mocked_context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    mocked_context.bot = MagicMock()
    mocked_context.bot.get_file = AsyncMock()
    return mocked_context


def success_result(text: str = "Recognized text") -> OCRProcessResult:
    """Create a successful OCR result."""
    return OCRProcessResult(
        success=True,
        text=text,
        engine="paddle",
        pages=1,
        duration_ms=250,
    )


class TestOCROptions:
    """Test OCR options parsed from captions."""

    def test_defaults(self) -> None:
        """Use automatic engine and all pages by default."""
        assert parse_ocr_options(None) == ("auto", None)

    def test_engine_and_pages(self) -> None:
        """Read engine and PDF page range from a caption."""
        assert parse_ocr_options("/ocr qwen pages=2-7") == ("qwen", "2-7")

    def test_hash_engine(self) -> None:
        """Accept a short hashtag engine selector."""
        assert parse_ocr_options("#glm") == ("glm", None)

    def test_jina_caption(self) -> None:
        """Select Jina explicitly for images and PDF page ranges."""
        assert parse_ocr_options("#jina") == ("jina", None)
        assert parse_ocr_options("/ocr jina pages=2-7") == ("jina", "2-7")


class TestDocumentOCRHandler:
    """Test the Telegram OCR handler."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("notification_fails", [False, True])
    async def test_jina_recognition_failure_notifies_without_replaying(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
        notification_fails: bool,
    ) -> None:
        """Show a manual-resend warning even when bounded server retries are exhausted."""
        update.effective_message.photo = [
            MagicMock(spec=PhotoSize, file_id="photo-id", file_size=1024)
        ]
        update.effective_message.caption = "#jina"
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"photo")
        context.bot.get_file.return_value = telegram_file
        processing = MagicMock()
        processing.edit_text = AsyncMock(side_effect=TimedOut() if notification_fails else None)
        update.effective_message.reply_text.return_value = processing
        if notification_fails:
            update.effective_message.reply_text.side_effect = [processing, TimedOut()]
        ocr_service.process_image.return_value = OCRProcessResult(
            success=False, error="server exhausted budget", retryable=True
        )
        with pytest.raises(Exception) as caught:
            await DocumentOCRHandler(ocr_service).handle(update, context)
        processing.edit_text.assert_awaited_once()
        warning = processing.edit_text.call_args.args[0]
        assert "不确定" in warning and "手动" in warning and "费用" in warning
        assert not is_retryable_error(caught.value)
        assert str(caught.value.__cause__) == "server exhausted budget"
        ocr_service.process_image.assert_awaited_once()

    @pytest.mark.asyncio
    @pytest.mark.parametrize("engine", ["jina", "auto"])
    async def test_cloud_send_failure_is_terminal_but_local_is_retryable(
        self, ocr_service: MagicMock, update: MagicMock, context: MagicMock, engine: str
    ) -> None:
        """A failed result delivery must not replay a potentially paid cloud request."""
        update.effective_message.photo = [
            MagicMock(spec=PhotoSize, file_id="photo-id", file_size=1024)
        ]
        update.effective_message.caption = f"#{engine}"
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"photo")
        context.bot.get_file.return_value = telegram_file
        processing = MagicMock()
        processing.delete = AsyncMock()
        update.effective_message.reply_text.side_effect = [processing, TimedOut(), None]
        ocr_service.process_image.return_value = success_result()
        with pytest.raises(Exception) as caught:
            await DocumentOCRHandler(ocr_service).handle(update, context)
        assert is_retryable_error(caught.value) is (engine != "jina")
        if engine == "jina":
            assert "手动重发" in update.effective_message.reply_text.call_args.args[0]
            assert update.effective_message.reply_text.await_count == 3

    @pytest.mark.asyncio
    @pytest.mark.parametrize("stage", ["download", "recognize", "processing", "delete"])
    async def test_jina_failures_never_replay_update(
        self, ocr_service: MagicMock, update: MagicMock, context: MagicMock, stage: str
    ) -> None:
        """Bounded server retries must not become unbounded Telegram job retries."""
        update.effective_message.photo = [
            MagicMock(spec=PhotoSize, file_id="photo-id", file_size=1024)
        ]
        update.effective_message.caption = "#jina"
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"photo")
        context.bot.get_file.return_value = telegram_file
        processing = MagicMock()
        processing.delete = AsyncMock()
        update.effective_message.reply_text.return_value = processing
        ocr_service.process_image.return_value = success_result()
        if stage == "download":
            context.bot.get_file.side_effect = TimedOut()
        elif stage == "recognize":
            ocr_service.process_image.return_value = OCRProcessResult(
                success=False, error="server exhausted budget", retryable=True
            )
        elif stage == "processing":
            update.effective_message.reply_text.side_effect = TimedOut()
        else:
            processing.delete.side_effect = TimedOut()
        with pytest.raises(Exception) as caught:
            await DocumentOCRHandler(ocr_service).handle(update, context)
        assert not is_retryable_error(caught.value)

    @pytest.mark.asyncio
    async def test_processes_telegram_photo(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Download the largest Telegram photo and OCR it."""
        photo = MagicMock(spec=PhotoSize)
        photo.file_id = "photo-id"
        photo.file_size = 1024
        update.effective_message.photo = [photo]
        update.effective_message.caption = "#glm"
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"photo-bytes")
        context.bot.get_file.return_value = telegram_file
        processing_message = MagicMock()
        processing_message.delete = AsyncMock()
        update.effective_message.reply_text.return_value = processing_message
        ocr_service.process_image.return_value = success_result()

        handler = DocumentOCRHandler(ocr_service)
        await handler.handle(update, context)

        ocr_service.process_image.assert_awaited_once_with(
            b"photo-bytes",
            filename="telegram-photo.jpg",
            content_type="image/jpeg",
            engine="glm",
        )
        update.effective_message.reply_text.assert_any_await("Recognized text")
        processing_message.delete.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_processes_pdf_document_with_page_range(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Download and OCR a PDF document with caption options."""
        document = MagicMock(spec=Document)
        document.file_id = "pdf-id"
        document.file_name = "report.pdf"
        document.mime_type = "application/pdf"
        document.file_size = 2048
        update.effective_message.document = document
        update.effective_message.caption = "/ocr qwen pages=3-5"
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"pdf-bytes")
        context.bot.get_file.return_value = telegram_file
        processing_message = MagicMock()
        processing_message.delete = AsyncMock()
        update.effective_message.reply_text.return_value = processing_message
        ocr_service.process_pdf.return_value = success_result("PDF text")

        handler = DocumentOCRHandler(ocr_service)
        await handler.handle(update, context)

        ocr_service.process_pdf.assert_awaited_once_with(
            b"pdf-bytes",
            filename="report.pdf",
            engine="qwen",
            pages="3-5",
        )
        update.effective_message.reply_text.assert_any_await("PDF text")

    @pytest.mark.asyncio
    async def test_processes_image_document_without_compression(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Support images sent as Telegram documents."""
        document = MagicMock(spec=Document)
        document.file_id = "image-id"
        document.file_name = "scan.png"
        document.mime_type = "image/png"
        document.file_size = 2048
        update.effective_message.document = document
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"png-bytes")
        context.bot.get_file.return_value = telegram_file
        processing_message = MagicMock()
        processing_message.delete = AsyncMock()
        update.effective_message.reply_text.return_value = processing_message
        ocr_service.process_image.return_value = success_result()

        handler = DocumentOCRHandler(ocr_service)
        await handler.handle(update, context)

        ocr_service.process_image.assert_awaited_once_with(
            b"png-bytes",
            filename="scan.png",
            content_type="image/png",
            engine="auto",
        )

    @pytest.mark.asyncio
    async def test_sends_long_result_as_markdown_file(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Avoid Telegram's message length limit for large OCR output."""
        photo = MagicMock(spec=PhotoSize, file_id="photo-id", file_size=1024)
        update.effective_message.photo = [photo]
        telegram_file = MagicMock()
        telegram_file.download_as_bytearray = AsyncMock(return_value=b"photo")
        context.bot.get_file.return_value = telegram_file
        processing_message = MagicMock()
        processing_message.delete = AsyncMock()
        update.effective_message.reply_text.return_value = processing_message
        ocr_service.process_image.return_value = success_result("x" * 4000)

        handler = DocumentOCRHandler(ocr_service)
        await handler.handle(update, context)

        update.effective_message.reply_document.assert_awaited_once()
        sent_document = update.effective_message.reply_document.call_args.kwargs["document"]
        assert sent_document.filename == "ocr-result.md"

    @pytest.mark.asyncio
    async def test_rejects_large_file(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Reject a Telegram document before downloading when it is too large."""
        document = MagicMock(spec=Document)
        document.file_size = 21 * 1024 * 1024
        document.mime_type = "application/pdf"
        update.effective_message.document = document

        handler = DocumentOCRHandler(ocr_service, max_file_size_bytes=20 * 1024 * 1024)
        await handler.handle(update, context)

        context.bot.get_file.assert_not_awaited()
        assert "文件太大" in update.effective_message.reply_text.call_args.args[0]

    @pytest.mark.asyncio
    async def test_ocr_help_command(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Explain supported attachments, engines, and page ranges."""
        handler = DocumentOCRHandler(ocr_service)

        await handler.show_help(update, context)

        help_text = update.effective_message.reply_text.call_args.args[0]
        assert "图片" in help_text
        assert "PDF" in help_text
        assert "pages=2-7" in help_text
        assert "paddle" in help_text
        assert "qwen" in help_text
        assert "jina" in help_text
        assert "上传" in help_text

    @pytest.mark.asyncio
    async def test_ocr_status_command(
        self,
        ocr_service: MagicMock,
        update: MagicMock,
        context: MagicMock,
    ) -> None:
        """Show OCR service and model availability."""
        ocr_service.health.return_value = OCRHealthResponse(
            status="ok",
            ollama=True,
            models=[
                OCRModelStatus(engine="paddle", name="paddle-model", available=True),
                OCRModelStatus(engine="glm", name="glm-model", available=False),
                OCRModelStatus(engine="qwen", name="qwen-model", available=True),
            ],
        )
        handler = DocumentOCRHandler(ocr_service)

        await handler.show_status(update, context)

        status_text = update.effective_message.reply_text.call_args.args[0]
        assert "OCR 服务正常" in status_text
        assert "✅ paddle" in status_text
        assert "❌ glm" in status_text
        assert "✅ qwen" in status_text
