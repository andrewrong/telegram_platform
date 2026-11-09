"""Telegram handler for OCR of photos, image documents, and PDF documents."""

import logging
import re
from dataclasses import dataclass

from telegram import InputFile, Message, Update
from telegram.ext import ContextTypes

from src.bot.handlers.base import BaseHandler
from src.exceptions import OCRServiceError
from src.models.schemas import OCRProcessResult
from src.services.ocr import VALID_ENGINES, OCRService
from src.utils.retry import RetryableJobError, TerminalJobError, is_retryable_error

logger = logging.getLogger(__name__)

TELEGRAM_TEXT_LIMIT = 3900
DEFAULT_MAX_FILE_SIZE = 20 * 1024 * 1024
PAGE_RANGE_PATTERN = re.compile(r"^\d+(?:-\d+)?$")


@dataclass(frozen=True)
class _Attachment:
    """A supported Telegram attachment ready for download."""

    file_id: str
    filename: str
    content_type: str
    file_size: int | None
    is_pdf: bool


def parse_ocr_options(caption: str | None) -> tuple[str, str | None]:
    """Parse OCR engine and optional PDF page range from a Telegram caption."""
    engine = "auto"
    pages = None
    if not caption:
        return engine, pages

    tokens = caption.lower().replace("#", " ").split()
    for token in tokens:
        if token in VALID_ENGINES:
            engine = token
        elif token.startswith("pages="):
            candidate = token.removeprefix("pages=")
            if PAGE_RANGE_PATTERN.fullmatch(candidate):
                pages = candidate
    return engine, pages


class DocumentOCRHandler(BaseHandler):
    """Recognize Telegram photos, uncompressed images, and PDF documents."""

    def __init__(
        self,
        ocr_service: OCRService,
        *,
        max_file_size_bytes: int = DEFAULT_MAX_FILE_SIZE,
        default_engine: str = "auto",
    ) -> None:
        """Initialize the document OCR handler."""
        self.ocr_service = ocr_service
        self.max_file_size_bytes = max_file_size_bytes
        self.default_engine = default_engine

    async def show_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show OCR frontend usage instructions."""
        del context
        message = update.effective_message
        if message is None:
            return
        await message.reply_text(
            "📝 OCR 使用说明\n\n"
            "直接发送以下内容：\n"
            "• Telegram 图片\n"
            "• 以文件发送的原始图片\n"
            "• PDF 文件\n\n"
            "Caption 可选择模型：auto、paddle、glm、qwen、jina\n"
            "#jina 或 /ocr jina 会将图片/PDF 页面上传 Jina 云端，可能产生费用。\n"
            "Jina 失败需手动重发，不自动重新执行整条任务。\n"
            "PDF 可指定页码，例如：/ocr qwen pages=2-7\n\n"
            "短结果直接回复，长结果会返回 Markdown 文件。\n\n"
            f"当前 Chat ID：{message.chat_id}\n"
            f"当前 Topic ID：{message.message_thread_id or '无'}"
        )

    async def show_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Show OCR service and model availability."""
        del context
        message = update.effective_message
        if message is None:
            return
        try:
            health = await self.ocr_service.health()
        except OCRServiceError as error:
            await message.reply_text(f"🔴 OCR 服务不可用：{error}")
            return

        heading = "🟢 OCR 服务正常" if health.status == "ok" else "🟡 OCR 服务降级"
        models = [
            f"{'✅' if model.available else '❌'} {model.engine} · {model.name}"
            for model in health.models
        ]
        cloud = (
            f"Jina 云端 · {'已配置' if health.jina.configured else '未配置'} · "
            f"{'元数据可达' if health.jina.reachable else '元数据不可达'} · "
            f"{'本地失败后兜底已启用' if health.jina.fallback_on_local_failure else '仅显式选择'}"
        )
        await message.reply_text(
            "\n".join(
                [heading, *models, cloud, "Jina 会上传图片到云端；状态不验证 Key 或推理容量。"]
            )
        )

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Download supported media, run OCR, and reply with Markdown."""
        message = update.effective_message
        if message is None:
            return

        attachment = self._resolve_attachment(message)
        if attachment is None:
            await message.reply_text("❌ 仅支持图片和 PDF 文件")
            return

        if attachment.file_size and attachment.file_size > self.max_file_size_bytes:
            actual_mb = attachment.file_size / 1024 / 1024
            max_mb = self.max_file_size_bytes / 1024 / 1024
            await message.reply_text(
                f"❌ 文件太大 ({actual_mb:.1f}MB)，请控制在 {max_mb:.0f}MB 以内"
            )
            return

        engine, pages = parse_ocr_options(message.caption)
        if engine == "auto":
            engine = self.default_engine
        processing_message: Message | None = None
        cloud_attempted = engine == "jina"
        try:
            telegram_file = await context.bot.get_file(attachment.file_id)
            file_data = bytes(await telegram_file.download_as_bytearray())
            processing_message = await message.reply_text(
                "🔍 正在识别 PDF，请稍候..." if attachment.is_pdf else "🔍 正在识别图片，请稍候..."
            )
            result = await self._recognize(attachment, file_data, engine, pages)
            cloud_attempted = cloud_attempted or result.engine == "jina" or result.cloud_attempted
            if not result.success:
                if cloud_attempted:
                    raise OCRServiceError(result.error)
                if result.retryable:
                    raise RetryableJobError(result.error)
                await processing_message.edit_text(f"❌ 文字识别失败：{result.error}")
                return

            await processing_message.delete()
            processing_message = None
            await self._send_result(message, result)
        except Exception as error:
            if cloud_attempted:
                await self._notify_cloud_failure(message, processing_message)
                raise TerminalJobError(
                    "Jina OCR update failed; resend manually to avoid duplicate cloud charges"
                ) from error
            if is_retryable_error(error):
                raise
            logger.exception("Failed to process Telegram OCR attachment")
            try:
                await processing_message.edit_text(f"❌ 下载或处理文件失败：{error}")
            except Exception:
                await message.reply_text("❌ 下载或处理文件失败")

    @staticmethod
    async def _notify_cloud_failure(message: Message, processing_message: Message | None) -> None:
        """Best-effort cloud failure feedback without triggering paid task replay."""
        warning = (
            "❌ Jina OCR 下载、识别或结果发送失败，云端处理结果可能不确定。"
            "不会自动重试整条任务；如需手动重发，可能产生额外用量和费用。"
        )
        if processing_message is not None:
            try:
                await processing_message.edit_text(warning)
                return
            except Exception:
                logger.warning("Could not edit Jina OCR failure notification")
        try:
            await message.reply_text(warning)
        except Exception:
            logger.warning("Could not deliver Jina OCR failure notification")

    async def _recognize(
        self,
        attachment: _Attachment,
        file_data: bytes,
        engine: str,
        pages: str | None,
    ) -> OCRProcessResult:
        """Dispatch an attachment to the matching OCR endpoint."""
        if attachment.is_pdf:
            return await self.ocr_service.process_pdf(
                file_data,
                filename=attachment.filename,
                engine=engine,
                pages=pages,
            )
        return await self.ocr_service.process_image(
            file_data,
            filename=attachment.filename,
            content_type=attachment.content_type,
            engine=engine,
        )

    @staticmethod
    def _resolve_attachment(message: Message) -> _Attachment | None:
        """Resolve the supported attachment contained in a message."""
        if message.photo:
            photo = message.photo[-1]
            return _Attachment(
                file_id=photo.file_id,
                filename="telegram-photo.jpg",
                content_type="image/jpeg",
                file_size=photo.file_size,
                is_pdf=False,
            )

        document = message.document
        if document is None or not document.mime_type:
            return None
        if document.mime_type == "application/pdf":
            is_pdf = True
        elif document.mime_type.startswith("image/"):
            is_pdf = False
        else:
            return None
        return _Attachment(
            file_id=document.file_id,
            filename=document.file_name or ("document.pdf" if is_pdf else "image"),
            content_type=document.mime_type,
            file_size=document.file_size,
            is_pdf=is_pdf,
        )

    @staticmethod
    async def _send_result(message: Message, result: OCRProcessResult) -> None:
        """Send short OCR output as text and long output as a Markdown file."""
        if len(result.text) <= TELEGRAM_TEXT_LIMIT:
            await message.reply_text(result.text)
            return

        document = InputFile(result.text.encode("utf-8"), filename="ocr-result.md")
        seconds = result.duration_ms / 1000
        await message.reply_document(
            document=document,
            caption=f"✅ OCR 完成 · {result.engine} · {result.pages} 页 · {seconds:.1f} 秒",
        )
