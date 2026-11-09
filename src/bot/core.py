"""Core Telegram bot implementation."""

import asyncio
import logging
from pathlib import Path

from telegram import Message
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
)
from telegram.request import HTTPXRequest

from src.bot.handlers.document import DocumentOCRHandler
from src.bot.handlers.ingredient import IngredientHandler
from src.bot.handlers.voice import VoiceHandler
from src.bot.health import HEALTH_PORT
from src.bot.reliable import ReliableUpdateRunner
from src.config.settings import Settings
from src.services.audio.asr import ASRService
from src.services.ingredient.analyzer import IngredientAnalyzer
from src.services.llm import LLMService
from src.services.ocr import OCRService
from src.utils.media_cache import MediaGroupManager

logger = logging.getLogger(__name__)


class _IngredientPhotoFilter(filters.MessageFilter):
    """Match photos sent to the configured ingredient-analysis destination."""

    def __init__(self, channel_id: int, topic_id: int | None) -> None:
        super().__init__(name="IngredientPhotoFilter")
        self.channel_id = channel_id
        self.topic_id = topic_id

    def filter(self, message: Message) -> bool:
        """Return whether a photo belongs to the configured channel and topic."""
        if not message.photo or message.chat_id != self.channel_id:
            return False
        return self.topic_id is None or message.message_thread_id == self.topic_id


class _OCRDestinationFilter(filters.MessageFilter):
    """Match messages sent to the configured OCR frontend destination."""

    def __init__(self, channel_id: int, topic_id: int | None) -> None:
        super().__init__(name="OCRDestinationFilter")
        self.channel_id = channel_id
        self.topic_id = topic_id

    def filter(self, message: Message) -> bool:
        """Return whether a message belongs to the OCR channel and topic."""
        if message.chat_id != self.channel_id:
            return False
        return self.topic_id is None or message.message_thread_id == self.topic_id


class TelegramBot:
    """Telegram bot main class.

    This class sets up and runs the Telegram bot with all handlers.

    Attributes:
        settings: Application settings.
        application: Telegram bot application instance.
    """

    def __init__(self, settings: Settings) -> None:
        """Initialize the bot.

        Args:
            settings: Application settings.
        """
        self.settings = settings
        logger.info(f"Initializing TelegramBot with ASR service: {settings.asr_service_url}")
        # The default 5-second PTB request timeout is too aggressive for a transient
        # DNS/network wobble. Durable retry still handles failures beyond these limits.
        request = HTTPXRequest(
            connection_pool_size=32,
            connect_timeout=15,
            read_timeout=30,
            write_timeout=30,
            media_write_timeout=60,
            pool_timeout=15,
        )
        polling_request = HTTPXRequest(
            connection_pool_size=4,
            connect_timeout=15,
            read_timeout=45,
            write_timeout=30,
            pool_timeout=15,
        )
        self.application = (
            Application.builder()
            .token(settings.telegram_bot_token)
            .request(request)
            .get_updates_request(polling_request)
            .build()
        )
        self._setup_handlers()
        logger.info("TelegramBot initialization complete")

    def _setup_handlers(self) -> None:
        """Set up all message handlers."""
        logger.info("Setting up handlers...")

        # Initialize services
        asr_service = ASRService(
            base_url=self.settings.asr_service_url,
            api_key=self.settings.typeless_api_key,
        )
        ocr_service = OCRService(
            base_url=self.settings.ocr_service_url,
            max_concurrent_requests=self.settings.ocr_max_concurrent,
            timeout_seconds=self.settings.ocr_request_timeout_seconds,
        )
        llm_service = LLMService(
            model=self.settings.llm_model,
            api_key=self.settings.llm_api_key,
            temperature=self.settings.llm_temperature,
            max_tokens=self.settings.llm_max_tokens,
        )
        ingredient_analyzer = IngredientAnalyzer(
            llm_service=llm_service,
            ocr_engine=self.settings.ingredient_ocr_engine,
        )

        # Store services in application for access from handlers
        self.application.ocr_service = ocr_service

        # Initialize media group manager for handling albums
        media_group_manager = MediaGroupManager(timeout_seconds=3.0)

        # Initialize handlers with configuration
        voice_handler = VoiceHandler(
            asr_service=asr_service,
            allowed_group_ids=self.settings.allowed_group_ids,
        )
        ingredient_handler = IngredientHandler(
            analyzer=ingredient_analyzer,
            ocr_service=ocr_service,
            media_group_manager=media_group_manager,
            channel_id=self.settings.ingredient_analysis_channel_id,
            topic_id=self.settings.ingredient_analysis_topic_id,
        )
        document_ocr_handler = DocumentOCRHandler(
            ocr_service=ocr_service,
            max_file_size_bytes=self.settings.ocr_max_file_size_mb * 1024 * 1024,
            default_engine=self.settings.ocr_default_engine,
        )

        # Register handlers
        self.application.add_handler(CommandHandler("start", self._start_command))
        self.application.add_handler(MessageHandler(filters.VOICE, voice_handler.handle))
        self.application.add_handler(MessageHandler(filters.AUDIO, voice_handler.handle_audio))
        if self.settings.ingredient_analysis_channel_id is not None:
            ingredient_filter = _IngredientPhotoFilter(
                self.settings.ingredient_analysis_channel_id,
                self.settings.ingredient_analysis_topic_id,
            )
            self.application.add_handler(
                MessageHandler(ingredient_filter, ingredient_handler.handle)
            )
        ocr_media_filter = filters.PHOTO | filters.Document.IMAGE | filters.Document.PDF
        ocr_command_filter = None
        if self.settings.ocr_channel_id is not None:
            ocr_destination_filter = _OCRDestinationFilter(
                self.settings.ocr_channel_id,
                self.settings.ocr_topic_id,
            )
            ocr_media_filter &= ocr_destination_filter
            ocr_command_filter = ocr_destination_filter

        self.application.add_handler(
            CommandHandler("ocr", document_ocr_handler.show_help, filters=ocr_command_filter)
        )
        self.application.add_handler(
            CommandHandler(
                "ocr_status",
                document_ocr_handler.show_status,
                filters=ocr_command_filter,
            )
        )
        self.application.add_handler(MessageHandler(ocr_media_filter, document_ocr_handler.handle))
        logger.info("Handlers registered: start, voice, audio, ingredient analysis, OCR frontend")

    async def _start_command(self, update, context) -> None:
        """Handle /start command."""
        logger.info("Start command received")
        await update.effective_message.reply_text(
            "👋 欢迎使用 Telegram 多媒体助手！\n"
            "发送图片、原图文件或 PDF，我会提取文字并返回 Markdown。\n"
            "Caption 可指定 paddle、glm、qwen、jina，以及 PDF 页码 pages=2-7。\n"
            "选择 jina 会将图片/PDF 页面上传云端并可能收费，失败需手动重发。\n"
            "使用 /ocr 查看 OCR 帮助，/ocr_status 检查模型状态。\n"
            "指定频道/话题中的配料表图片仍会进行成分分析，也支持语音转文字。"
        )

    def run(self) -> None:
        """Start the bot."""
        logger.info("Starting durable Telegram polling...")
        runner = ReliableUpdateRunner(
            self.application,
            database_path=Path(self.settings.update_store_path),
            worker_count=self.settings.update_worker_count,
            health_port=HEALTH_PORT,
        )
        asyncio.run(runner.run())


async def start_bot(settings: Settings) -> None:
    """Start the bot with given settings.

    Args:
        settings: Application settings.
    """
    bot = TelegramBot(settings)
    bot.run()
