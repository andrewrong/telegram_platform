"""Ingredient analysis handler for Telegram bot."""

import asyncio
import logging

from telegram import Message, Update
from telegram.ext import ContextTypes

from src.bot.handlers.base import BaseHandler
from src.services.ingredient.analyzer import IngredientAnalyzer
from src.services.ocr import OCRService
from src.utils.media_cache import MediaGroupManager

logger = logging.getLogger(__name__)

# Maximum image file size: 10MB
MAX_IMAGE_FILE_SIZE = 10 * 1024 * 1024

# Separator for merging multiple OCR results
IMAGE_SEPARATOR = "\n\n=== 图片 {index} ===\n\n"


class IngredientHandler(BaseHandler):
    """Handler for ingredient list analysis.

    This handler processes images containing food ingredient lists,
    extracts text via OCR, and uses LLM to provide health analysis.

    Supports both single images and media groups (multiple images sent
    as an album), merging OCR results from multiple images.

    Attributes:
        analyzer: Ingredient analysis service.
        ocr_service: OCR service for text extraction.
        media_group_manager: Manager for handling media groups.
        allowed_group_ids: Set of group IDs to monitor (empty = process all).
    """

    def __init__(
        self,
        analyzer: IngredientAnalyzer,
        ocr_service: OCRService,
        media_group_manager: MediaGroupManager,
        channel_id: int | None = None,
        topic_id: int | None = None,
    ) -> None:
        """Initialize ingredient handler.

        Args:
            analyzer: Ingredient analyzer instance.
            ocr_service: OCR service for text extraction.
            media_group_manager: Manager for handling media groups.
            channel_id: Specific channel/group ID where analysis is enabled.
                       If None, all channels/groups will be processed.
            topic_id: Specific topic ID within the channel (for forum channels).
                     If None, all topics will be processed.
        """
        self.analyzer = analyzer
        self.ocr_service = ocr_service
        self.media_group_manager = media_group_manager
        self.channel_id = channel_id
        self.topic_id = topic_id
        self._has_channel_filter = channel_id is not None
        self._has_topic_filter = topic_id is not None
        logger.info(f"IngredientHandler initialized (channel_id={channel_id}, topic_id={topic_id})")

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an image message for ingredient analysis.

        Supports both single images and media groups (albums).
        For media groups, images are cached and processed together
        after all images are received or a timeout occurs.

        Args:
            update: Telegram update containing the image message.
            context: Telegram context object.
        """
        user = update.effective_user
        message = update.effective_message

        # Log chat info
        chat = message.chat if message else None
        chat_type = chat.type if chat else "unknown"
        chat_id = chat.id if chat else None
        chat_title = chat.title if chat else (chat.username if chat else "Private Chat")

        logger.info(
            f"Received ingredient analysis request from user_id={user.id if user else 'unknown'}, "
            f"username={user.username if user else 'unknown'}, "
            f"chat_id={chat_id}, chat_type={chat_type}, chat_title={chat_title}"
        )

        if not message or not message.photo:
            logger.warning("No photo detected in update")
            await message.reply_text("❌ 未检测到图片")
            return

        # Check channel/topic filters first
        if not self._check_filters(chat_id, message):
            return

        # Check file size before downloading
        photo = message.photo[-1]
        if photo.file_size and photo.file_size > MAX_IMAGE_FILE_SIZE:
            size_mb = photo.file_size / 1024 / 1024
            max_mb = MAX_IMAGE_FILE_SIZE / 1024 / 1024
            logger.warning(f"Image file too large: {photo.file_size} bytes (max {max_mb:.0f}MB)")
            await message.reply_text(
                f"❌ 图片文件太大 ({size_mb:.1f}MB)，请控制在 {max_mb:.0f}MB 以内"
            )
            return

        # Check if this is part of a media group
        media_group_id = message.media_group_id

        if media_group_id:
            # Handle media group (album)
            await self._handle_media_group(
                media_group_id=media_group_id,
                message=message,
                context=context,
            )
        else:
            # Handle single image
            await self._handle_single_image(
                message=message,
                context=context,
            )

    def _check_filters(self, chat_id: int | None, message: Message) -> bool:
        """Check if message passes channel and topic filters.

        Args:
            chat_id: ID of the chat.
            message: Telegram message object.

        Returns:
            True if message passes filters, False otherwise.
        """
        # Check if channel filtering is enabled
        if self._has_channel_filter:
            if chat_id != self.channel_id:
                logger.info(
                    f"Ignoring message from chat {chat_id} - "
                    f"not the configured channel {self.channel_id}"
                )
                return False
            logger.info(f"Processing message from configured channel {chat_id}")

        # Check if topic filtering is enabled (for forum channels)
        if self._has_topic_filter:
            message_topic_id = getattr(message, "message_thread_id", None)
            if message_topic_id != self.topic_id:
                logger.info(
                    f"Ignoring message from topic {message_topic_id} - "
                    f"not the configured topic {self.topic_id}"
                )
                return False
            logger.info(f"Processing message from configured topic {self.topic_id}")

        return True

    async def _handle_single_image(
        self,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle a single image message.

        Args:
            message: Telegram message object.
            context: Telegram context object.
        """
        photo = message.photo[-1]
        logger.info(
            f"Processing single photo: file_id={photo.file_id}, "
            f"width={photo.width}, height={photo.height}, "
            f"file_size={photo.file_size} bytes"
        )

        try:
            # Download image
            file = await context.bot.get_file(photo.file_id)
            image_bytes = await file.download_as_bytearray()
            logger.info(f"Downloaded {len(image_bytes)} bytes")

            # Send processing message
            processing_msg = await message.reply_text("🔍 正在分析配料表，请稍候...")

            # Analyze
            result = await self.analyzer.analyze_from_image(bytes(image_bytes), self.ocr_service)

            # Delete processing message and send result
            await processing_msg.delete()
            await self._send_result(message, result)

        except Exception as e:
            logger.exception(f"Error processing single image: {e}")
            await message.reply_text("❌ 处理图片失败")

    async def _handle_media_group(
        self,
        media_group_id: str,
        message: Message,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Handle an image that is part of a media group (album).

        Images are cached and processed together after all images
        are received or a timeout occurs.

        Args:
            media_group_id: Unique identifier for the media group.
            message: Telegram message object.
            context: Telegram context object.
        """
        photo = message.photo[-1]
        logger.info(
            f"Processing media group image: media_group_id={media_group_id}, "
            f"file_id={photo.file_id}"
        )

        try:
            # Download image
            file = await context.bot.get_file(photo.file_id)
            image_bytes = await file.download_as_bytearray()

            # Add to media group manager
            is_first = await self.media_group_manager.add_image(
                media_group_id=media_group_id,
                image_data=bytes(image_bytes),
                message=message,
                process_callback=self._process_media_group,
                context=context,
            )

            # Send "processing" message only for the first image
            if is_first:
                cache = self.media_group_manager._caches.get(media_group_id)
                if cache:
                    cache.processing_message = await message.reply_text(
                        "🔍 检测到多张图片，正在分析配料表，请稍候..."
                    )

        except Exception as e:
            logger.exception(f"Error handling media group image: {e}")

    async def _process_media_group(
        self,
        media_group_id: str,
        images: list[bytes],
        chat_id: int,
        message_thread_id: int | None,
        context: ContextTypes.DEFAULT_TYPE,
    ) -> None:
        """Process a complete media group.

        This callback is called by MediaGroupManager when all images
        have been collected or the timeout occurs.

        Args:
            media_group_id: The media group ID.
            images: List of image data bytes.
            chat_id: ID of the chat.
            message_thread_id: Topic ID for forum channels.
            context: Telegram context object.
        """
        logger.info(f"Processing media group {media_group_id} with {len(images)} images")

        # Get cache to retrieve processing message
        cache = self.media_group_manager._caches.get(media_group_id)
        processing_msg = cache.processing_message if cache else None

        try:
            # Process all images with OCR
            ocr_results = await self._process_multiple_images(images)

            if not ocr_results:
                if processing_msg:
                    try:
                        await processing_msg.edit_text("❌ 无法从图片中识别出文字")
                    except Exception:
                        # Message might be expired, send new message
                        await context.bot.send_message(
                            chat_id=chat_id,
                            message_thread_id=message_thread_id,
                            text="❌ 无法从图片中识别出文字",
                        )
                return

            # Merge OCR results
            merged_text = self._merge_ocr_results(ocr_results)

            # Analyze merged text
            result = await self.analyzer.analyze_from_text(merged_text)

            # Delete processing message (ignore errors if message expired)
            if processing_msg:
                try:
                    await processing_msg.delete()
                except Exception:
                    pass  # Message might be expired, ignore

            # Send result directly to chat (don't rely on reply_to message)
            text, entities = self.analyzer.format_for_telegram(result)
            try:
                await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=text,
                    entities=entities,
                )
            except Exception as send_error:
                logger.error(f"Failed to send message: {send_error}")
                # Fallback: send plain text
                await context.bot.send_message(
                    chat_id=chat_id,
                    message_thread_id=message_thread_id,
                    text=result[:4000],  # Limit length
                )

        except Exception as e:
            logger.exception(f"Error processing media group: {e}")
            error_text = "❌ 分析过程中出现错误"
            if processing_msg:
                try:
                    await processing_msg.edit_text(error_text)
                except Exception:
                    # Message might be expired, send new message
                    try:
                        await context.bot.send_message(
                            chat_id=chat_id,
                            message_thread_id=message_thread_id,
                            text=error_text,
                        )
                    except Exception as send_err:
                        logger.error(f"Failed to send error message: {send_err}")

    async def _process_multiple_images(self, images: list[bytes]) -> list[str]:
        """Process multiple images with OCR concurrently.

        Args:
            images: List of image data bytes.

        Returns:
            List of OCR results (empty strings for failed images).
        """
        logger.info(f"Processing {len(images)} images with OCR")

        # Create tasks for all images
        tasks = [
            self.ocr_service.process_image(
                image,
                engine=self.analyzer.ocr_engine,
            )
            for image in images
        ]

        # Run OCR concurrently (limited by OCRService semaphore)
        results = await asyncio.gather(*tasks, return_exceptions=True)

        ocr_texts = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"OCR failed for image {i + 1}: {result}")
                ocr_texts.append("")
            elif result.success and self.analyzer.is_plausible_ingredient_text(result.text):
                ocr_texts.append(result.text)
            elif result.success:
                logger.warning("Rejected implausible OCR text for image %d", i + 1)
                ocr_texts.append("")
            else:
                logger.error(f"OCR failed for image {i + 1}: {result.error}")
                ocr_texts.append("")

        return ocr_texts

    def _merge_ocr_results(self, ocr_results: list[str]) -> str:
        """Merge multiple OCR results into a single text.

        Args:
            ocr_results: List of OCR text results.

        Returns:
            Merged text with separators.
        """
        # Filter out empty results
        valid_results = [(i + 1, text) for i, text in enumerate(ocr_results) if text.strip()]

        if not valid_results:
            return ""

        # Build merged text with separators
        parts = []
        for index, text in valid_results:
            if len(valid_results) > 1:
                parts.append(IMAGE_SEPARATOR.format(index=index))
            parts.append(text)

        return "".join(parts)

    async def _send_result(self, message: Message, result: str) -> None:
        """Send analysis result to user.

        Args:
            message: Message to reply to.
            result: Analysis result text.
        """
        if result.startswith("❌"):
            logger.error(f"Analysis failed: {result}")
            await message.reply_text(result)
        else:
            logger.info(f"Analysis successful, result length: {len(result)}")
            # Format for Telegram using telegramify-markdown
            text, entities = self.analyzer.format_for_telegram(result)
            await message.reply_text(text, entities=entities)
