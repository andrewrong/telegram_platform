"""Voice message handler."""

import logging

from telegram import Update
from telegram.ext import ContextTypes

from src.bot.handlers.base import BaseHandler
from src.services.audio.asr import ASRService
from src.utils.retry import RetryableJobError, is_retryable_error

logger = logging.getLogger(__name__)

# Maximum voice file size: 10MB
MAX_VOICE_FILE_SIZE = 10 * 1024 * 1024

# Maximum audio file size: 50MB (audio files can be larger than voice messages)
MAX_AUDIO_FILE_SIZE = 50 * 1024 * 1024


class VoiceHandler(BaseHandler):
    """Handler for voice messages.

    This handler downloads voice messages from Telegram and
    sends them to the ASR service for transcription.

    When configured with allowed_group_ids, only messages from
    those specific groups/channels will be processed.

    Attributes:
        asr_service: Service for audio transcription.
        allowed_group_ids: Set of group IDs to monitor (empty = process all).
    """

    def __init__(
        self,
        asr_service: ASRService,
        allowed_group_ids: list[int] | None = None,
    ) -> None:
        """Initialize voice handler.

        Args:
            asr_service: ASR service instance.
            allowed_group_ids: List of group IDs to monitor. If empty/None,
                             all voice messages will be processed.
        """
        self.asr_service = asr_service
        self.allowed_group_ids = set(allowed_group_ids or [])
        self._has_filter = bool(self.allowed_group_ids)
        logger.info(
            f"VoiceHandler initialized (filter_enabled={self._has_filter}, "
            f"allowed_groups={self.allowed_group_ids})"
        )

    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle a voice message.

        Args:
            update: Telegram update containing the voice message.
            context: Telegram context object.
        """
        user = update.effective_user
        message = update.effective_message

        # Log chat info for group ID identification
        chat = message.chat if message else None
        chat_type = chat.type if chat else "unknown"
        chat_id = chat.id if chat else None
        chat_title = chat.title if chat else (chat.username if chat else "Private Chat")

        logger.info(
            f"Received voice message from user_id={user.id if user else 'unknown'}, "
            f"username={user.username if user else 'unknown'}, "
            f"chat_id={chat_id}, chat_type={chat_type}, chat_title={chat_title}"
        )

        if not message or not message.voice:
            logger.warning("No voice message detected in update")
            await message.reply_text("❌ 未检测到语音消息")
            return

        voice = message.voice

        # Check file size before downloading
        if voice.file_size and voice.file_size > MAX_VOICE_FILE_SIZE:
            size_mb = voice.file_size / 1024 / 1024
            max_mb = MAX_VOICE_FILE_SIZE / 1024 / 1024
            logger.warning(f"Voice file too large: {voice.file_size} bytes (max {max_mb:.0f}MB)")
            await message.reply_text(
                f"❌ 语音文件太大 ({size_mb:.1f}MB)，请控制在 {max_mb:.0f}MB 以内"
            )
            return

        # Check if group filtering is enabled
        if self._has_filter:
            if chat_id not in self.allowed_group_ids:
                logger.info(f"Ignoring message from chat {chat_id} - not in allowed groups")
                return
            logger.info(f"Processing message from allowed group {chat_id}")
        logger.info(
            f"Voice info: file_id={voice.file_id}, duration={voice.duration}s, "
            f"file_size={voice.file_size} bytes"
        )

        try:
            # Get voice file info
            logger.info(f"Requesting file info for file_id={voice.file_id}")
            file = await context.bot.get_file(voice.file_id)
            logger.info("Got voice file metadata")

            # Download audio data
            logger.info("Downloading voice data...")
            audio_bytes = await file.download_as_bytearray()
            logger.info(f"Downloaded {len(audio_bytes)} bytes")

            # Process with ASR service
            logger.info("Sending to ASR service...")
            result = await self.asr_service.process_audio(
                bytes(audio_bytes),
                duration=voice.duration,
                postprocess_mode="standard",
            )

            if result.success:
                logger.info(f"ASR success, text length: {len(result.text)}")
                await message.reply_text(result.text)
            else:
                if result.retryable:
                    raise RetryableJobError(result.error)
                logger.error(f"ASR failed: {result.error}")
                await message.reply_text(f"❌ 转写失败: {result.error}")

        except Exception as e:
            if is_retryable_error(e):
                raise
            logger.exception(f"Error processing voice message: {e}")
            await message.reply_text("❌ 下载语音消息失败")

    async def handle_audio(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
        """Handle an audio file message (MP3, etc.).

        Args:
            update: Telegram update containing the audio file.
            context: Telegram context object.
        """
        user = update.effective_user
        message = update.effective_message

        # Log chat info for group ID identification
        chat = message.chat if message else None
        chat_type = chat.type if chat else "unknown"
        chat_id = chat.id if chat else None
        chat_title = chat.title if chat else (chat.username if chat else "Private Chat")

        logger.info(
            f"Received audio file from user_id={user.id if user else 'unknown'}, "
            f"username={user.username if user else 'unknown'}, "
            f"chat_id={chat_id}, chat_type={chat_type}, chat_title={chat_title}"
        )

        if not message or not message.audio:
            logger.warning("No audio file detected in update")
            await message.reply_text("❌ 未检测到音频文件")
            return

        audio = message.audio

        # Check file size before downloading
        if audio.file_size and audio.file_size > MAX_AUDIO_FILE_SIZE:
            size_mb = audio.file_size / 1024 / 1024
            max_mb = MAX_AUDIO_FILE_SIZE / 1024 / 1024
            logger.warning(f"Audio file too large: {audio.file_size} bytes (max {max_mb:.0f}MB)")
            await message.reply_text(
                f"❌ 音频文件太大 ({size_mb:.1f}MB)，请控制在 {max_mb:.0f}MB 以内"
            )
            return

        # Check if group filtering is enabled
        if self._has_filter:
            if chat_id not in self.allowed_group_ids:
                logger.info(f"Ignoring message from chat {chat_id} - not in allowed groups")
                return
            logger.info(f"Processing message from allowed group {chat_id}")

        file_name = audio.file_name if audio.file_name else "unknown"
        mime_type = audio.mime_type if audio.mime_type else "unknown"
        logger.info(
            f"Audio info: file_id={audio.file_id}, duration={audio.duration}s, "
            f"file_size={audio.file_size} bytes, file_name={file_name}, mime_type={mime_type}"
        )

        try:
            # Get audio file info
            logger.info(f"Requesting file info for file_id={audio.file_id}")
            file = await context.bot.get_file(audio.file_id)
            logger.info("Got audio file metadata")

            # Download audio data
            logger.info("Downloading audio data...")
            audio_bytes = await file.download_as_bytearray()
            logger.info(f"Downloaded {len(audio_bytes)} bytes")

            # Process with ASR service
            logger.info("Sending to ASR service...")
            result = await self.asr_service.process_audio(
                bytes(audio_bytes),
                duration=audio.duration,
                postprocess_mode="standard",
            )

            if result.success:
                logger.info(f"ASR success, text length: {len(result.text)}")
                await message.reply_text(result.text)
            else:
                if result.retryable:
                    raise RetryableJobError(result.error)
                logger.error(f"ASR failed: {result.error}")
                await message.reply_text(f"❌ 转写失败: {result.error}")

        except Exception as e:
            if is_retryable_error(e):
                raise
            logger.exception(f"Error processing audio file: {e}")
            await message.reply_text("❌ 下载音频文件失败")
