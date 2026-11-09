"""Tests for audio file handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Audio, Chat, Message, Update, User
from telegram.ext import ContextTypes

from src.bot.handlers.voice import VoiceHandler
from src.models.schemas import AudioProcessResult


@pytest.fixture
def mock_asr_service():
    """Create a mock ASR service."""
    service = MagicMock()
    service.process_audio = AsyncMock()
    return service


@pytest.fixture
def voice_handler(mock_asr_service):
    """Create a VoiceHandler instance."""
    return VoiceHandler(asr_service=mock_asr_service)


@pytest.fixture
def mock_update_with_audio():
    """Create a mock Update with audio file."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_user.username = "test_user"

    update.effective_message = MagicMock(spec=Message)
    update.effective_message.chat = MagicMock(spec=Chat)
    update.effective_message.chat.id = 67890
    update.effective_message.chat.type = "private"
    update.effective_message.chat.title = None
    update.effective_message.chat.username = "testuser"

    # Create audio object (MP3 file, etc.)
    audio = MagicMock(spec=Audio)
    audio.file_id = "audio_file_123"
    audio.file_size = 1024 * 1024  # 1MB
    audio.duration = 60  # 60 seconds
    audio.file_name = "test_audio.mp3"
    audio.mime_type = "audio/mpeg"

    update.effective_message.audio = audio
    update.effective_message.voice = None
    update.effective_message.reply_text = AsyncMock()

    return update


@pytest.fixture
def mock_context():
    """Create a mock context."""
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot = MagicMock()
    return context


@pytest.fixture
def mock_file():
    """Create a mock file."""
    file = MagicMock()
    file.file_path = "https://api.telegram.org/file/test/audio.mp3"
    file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
    return file


class TestAudioHandler:
    """Test cases for audio file handling."""

    async def test_handle_audio_success(
        self, voice_handler, mock_update_with_audio, mock_context, mock_file, mock_asr_service
    ):
        """Test successful audio file processing."""
        # Arrange
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)
        mock_asr_service.process_audio.return_value = AudioProcessResult(
            success=True, text="这是音频转录的文本", error=""
        )

        # Act
        await voice_handler.handle_audio(mock_update_with_audio, mock_context)

        # Assert
        mock_context.bot.get_file.assert_called_once_with("audio_file_123")
        mock_file.download_as_bytearray.assert_called_once()
        mock_asr_service.process_audio.assert_called_once_with(
            b"audio data",
            duration=60,
            postprocess_mode="standard",
        )
        mock_update_with_audio.effective_message.reply_text.assert_called_once_with(
            "这是音频转录的文本"
        )

    async def test_handle_audio_file_too_large(
        self, voice_handler, mock_update_with_audio, mock_context
    ):
        """Test that large audio files are rejected."""
        # Arrange - set file size to 60MB (over the 50MB limit)
        mock_update_with_audio.effective_message.audio.file_size = 60 * 1024 * 1024

        # Act
        await voice_handler.handle_audio(mock_update_with_audio, mock_context)

        # Assert
        mock_update_with_audio.effective_message.reply_text.assert_called_once()
        call_args = mock_update_with_audio.effective_message.reply_text.call_args[0][0]
        assert "太大" in call_args

    async def test_handle_audio_no_audio_detected(
        self, voice_handler, mock_update_with_audio, mock_context
    ):
        """Test handling when no audio is detected."""
        # Arrange
        mock_update_with_audio.effective_message.audio = None

        # Act
        await voice_handler.handle_audio(mock_update_with_audio, mock_context)

        # Assert
        mock_update_with_audio.effective_message.reply_text.assert_called_once_with(
            "❌ 未检测到音频文件"
        )

    async def test_handle_audio_asr_failure(
        self, voice_handler, mock_update_with_audio, mock_context, mock_file, mock_asr_service
    ):
        """Test handling when ASR service fails."""
        # Arrange
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)
        mock_asr_service.process_audio.return_value = AudioProcessResult(
            success=False, text="", error="ASR service unavailable"
        )

        # Act
        await voice_handler.handle_audio(mock_update_with_audio, mock_context)

        # Assert
        mock_update_with_audio.effective_message.reply_text.assert_called_once()
        call_args = mock_update_with_audio.effective_message.reply_text.call_args[0][0]
        assert "转写失败" in call_args

    async def test_handle_audio_download_error(
        self, voice_handler, mock_update_with_audio, mock_context, mock_asr_service
    ):
        """Test handling when file download fails."""
        # Arrange
        mock_context.bot.get_file = AsyncMock(side_effect=Exception("Network error"))

        # Act
        await voice_handler.handle_audio(mock_update_with_audio, mock_context)

        # Assert
        mock_update_with_audio.effective_message.reply_text.assert_called_once_with(
            "❌ 下载音频文件失败"
        )
