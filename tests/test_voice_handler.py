"""Tests for voice message handler."""

from unittest.mock import AsyncMock, MagicMock

import pytest
from telegram import Chat, File, Message, Update, Voice
from telegram.error import TimedOut
from telegram.ext import ContextTypes

from src.bot.handlers.voice import VoiceHandler
from src.models.schemas import AudioProcessResult


class TestVoiceHandler:
    """Test cases for VoiceHandler."""

    @pytest.fixture
    def mock_asr(self) -> AsyncMock:
        """Create mock ASR service."""
        return AsyncMock()

    @pytest.fixture
    def handler(self, mock_asr: AsyncMock) -> VoiceHandler:
        """Create voice handler with mocked ASR service."""
        return VoiceHandler(asr_service=mock_asr)

    @pytest.fixture
    def handler_with_filter(self, mock_asr: AsyncMock) -> VoiceHandler:
        """Create voice handler with group filter configured."""
        return VoiceHandler(
            asr_service=mock_asr,
            allowed_group_ids=[-1001234567890],
        )

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create mock context object."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot = AsyncMock()
        return context

    @pytest.fixture
    def mock_group_update(self) -> MagicMock:
        """Create mock update from allowed group."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = 123456
        update.effective_user.username = "testuser"
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.voice = MagicMock(spec=Voice)
        update.effective_message.voice.file_id = "test_file_id"
        update.effective_message.voice.duration = 5
        update.effective_message.voice.file_size = 1024
        update.effective_message.chat = MagicMock(spec=Chat)
        update.effective_message.chat.id = -1001234567890
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_handle_voice_message_success(
        self,
        handler: VoiceHandler,
        mock_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test handling voice message successfully."""
        # Arrange
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "voice/test.oga"
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        handler.asr_service.process_audio = AsyncMock(
            return_value=AudioProcessResult(success=True, text="Hello world", error="")
        )

        # Act
        await handler.handle(mock_group_update, mock_context)

        # Assert
        mock_context.bot.get_file.assert_called_once_with("test_file_id")
        mock_group_update.effective_message.reply_text.assert_called_once_with("Hello world")

    @pytest.mark.asyncio
    async def test_handle_voice_message_no_voice(
        self,
        handler: VoiceHandler,
        mock_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test handling message without voice."""
        # Arrange
        mock_group_update.effective_message.voice = None

        # Act
        await handler.handle(mock_group_update, mock_context)

        # Assert
        mock_group_update.effective_message.reply_text.assert_called_once_with(
            "❌ 未检测到语音消息"
        )

    @pytest.mark.asyncio
    async def test_handle_voice_message_transcription_failure(
        self,
        handler: VoiceHandler,
        mock_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test handling when transcription fails."""
        # Arrange
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "voice/test.oga"
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        handler.asr_service.process_audio = AsyncMock(
            return_value=AudioProcessResult(success=False, text="", error="Service timeout")
        )

        # Act
        await handler.handle(mock_group_update, mock_context)

        # Assert
        mock_group_update.effective_message.reply_text.assert_called_once_with(
            "❌ 转写失败: Service timeout"
        )

    @pytest.mark.asyncio
    async def test_handle_voice_message_download_error(
        self,
        handler: VoiceHandler,
        mock_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test handling when audio download fails."""
        # Arrange
        mock_context.bot.get_file = AsyncMock(side_effect=Exception("Network error"))

        # Act
        await handler.handle(mock_group_update, mock_context)

        # Assert
        mock_group_update.effective_message.reply_text.assert_called_once_with(
            "❌ 下载语音消息失败"
        )

    @pytest.mark.asyncio
    async def test_handle_voice_message_reraises_transient_download_error(
        self,
        handler: VoiceHandler,
        mock_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """The durable worker must see timeouts so it can retry this same update."""
        mock_context.bot.get_file = AsyncMock(side_effect=TimedOut())

        with pytest.raises(TimedOut):
            await handler.handle(mock_group_update, mock_context)

        mock_group_update.effective_message.reply_text.assert_not_called()


class TestVoiceHandlerFileSizeLimit:
    """Test cases for voice handler file size limiting."""

    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

    @pytest.fixture
    def mock_asr(self) -> AsyncMock:
        """Create mock ASR service."""
        return AsyncMock()

    @pytest.fixture
    def handler(self, mock_asr: AsyncMock) -> VoiceHandler:
        """Create voice handler."""
        return VoiceHandler(asr_service=mock_asr)

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create mock context object."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot = AsyncMock()
        return context

    @pytest.fixture
    def mock_small_file_update(self) -> MagicMock:
        """Create mock update with small file (under limit)."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = 123456
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.voice = MagicMock(spec=Voice)
        update.effective_message.voice.file_id = "test_file_id"
        update.effective_message.voice.file_size = 1024 * 1024  # 1MB
        update.effective_message.chat = MagicMock(spec=Chat)
        update.effective_message.chat.id = 123456
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_large_file_update(self) -> MagicMock:
        """Create mock update with large file (over limit)."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = 123456
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.voice = MagicMock(spec=Voice)
        update.effective_message.voice.file_id = "large_file_id"
        update.effective_message.voice.file_size = 20 * 1024 * 1024  # 20MB
        update.effective_message.chat = MagicMock(spec=Chat)
        update.effective_message.chat.id = 123456
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.mark.asyncio
    async def test_reject_oversized_file(
        self,
        handler: VoiceHandler,
        mock_large_file_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that oversized files are rejected without downloading."""
        # Act
        await handler.handle(mock_large_file_update, mock_context)

        # Assert - Should not download or process
        mock_context.bot.get_file.assert_not_called()
        handler.asr_service.process_audio.assert_not_called()
        mock_large_file_update.effective_message.reply_text.assert_called_once()
        # Check error message mentions file size
        call_args = mock_large_file_update.effective_message.reply_text.call_args[0][0]
        assert "太大" in call_args or "size" in call_args.lower() or "10MB" in call_args

    @pytest.mark.asyncio
    async def test_accept_normal_file(
        self,
        handler: VoiceHandler,
        mock_small_file_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that normal sized files are processed."""
        # Arrange
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "voice/test.oga"
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        handler.asr_service.process_audio = AsyncMock(
            return_value=AudioProcessResult(success=True, text="Hello", error="")
        )

        # Act
        await handler.handle(mock_small_file_update, mock_context)

        # Assert - Should process normally
        mock_context.bot.get_file.assert_called_once()
        handler.asr_service.process_audio.assert_called_once()


class TestVoiceHandlerWithGroupFilter:
    """Test cases for voice handler with group filtering."""

    @pytest.fixture
    def mock_asr(self) -> AsyncMock:
        """Create mock ASR service."""
        return AsyncMock()

    @pytest.fixture
    def handler_with_filter(self, mock_asr: AsyncMock) -> VoiceHandler:
        """Create voice handler with group filter configured."""
        return VoiceHandler(
            asr_service=mock_asr,
            allowed_group_ids=[-1001234567890],
        )

    @pytest.fixture
    def mock_allowed_group_update(self) -> MagicMock:
        """Create mock update from allowed group."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = 123456
        update.effective_user.username = "testuser"
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.voice = MagicMock(spec=Voice)
        update.effective_message.voice.file_id = "test_file_id"
        update.effective_message.voice.duration = 5
        update.effective_message.voice.file_size = 1024
        update.effective_message.chat = MagicMock(spec=Chat)
        update.effective_message.chat.id = -1001234567890  # Allowed group
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_unauthorized_group_update(self) -> MagicMock:
        """Create mock update from unauthorized group."""
        update = MagicMock(spec=Update)
        update.effective_user = MagicMock()
        update.effective_user.id = 123456
        update.effective_user.username = "testuser"
        update.effective_message = MagicMock(spec=Message)
        update.effective_message.voice = MagicMock(spec=Voice)
        update.effective_message.voice.file_id = "test_file_id"
        update.effective_message.voice.file_size = 1024  # Small file
        update.effective_message.chat = MagicMock(spec=Chat)
        update.effective_message.chat.id = -1009999999999  # Unauthorized group
        update.effective_message.reply_text = AsyncMock()
        return update

    @pytest.fixture
    def mock_context(self) -> MagicMock:
        """Create mock context object."""
        context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
        context.bot = AsyncMock()
        return context

    @pytest.mark.asyncio
    async def test_process_allowed_group(
        self,
        handler_with_filter: VoiceHandler,
        mock_allowed_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that messages from allowed group are processed."""
        # Arrange
        mock_file = MagicMock(spec=File)
        mock_file.file_path = "voice/test.oga"
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        handler_with_filter.asr_service.process_audio = AsyncMock(
            return_value=AudioProcessResult(success=True, text="Transcribed text", error="")
        )

        # Act
        await handler_with_filter.handle(mock_allowed_group_update, mock_context)

        # Assert
        mock_context.bot.get_file.assert_called_once()
        handler_with_filter.asr_service.process_audio.assert_called_once()
        mock_allowed_group_update.effective_message.reply_text.assert_called_once_with(
            "Transcribed text"
        )

    @pytest.mark.asyncio
    async def test_ignore_unauthorized_group(
        self,
        handler_with_filter: VoiceHandler,
        mock_unauthorized_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that messages from unauthorized groups are ignored."""
        # Act
        await handler_with_filter.handle(mock_unauthorized_group_update, mock_context)

        # Assert
        mock_context.bot.get_file.assert_not_called()
        handler_with_filter.asr_service.process_audio.assert_not_called()
        mock_unauthorized_group_update.effective_message.reply_text.assert_not_called()

    @pytest.mark.asyncio
    async def test_handler_without_filter_processes_all(
        self,
        mock_asr: AsyncMock,
        mock_unauthorized_group_update: MagicMock,
        mock_context: MagicMock,
    ) -> None:
        """Test that handler without filter processes all messages."""
        # Arrange - handler without filter
        handler = VoiceHandler(asr_service=mock_asr)

        mock_file = MagicMock(spec=File)
        mock_file.file_path = "voice/test.oga"
        mock_file.download_as_bytearray = AsyncMock(return_value=bytearray(b"audio data"))
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        handler.asr_service.process_audio = AsyncMock(
            return_value=AudioProcessResult(success=True, text="Hello world", error="")
        )

        # Act
        await handler.handle(mock_unauthorized_group_update, mock_context)

        # Assert - should process normally
        mock_context.bot.get_file.assert_called_once()
        handler.asr_service.process_audio.assert_called_once()
