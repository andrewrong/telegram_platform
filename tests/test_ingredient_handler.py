"""Unit tests for ingredient handler."""

import logging
from unittest.mock import AsyncMock, MagicMock, call, patch

import pytest
from telegram import Chat, Message, PhotoSize, Update, User
from telegram.ext import ContextTypes

from src.bot.handlers.ingredient import IngredientHandler
from src.services.ingredient.analyzer import IngredientAnalyzer
from src.services.ocr import OCRService
from src.utils.media_cache import MediaGroupManager


@pytest.fixture
def mock_analyzer():
    """Create a mock ingredient analyzer."""
    analyzer = MagicMock(spec=IngredientAnalyzer)
    analyzer.ocr_engine = "glm"
    analyzer.analyze_from_image = AsyncMock()
    analyzer.analyze_from_text = AsyncMock()
    analyzer.is_plausible_ingredient_text.side_effect = (
        IngredientAnalyzer.is_plausible_ingredient_text
    )
    analyzer.format_for_telegram = MagicMock(return_value=("分析结果文本", []))
    return analyzer


@pytest.fixture
def mock_ocr_service():
    """Create a mock OCR service."""
    ocr = MagicMock(spec=OCRService)
    ocr.process_image = AsyncMock()
    return ocr


@pytest.fixture
def mock_media_manager():
    """Create a mock media group manager."""
    manager = MagicMock(spec=MediaGroupManager)
    manager.add_image = AsyncMock(return_value=True)
    manager._caches = {}
    return manager


@pytest.fixture
def mock_update():
    """Create a mock Telegram update."""
    update = MagicMock(spec=Update)
    update.effective_user = MagicMock(spec=User)
    update.effective_user.id = 12345
    update.effective_user.username = "testuser"
    update.effective_message = MagicMock(spec=Message)
    update.effective_message.chat = MagicMock(spec=Chat)
    update.effective_message.chat.id = -1001234567890
    update.effective_message.chat.type = "group"
    update.effective_message.chat.title = "Test Group"
    update.effective_message.media_group_id = None
    update.effective_message.message_thread_id = None
    return update


@pytest.fixture
def mock_context():
    """Create a mock Telegram context."""
    context = MagicMock(spec=ContextTypes.DEFAULT_TYPE)
    context.bot = MagicMock()
    return context


class TestIngredientHandler:
    """Test cases for IngredientHandler."""

    def test_init(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test handler initialization."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        assert handler.analyzer == mock_analyzer
        assert handler.ocr_service == mock_ocr_service
        assert handler.media_group_manager == mock_media_manager
        assert not handler._has_channel_filter
        assert not handler._has_topic_filter

    def test_init_with_channel_filter(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test handler initialization with channel filter."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
            channel_id=-1001234567890,
        )
        assert handler._has_channel_filter
        assert handler.channel_id == -1001234567890

    def test_init_with_topic_filter(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test handler initialization with channel and topic filter."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
            channel_id=-1001234567890,
            topic_id=123,
        )
        assert handler._has_channel_filter
        assert handler._has_topic_filter
        assert handler.channel_id == -1001234567890
        assert handler.topic_id == 123

    @pytest.mark.asyncio
    async def test_handle_no_photo(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling message without photo."""
        mock_update.effective_message.photo = None

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once_with("❌ 未检测到图片")
        mock_analyzer.analyze_from_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_photo_too_large(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling oversized photo."""
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 15 * 1024 * 1024  # 15MB
        mock_update.effective_message.photo = [photo]

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        mock_update.effective_message.reply_text.assert_called_once()
        assert "太大" in mock_update.effective_message.reply_text.call_args[0][0]
        mock_analyzer.analyze_from_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_filtered_channel(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling message from filtered out channel."""
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        mock_update.effective_message.photo = [photo]
        mock_update.effective_message.chat.id = -9999999999

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
            channel_id=-1001234567890,
        )
        await handler.handle(mock_update, mock_context)

        # Should not process
        mock_analyzer.analyze_from_image.assert_not_called()

    @pytest.mark.asyncio
    async def test_handle_single_image_success(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test successful single image ingredient analysis."""
        # Setup photo (single image, no media_group_id)
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        photo.file_id = "photo123"
        photo.width = 100
        photo.height = 100
        mock_update.effective_message.photo = [photo]
        mock_update.effective_message.media_group_id = None

        # Setup file download
        mock_file = MagicMock()
        mock_file.file_path = "path/to/photo.jpg"
        mock_file.download_as_bytearray = AsyncMock(return_value=b"fake_image_data")
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        # Setup OCR service response
        mock_ocr_result = MagicMock()
        mock_ocr_result.success = True
        mock_ocr_result.text = "配料：水、糖、盐"
        mock_ocr_result.error = ""
        mock_ocr_service.process_image = AsyncMock(return_value=mock_ocr_result)

        # Setup analyzer response
        mock_analyzer.analyze_from_image.return_value = "## 配料表分析\n\n健康等级：⭐⭐⭐⭐⭐"
        mock_analyzer.format_for_telegram.return_value = ("配料表分析\n健康等级：⭐⭐⭐⭐⭐", [])

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        # Should use analyze_from_image for single image
        mock_analyzer.analyze_from_image.assert_called_once()
        mock_analyzer.format_for_telegram.assert_called_once()

    @pytest.mark.asyncio
    async def test_handle_media_group(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling media group (album) message."""
        # Setup photo with media_group_id
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        photo.file_id = "photo123"
        mock_update.effective_message.photo = [photo]
        mock_update.effective_message.media_group_id = "media_group_123"

        # Setup file download
        mock_file = MagicMock()
        mock_file.download_as_bytearray = AsyncMock(return_value=b"fake_image_data")
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        # Setup media manager to indicate first image
        mock_media_manager.add_image = AsyncMock(return_value=True)

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        # Should add to media group manager
        mock_media_manager.add_image.assert_called_once()
        call_args = mock_media_manager.add_image.call_args[1]
        assert call_args["media_group_id"] == "media_group_123"

    @pytest.mark.asyncio
    async def test_handle_analysis_error(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling when analysis returns error."""
        # Setup photo
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        photo.file_id = "photo123"
        mock_update.effective_message.photo = [photo]
        mock_update.effective_message.media_group_id = None

        # Setup file download
        mock_file = MagicMock()
        mock_file.file_path = "path/to/photo.jpg"
        mock_file.download_as_bytearray = AsyncMock(return_value=b"fake_image_data")
        mock_context.bot.get_file = AsyncMock(return_value=mock_file)

        # Setup OCR service response
        mock_ocr_result = MagicMock()
        mock_ocr_result.success = True
        mock_ocr_result.text = "配料"
        mock_ocr_result.error = ""
        mock_ocr_service.process_image = AsyncMock(return_value=mock_ocr_result)

        # Setup analyzer error response
        mock_analyzer.analyze_from_image.return_value = "❌ 图片识别失败: OCR failed"

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        # Check that error message was sent
        calls = mock_update.effective_message.reply_text.call_args_list
        assert any("图片识别失败" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_handle_download_error(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test handling download error."""
        # Setup photo
        photo = MagicMock(spec=PhotoSize)
        photo.file_size = 1024
        photo.file_id = "photo123"
        mock_update.effective_message.photo = [photo]
        mock_update.effective_message.media_group_id = None

        # Setup download to fail
        mock_context.bot.get_file = AsyncMock(side_effect=Exception("Network error"))

        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )
        await handler.handle(mock_update, mock_context)

        # Check that error message was sent
        calls = mock_update.effective_message.reply_text.call_args_list
        assert any("处理图片失败" in str(call) for call in calls)

    @pytest.mark.asyncio
    async def test_process_multiple_images(
        self, mock_analyzer, mock_ocr_service, mock_media_manager
    ):
        """Test processing multiple images with OCR."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        # Setup OCR responses
        mock_results = [
            MagicMock(success=True, text="配料1", error=""),
            MagicMock(success=True, text="配料2", error=""),
            MagicMock(success=False, text="", error="OCR failed"),
            MagicMock(success=True, text="10000000000000000000000000000000", error=""),
        ]
        mock_ocr_service.process_image = AsyncMock(side_effect=mock_results)

        images = [b"image1", b"image2", b"image3", b"image4"]
        results = await handler._process_multiple_images(images)

        assert len(results) == 4
        assert results[0] == "配料1"
        assert results[1] == "配料2"
        assert results[2] == ""  # Failed OCR returns empty string
        assert results[3] == ""  # Implausible OCR is excluded before merging
        mock_ocr_service.process_image.assert_has_awaits(
            [
                call(b"image1", engine="glm"),
                call(b"image2", engine="glm"),
                call(b"image3", engine="glm"),
                call(b"image4", engine="glm"),
            ]
        )

    def test_merge_ocr_results_single(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test merging single OCR result."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        results = ["配料：水、糖"]
        merged = handler._merge_ocr_results(results)

        # Single result should not have separator
        assert "=== 图片" not in merged
        assert merged == "配料：水、糖"

    def test_merge_ocr_results_multiple(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test merging multiple OCR results with separators."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        results = ["配料A", "配料B"]
        merged = handler._merge_ocr_results(results)

        # Multiple results should have separators
        assert "=== 图片 1 ===" in merged
        assert "=== 图片 2 ===" in merged
        assert "配料A" in merged
        assert "配料B" in merged

    def test_merge_ocr_results_with_empty(
        self, mock_analyzer, mock_ocr_service, mock_media_manager
    ):
        """Test merging OCR results with empty entries."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        results = ["配料A", "", "配料B", "   "]
        merged = handler._merge_ocr_results(results)

        # Empty results should be skipped
        assert "配料A" in merged
        assert "配料B" in merged
        # Original indices are preserved (1 and 3)
        assert "=== 图片 1 ===" in merged
        assert "=== 图片 3 ===" in merged
        # Empty image 2 and 4 should not appear
        assert "=== 图片 2 ===" not in merged
        assert "=== 图片 4 ===" not in merged

    def test_merge_ocr_results_all_empty(self, mock_analyzer, mock_ocr_service, mock_media_manager):
        """Test merging when all OCR results are empty."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        results = ["", "   ", ""]
        merged = handler._merge_ocr_results(results)

        assert merged == ""

    @pytest.mark.asyncio
    async def test_process_media_group(  # noqa: PLR0913
        self, mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context
    ):
        """Test processing a complete media group."""
        handler = IngredientHandler(
            analyzer=mock_analyzer,
            ocr_service=mock_ocr_service,
            media_group_manager=mock_media_manager,
        )

        # Setup cache with processing message
        mock_cache = MagicMock()
        mock_cache.processing_message = mock_update.effective_message
        mock_media_manager._caches = {"group1": mock_cache}

        # Setup OCR and analyzer
        with patch.object(
            handler, "_process_multiple_images", new_callable=AsyncMock
        ) as mock_process:
            mock_process.return_value = ["配料A", "配料B"]
            mock_analyzer.analyze_from_text = AsyncMock(return_value="## 分析结果")
            mock_analyzer.format_for_telegram.return_value = ("分析结果", [])

            await handler._process_media_group(
                media_group_id="group1",
                images=[b"img1", b"img2"],
                chat_id=12345,
                message_thread_id=None,
                context=mock_context,
            )

            # Should process images and analyze merged text
            mock_process.assert_called_once_with([b"img1", b"img2"])
            mock_analyzer.analyze_from_text.assert_called_once()


@pytest.mark.asyncio
async def test_handler_logs_no_identifiers(
    mock_analyzer, mock_ocr_service, mock_media_manager, mock_update, mock_context, caplog
):
    """Constructor and a handled photo log no chat, user, file, channel or topic id."""
    caplog.set_level(logging.DEBUG)
    mock_update.effective_user.username = "secret_user_name"
    mock_update.effective_user.id = 424242
    chat = mock_update.effective_message.chat
    chat.id = -1009876543210
    chat.title = "Secret Chat Title"
    mock_update.effective_message.message_thread_id = 777001
    photo = MagicMock(spec=PhotoSize)
    photo.file_size = 1024
    photo.file_id = "secret_file_id"
    photo.width = 100
    photo.height = 100
    mock_update.effective_message.photo = [photo]
    mock_update.effective_message.reply_text = AsyncMock()
    mock_file = MagicMock()
    mock_file.file_path = "p.jpg"
    mock_file.download_as_bytearray = AsyncMock(return_value=b"img")
    mock_context.bot.get_file = AsyncMock(return_value=mock_file)
    ocr = MagicMock(success=True, text="配料：水", error="")
    mock_ocr_service.process_image = AsyncMock(return_value=ocr)
    mock_analyzer.analyze_from_image.return_value = "ok"
    mock_analyzer.format_for_telegram.return_value = ("ok", [])

    handler = IngredientHandler(
        analyzer=mock_analyzer,
        ocr_service=mock_ocr_service,
        media_group_manager=mock_media_manager,
        channel_id=-1009876543210,
        topic_id=777001,
    )
    await handler.handle(mock_update, mock_context)
    # A message from another chat and topic exercises the ignore paths.
    chat.id = -1005550001111
    await handler.handle(mock_update, mock_context)

    assert caplog.records
    text = "\n".join(record.getMessage() for record in caplog.records)
    for value in (
        "1009876543210",
        "1005550001111",
        "424242",
        "secret_user_name",
        "Secret Chat Title",
        "secret_file_id",
        "777001",
    ):
        assert value not in text
