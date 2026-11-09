"""Unit tests for ingredient analyzer."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.services.ingredient.analyzer import IngredientAnalyzer
from src.services.llm import LLMService
from src.services.ocr import OCRService


@pytest.fixture
def mock_llm_service():
    """Create a mock LLM service."""
    service = MagicMock(spec=LLMService)
    service.chat = AsyncMock()
    return service


@pytest.fixture
def mock_ocr_service():
    """Create a mock OCR service."""
    service = MagicMock(spec=OCRService)
    service.process_image = AsyncMock()
    return service


class TestIngredientAnalyzer:
    """Test cases for IngredientAnalyzer."""

    def test_init(self, mock_llm_service):
        """Test analyzer initialization."""
        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        assert analyzer.llm_service == mock_llm_service

    @pytest.mark.asyncio
    async def test_analyze_from_text_success(self, mock_llm_service):
        """Test analyzing ingredient text successfully."""
        mock_llm_service.chat.return_value = "## 配料表分析\n\n这是健康的食品。"

        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        result = await analyzer.analyze_from_text("水, 糖, 盐")

        assert "## 配料表分析" in result
        mock_llm_service.chat.assert_called_once()

        # Verify the messages passed to LLM
        call_args = mock_llm_service.chat.call_args
        if call_args and len(call_args) >= 1 and call_args[0]:
            messages = call_args[0][0]
        elif call_args and "messages" in call_args.kwargs:
            messages = call_args.kwargs["messages"]
        else:
            messages = []

        assert len(messages) >= 2  # System + user messages
        assert any(msg["role"] == "system" for msg in messages)
        assert any(msg["role"] == "user" for msg in messages)
        assert "水, 糖, 盐" in str(messages)

    @pytest.mark.asyncio
    async def test_analyze_from_text_empty_input(self, mock_llm_service):
        """Test analyzing empty ingredient text."""
        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        result = await analyzer.analyze_from_text("")

        assert "无法分析" in result or "empty" in result.lower()
        mock_llm_service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_from_text_whitespace_only(self, mock_llm_service):
        """Test analyzing whitespace-only ingredient text."""
        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        result = await analyzer.analyze_from_text("   \n\t  ")

        assert "无法分析" in result or "empty" in result.lower()
        mock_llm_service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_from_image_success(self, mock_llm_service, mock_ocr_service):
        """Test analyzing ingredient image successfully."""
        mock_ocr_service.process_image.return_value = MagicMock(
            success=True, text="水, 糖, 盐, 面粉", error=""
        )
        mock_llm_service.chat.return_value = "## 配料表分析\n\n这是健康的食品。"

        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        image_data = b"fake_image_data"

        result = await analyzer.analyze_from_image(image_data, mock_ocr_service)

        assert "## 配料表分析" in result
        mock_ocr_service.process_image.assert_called_once_with(image_data, engine="glm")
        mock_llm_service.chat.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_from_image_ocr_failure(self, mock_llm_service, mock_ocr_service):
        """Test analyzing image when OCR fails."""
        mock_ocr_service.process_image.return_value = MagicMock(
            success=False, text="", error="OCR failed: image too blurry"
        )

        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        image_data = b"fake_image_data"

        result = await analyzer.analyze_from_image(image_data, mock_ocr_service)

        assert "❌" in result or "失败" in result
        assert "OCR failed" in result
        mock_llm_service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_from_image_empty_ocr(self, mock_llm_service, mock_ocr_service):
        """Test analyzing image when OCR returns empty text."""
        mock_ocr_service.process_image.return_value = MagicMock(success=True, text="", error="")

        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        image_data = b"fake_image_data"

        result = await analyzer.analyze_from_image(image_data, mock_ocr_service)

        assert "无法识别" in result or "无法从图片中识别" in result
        mock_llm_service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_from_image_rejects_numeric_only_ocr(
        self, mock_llm_service, mock_ocr_service
    ):
        """Do not send a degenerate numeric-only OCR result to the ingredient LLM."""
        mock_ocr_service.process_image.return_value = MagicMock(
            success=True,
            text="10000000000000000000000000000000",
            error="",
        )
        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)

        result = await analyzer.analyze_from_image(b"image", mock_ocr_service)

        assert "未识别到有效配料文字" in result
        mock_llm_service.chat.assert_not_called()

    @pytest.mark.asyncio
    async def test_analyze_from_image_llm_failure(self, mock_llm_service, mock_ocr_service):
        """Test analyzing image when LLM fails."""
        mock_ocr_service.process_image.return_value = MagicMock(
            success=True, text="水, 糖, 盐", error=""
        )
        from src.exceptions import LLMServiceError

        mock_llm_service.chat.side_effect = LLMServiceError("API rate limit exceeded")

        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        image_data = b"fake_image_data"

        result = await analyzer.analyze_from_image(image_data, mock_ocr_service)

        assert "❌" in result or "失败" in result
        assert "API rate limit" in result

    def test_get_system_prompt_not_empty(self, mock_llm_service):
        """Test that system prompt is not empty."""
        analyzer = IngredientAnalyzer(llm_service=mock_llm_service)
        prompt = analyzer._get_system_prompt()

        assert len(prompt) > 0
        assert "配料" in prompt or "ingredient" in prompt.lower()
