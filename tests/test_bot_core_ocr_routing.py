"""Tests for Telegram OCR route selection."""

from unittest.mock import MagicMock

from telegram import Message

from src.bot.core import _IngredientPhotoFilter, _OCRDestinationFilter


def make_message(chat_id: int, topic_id: int | None, *, has_photo: bool = True) -> MagicMock:
    """Create a minimal Telegram message for route filter tests."""
    message = MagicMock(spec=Message)
    message.chat_id = chat_id
    message.message_thread_id = topic_id
    message.photo = [MagicMock()] if has_photo else None
    return message


class TestIngredientPhotoFilter:
    """Test the boundary between ingredient analysis and generic OCR."""

    def test_matches_configured_channel_and_topic(self) -> None:
        """Route a matching photo to ingredient analysis."""
        message_filter = _IngredientPhotoFilter(-1001, 42)

        assert message_filter.filter(make_message(-1001, 42)) is True

    def test_rejects_other_topic_for_generic_ocr(self) -> None:
        """Leave photos from another topic for the generic OCR handler."""
        message_filter = _IngredientPhotoFilter(-1001, 42)

        assert message_filter.filter(make_message(-1001, 99)) is False

    def test_rejects_non_photo(self) -> None:
        """Leave PDF and image documents for the generic OCR handler."""
        message_filter = _IngredientPhotoFilter(-1001, None)

        assert message_filter.filter(make_message(-1001, None, has_photo=False)) is False


class TestOCRDestinationFilter:
    """Test dedicated OCR channel and topic routing."""

    def test_matches_configured_channel_and_topic(self) -> None:
        """Accept messages sent to the OCR frontend topic."""
        message_filter = _OCRDestinationFilter(-1001, 88)

        assert message_filter.filter(make_message(-1001, 88)) is True

    def test_rejects_other_topic(self) -> None:
        """Ignore media sent outside the OCR frontend topic."""
        message_filter = _OCRDestinationFilter(-1001, 88)

        assert message_filter.filter(make_message(-1001, 99)) is False

    def test_channel_only_accepts_every_topic(self) -> None:
        """Allow the whole OCR channel when no topic is configured."""
        message_filter = _OCRDestinationFilter(-1001, None)

        assert message_filter.filter(make_message(-1001, 99)) is True
