"""Unit tests for media group cache manager."""

import asyncio
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.media_cache import MediaGroupCache, MediaGroupManager


class TestMediaGroupCache:
    """Test cases for MediaGroupCache dataclass."""

    def test_cache_initialization(self):
        """Test MediaGroupCache initialization."""
        cache = MediaGroupCache(
            media_group_id="test123",
            chat_id=12345,
            message_thread_id=678,
        )
        assert cache.media_group_id == "test123"
        assert cache.chat_id == 12345
        assert cache.message_thread_id == 678
        assert cache.images == []
        assert cache.processing_message is None
        assert cache.timer_task is None
        assert cache.is_processing is False


class TestMediaGroupManager:
    """Test cases for MediaGroupManager."""

    def test_init_default_timeout(self):
        """Test initialization with default timeout."""
        manager = MediaGroupManager()
        assert manager.timeout_seconds == 3.0
        assert manager._caches == {}

    def test_init_custom_timeout(self):
        """Test initialization with custom timeout."""
        manager = MediaGroupManager(timeout_seconds=5.0)
        assert manager.timeout_seconds == 5.0

    @pytest.mark.asyncio
    async def test_add_first_image_creates_cache(self):
        """Test adding first image creates new cache entry."""
        manager = MediaGroupManager(timeout_seconds=0.1)
        process_callback = AsyncMock()

        image_data = b"fake_image_data"
        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        result = await manager.add_image(
            media_group_id="group1",
            image_data=image_data,
            message=message,
            process_callback=process_callback,
        )

        # Should return True indicating first image
        assert result is True
        assert "group1" in manager._caches
        assert len(manager._caches["group1"].images) == 1
        assert manager._caches["group1"].images[0] == image_data

    @pytest.mark.asyncio
    async def test_add_subsequent_images(self):
        """Test adding subsequent images to same group."""
        manager = MediaGroupManager(timeout_seconds=0.1)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        # First image
        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message,
            process_callback=process_callback,
        )

        # Second image
        result = await manager.add_image(
            media_group_id="group1",
            image_data=b"image2",
            message=message,
            process_callback=process_callback,
        )

        # Should return False indicating not first image
        assert result is False
        assert len(manager._caches["group1"].images) == 2

    @pytest.mark.asyncio
    async def test_timeout_triggers_processing(self):
        """Test that timeout triggers processing."""
        manager = MediaGroupManager(timeout_seconds=0.05)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message,
            process_callback=process_callback,
        )

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Process callback should have been called
        process_callback.assert_called_once()
        call_args = process_callback.call_args[0]
        assert call_args[0] == "group1"  # media_group_id
        assert call_args[1] == [b"image1"]  # images
        assert call_args[2] == 12345  # chat_id

    @pytest.mark.asyncio
    async def test_multiple_images_merge_on_timeout(self):
        """Test multiple images are merged on timeout."""
        manager = MediaGroupManager(timeout_seconds=0.05)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = 678

        # Add 3 images
        for i in range(3):
            await manager.add_image(
                media_group_id="group1",
                image_data=f"image{i}".encode(),
                message=message,
                process_callback=process_callback,
            )

        # Wait for timeout
        await asyncio.sleep(0.1)

        # All images should be passed to callback
        process_callback.assert_called_once()
        call_args = process_callback.call_args[0]
        images = call_args[1]
        assert len(images) == 3
        assert images == [b"image0", b"image1", b"image2"]

    @pytest.mark.asyncio
    async def test_cache_cleanup_after_processing(self):
        """Test cache is cleaned up after processing."""
        manager = MediaGroupManager(timeout_seconds=0.05)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message,
            process_callback=process_callback,
        )

        # Wait for processing
        await asyncio.sleep(0.1)

        # Cache should be removed
        assert "group1" not in manager._caches

    @pytest.mark.asyncio
    async def test_concurrent_groups_isolated(self):
        """Test different media groups are isolated."""
        manager = MediaGroupManager(timeout_seconds=0.1)
        process_callback = AsyncMock()

        message1 = MagicMock()
        message1.chat.id = 111
        message1.message_thread_id = None

        message2 = MagicMock()
        message2.chat.id = 222
        message2.message_thread_id = None

        # Add to different groups
        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message1,
            process_callback=process_callback,
        )
        await manager.add_image(
            media_group_id="group2",
            image_data=b"image2",
            message=message2,
            process_callback=process_callback,
        )

        assert len(manager._caches) == 2
        assert manager._caches["group1"].chat_id == 111
        assert manager._caches["group2"].chat_id == 222

    @pytest.mark.asyncio
    async def test_processing_flag_prevents_duplicate(self):
        """Test processing flag prevents duplicate processing."""
        manager = MediaGroupManager(timeout_seconds=0.05)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message,
            process_callback=process_callback,
        )

        # Manually set processing flag (simulating race condition)
        manager._caches["group1"].is_processing = True

        # Wait for timeout
        await asyncio.sleep(0.1)

        # Callback should not be called since is_processing is True
        process_callback.assert_not_called()

    @pytest.mark.asyncio
    async def test_cancel_timer_on_cleanup(self):
        """Test timer is cancelled on cleanup."""
        manager = MediaGroupManager(timeout_seconds=10.0)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        await manager.add_image(
            media_group_id="group1",
            image_data=b"image1",
            message=message,
            process_callback=process_callback,
        )

        timer_task = manager._caches["group1"].timer_task
        assert timer_task is not None
        assert not timer_task.done()

        # Cleanup
        manager._cleanup("group1")

        # Give a moment for cancellation to take effect
        await asyncio.sleep(0)

        # Timer should be cancelled or done
        assert timer_task.cancelled() or timer_task.done()

    @pytest.mark.asyncio
    async def test_close_cancels_all_timers(self):
        """Test close cancels all pending timers."""
        manager = MediaGroupManager(timeout_seconds=10.0)
        process_callback = AsyncMock()

        message = MagicMock()
        message.chat.id = 12345
        message.message_thread_id = None

        # Add multiple groups
        for i in range(3):
            await manager.add_image(
                media_group_id=f"group{i}",
                image_data=b"image",
                message=message,
                process_callback=process_callback,
            )

        assert len(manager._caches) == 3

        # Close manager
        manager.close()

        # All caches should be cleared
        assert len(manager._caches) == 0
