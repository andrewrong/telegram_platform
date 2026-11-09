"""Media group cache manager for handling multi-image messages.

This module provides functionality to cache and merge multiple images
sent as a media group (album) in Telegram, processing them as a single
unit after all images are received or a timeout occurs.
"""

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Callable

from telegram import Message

logger = logging.getLogger(__name__)


@dataclass
class MediaGroupCache:
    """Cache entry for a media group being processed.

    Attributes:
        media_group_id: Unique identifier for the media group.
        chat_id: ID of the chat where the media group was sent.
        message_thread_id: Topic ID for forum channels (optional).
        images: List of downloaded image data.
        processing_message: Reference to the "processing" message sent to user.
        timer_task: Asyncio task for the timeout timer.
        is_processing: Flag to prevent duplicate processing.
        context: Telegram context object for sending messages.
    """

    media_group_id: str
    chat_id: int
    message_thread_id: int | None
    images: list[bytes] = field(default_factory=list)
    processing_message: Message | None = None
    timer_task: asyncio.Task | None = None
    is_processing: bool = False
    context: any = None


class MediaGroupManager:
    """Manager for handling media group (album) messages.

        This class caches images from the same media group and processes them
    together after all images are received or a timeout occurs.

        Attributes:
            timeout_seconds: Time to wait for all images in a group.
            _caches: Dictionary mapping media_group_id to MediaGroupCache.
    """

    def __init__(self, timeout_seconds: float = 3.0) -> None:
        """Initialize the media group manager.

        Args:
            timeout_seconds: Time to wait for all images in a group.
                             Default is 3.0 seconds.
        """
        self.timeout_seconds = timeout_seconds
        self._caches: dict[str, MediaGroupCache] = {}

    async def add_image(
        self,
        media_group_id: str,
        image_data: bytes,
        message: Message,
        process_callback: Callable[[str, list[bytes], int, int | None, any], None],
        context: any = None,
    ) -> bool:
        """Add an image to the media group cache.

        Args:
            media_group_id: Unique identifier for the media group.
            image_data: Raw image data bytes.
            message: Telegram message object (for chat info).
            process_callback: Callback function to call when processing.
                              Signature: (media_group_id, images, chat_id,
                              message_thread_id, context)
            context: Telegram context object for sending messages.

        Returns:
            True if this is the first image in the group, False otherwise.
        """
        is_first = media_group_id not in self._caches

        if is_first:
            # Create new cache entry
            logger.info(
                f"Creating new media group cache: {media_group_id}, "
                f"chat_id={message.chat.id}, timeout={self.timeout_seconds}s"
            )
            cache = MediaGroupCache(
                media_group_id=media_group_id,
                chat_id=message.chat.id,
                message_thread_id=getattr(message, "message_thread_id", None),
                context=context,
            )
            self._caches[media_group_id] = cache

            # Start timeout timer
            cache.timer_task = asyncio.create_task(
                self._timer_callback(media_group_id, process_callback)
            )
        else:
            cache = self._caches[media_group_id]
            logger.debug(f"Adding image to existing media group: {media_group_id}")

        # Add image to cache
        cache.images.append(image_data)

        return is_first

    async def _timer_callback(
        self,
        media_group_id: str,
        process_callback: Callable[[str, list[bytes], int, int | None, any], None],
    ) -> None:
        """Timer callback that processes the group after timeout.

        Args:
            media_group_id: The media group ID to process.
            process_callback: Callback function for processing.
        """
        try:
            await asyncio.sleep(self.timeout_seconds)
            await self._process_group(media_group_id, process_callback)
        except asyncio.CancelledError:
            logger.debug(f"Timer cancelled for media group: {media_group_id}")
        except Exception as e:
            logger.exception(f"Error in timer callback for {media_group_id}: {e}")

    async def _process_group(
        self,
        media_group_id: str,
        process_callback: Callable[[str, list[bytes], int, int | None, any], None],
    ) -> None:
        """Process a complete media group.

        Args:
            media_group_id: The media group ID to process.
            process_callback: Callback function for processing.
        """
        cache = self._caches.get(media_group_id)
        if not cache:
            logger.warning(f"Cache not found for media group: {media_group_id}")
            return

        if cache.is_processing:
            logger.debug(f"Media group {media_group_id} is already being processed")
            return

        cache.is_processing = True

        try:
            logger.info(f"Processing media group {media_group_id} with {len(cache.images)} images")
            await process_callback(
                media_group_id,
                cache.images,
                cache.chat_id,
                cache.message_thread_id,
                cache.context,
            )
        except Exception as e:
            logger.exception(f"Error processing media group {media_group_id}: {e}")
        finally:
            self._cleanup(media_group_id)

    def _cleanup(self, media_group_id: str) -> None:
        """Clean up the cache entry for a media group.

        Args:
            media_group_id: The media group ID to clean up.
        """
        cache = self._caches.pop(media_group_id, None)
        if cache and cache.timer_task and not cache.timer_task.done():
            cache.timer_task.cancel()
            logger.debug(f"Cancelled timer for media group: {media_group_id}")

    def close(self) -> None:
        """Close the manager and cancel all pending timers."""
        logger.info(f"Closing MediaGroupManager, cancelling {len(self._caches)} timers")
        for media_group_id in list(self._caches.keys()):
            self._cleanup(media_group_id)
