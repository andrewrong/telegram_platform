"""Base handler class for Telegram bot handlers."""

from abc import ABC, abstractmethod
from typing import Any

from telegram import Update
from telegram.ext import ContextTypes


class BaseHandler(ABC):
    """Base class for all Telegram bot handlers.

    All handlers should inherit from this class and implement
    the handle method.
    """

    @abstractmethod
    async def handle(self, update: Update, context: ContextTypes.DEFAULT_TYPE) -> Any:
        """Handle an incoming update.

        Args:
            update: The Telegram update object.
            context: The Telegram context object.

        Returns:
            Any result from handling the update.
        """
        pass
