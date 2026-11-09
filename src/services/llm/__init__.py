"""LLM (Large Language Model) service using OpenRouter and Gemini SDKs.

This module provides a unified interface for interacting with LLM providers
without using LiteLLM, avoiding thread explosion issues in Docker containers.
"""

import logging
from typing import Any

from google import genai
from google.genai import types as genai_types
from openrouter import OpenRouter

from src.exceptions import LLMServiceError

logger = logging.getLogger(__name__)


class LLMService:
    """Service for interacting with Large Language Models.

    This service provides a unified interface for multiple LLM providers
    (OpenRouter, Gemini) using their official SDKs directly.

    Attributes:
        model: The model identifier (e.g., "gemini-3.6-flash" or an OpenRouter model).
        api_key: API key for authentication (optional, can use env vars).
        temperature: Sampling temperature (0.0 to 1.0).
        max_tokens: Maximum tokens in the response.
    """

    DEFAULT_MODEL = "gemini-3.6-flash"
    DEFAULT_TEMPERATURE = 0.7
    DEFAULT_MAX_TOKENS = 4096

    def __init__(
        self,
        model: str = DEFAULT_MODEL,
        api_key: str | None = None,
        temperature: float = DEFAULT_TEMPERATURE,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        """Initialize LLM service.

        Args:
            model: Model identifier. Defaults to "gemini-3.6-flash".
            api_key: API key. If None, uses environment variables.
            temperature: Sampling temperature. Defaults to 0.7.
            max_tokens: Maximum response tokens. Defaults to 4096.
        """
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Detect provider based on model name
        # OpenRouter format: "openrouter/provider/model" or "gemini/model" (Gemini via OpenRouter)
        # Gemini native format: "gemini-model-name"
        self._is_gemini = model.startswith("gemini-") and "/" not in model
        self._is_openrouter = not self._is_gemini

        # Initialize clients lazily
        self._openrouter_client: OpenRouter | None = None
        self._gemini_client: genai.Client | None = None

        logger.info(
            f"LLMService initialized: model={model}, "
            f"provider={'openrouter' if self._is_openrouter else 'gemini'}, "
            f"temperature={temperature}, max_tokens={max_tokens}"
        )

    def _get_openrouter_model(self) -> str:
        """Extract the actual model name for OpenRouter.

        Returns:
            Model name with 'openrouter/' prefix removed.
        """
        if self.model.startswith("openrouter/"):
            return self.model[11:]  # Remove 'openrouter/' prefix
        return self.model

    def _convert_messages_to_gemini_format(
        self, messages: list[dict[str, str]]
    ) -> tuple[str | None, list[str]]:
        """Convert OpenAI-style messages to Gemini format.

        Gemini uses a different message format with system instruction and contents.

        Args:
            messages: List of message dicts with "role" and "content" keys.

        Returns:
            Tuple of (system_instruction, list_of_contents).
        """
        system_instruction: str | None = None
        contents: list[str] = []

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", "")

            if role == "system":
                system_instruction = content
            elif role == "user":
                contents.append(content)
            elif role == "assistant":
                # For assistant messages, we include them as context
                contents.append(f"Assistant: {content}")
            else:
                contents.append(content)

        return system_instruction, contents

    async def _chat_openrouter(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat request using OpenRouter SDK.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Generated text response.

        Raises:
            LLMServiceError: If the request fails or returns empty response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens
        model_name = self._get_openrouter_model()

        logger.debug(f"Sending OpenRouter request: model={model_name}, messages={len(messages)}")

        try:
            # Use synchronous context manager (OpenRouter SDK uses sync context)
            with OpenRouter(api_key=self.api_key or "") as client:
                response = await client.chat.send_async(
                    messages=messages,  # type: ignore[arg-type]
                    model=model_name,
                    temperature=temp,
                    max_tokens=tokens,
                )

                content = response.choices[0].message.content

                if not content or not content.strip():
                    logger.error("OpenRouter returned empty response")
                    raise LLMServiceError("LLM returned empty response")

                logger.info(f"OpenRouter completion successful, response length: {len(content)}")
                return content

        except LLMServiceError:
            raise
        except Exception as e:
            logger.error(f"OpenRouter request failed: {e}")
            raise LLMServiceError(f"LLM request failed: {e}") from e

    async def _chat_gemini(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Send chat request using Gemini SDK.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.

        Returns:
            Generated text response.

        Raises:
            LLMServiceError: If the request fails or returns empty response.
        """
        temp = temperature if temperature is not None else self.temperature
        tokens = max_tokens if max_tokens is not None else self.max_tokens

        logger.debug(f"Sending Gemini request: model={self.model}, messages={len(messages)}")

        try:
            # Initialize async client if needed
            if self._gemini_client is None:
                self._gemini_client = genai.Client(api_key=self.api_key)

            system_instruction, contents = self._convert_messages_to_gemini_format(messages)

            # Combine all user contents into a single string
            combined_content = "\n\n".join(contents)

            # Build config
            config_options: dict[str, Any] = {"max_output_tokens": tokens}
            if not self.model.startswith("gemini-3."):
                config_options["temperature"] = temp
            config = genai_types.GenerateContentConfig(**config_options)
            if system_instruction:
                config.system_instruction = system_instruction

            # Use async client via .aio accessor
            response = await self._gemini_client.aio.models.generate_content(
                model=self.model,
                contents=combined_content,
                config=config,
            )

            content = response.text

            if not content or not content.strip():
                logger.error("Gemini returned empty response")
                raise LLMServiceError("LLM returned empty response")

            logger.info(f"Gemini completion successful, response length: {len(content)}")
            return content

        except LLMServiceError:
            raise
        except Exception as e:
            logger.error(f"Gemini request failed: {e}")
            raise LLMServiceError(f"LLM request failed: {e}") from e

    async def chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Send chat completion request to LLM.

        Automatically routes to the appropriate provider based on model name.

        Args:
            messages: List of message dicts with "role" and "content" keys.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            **kwargs: Additional parameters (ignored, for compatibility).

        Returns:
            Generated text response.

        Raises:
            LLMServiceError: If the request fails or returns empty response.
        """
        if self._is_gemini:
            return await self._chat_gemini(messages, temperature, max_tokens)
        else:
            return await self._chat_openrouter(messages, temperature, max_tokens)

    def chat_sync(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        **kwargs: Any,
    ) -> str:
        """Synchronous chat completion (not implemented).

        This method exists for interface consistency but raises NotImplementedError.
        Use chat() for async operations.

        Args:
            messages: List of message dicts.
            temperature: Override default temperature.
            max_tokens: Override default max_tokens.
            **kwargs: Additional parameters.

        Raises:
            NotImplementedError: Always raised. Use chat() instead.
        """
        raise NotImplementedError("Use chat() for async operations")

    async def close(self) -> None:
        """Close the LLM service and release resources.

        Closes any open client connections and releases resources.
        """
        if self._gemini_client is not None:
            # Gemini client doesn't require explicit cleanup
            self._gemini_client = None

        # OpenRouter client is managed via context manager, no cleanup needed
        self._openrouter_client = None

        logger.info("LLMService closed")
