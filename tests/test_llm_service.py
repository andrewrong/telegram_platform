"""Unit tests for LLM service."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.exceptions import LLMServiceError
from src.services.llm import LLMService


class TestLLMService:
    """Test cases for LLMService."""

    def test_init_default_values(self) -> None:
        """Test LLMService initialization with default values."""
        service = LLMService()
        assert service.model == "gemini-3.6-flash"
        assert service.api_key is None
        assert service.temperature == 0.7
        assert service.max_tokens == 4096

    def test_init_custom_values(self) -> None:
        """Test LLMService initialization with custom values."""
        service = LLMService(
            model="gemini-2.0-flash",
            api_key="test-key",
            temperature=0.5,
            max_tokens=2048,
        )
        assert service.model == "gemini-2.0-flash"
        assert service.api_key == "test-key"
        assert service.temperature == 0.5
        assert service.max_tokens == 2048

    def test_init_openrouter_model(self) -> None:
        """Test LLMService detects OpenRouter model correctly."""
        service = LLMService(model="openrouter/anthropic/claude-3.5-sonnet")
        assert service._is_openrouter is True
        assert service._is_gemini is False

    def test_init_gemini_model(self) -> None:
        """Test LLMService detects Gemini model correctly."""
        service = LLMService(model="gemini-2.0-flash")
        assert service._is_openrouter is False
        assert service._is_gemini is True

    def test_init_defaults_to_openrouter(self) -> None:
        """Test LLMService defaults to OpenRouter for unknown models."""
        service = LLMService(model="some-random-model")
        assert service._is_openrouter is True
        assert service._is_gemini is False

    @pytest.mark.asyncio
    async def test_chat_openrouter_success(self) -> None:
        """Test successful chat completion with OpenRouter."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        # Mock the OpenRouter SDK response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "Hello! How can I help you?"

        # Mock the internal _chat_openrouter method
        with patch.object(service, "_chat_openrouter", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Hello! How can I help you?"
            result = await service.chat(messages)
            assert result == "Hello! How can I help you?"
            mock_chat.assert_called_once_with(messages, None, None)

    @pytest.mark.asyncio
    async def test_chat_gemini_success(self) -> None:
        """Test successful chat completion with Gemini."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        # Mock the internal _chat_gemini method
        with patch.object(service, "_chat_gemini", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Hello! How can I help you?"
            result = await service.chat(messages)
            assert result == "Hello! How can I help you?"
            mock_chat.assert_called_once_with(messages, None, None)

    @pytest.mark.asyncio
    async def test_chat_with_system_message(self) -> None:
        """Test chat with system message."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        with patch.object(service, "_chat_openrouter", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Hi there!"
            result = await service.chat(messages)
            assert result == "Hi there!"
            mock_chat.assert_called_once_with(messages, None, None)

    @pytest.mark.asyncio
    async def test_chat_empty_response_openrouter(self) -> None:
        """Test chat with empty response from OpenRouter."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        # Test that empty response raises LLMServiceError
        with patch("src.services.llm.OpenRouter") as mock_openrouter_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.choices = [MagicMock()]
            mock_response.choices[0].message.content = ""
            mock_client.chat.send_async = AsyncMock(return_value=mock_response)
            mock_openrouter_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_openrouter_class.return_value.__exit__ = MagicMock(return_value=None)

            with pytest.raises(LLMServiceError, match="LLM returned empty response"):
                await service._chat_openrouter(messages)

    @pytest.mark.asyncio
    async def test_chat_empty_response_gemini(self) -> None:
        """Test chat with empty response from Gemini."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_response = MagicMock()
            mock_response.text = ""
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMServiceError, match="LLM returned empty response"):
                await service._chat_gemini(messages)

    @pytest.mark.asyncio
    async def test_chat_api_error_openrouter(self) -> None:
        """Test chat with API error from OpenRouter."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("src.services.llm.OpenRouter") as mock_openrouter_class:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(
                side_effect=Exception("API Error: Rate limit exceeded")
            )
            mock_openrouter_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_openrouter_class.return_value.__exit__ = MagicMock(return_value=None)

            with pytest.raises(LLMServiceError, match="LLM request failed"):
                await service._chat_openrouter(messages)

    @pytest.mark.asyncio
    async def test_chat_api_error_gemini(self) -> None:
        """Test chat with API error from Gemini."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(
                side_effect=Exception("API Error: Invalid API key")
            )
            mock_client_class.return_value = mock_client

            with pytest.raises(LLMServiceError, match="LLM request failed"):
                await service._chat_gemini(messages)

    @pytest.mark.asyncio
    async def test_chat_with_temperature_override(self) -> None:
        """Test chat with temperature override."""
        service = LLMService(
            model="openrouter/google/gemini-2.0-flash",
            api_key="test-key",
            temperature=0.7,
        )
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(service, "_chat_openrouter", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Response"
            await service.chat(messages, temperature=0.2)

            # Verify temperature was overridden
            mock_chat.assert_called_once_with(messages, 0.2, None)

    @pytest.mark.asyncio
    async def test_chat_with_max_tokens_override(self) -> None:
        """Test chat with max_tokens override."""
        service = LLMService(
            model="gemini-2.0-flash",
            api_key="test-key",
            max_tokens=4096,
        )
        messages = [{"role": "user", "content": "Hello"}]

        with patch.object(service, "_chat_gemini", new_callable=AsyncMock) as mock_chat:
            mock_chat.return_value = "Response"
            await service.chat(messages, max_tokens=1024)

            # Verify max_tokens was overridden
            mock_chat.assert_called_once_with(messages, None, 1024)

    @pytest.mark.asyncio
    async def test_close_with_openrouter(self) -> None:
        """Test close method with OpenRouter client."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")

        # Close should not raise
        await service.close()
        assert service._openrouter_client is None
        assert service._gemini_client is None

    @pytest.mark.asyncio
    async def test_close_with_gemini(self) -> None:
        """Test close method with Gemini client."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")

        # Initialize mock client
        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client_class.return_value = mock_client

            mock_response = MagicMock()
            mock_response.text = "test"
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)

            # Trigger client initialization
            await service._chat_gemini([{"role": "user", "content": "test"}])
            assert service._gemini_client is not None

        # Close should clean up
        await service.close()
        assert service._gemini_client is None

    @pytest.mark.asyncio
    async def test_openrouter_internal_success(self) -> None:
        """Test internal OpenRouter chat method with mocked SDK."""
        service = LLMService(model="openrouter/google/gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "OpenRouter response"

        with patch("src.services.llm.OpenRouter") as mock_openrouter_class:
            mock_client = MagicMock()
            mock_client.chat.send_async = AsyncMock(return_value=mock_response)
            mock_openrouter_class.return_value.__enter__ = MagicMock(return_value=mock_client)
            mock_openrouter_class.return_value.__exit__ = MagicMock(return_value=None)

            result = await service._chat_openrouter(messages)

            assert result == "OpenRouter response"
            mock_client.chat.send_async.assert_called_once()
            call_kwargs = mock_client.chat.send_async.call_args[1]
            assert call_kwargs["model"] == "google/gemini-2.0-flash"
            assert call_kwargs["temperature"] == 0.7
            assert call_kwargs["max_tokens"] == 4096

    @pytest.mark.asyncio
    async def test_gemini_internal_success(self) -> None:
        """Test internal Gemini chat method with mocked SDK."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")
        messages = [{"role": "user", "content": "Hello"}]

        mock_response = MagicMock()
        mock_response.text = "Gemini response"

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await service._chat_gemini(messages)

            assert result == "Gemini response"
            mock_client.aio.models.generate_content.assert_called_once()

    @pytest.mark.asyncio
    async def test_gemini_3_omits_deprecated_temperature(self) -> None:
        """Do not send deprecated sampling parameters to Gemini 3.x."""
        service = LLMService(model="gemini-3.6-flash", api_key="test-key")
        mock_response = MagicMock(text="Gemini 3 response")

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            await service._chat_gemini([{"role": "user", "content": "Hello"}])

        config = mock_client.aio.models.generate_content.call_args.kwargs["config"]
        assert config.temperature is None
        assert config.max_output_tokens == 4096

    @pytest.mark.asyncio
    async def test_gemini_with_system_message(self) -> None:
        """Test Gemini chat with system message."""
        service = LLMService(model="gemini-2.0-flash", api_key="test-key")
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "Hello"},
        ]

        mock_response = MagicMock()
        mock_response.text = "Hi there!"

        with patch("google.genai.Client") as mock_client_class:
            mock_client = MagicMock()
            mock_client.aio.models.generate_content = AsyncMock(return_value=mock_response)
            mock_client_class.return_value = mock_client

            result = await service._chat_gemini(messages)

            assert result == "Hi there!"
            call_kwargs = mock_client.aio.models.generate_content.call_args[1]
            config = call_kwargs.get("config")
            assert config is not None
            assert config.system_instruction == "You are a helpful assistant."

    def test_chat_sync_not_implemented(self) -> None:
        """Test that sync chat raises NotImplementedError."""
        service = LLMService(model="gemini-2.0-flash")
        messages = [{"role": "user", "content": "Hello"}]

        with pytest.raises(NotImplementedError, match=r"Use chat\(\) for async operations"):
            service.chat_sync(messages)
