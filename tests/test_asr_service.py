"""Tests for ASR service."""

import asyncio

import pytest
import respx
from httpx import Response

from src.exceptions import ASRServiceError
from src.services.audio.asr import ASRService


class TestASRService:
    """Test cases for ASRService."""

    @pytest.fixture
    def service(self) -> ASRService:
        """Create ASR service instance for testing."""
        return ASRService(base_url="http://127.0.0.1:28111")

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_uses_typeless_v1_contract(self, service: ASRService) -> None:
        """Send every audio duration through the stable Typeless v1 endpoint."""
        # Arrange
        expected_text = "Hello world, processed."
        route = respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(
                200,
                json={
                    "id": "tr_test",
                    "object": "audio.transcription",
                    "status": "completed",
                    "text": expected_text,
                    "raw_text": "Hello world",
                    "language": "auto",
                    "model": "sensevoice",
                    "duration_seconds": 3.0,
                },
            )
        )

        # Act
        result = await service.transcribe(b"fake audio data", duration=90)

        # Assert - duration no longer changes the public endpoint.
        assert result == expected_text
        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_sends_standard_postprocess_and_optional_api_key(self) -> None:
        """Use rule-based processing and pass a Typeless API key when configured."""
        # Arrange
        captured_request = {}

        def capture_request(request):
            captured_request["headers"] = dict(request.headers)
            captured_request["body"] = request.content.decode()
            return Response(200, json={"text": "Hello world"})

        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            side_effect=capture_request
        )
        service = ASRService(base_url="http://127.0.0.1:28111", api_key="test-key")

        # Act
        result = await service.transcribe(b"data")

        # Assert
        assert result == "Hello world"
        assert captured_request["headers"]["x-api-key"] == "test-key"
        assert 'name="postprocess_mode"' in captured_request["body"]
        assert "standard" in captured_request["body"]

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_includes_postprocess_mode(self, service: ASRService) -> None:
        """Test that postprocess_mode parameter is included in request URL."""
        # Arrange
        captured_query = {}

        def capture_request(request):
            # Extract query string parameters from URL
            captured_query["url"] = str(request.url)
            return Response(200, json={"text": "test", "raw_text": "test"})

        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            side_effect=capture_request
        )

        # Act
        await service.transcribe(b"data")

        # Assert - the v1 endpoint does not encode processing options in the URL.
        assert captured_query["url"] == "http://127.0.0.1:28111/v1/audio/transcriptions"

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_success_without_processed(self, service: ASRService) -> None:
        """Test successful transcription without processed_transcript."""
        # Arrange
        expected_text = "Hello world"
        route = respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(
                200,
                json={
                    "text": expected_text,
                    "raw_text": expected_text,
                },
            )
        )

        # Act
        result = await service.transcribe(b"fake audio data")

        # Assert
        assert result == expected_text
        assert route.called

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_empty_response(self, service: ASRService) -> None:
        """Test transcription when API returns empty response."""
        # Arrange
        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(
                200,
                json={
                    "text": "",
                    "raw_text": "",
                },
            )
        )

        # Act & Assert
        with pytest.raises(ASRServiceError, match="empty transcript"):
            await service.transcribe(b"fake audio data")

    @pytest.mark.asyncio
    @respx.mock
    async def test_transcribe_http_error(self, service: ASRService) -> None:
        """Test transcription when HTTP request fails."""
        # Arrange
        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(500, text="Internal Server Error")
        )

        # Act & Assert
        with pytest.raises(ASRServiceError, match="ASR service request failed"):
            await service.transcribe(b"fake audio data")

    @pytest.mark.asyncio
    @respx.mock
    async def test_process_audio_success(self, service: ASRService) -> None:
        """Test process_audio wrapper returns correct result on success."""
        # Arrange
        expected_text = "Transcribed text"
        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(
                200,
                json={
                    "text": expected_text,
                    "raw_text": "raw text",
                },
            )
        )

        # Act
        result = await service.process_audio(b"audio data")

        # Assert
        assert result.success is True
        assert result.text == expected_text
        assert result.error == ""

    @pytest.mark.asyncio
    @respx.mock
    async def test_process_audio_failure(self, service: ASRService) -> None:
        """Test process_audio wrapper returns error result on failure."""
        # Arrange
        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(
            return_value=Response(500)
        )

        # Act
        result = await service.process_audio(b"audio data")

        # Assert
        assert result.success is False
        assert result.text == ""
        assert "ASR service request failed" in result.error


class TestASRServiceConcurrency:
    """Test cases for ASR service concurrency limiting."""

    @pytest.fixture
    def limited_service(self) -> ASRService:
        """Create ASR service with low concurrency limit for testing."""
        return ASRService(
            base_url="http://127.0.0.1:28111",
            max_concurrent_requests=2,
        )

    @pytest.mark.asyncio
    @respx.mock
    async def test_concurrency_limit_enforced(self, limited_service: ASRService) -> None:
        """Test that concurrent requests are limited by semaphore."""
        # Arrange - mock slow endpoint
        active_requests = 0
        max_active = 0

        async def slow_handler(request):
            nonlocal active_requests, max_active
            active_requests += 1
            max_active = max(max_active, active_requests)
            await asyncio.sleep(0.1)  # Simulate slow processing
            active_requests -= 1
            return Response(200, json={"text": "test", "raw_text": ""})

        respx.post("http://127.0.0.1:28111/v1/audio/transcriptions").mock(side_effect=slow_handler)

        # Act - Start 5 concurrent requests with limit of 2
        tasks = [
            limited_service.transcribe(b"data"),
            limited_service.transcribe(b"data"),
            limited_service.transcribe(b"data"),
            limited_service.transcribe(b"data"),
            limited_service.transcribe(b"data"),
        ]
        await asyncio.gather(*tasks)

        # Assert - Max concurrent should not exceed limit
        assert max_active <= 2, f"Expected max 2 concurrent requests, got {max_active}"

    @pytest.mark.asyncio
    async def test_close_releases_resources(self) -> None:
        """Test that close() properly releases resources."""
        # Arrange
        service = ASRService(base_url="http://127.0.0.1:28111")

        # Act
        await service.close()

        # Assert - Should not raise any errors
        assert service._client.is_closed

    def test_default_concurrency_limit(self) -> None:
        """Test default concurrency limit is 5."""
        # Act
        service = ASRService()

        # Assert
        assert service._semaphore._value == 5
