"""ASR (Automatic Speech Recognition) service."""

import asyncio
import logging

import httpx

from src.exceptions import ASRServiceError
from src.models.schemas import AudioProcessResult, TypelessTranscriptionResponse
from src.utils.retry import is_retryable_error

logger = logging.getLogger(__name__)


class ASRService:
    """Service for transcribing audio to text.

    This service communicates with a local ASR service via HTTP API.
    Uses connection pooling and concurrency limiting for efficient resource use.

    Attributes:
        base_url: Base URL of the ASR service.
        _client: Reusable httpx AsyncClient with connection pool.
        _semaphore: Semaphore for limiting concurrent requests.
    """

    DEFAULT_MAX_CONCURRENT = 5

    TRANSCRIPTION_PATH = "/v1/audio/transcriptions"
    DEFAULT_POSTPROCESS_MODE = "standard"

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:28111",
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT,
        api_key: str | None = None,
    ) -> None:
        """Initialize ASR service with connection pool and concurrency limit.

        Args:
            base_url: Base URL of the ASR service.
            max_concurrent_requests: Maximum number of concurrent ASR requests.
                                     Defaults to 5.
            api_key: Optional Typeless API key.
        """
        self.base_url = base_url.rstrip("/")
        self._endpoint = f"{self.base_url}{self.TRANSCRIPTION_PATH}"
        self._headers = {"X-API-Key": api_key} if api_key else {}
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        # Create client with connection pool (limits: 100 connections, 20 per host)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(300.0),  # 5 minutes for long audio
            limits=httpx.Limits(
                max_connections=100,
                max_keepalive_connections=20,
                keepalive_expiry=30.0,  # Keep idle connections for 30 seconds
            ),
        )
        logger.info(
            f"ASRService initialized with connection pool and "
            f"concurrency limit={max_concurrent_requests}: {self.base_url}"
        )

    async def transcribe(
        self,
        audio_data: bytes,
        duration: int = 0,
        postprocess_mode: str = DEFAULT_POSTPROCESS_MODE,
    ) -> str:
        """Transcribe audio data to text.

        Uses semaphore to limit concurrent requests to the ASR service.
        Uses the stable Typeless v1 contract for every audio duration.

        Args:
            audio_data: Raw audio file bytes.
            duration: Audio duration in seconds.
            postprocess_mode: Typeless post-processing mode. Defaults to the
                local-rule ``standard`` mode.

        Returns:
            Transcribed text.

        Raises:
            ASRServiceError: If transcription fails.
        """
        logger.info(
            f"Sending {len(audio_data)} bytes to Typeless v1 at {self._endpoint} "
            f"(duration={duration}s, postprocess_mode={postprocess_mode})"
        )

        # Acquire semaphore to limit concurrent requests
        async with self._semaphore:
            logger.debug("Acquired semaphore, active requests limited")
            try:
                # Telegram voice is .oga format, but the v1 endpoint accepts it
                # through the conventional OGG multipart media type.
                files = {"file": ("audio.ogg", audio_data, "audio/ogg")}
                data = {
                    "model": "default",
                    "language": "auto",
                    "postprocess_mode": postprocess_mode,
                }
                response = await self._client.post(
                    self._endpoint,
                    files=files,
                    data=data,
                    headers=self._headers,
                )
                logger.info(f"ASR service responded with status: {response.status_code}")
                response.raise_for_status()
            except httpx.HTTPError as e:
                logger.error(f"ASR service HTTP error: {e}")
                raise ASRServiceError(f"ASR service request failed: {e}") from e

            try:
                result = TypelessTranscriptionResponse.model_validate_json(response.text)
                logger.info(
                    f"Typeless response parsed: raw={result.raw_text[:50]}..., "
                    f"text={result.text[:50]}..."
                )
            except Exception as e:
                logger.error(
                    f"Failed to parse ASR response: {e}, raw response: {response.text[:500]}"
                )
                raise ASRServiceError(f"Invalid response format: {e}") from e

            text = result.text or result.raw_text
            if not text:
                logger.error("ASR service returned empty transcript")
                raise ASRServiceError("ASR service returned empty transcript")

            logger.info(f"Transcription successful, text length: {len(text)}")
            return text

    async def process_audio(
        self,
        audio_data: bytes,
        duration: int = 0,
        postprocess_mode: str = DEFAULT_POSTPROCESS_MODE,
    ) -> AudioProcessResult:
        """Process audio with error handling wrapper.

        This is a convenience method that wraps transcribe() and
        returns a result object instead of raising exceptions.

        Args:
            audio_data: Raw audio file bytes.
            duration: Audio duration in seconds.
            postprocess_mode: Typeless post-processing mode.

        Returns:
            AudioProcessResult with success status and text or error.
        """
        logger.info(f"Processing audio: {len(audio_data)} bytes, duration={duration}s")
        try:
            text = await self.transcribe(
                audio_data,
                duration=duration,
                postprocess_mode=postprocess_mode,
            )
            logger.info("Audio processing successful")
            return AudioProcessResult(success=True, text=text, error="")
        except ASRServiceError as e:
            logger.error(f"ASR service error in process_audio: {e}")
            return AudioProcessResult(
                success=False,
                text="",
                error=str(e),
                retryable=is_retryable_error(e),
            )
        except Exception as e:
            logger.exception(f"Unexpected error in process_audio: {e}")
            return AudioProcessResult(
                success=False,
                text="",
                error=f"Unexpected error: {e}",
                retryable=is_retryable_error(e),
            )

    async def close(self) -> None:
        """Close the HTTP client and connection pool.

        Should be called when the service is no longer needed.
        """
        await self._client.aclose()
        logger.info("ASRService connection pool closed")
