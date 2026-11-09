"""Async client for the local image and PDF OCR service."""

import asyncio
import json
import logging
from typing import Literal, cast

import httpx
from pydantic import ValidationError

from src.exceptions import OCRServiceError
from src.models.schemas import OCRHealthResponse, OCRProcessResult, OCRResponse
from src.utils.retry import is_retryable_error

logger = logging.getLogger(__name__)

OCREngine = Literal["auto", "paddle", "glm", "qwen", "jina"]
VALID_ENGINES = frozenset({"auto", "paddle", "glm", "qwen", "jina"})


class OCRService:
    """Upload images and PDFs to the local OCR HTTP service."""

    DEFAULT_MAX_CONCURRENT = 4
    DEFAULT_TIMEOUT_SECONDS = 900.0

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:18100",
        max_concurrent_requests: int = DEFAULT_MAX_CONCURRENT,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Initialize the shared OCR HTTP client.

        Args:
            base_url: Base URL of the deployed OCR service.
            max_concurrent_requests: Maximum concurrent OCR uploads.
            timeout_seconds: End-to-end timeout for a single OCR request.
        """
        self.base_url = base_url.rstrip("/")
        self.image_endpoint = f"{self.base_url}/ocr/image"
        self.pdf_endpoint = f"{self.base_url}/ocr/pdf"
        self.health_endpoint = f"{self.base_url}/ocr/health"
        self._semaphore = asyncio.Semaphore(max_concurrent_requests)
        self._client = httpx.AsyncClient(
            timeout=httpx.Timeout(timeout_seconds),
            limits=httpx.Limits(
                max_connections=max_concurrent_requests,
                max_keepalive_connections=max_concurrent_requests,
                keepalive_expiry=30.0,
            ),
        )
        logger.info(
            "OCRService initialized: base_url=%s concurrency=%d",
            self.base_url,
            max_concurrent_requests,
        )

    @staticmethod
    def _clean_markdown_code_blocks(text: str) -> str:
        """Remove a single wrapping Markdown code fence."""
        cleaned = text.strip()
        if cleaned.startswith("```"):
            first_newline = cleaned.find("\n")
            cleaned = cleaned[first_newline + 1 :] if first_newline >= 0 else cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        return cleaned.strip()

    @staticmethod
    def _validate_engine(engine: str) -> OCREngine:
        """Validate and normalize an OCR engine name."""
        normalized = engine.strip().lower()
        if normalized not in VALID_ENGINES:
            raise OCRServiceError(
                f"Unsupported OCR engine {engine!r}; expected auto, paddle, glm, qwen, or jina"
            )
        return cast(OCREngine, normalized)

    async def _post(
        self,
        endpoint: str,
        *,
        files: dict[str, tuple[str, bytes, str]],
        data: dict[str, str],
    ) -> OCRResponse:
        """Send one OCR request and validate its response."""
        async with self._semaphore:
            try:
                response = await self._client.post(endpoint, files=files, data=data)
            except httpx.HTTPError as error:
                raise OCRServiceError(f"OCR service request failed: {error}") from error

        if response.status_code >= 400:
            detail = response.text.strip()
            try:
                detail = str(json.loads(detail).get("error", detail))
            except (json.JSONDecodeError, AttributeError):
                pass
            error = OCRServiceError(
                f"OCR service returned HTTP {response.status_code}: {detail or 'unknown error'}"
            )
            # Preserve retryability after translating the HTTP response into the
            # handler-friendly OCRServiceError.
            error.retryable = response.status_code == 429 or response.status_code >= 500
            raise error

        try:
            result = OCRResponse.model_validate_json(response.text)
        except ValidationError as error:
            raise OCRServiceError(f"Invalid OCR response: {error}") from error

        result.markdown = self._clean_markdown_code_blocks(result.markdown)
        if not result.markdown:
            raise OCRServiceError("OCR service returned empty text")
        return result

    async def recognize_image(
        self,
        image_data: bytes,
        *,
        filename: str = "image.png",
        content_type: str = "image/png",
        engine: str = "auto",
    ) -> OCRResponse:
        """Recognize an image and return Markdown with metadata."""
        selected_engine = self._validate_engine(engine)
        return await self._post(
            self.image_endpoint,
            files={"file": (filename, image_data, content_type)},
            data={"engine": selected_engine},
        )

    async def recognize_pdf(
        self,
        pdf_data: bytes,
        *,
        filename: str = "document.pdf",
        engine: str = "auto",
        pages: str | None = None,
    ) -> OCRResponse:
        """Recognize a PDF, optionally limiting it to a page or inclusive range."""
        selected_engine = self._validate_engine(engine)
        data = {"engine": selected_engine}
        if pages:
            data["page_range"] = pages
        return await self._post(
            self.pdf_endpoint,
            files={"file": (filename, pdf_data, "application/pdf")},
            data=data,
        )

    async def health(self) -> OCRHealthResponse:
        """Return OCR and Ollama model availability."""
        try:
            response = await self._client.get(self.health_endpoint)
        except httpx.HTTPError as error:
            raise OCRServiceError(f"OCR health check failed: {error}") from error
        try:
            return OCRHealthResponse.model_validate_json(response.text)
        except ValidationError as error:
            raise OCRServiceError(f"Invalid OCR health response: {error}") from error

    async def process_image(
        self,
        image_data: bytes,
        *,
        filename: str = "image.png",
        content_type: str = "image/png",
        engine: str = "auto",
    ) -> OCRProcessResult:
        """Recognize an image and convert failures to a result object."""
        try:
            result = await self.recognize_image(
                image_data,
                filename=filename,
                content_type=content_type,
                engine=engine,
            )
            return self._to_process_result(result)
        except OCRServiceError as error:
            logger.error("Image OCR failed: %s", error)
            return OCRProcessResult(
                success=False, error=str(error), retryable=is_retryable_error(error)
            )

    async def process_pdf(
        self,
        pdf_data: bytes,
        *,
        filename: str = "document.pdf",
        engine: str = "auto",
        pages: str | None = None,
    ) -> OCRProcessResult:
        """Recognize a PDF and convert failures to a result object."""
        try:
            result = await self.recognize_pdf(
                pdf_data,
                filename=filename,
                engine=engine,
                pages=pages,
            )
            return self._to_process_result(result)
        except OCRServiceError as error:
            logger.error("PDF OCR failed: %s", error)
            return OCRProcessResult(
                success=False, error=str(error), retryable=is_retryable_error(error)
            )

    @staticmethod
    def _to_process_result(result: OCRResponse) -> OCRProcessResult:
        """Convert the HTTP schema to the handler-friendly schema."""
        return OCRProcessResult(
            success=True,
            text=result.markdown,
            engine=result.engine,
            pages=result.pages,
            duration_ms=result.duration_ms,
        )

    async def close(self) -> None:
        """Close the shared HTTP client and connection pool."""
        await self._client.aclose()
        logger.info("OCRService connection pool closed")
