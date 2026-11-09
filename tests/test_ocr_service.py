"""Unit tests for the OCR HTTP client."""

from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from src.exceptions import OCRServiceError
from src.models.schemas import OCRHealthResponse, OCRProcessResult, OCRResponse
from src.services.ocr import OCRService


@pytest.fixture
def ocr_service() -> OCRService:
    """Create an OCR service instance for testing."""
    return OCRService(base_url="http://localhost:8000")


def make_response(
    *,
    markdown: str = "Hello World",
    engine: str = "paddle",
    pages: int = 1,
    duration_ms: int = 123,
    status_code: int = 200,
) -> MagicMock:
    """Create a mock OCR HTTP response."""
    response = MagicMock()
    response.status_code = status_code
    response.text = (
        f'{{"markdown": {markdown!r}, "engine": "{engine}", '
        f'"pages": {pages}, "duration_ms": {duration_ms}}}'
    ).replace("'", '"')
    return response


class TestOCRService:
    """Test the OCR service client."""

    def test_cloud_health_retains_independent_backend_state(self) -> None:
        """Cloud availability does not imply local backend readiness."""
        health = OCRHealthResponse.model_validate(
            {
                "status": "ok",
                "ollama": False,
                "backend": "lmstudio",
                "backend_ready": False,
                "jina": {"configured": True, "reachable": True},
                "models": [{"engine": "jina", "name": "jina-ocr-v1", "available": True}],
            }
        )
        assert health.jina.configured and health.jina.reachable
        assert health.backend == "lmstudio" and not health.backend_ready

    @pytest.mark.asyncio
    @pytest.mark.parametrize("returned_engine", ["jina", "mixed"])
    async def test_jina_pdf_accepts_actual_engine(
        self, ocr_service: OCRService, returned_engine: str
    ) -> None:
        """Jina PDF requests may return cloud or mixed fallback results."""
        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as post:
            post.return_value = make_response(engine=returned_engine, pages=2)
            result = await ocr_service.recognize_pdf(b"pdf", engine="jina", pages="1-2")
        assert result.engine == returned_engine
        assert post.call_args.kwargs["data"] == {"engine": "jina", "page_range": "1-2"}

    def test_init_default_url(self) -> None:
        """Use the deployed OCR service by default."""
        service = OCRService()

        assert service.base_url == "http://127.0.0.1:18100"
        assert service.image_endpoint == "http://127.0.0.1:18100/ocr/image"
        assert service.pdf_endpoint == "http://127.0.0.1:18100/ocr/pdf"

    @pytest.mark.asyncio
    async def test_recognize_image_uses_current_api(self, ocr_service: OCRService) -> None:
        """Upload an image with an explicit OCR engine."""
        response = make_response(engine="glm")

        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            result = await ocr_service.recognize_image(
                b"image-data",
                filename="scan.jpg",
                content_type="image/jpeg",
                engine="glm",
            )

        assert result == OCRResponse(
            markdown="Hello World",
            engine="glm",
            pages=1,
            duration_ms=123,
        )
        mock_post.assert_awaited_once_with(
            "http://localhost:8000/ocr/image",
            files={"file": ("scan.jpg", b"image-data", "image/jpeg")},
            data={"engine": "glm"},
        )

    @pytest.mark.asyncio
    async def test_recognize_pdf_supports_page_range(self, ocr_service: OCRService) -> None:
        """Upload a PDF with page selection and engine selection."""
        response = make_response(markdown="# Pages", engine="qwen", pages=3)

        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            result = await ocr_service.recognize_pdf(
                b"pdf-data",
                filename="report.pdf",
                engine="qwen",
                pages="2-4",
            )

        assert result.markdown == "# Pages"
        assert result.pages == 3
        mock_post.assert_awaited_once_with(
            "http://localhost:8000/ocr/pdf",
            files={"file": ("report.pdf", b"pdf-data", "application/pdf")},
            data={"engine": "qwen", "page_range": "2-4"},
        )

    @pytest.mark.asyncio
    async def test_rejects_unsupported_engine(self, ocr_service: OCRService) -> None:
        """Reject an invalid engine before making a request."""
        with pytest.raises(OCRServiceError, match="Unsupported OCR engine"):
            await ocr_service.recognize_image(b"image", engine="unknown")

    @pytest.mark.asyncio
    async def test_surfaces_service_error_body(self, ocr_service: OCRService) -> None:
        """Include the OCR API error body in the client error."""
        response = MagicMock(status_code=503, text='{"error":"Ollama unavailable"}')

        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            with pytest.raises(OCRServiceError, match="Ollama unavailable"):
                await ocr_service.recognize_image(b"image")

    @pytest.mark.asyncio
    async def test_wraps_network_error(self, ocr_service: OCRService) -> None:
        """Convert HTTP client failures into the domain exception."""
        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("Connection failed")
            with pytest.raises(OCRServiceError, match="Connection failed"):
                await ocr_service.recognize_image(b"image")

    @pytest.mark.asyncio
    async def test_rejects_invalid_response(self, ocr_service: OCRService) -> None:
        """Reject a response that does not follow the OCR schema."""
        response = MagicMock(status_code=200, text="not-json")

        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            with pytest.raises(OCRServiceError, match="Invalid OCR response"):
                await ocr_service.recognize_image(b"image")

    @pytest.mark.asyncio
    async def test_process_image_preserves_metadata(self, ocr_service: OCRService) -> None:
        """Return OCR metadata from the compatibility wrapper."""
        response = make_response(engine="glm", duration_ms=456)

        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.return_value = response
            result = await ocr_service.process_image(b"image", engine="glm")

        assert result == OCRProcessResult(
            success=True,
            text="Hello World",
            engine="glm",
            pages=1,
            duration_ms=456,
        )

    @pytest.mark.asyncio
    async def test_process_pdf_returns_failure_result(self, ocr_service: OCRService) -> None:
        """Return a structured failure instead of raising from process_pdf."""
        with patch.object(ocr_service._client, "post", new_callable=AsyncMock) as mock_post:
            mock_post.side_effect = httpx.ConnectError("offline")
            result = await ocr_service.process_pdf(b"pdf")

        assert result.success is False
        assert result.text == ""
        assert "offline" in result.error

    @pytest.mark.asyncio
    async def test_close(self, ocr_service: OCRService) -> None:
        """Close the shared HTTP client."""
        with patch.object(ocr_service._client, "aclose", new_callable=AsyncMock) as mock_close:
            await ocr_service.close()

        mock_close.assert_awaited_once()


class TestOCRTextCleaning:
    """Test Markdown fence cleanup."""

    def test_clean_markdown_fence(self) -> None:
        """Remove a wrapping Markdown fence."""
        assert OCRService._clean_markdown_code_blocks("```text\nHello\n```") == "Hello"

    def test_preserve_plain_markdown(self) -> None:
        """Keep unfenced Markdown unchanged."""
        assert OCRService._clean_markdown_code_blocks("# Heading") == "# Heading"
