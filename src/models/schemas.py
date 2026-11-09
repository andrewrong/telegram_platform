"""Pydantic schemas for data validation."""

from typing import Literal

from pydantic import BaseModel, Field


class TypelessTranscriptionResponse(BaseModel):
    """Stable response schema returned by the Typeless v1 transcription API.

    Attributes:
        text: Post-processed transcription result.
        raw_text: Model output before post-processing.
    """

    text: str = Field(default="", description="Post-processed transcription result")
    raw_text: str = Field(default="", description="Raw model transcription result")


class AudioProcessResult(BaseModel):
    """Result of processing audio.

    Attributes:
        success: Whether processing was successful.
        text: Transcribed text if successful.
        error: Error message if failed.
    """

    success: bool
    text: str = ""
    error: str = ""
    retryable: bool = False


class OCRResponse(BaseModel):
    """Successful response from the OCR HTTP service."""

    markdown: str = Field(min_length=1, description="Recognized Markdown content")
    engine: Literal["paddle", "glm", "qwen", "jina", "mixed"]
    pages: int = Field(ge=1, description="Number of processed pages")
    duration_ms: int = Field(ge=0, description="Processing duration in milliseconds")


class OCRModelStatus(BaseModel):
    """Availability of one OCR engine."""

    engine: Literal["paddle", "glm", "qwen", "jina"]
    name: str
    available: bool


class JinaHealthStatus(BaseModel):
    """Cloud metadata reachability, not proof of valid credentials or inference capacity."""

    configured: bool = False
    reachable: bool = False


class OCRHealthResponse(BaseModel):
    """Health response from the OCR HTTP service."""

    status: Literal["ok", "degraded"]
    ollama: bool
    backend: str | None = None
    backend_ready: bool | None = None
    jina: JinaHealthStatus = Field(default_factory=JinaHealthStatus)
    models: list[OCRModelStatus]


class OCRProcessResult(BaseModel):
    """Non-raising OCR result used by Telegram handlers."""

    success: bool
    text: str = ""
    error: str = ""
    engine: str = ""
    pages: int = 0
    duration_ms: int = 0
    retryable: bool = False
