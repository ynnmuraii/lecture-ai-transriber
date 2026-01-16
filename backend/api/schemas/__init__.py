"""API schemas package for the Lecture Transcriber API."""

from backend.api.schemas.request import (
    TranscribeRequest,
    DownloadRequest,
)
from backend.api.schemas.response import (
    UploadResponse,
    TaskResponse,
    StatusResponse,
    TranscriptionSegment,
    ResultResponse,
    ErrorDetail,
    ErrorResponse,
    HealthResponse,
)

__all__ = [
    # Request schemas
    "TranscribeRequest",
    "DownloadRequest",
    # Response schemas
    "UploadResponse",
    "TaskResponse",
    "StatusResponse",
    "TranscriptionSegment",
    "ResultResponse",
    "ErrorDetail",
    "ErrorResponse",
    "HealthResponse",
]
