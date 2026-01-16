"""Response schemas for the Lecture Transcriber API."""

from datetime import datetime
from typing import Optional, List, Dict, Any
from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response model for file upload endpoint."""
    
    file_id: str = Field(..., description="Unique identifier for the uploaded file")
    filename: str = Field(..., description="Original filename")
    size: int = Field(..., ge=0, description="File size in bytes")
    duration: Optional[float] = Field(None, ge=0, description="Video duration in seconds")


class TaskResponse(BaseModel):
    """Response model for transcription task creation."""
    
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current task status")
    created_at: datetime = Field(..., description="Task creation timestamp")


class StatusResponse(BaseModel):
    """Response model for task status endpoint."""
    
    task_id: str = Field(..., description="Unique identifier for the task")
    status: str = Field(..., description="Current status: pending, processing, completed, failed")
    progress: float = Field(..., ge=0, le=100, description="Processing progress percentage")
    message: Optional[str] = Field(None, description="Status message or error description")
    result_url: Optional[str] = Field(None, description="URL to download results when completed")


class TranscriptionSegment(BaseModel):
    """Model for a single transcription segment."""
    
    text: str = Field(..., description="Transcribed text")
    start_time: float = Field(..., ge=0, description="Segment start time in seconds")
    end_time: float = Field(..., ge=0, description="Segment end time in seconds")
    confidence: Optional[float] = Field(None, ge=0, le=1, description="Transcription confidence")


class ResultResponse(BaseModel):
    """Response model for transcription results."""
    
    task_id: str = Field(..., description="Task identifier")
    status: str = Field(..., description="Task status")
    content: Optional[str] = Field(None, description="Transcribed content in Markdown format")
    segments: Optional[List[TranscriptionSegment]] = Field(None, description="Transcription segments")
    metadata: Optional[Dict[str, Any]] = Field(None, description="Processing metadata")


class ErrorDetail(BaseModel):
    """Model for error details."""
    
    reason: Optional[str] = Field(None, description="Error reason")
    suggestion: Optional[str] = Field(None, description="Suggested action")


class ErrorResponse(BaseModel):
    """Response model for API errors."""
    
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    details: Optional[ErrorDetail] = Field(None, description="Additional error details")


class HealthResponse(BaseModel):
    """Response model for health check endpoint."""
    
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="API version")
    gpu_available: bool = Field(..., description="Whether GPU is available")
