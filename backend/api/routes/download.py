"""
Download endpoint for the Lecture Transcriber API.

This module handles result file serving with:
- Download by task_id
- Support for multiple output formats (MD, JSON)
- Proper content-type headers for file downloads
"""

import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, status, Query
from fastapi.responses import FileResponse

from backend.api.schemas.response import ErrorResponse
from backend.api.routes.transcribe import get_task, TaskStatus, get_output_dir

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Supported output formats
SUPPORTED_FORMATS = {"md", "json"}

# Content type mapping
CONTENT_TYPE_MAP = {
    "md": "text/markdown; charset=utf-8",
    "json": "application/json; charset=utf-8",
}


def get_result_file_path(task_id: str, format: str) -> Optional[Path]:
    """
    Get the path to a result file for a given task and format.
    
    Args:
        task_id: The task identifier
        format: The output format (md or json)
        
    Returns:
        Path to the result file if it exists, None otherwise
    """
    output_dir = get_output_dir()
    file_path = output_dir / f"{task_id}.{format}"
    
    if file_path.exists():
        return file_path
    
    return None


@router.get(
    "/download/{task_id}",
    responses={
        200: {
            "description": "Result file download",
            "content": {
                "text/markdown": {},
                "application/json": {},
            }
        },
        400: {"model": ErrorResponse, "description": "Invalid format"},
        404: {"model": ErrorResponse, "description": "Task or file not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Download transcription result",
    description="Download the transcription result file in the specified format (MD or JSON)."
)
async def download_result(
    task_id: str,
    format: str = Query(
        default="md",
        description="Output format: 'md' for Markdown, 'json' for JSON metadata",
        pattern="^(md|json)$"
    )
) -> FileResponse:
    """
    Download the result file for a completed transcription task.
    
    Args:
        task_id: The unique identifier of the task
        format: The output format (md or json)
        
    Returns:
        FileResponse with the result file
        
    Raises:
        HTTPException: If task not found, not completed, or file not found
    """
    # Validate task_id
    if not task_id or not task_id.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_TASK_ID",
                    "message": "Task ID is required",
                    "details": {
                        "reason": "Empty or invalid task ID provided",
                        "suggestion": "Provide a valid task ID from the transcribe endpoint"
                    }
                }
            }
        )
    
    task_id = task_id.strip()
    format = format.lower().strip()
    
    # Validate format
    if format not in SUPPORTED_FORMATS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FORMAT",
                    "message": f"Unsupported format: {format}",
                    "details": {
                        "reason": f"Format must be one of: {', '.join(SUPPORTED_FORMATS)}",
                        "suggestion": "Use 'md' for Markdown or 'json' for JSON metadata"
                    }
                }
            }
        )
    
    # Get task from storage
    task = get_task(task_id)
    
    if not task:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "TASK_NOT_FOUND",
                    "message": "Task not found",
                    "details": {
                        "reason": f"No task found with ID: {task_id}",
                        "suggestion": "Check the task ID or create a new transcription task"
                    }
                }
            }
        )
    
    # Check if task is completed
    if task.status != TaskStatus.COMPLETED:
        status_message = {
            TaskStatus.PENDING: "Task is still pending",
            TaskStatus.PROCESSING: "Task is still processing",
            TaskStatus.FAILED: "Task failed and has no results",
        }.get(task.status, f"Task status is {task.status.value}")
        
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "TASK_NOT_COMPLETED",
                    "message": status_message,
                    "details": {
                        "reason": f"Task status is '{task.status.value}', not 'completed'",
                        "suggestion": "Wait for the task to complete or check status at /api/status/{task_id}"
                    }
                }
            }
        )
    
    # Get the result file path
    file_path = get_result_file_path(task_id, format)
    
    if not file_path:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail={
                "error": {
                    "code": "FILE_NOT_FOUND",
                    "message": f"Result file not found for format: {format}",
                    "details": {
                        "reason": f"The {format.upper()} file for this task does not exist",
                        "suggestion": "Try a different format or re-run the transcription"
                    }
                }
            }
        )
    
    # Determine filename for download
    original_filename = f"transcription_{task_id[:8]}.{format}"
    
    logger.info(f"Serving download for task {task_id}, format: {format}")
    
    return FileResponse(
        path=str(file_path),
        media_type=CONTENT_TYPE_MAP.get(format, "application/octet-stream"),
        filename=original_filename,
        headers={
            "Content-Disposition": f'attachment; filename="{original_filename}"'
        }
    )
