"""
Status endpoint for the Lecture Transcriber API.

This module handles task status tracking with:
- Task status retrieval by task_id
- Progress and status information
- Result URL when completed
"""

import logging
from typing import Optional

from fastapi import APIRouter, HTTPException, status

from backend.api.schemas.response import StatusResponse, ErrorResponse
from backend.api.routes.transcribe import get_task, TaskStatus

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()


@router.get(
    "/status/{task_id}",
    response_model=StatusResponse,
    responses={
        404: {"model": ErrorResponse, "description": "Task not found"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Get task status",
    description="Get the current status and progress of a transcription task."
)
async def get_task_status(task_id: str) -> StatusResponse:
    """
    Get the status of a transcription task.
    
    Returns the current status, progress percentage, and result URL
    when the task is completed.
    
    Args:
        task_id: The unique identifier of the task
        
    Returns:
        StatusResponse with task_id, status, progress, message, and result_url
        
    Raises:
        HTTPException: If task not found
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
    
    # Get task from storage
    task = get_task(task_id.strip())
    
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
    
    # Build result URL if task is completed
    result_url: Optional[str] = None
    if task.status == TaskStatus.COMPLETED and task.result_path:
        result_url = f"/api/download/{task_id}"
    
    # Build status message
    message = task.message
    if task.status == TaskStatus.FAILED and task.error_message:
        message = task.error_message
    
    logger.debug(f"Status check for task {task_id}: {task.status.value} ({task.progress}%)")
    
    return StatusResponse(
        task_id=task.task_id,
        status=task.status.value,
        progress=task.progress,
        message=message,
        result_url=result_url
    )
