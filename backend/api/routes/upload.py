"""
Upload endpoint for the Lecture Transcriber API.

This module handles video file uploads with validation, storage in temp directory,
and returns file metadata including duration when available.
"""

import uuid
import logging
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, HTTPException, status
from fastapi.responses import JSONResponse

from backend.api.schemas.response import UploadResponse, ErrorResponse, ErrorDetail
from backend.core.processing.audio_extractor import AudioExtractor

# Configure logging
logger = logging.getLogger(__name__)

# Create router
router = APIRouter()

# Supported video formats (matching AudioExtractor)
SUPPORTED_FORMATS = {'.mp4', '.mkv', '.webm', '.avi', '.mov'}
SUPPORTED_MIME_TYPES = {
    'video/mp4',
    'video/x-matroska',
    'video/webm',
    'video/x-msvideo',
    'video/quicktime',
    'application/octet-stream',
}

# Maximum file size (2GB)
MAX_FILE_SIZE = 2 * 1024 * 1024 * 1024

# Temp directory path (will be set from main.py)
TEMP_DIR: Optional[Path] = None


def set_temp_dir(temp_dir: Path) -> None:
    """Set the temp directory for file uploads."""
    global TEMP_DIR
    TEMP_DIR = temp_dir
    TEMP_DIR.mkdir(exist_ok=True)


def get_temp_dir() -> Path:
    """Get the temp directory, creating default if not set."""
    global TEMP_DIR
    if TEMP_DIR is None:
        TEMP_DIR = Path("./temp")
        TEMP_DIR.mkdir(exist_ok=True)
    return TEMP_DIR


def validate_file_extension(filename: str) -> bool:
    """Check if the file has a supported extension."""
    if not filename:
        return False
    extension = Path(filename).suffix.lower()
    return extension in SUPPORTED_FORMATS


def validate_content_type(content_type: Optional[str]) -> bool:
    """Check if the content type is valid for video files."""
    if not content_type:
        return True  # Allow if not specified, will validate by extension
    return content_type in SUPPORTED_MIME_TYPES


async def get_video_duration(file_path: Path) -> Optional[float]:
    """
    Get video duration using ffprobe.
    
    Returns None if duration cannot be determined.
    """
    try:
        import ffmpeg
        probe = ffmpeg.probe(str(file_path))
        
        # Try to get duration from format
        if 'format' in probe and 'duration' in probe['format']:
            return float(probe['format']['duration'])
        
        # Try to get duration from video stream
        for stream in probe.get('streams', []):
            if stream.get('codec_type') == 'video' and 'duration' in stream:
                return float(stream['duration'])
        
        return None
    except Exception as e:
        logger.warning(f"Could not determine video duration: {e}")
        return None


@router.post(
    "/upload",
    response_model=UploadResponse,
    responses={
        400: {"model": ErrorResponse, "description": "Invalid file"},
        413: {"model": ErrorResponse, "description": "File too large"},
        500: {"model": ErrorResponse, "description": "Server error"},
    },
    summary="Upload a video file",
    description="Upload a video file for transcription. Supports MP4, MKV, WebM, AVI, and MOV formats."
)
async def upload_video(
    file: UploadFile = File(..., description="Video file to upload")
) -> UploadResponse:
    """
    Upload a video file for transcription.
    
    The file is validated for format and size, stored in the temp directory,
    and metadata is extracted and returned.
    
    Args:
        file: The video file to upload
        
    Returns:
        UploadResponse with file_id, filename, size, and duration
        
    Raises:
        HTTPException: If validation fails or upload errors occur
    """
    # Validate filename
    if not file.filename:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_FILENAME",
                    "message": "Filename is required",
                    "details": {
                        "reason": "No filename provided",
                        "suggestion": "Provide a file with a valid filename"
                    }
                }
            }
        )
    
    # Validate file extension
    if not validate_file_extension(file.filename):
        supported_list = ', '.join(sorted(ext.lstrip('.') for ext in SUPPORTED_FORMATS))
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "UNSUPPORTED_FORMAT",
                    "message": f"Unsupported file format",
                    "details": {
                        "reason": f"File extension not supported",
                        "suggestion": f"Use one of the supported formats: {supported_list}"
                    }
                }
            }
        )
    
    # Validate content type (if provided)
    if not validate_content_type(file.content_type):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail={
                "error": {
                    "code": "INVALID_CONTENT_TYPE",
                    "message": "Invalid content type for video file",
                    "details": {
                        "reason": f"Content type '{file.content_type}' is not a valid video type",
                        "suggestion": "Upload a valid video file"
                    }
                }
            }
        )
    
    # Generate unique file ID
    file_id = str(uuid.uuid4())
    
    # Get file extension
    original_extension = Path(file.filename).suffix.lower()
    
    # Create storage path
    temp_dir = get_temp_dir()
    storage_filename = f"{file_id}{original_extension}"
    storage_path = temp_dir / storage_filename
    
    try:
        # Read and save file in chunks to handle large files
        total_size = 0
        chunk_size = 1024 * 1024  # 1MB chunks
        
        with open(storage_path, "wb") as buffer:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                
                total_size += len(chunk)
                
                # Check file size limit
                if total_size > MAX_FILE_SIZE:
                    # Clean up partial file
                    buffer.close()
                    storage_path.unlink(missing_ok=True)
                    raise HTTPException(
                        status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                        detail={
                            "error": {
                                "code": "FILE_TOO_LARGE",
                                "message": "File size exceeds maximum allowed",
                                "details": {
                                    "reason": f"File exceeds {MAX_FILE_SIZE // (1024*1024*1024)}GB limit",
                                    "suggestion": "Upload a smaller file or compress the video"
                                }
                            }
                        }
                    )
                
                buffer.write(chunk)
        
        logger.info(f"File uploaded: {file.filename} -> {storage_path} ({total_size} bytes)")
        
        # Get video duration
        duration = await get_video_duration(storage_path)
        
        return UploadResponse(
            file_id=file_id,
            filename=file.filename,
            size=total_size,
            duration=duration
        )
        
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        # Clean up on error
        storage_path.unlink(missing_ok=True)
        logger.error(f"Upload failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail={
                "error": {
                    "code": "UPLOAD_FAILED",
                    "message": "Failed to save uploaded file",
                    "details": {
                        "reason": str(e),
                        "suggestion": "Try uploading again"
                    }
                }
            }
        )
