"""
Hallucination Filter for Whisper Transcriptions

This module provides filtering capabilities to remove common Whisper hallucinations
including YouTube artifacts, repeated phrases, and low-confidence segments.

"""

import re
import logging
from typing import List, Dict, Set, Optional
from dataclasses import dataclass

from backend.core.models.data_models import TranscriptionSegment


logger = logging.getLogger(__name__)


@dataclass
class HallucinationFilterConfig:
    """Configuration for hallucination filtering."""
    # Compression ratio threshold - segments above this are likely repetitive
    compression_ratio_threshold: float = 1.8
    
    # Log probability threshold - segments below this are low confidence
    logprob_threshold: float = -0.8
    
    # Enable/disable specific filters
    enable_blacklist_filter: bool = True
    enable_pattern_filter: bool = True
    enable_compression_filter: bool = True
    enable_logprob_filter: bool = True
    
    # Custom blacklist phrases (in addition to defaults)
    custom_blacklist: List[str] = None
    
    def __post_init__(self):
        """Initialize custom blacklist if not provided."""
        if self.custom_blacklist is None:
            self.custom_blacklist = []


class HallucinationFilter:
    """
    Filter for removing Whisper hallucinations from transcription segments.
    
    This filter identifies and removes common hallucination patterns including:
    - YouTube artifacts (subtitles, channel promotions)
    - Generic closing phrases
    - Highly repetitive content (high compression ratio)
    - Low confidence segments (low log probability)
    """
    
    # Default blacklist of common hallucinations
    DEFAULT_BLACKLIST = {
        # YouTube artifacts (Russian)
        "спасибо за просмотр",
        "подписывайтесь на канал",
        "ставьте лайки",
        "не забудьте подписаться",
        "жмите на колокольчик",
        "оставляйте комментарии",
        "делитесь видео",
        "смотрите другие видео",
        "ссылка в описании",
        "переходите по ссылке",
        
        # YouTube artifacts (English)
        "thanks for watching",
        "subscribe to the channel",
        "like and subscribe",
        "don't forget to subscribe",
        "hit the bell icon",
        "leave a comment",
        "share this video",
        "check out my other videos",
        "link in the description",
        "click the link",
        
        # Generic closing phrases
        "до свидания",
        "до встречи",
        "пока пока",
        "увидимся",
        "всем пока",
        "goodbye",
        "see you later",
        "bye bye",
        
        # Common subtitle artifacts
        "субтитры",
        "subtitles",
        "перевод",
        "translation",
        "автоматические субтитры",
        "automatic subtitles",
        
        # Music/sound indicators
        "[музыка]",
        "[music]",
        "[аплодисменты]",
        "[applause]",
        "[смех]",
        "[laughter]",
    }
    
    # Regex patterns for detecting hallucinations
    HALLUCINATION_PATTERNS = [
        # URLs and links
        r'https?://\S+',
        r'www\.\S+',
        
        # Social media handles
        r'@\w+',
        r'#\w+',
        
        # Repeated punctuation
        r'\.{4,}',
        r'\?{3,}',
        r'!{3,}',
        
        # Excessive repetition of short phrases (e.g., "да да да да да")
        r'\b(\w{1,3})\s+\1\s+\1\s+\1',
        
        # Channel/video references
        r'канал\s+\w+',
        r'channel\s+\w+',
        
        # Promotional language patterns
        r'подпис(ывайтесь|аться|ка)',
        r'subscrib(e|ing)',
    ]
    
    def __init__(self, config: Optional[HallucinationFilterConfig] = None):
        """
        Initialize the hallucination filter.
        
        Args:
            config: Configuration for filtering behavior. Uses defaults if not provided.
        """
        self.config = config or HallucinationFilterConfig()
        
        # Combine default and custom blacklist
        self.blacklist = self.DEFAULT_BLACKLIST.copy()
        if self.config.custom_blacklist:
            self.blacklist.update(phrase.lower() for phrase in self.config.custom_blacklist)
        
        # Compile regex patterns for efficiency
        self.compiled_patterns = [
            re.compile(pattern, re.IGNORECASE | re.UNICODE)
            for pattern in self.HALLUCINATION_PATTERNS
        ]
        
        logger.info(f"Initialized HallucinationFilter with {len(self.blacklist)} blacklist phrases")
    
    def filter_segments(
        self,
        segments: List[TranscriptionSegment],
        compression_ratios: Optional[List[float]] = None,
        logprobs: Optional[List[float]] = None
    ) -> List[TranscriptionSegment]:
        """
        Filter hallucinations from transcription segments.
        
        Args:
            segments: List of transcription segments to filter
            compression_ratios: Optional list of compression ratios for each segment
            logprobs: Optional list of log probabilities for each segment
        
        Returns:
            Filtered list of segments with hallucinations removed
        """
        if not segments:
            return []
        
        filtered_segments = []
        filtered_count = 0
        
        for i, segment in enumerate(segments):
            # Get compression ratio and logprob for this segment if available
            compression_ratio = compression_ratios[i] if compression_ratios and i < len(compression_ratios) else None
            logprob = logprobs[i] if logprobs and i < len(logprobs) else None
            
            # Check if segment should be filtered
            if self._should_filter_segment(segment, compression_ratio, logprob):
                filtered_count += 1
                logger.debug(f"Filtered segment: '{segment.text[:50]}...'")
                continue
            
            filtered_segments.append(segment)
        
        if filtered_count > 0:
            logger.info(f"Filtered {filtered_count} hallucinated segments out of {len(segments)}")
        
        return filtered_segments
    
    def _should_filter_segment(
        self,
        segment: TranscriptionSegment,
        compression_ratio: Optional[float] = None,
        logprob: Optional[float] = None
    ) -> bool:
        """
        Determine if a segment should be filtered as a hallucination.
        
        Args:
            segment: Segment to check
            compression_ratio: Compression ratio for the segment
            logprob: Log probability for the segment
        
        Returns:
            True if segment should be filtered, False otherwise
        """
        text = segment.text.strip()
        
        # Skip empty segments
        if not text:
            return True
        
        # Check blacklist
        if self.config.enable_blacklist_filter and self._matches_blacklist(text):
            logger.debug(f"Blacklist match: '{text[:50]}'")
            return True
        
        # Check patterns
        if self.config.enable_pattern_filter and self._matches_pattern(text):
            logger.debug(f"Pattern match: '{text[:50]}'")
            return True
        
        # Check compression ratio
        if (self.config.enable_compression_filter and 
            compression_ratio is not None and 
            compression_ratio > self.config.compression_ratio_threshold):
            logger.debug(f"High compression ratio ({compression_ratio:.2f}): '{text[:50]}'")
            return True
        
        # Check log probability
        if (self.config.enable_logprob_filter and 
            logprob is not None and 
            logprob < self.config.logprob_threshold):
            logger.debug(f"Low log probability ({logprob:.2f}): '{text[:50]}'")
            return True
        
        return False
    
    def _matches_blacklist(self, text: str) -> bool:
        """
        Check if text matches any blacklist phrase.
        
        Args:
            text: Text to check
        
        Returns:
            True if text contains a blacklisted phrase
        """
        text_lower = text.lower()
        
        # Check for exact phrase matches
        for phrase in self.blacklist:
            if phrase in text_lower:
                return True
        
        return False
    
    def _matches_pattern(self, text: str) -> bool:
        """
        Check if text matches any hallucination pattern.
        
        Args:
            text: Text to check
        
        Returns:
            True if text matches a hallucination pattern
        """
        for pattern in self.compiled_patterns:
            if pattern.search(text):
                return True
        
        return False
    
    def get_filter_statistics(
        self,
        original_segments: List[TranscriptionSegment],
        filtered_segments: List[TranscriptionSegment]
    ) -> Dict[str, any]:
        """
        Get statistics about filtering operation.
        
        Args:
            original_segments: Original segments before filtering
            filtered_segments: Segments after filtering
        
        Returns:
            Dictionary with filtering statistics
        """
        original_count = len(original_segments)
        filtered_count = len(filtered_segments)
        removed_count = original_count - filtered_count
        
        original_duration = sum(s.end_time - s.start_time for s in original_segments)
        filtered_duration = sum(s.end_time - s.start_time for s in filtered_segments)
        removed_duration = original_duration - filtered_duration
        
        return {
            "original_segment_count": original_count,
            "filtered_segment_count": filtered_count,
            "removed_segment_count": removed_count,
            "removal_percentage": (removed_count / original_count * 100) if original_count > 0 else 0,
            "original_duration_seconds": original_duration,
            "filtered_duration_seconds": filtered_duration,
            "removed_duration_seconds": removed_duration,
        }
