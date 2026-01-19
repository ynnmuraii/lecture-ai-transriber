"""
Unit tests for HallucinationFilter

Tests the filtering of common Whisper hallucinations including:
- Blacklist matching
- Pattern detection
- Compression ratio filtering
- Log probability filtering
"""

import pytest
from backend.core.processing.hallucination_filter import (
    HallucinationFilter,
    HallucinationFilterConfig
)
from backend.core.models.data_models import TranscriptionSegment


class TestHallucinationFilter:
    """Test suite for HallucinationFilter."""
    
    def test_filter_initialization(self):
        """Test that filter initializes with default config."""
        filter = HallucinationFilter()
        assert filter.config is not None
        assert len(filter.blacklist) > 0
        assert len(filter.compiled_patterns) > 0
    
    def test_filter_initialization_with_custom_config(self):
        """Test filter initialization with custom configuration."""
        config = HallucinationFilterConfig(
            compression_ratio_threshold=2.0,
            logprob_threshold=-1.0,
            custom_blacklist=["custom phrase"]
        )
        filter = HallucinationFilter(config)
        assert filter.config.compression_ratio_threshold == 2.0
        assert filter.config.logprob_threshold == -1.0
        assert "custom phrase" in filter.blacklist
    
    def test_blacklist_filtering_russian(self):
        """Test filtering of Russian YouTube artifacts."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("Это нормальный текст лекции", 0.0, 5.0, 0.9),
            TranscriptionSegment("Спасибо за просмотр", 5.0, 7.0, 0.9),
            TranscriptionSegment("Подписывайтесь на канал", 7.0, 9.0, 0.9),
            TranscriptionSegment("Продолжаем лекцию", 9.0, 12.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 2
        assert filtered[0].text == "Это нормальный текст лекции"
        assert filtered[1].text == "Продолжаем лекцию"
    
    def test_blacklist_filtering_english(self):
        """Test filtering of English YouTube artifacts."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("This is normal lecture content", 0.0, 5.0, 0.9),
            TranscriptionSegment("Thanks for watching", 5.0, 7.0, 0.9),
            TranscriptionSegment("Don't forget to subscribe", 7.0, 9.0, 0.9),
            TranscriptionSegment("Back to the lecture", 9.0, 12.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 2
        assert filtered[0].text == "This is normal lecture content"
        assert filtered[1].text == "Back to the lecture"
    
    def test_pattern_filtering_urls(self):
        """Test filtering of URLs and links."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("Visit https://example.com for more", 0.0, 3.0, 0.9),
            TranscriptionSegment("Check out www.example.com", 3.0, 6.0, 0.9),
            TranscriptionSegment("Normal lecture content", 6.0, 9.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 1
        assert filtered[0].text == "Normal lecture content"
    
    def test_pattern_filtering_repetition(self):
        """Test filtering of excessive repetition."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("да да да да", 0.0, 2.0, 0.9),
            TranscriptionSegment("Normal content here", 2.0, 5.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 1
        assert filtered[0].text == "Normal content here"
    
    def test_compression_ratio_filtering(self):
        """Test filtering based on compression ratio."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("Normal text", 0.0, 2.0, 0.9),
            TranscriptionSegment("Repetitive text", 2.0, 4.0, 0.9),
            TranscriptionSegment("More normal text", 4.0, 6.0, 0.9),
        ]
        compression_ratios = [1.2, 2.5, 1.3]  # Middle one is high
        
        filtered = filter.filter_segments(segments, compression_ratios=compression_ratios)
        
        assert len(filtered) == 2
        assert filtered[0].text == "Normal text"
        assert filtered[1].text == "More normal text"
    
    def test_logprob_filtering(self):
        """Test filtering based on log probability."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("High confidence text", 0.0, 2.0, 0.9),
            TranscriptionSegment("Low confidence text", 2.0, 4.0, 0.9),
            TranscriptionSegment("Another high confidence", 4.0, 6.0, 0.9),
        ]
        logprobs = [-0.5, -1.2, -0.6]  # Middle one is low
        
        filtered = filter.filter_segments(segments, logprobs=logprobs)
        
        assert len(filtered) == 2
        assert filtered[0].text == "High confidence text"
        assert filtered[1].text == "Another high confidence"
    
    def test_empty_segments_filtered(self):
        """Test that empty segments are filtered out."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("Normal text", 0.0, 2.0, 0.9),
            TranscriptionSegment("", 2.0, 4.0, 0.9),
            TranscriptionSegment("   ", 4.0, 6.0, 0.9),
            TranscriptionSegment("More text", 6.0, 8.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 2
        assert filtered[0].text == "Normal text"
        assert filtered[1].text == "More text"
    
    def test_filter_statistics(self):
        """Test that filter statistics are calculated correctly."""
        filter = HallucinationFilter()
        original = [
            TranscriptionSegment("Text 1", 0.0, 2.0, 0.9),
            TranscriptionSegment("Спасибо за просмотр", 2.0, 4.0, 0.9),
            TranscriptionSegment("Text 2", 4.0, 6.0, 0.9),
        ]
        filtered = filter.filter_segments(original)
        
        stats = filter.get_filter_statistics(original, filtered)
        
        assert stats["original_segment_count"] == 3
        assert stats["filtered_segment_count"] == 2
        assert stats["removed_segment_count"] == 1
        assert stats["removal_percentage"] == pytest.approx(33.33, rel=0.1)
    
    def test_disable_specific_filters(self):
        """Test that specific filters can be disabled."""
        config = HallucinationFilterConfig(
            enable_blacklist_filter=False,
            enable_pattern_filter=True
        )
        filter = HallucinationFilter(config)
        
        segments = [
            TranscriptionSegment("Спасибо за просмотр", 0.0, 2.0, 0.9),
            TranscriptionSegment("Visit https://example.com", 2.0, 4.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        # Blacklist disabled, so first segment should pass
        # Pattern enabled, so second segment should be filtered
        assert len(filtered) == 1
        assert filtered[0].text == "Спасибо за просмотр"
    
    def test_no_filtering_when_all_disabled(self):
        """Test that no filtering occurs when all filters are disabled."""
        config = HallucinationFilterConfig(
            enable_blacklist_filter=False,
            enable_pattern_filter=False,
            enable_compression_filter=False,
            enable_logprob_filter=False
        )
        filter = HallucinationFilter(config)
        
        segments = [
            TranscriptionSegment("Спасибо за просмотр", 0.0, 2.0, 0.9),
            TranscriptionSegment("Visit https://example.com", 2.0, 4.0, 0.9),
        ]
        
        filtered = filter.filter_segments(segments)
        
        assert len(filtered) == 2
    
    def test_combined_filtering(self):
        """Test that multiple filter types work together."""
        filter = HallucinationFilter()
        segments = [
            TranscriptionSegment("Normal lecture content", 0.0, 2.0, 0.9),
            TranscriptionSegment("Спасибо за просмотр", 2.0, 4.0, 0.9),  # Blacklist
            TranscriptionSegment("Visit www.example.com", 4.0, 6.0, 0.9),  # Pattern
            TranscriptionSegment("More content", 6.0, 8.0, 0.9),
            TranscriptionSegment("Repetitive", 8.0, 10.0, 0.9),  # High compression
        ]
        compression_ratios = [1.2, 1.3, 1.4, 1.1, 2.5]
        
        filtered = filter.filter_segments(segments, compression_ratios=compression_ratios)
        
        assert len(filtered) == 2
        assert filtered[0].text == "Normal lecture content"
        assert filtered[1].text == "More content"
