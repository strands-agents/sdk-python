"""Tests for memory management configuration."""

from strands.agent.memory.config import MemoryCategory, MemoryConfig, MemoryThresholds


def test_memory_category_enum():
    """Test MemoryCategory enum values."""
    assert MemoryCategory.ACTIVE.value == "active"
    assert MemoryCategory.CACHED.value == "cached"
    assert MemoryCategory.ARCHIVED.value == "archived"
    assert MemoryCategory.METADATA.value == "metadata"


def test_memory_thresholds_defaults():
    """Test MemoryThresholds default values."""
    thresholds = MemoryThresholds()

    assert thresholds.active_memory_limit == 8192
    assert thresholds.cached_memory_limit == 32768
    assert thresholds.total_memory_limit == 131072
    assert thresholds.cleanup_threshold == 0.8
    assert thresholds.emergency_threshold == 0.95
    assert thresholds.cache_ttl == 3600
    assert thresholds.archive_after == 86400
    assert thresholds.cleanup_ratio == 0.3
    assert thresholds.emergency_cleanup_ratio == 0.5


def test_memory_thresholds_custom_values():
    """Test MemoryThresholds with custom values."""
    thresholds = MemoryThresholds(
        active_memory_limit=4096,
        cached_memory_limit=16384,
        total_memory_limit=65536,
        cleanup_threshold=0.7,
        emergency_threshold=0.9,
    )

    assert thresholds.active_memory_limit == 4096
    assert thresholds.cached_memory_limit == 16384
    assert thresholds.total_memory_limit == 65536
    assert thresholds.cleanup_threshold == 0.7
    assert thresholds.emergency_threshold == 0.9


def test_memory_config_defaults():
    """Test MemoryConfig default values."""
    config = MemoryConfig()

    assert config.enable_categorization is True
    assert config.enable_lifecycle is True
    assert config.enable_metrics is True
    assert config.enable_archival is True
    assert config.cleanup_strategy == "lru"
    assert config.strict_validation is True
    assert config.thresholds is not None
    assert isinstance(config.thresholds, MemoryThresholds)


def test_memory_config_with_custom_thresholds():
    """Test MemoryConfig with custom thresholds."""
    thresholds = MemoryThresholds(active_memory_limit=2048)
    config = MemoryConfig(thresholds=thresholds)

    assert config.thresholds.active_memory_limit == 2048
    assert config.thresholds.cached_memory_limit == 32768  # Default value


def test_memory_config_conservative():
    """Test conservative memory configuration."""
    config = MemoryConfig.conservative()

    assert config.thresholds.active_memory_limit == 4096
    assert config.thresholds.cached_memory_limit == 16384
    assert config.thresholds.total_memory_limit == 65536
    assert config.thresholds.cleanup_threshold == 0.7
    assert config.thresholds.cleanup_ratio == 0.4


def test_memory_config_aggressive():
    """Test aggressive memory configuration."""
    config = MemoryConfig.aggressive()

    assert config.thresholds.active_memory_limit == 16384
    assert config.thresholds.cached_memory_limit == 65536
    assert config.thresholds.total_memory_limit == 262144
    assert config.thresholds.cleanup_threshold == 0.9
    assert config.thresholds.cleanup_ratio == 0.2


def test_memory_config_minimal():
    """Test minimal memory configuration."""
    config = MemoryConfig.minimal()

    assert config.enable_lifecycle is False
    assert config.enable_metrics is False
    assert config.enable_archival is False
    assert config.enable_categorization is True  # Still enabled by default

    assert config.thresholds.active_memory_limit == 2048
    assert config.thresholds.cached_memory_limit == 8192
    assert config.thresholds.total_memory_limit == 32768


def test_memory_config_custom_features():
    """Test MemoryConfig with custom feature toggles."""
    config = MemoryConfig(
        enable_categorization=False,
        enable_lifecycle=False,
        enable_metrics=False,
        enable_archival=False,
        cleanup_strategy="fifo",
        strict_validation=False,
    )

    assert config.enable_categorization is False
    assert config.enable_lifecycle is False
    assert config.enable_metrics is False
    assert config.enable_archival is False
    assert config.cleanup_strategy == "fifo"
    assert config.strict_validation is False


def test_memory_config_post_init():
    """Test MemoryConfig __post_init__ method."""
    # Test with None thresholds (should create default)
    config = MemoryConfig(thresholds=None)
    assert config.thresholds is not None
    assert isinstance(config.thresholds, MemoryThresholds)

    # Test with provided thresholds (should keep them)
    custom_thresholds = MemoryThresholds(active_memory_limit=1024)
    config = MemoryConfig(thresholds=custom_thresholds)
    assert config.thresholds is custom_thresholds
    assert config.thresholds.active_memory_limit == 1024
