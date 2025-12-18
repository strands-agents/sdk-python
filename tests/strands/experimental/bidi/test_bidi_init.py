"""Test import error handling for bidi module."""

import sys
from unittest.mock import patch

import pytest


def test_bidi_audio_io_import_from_bidi_module_without_pyaudio():
    """Test that importing and instantiating BidiAudioIO from bidi module raises helpful error without pyaudio."""
    # Mock pyaudio as missing
    with patch.dict(sys.modules, {"pyaudio": None}):
        # Remove pyaudio from sys.modules to simulate it not being installed
        sys.modules.pop("pyaudio", None)

        # Import succeeds via lazy loading, but instantiation should raise
        from strands.experimental.bidi import BidiAudioIO

        with pytest.raises(ImportError) as exc_info:
            BidiAudioIO()

        # Verify the error message is helpful
        assert "BidiAudioIO requires pyaudio" in str(exc_info.value)
        assert "pip install strands-agents[bidi-io]" in str(exc_info.value)
