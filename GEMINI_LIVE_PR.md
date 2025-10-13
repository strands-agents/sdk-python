# Add Gemini Live API Support for Bidirectional Streaming

## Description

This PR adds support for Google's Gemini Live API as a bidirectional streaming model provider, enabling real-time audio conversations with native audio input/output, image/video input, and automatic transcription.

### Key Features

**Gemini Live Model Provider (`gemini_live.py`)**
- Uses official `google-genai` SDK for robust WebSocket communication
- Native audio streaming with 16kHz input and 24kHz output
- Real-time audio transcription (both input and output)
- Image/video frame input support for multimodal conversations
- Automatic VAD-based interruption handling
- Tool calling integration
- Message history support

**Enhanced Bidirectional Streaming**
- Added `ImageInputEvent` type for sending images/video frames
- Added `TranscriptEvent` type for audio transcriptions (separate from text output)
- Extended `BidirectionalAgent.send()` to accept text, audio, and image inputs
- Updated abstract `BidirectionalModelSession` interface with `send_image_content()`

**Test Suite Enhancements**
- Updated test to support both Gemini Live and Nova Sonic
- Added camera capture for real-time video frame streaming (1 FPS)
- Demonstrates audio + video multimodal interaction
- Falls back to Nova Sonic if no Gemini API key provided

### Implementation Details

The implementation follows the same architectural patterns as Nova Sonic:
- Provider-agnostic event conversion
- Clean separation between session management and model interface
- Simplified configuration - all Gemini Live API parameters pass through directly
- Proper async/await patterns with context manager for connection lifecycle

### Configuration Example

```python
from strands.experimental.bidirectional_streaming.models.gemini_live import GeminiLiveBidirectionalModel

model = GeminiLiveBidirectionalModel(
    model_id="gemini-2.5-flash-native-audio-preview-09-2025",
    api_key="your-api-key",
    params={
        "response_modalities": ["AUDIO"],
        "input_audio_transcription": {},   # Enable input transcription
        "output_audio_transcription": {},  # Enable output transcription
    }
)
```

## Related Issues

<!-- Link to related issues using #issue-number format -->

## Documentation PR

<!-- Link to related associated PR in the agent-docs repo -->

## Type of Change

New feature

## Testing

### How have you tested the change?

- [x] Tested real-time audio conversations with Gemini Live API
- [x] Verified audio transcription (input and output) works correctly
- [x] Tested image/video frame streaming from camera
- [x] Verified tool calling integration
- [x] Tested message history support
- [x] Confirmed interruption handling via VAD
- [x] Verified fallback to Nova Sonic when no API key provided
- [x] Ran `hatch fmt` for code formatting

### Test Environment
- Python 3.12+
- Dependencies: `google-genai`, `pyaudio`, `opencv-python`, `pillow`
- Tested with `GOOGLE_AI_API_KEY` environment variable

### Files Changed
1. **New**: `src/strands/experimental/bidirectional_streaming/models/gemini_live.py` (501 lines)
2. **Modified**: `src/strands/experimental/bidirectional_streaming/agent/agent.py` - Added image input support
3. **Modified**: `src/strands/experimental/bidirectional_streaming/models/bidirectional_model.py` - Added abstract `send_image_content()` method
4. **Modified**: `src/strands/experimental/bidirectional_streaming/models/novasonic.py` - Added stub for image input (not supported)
5. **Modified**: `src/strands/experimental/bidirectional_streaming/models/__init__.py` - Export Gemini Live model classes
6. **Modified**: `src/strands/experimental/bidirectional_streaming/types/bidirectional_streaming.py` - Added `ImageInputEvent` and `TranscriptEvent` types
7. **Modified**: `src/strands/experimental/bidirectional_streaming/types/__init__.py` - Export new event types
8. **Modified**: `src/strands/experimental/bidirectional_streaming/tests/test_bidirectional_streaming.py` - Enhanced test with Gemini Live and camera support

Verify that the changes do not break functionality or introduce warnings in consuming repositories: agents-docs, agents-tools, agents-cli

- [ ] I ran `hatch run prepare`

## Checklist

- [x] I have read the CONTRIBUTING document
- [x] I have added any necessary tests that prove my fix is effective or my feature works
- [ ] I have updated the documentation accordingly
- [ ] I have added an appropriate example to the documentation to outline the feature, or no new docs are needed
- [x] My changes generate no new warnings
- [x] Any dependent changes have been merged and published

---

By submitting this pull request, I confirm that you can use, modify, copy, and redistribute this contribution, under the terms of your choice.
