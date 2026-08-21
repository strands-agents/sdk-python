Microphone audio processing for bidirectional streaming.

Provides acoustic echo cancellation, noise suppression, and automatic gain control for microphone input, backed by pywebrtc-audio (pip install strands-agents\[bidi-aec\]). Pass an `AudioProcessorConfig` to `BidiAudioIO` to enable it. When echo cancellation is enabled, the agent’s speaker output is used as a reference signal to cancel echo from the microphone input so the model does not hear its own voice.

## AudioProcessorConfig

```python
@dataclass
class AudioProcessorConfig()
```

Defined in: [src/strands/experimental/bidi/audio.py:26](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L26)

Configuration for microphone audio processing.

Passing an instance of this config to `BidiAudioIO` enables microphone processing via WebRTC. Requires pywebrtc-audio (pip install strands-agents\[bidi-aec\]).

Processing always applies noise suppression and automatic gain control to the microphone signal. Echo cancellation additionally removes the agent’s own speaker audio from the mic input; it needs the speaker signal as a reference and therefore only works when the same `BidiAudioIO` produces both `input()` and `output()`. Disable it for setups with no acoustic echo (for example a headset) while still benefiting from noise suppression and gain control.

**Attributes**:

-   `echo_cancellation` - Cancel the agent’s own speaker audio from the mic input.
-   `stream_delay_ms` - Playback-to-capture delay hint in ms for AEC. 0 lets AEC3 auto-estimate the delay. Advanced tuning knob; only set a non-zero value if echo cancellation is measurably failing on hardware with large or fixed playback-to-capture latency (e.g. Bluetooth).

#### \_\_post\_init\_\_

```python
def __post_init__() -> None
```

Defined in: [src/strands/experimental/bidi/audio.py:48](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L48)

Validate configuration values.

**Raises**:

-   `ValueError` - If stream\_delay\_ms is out of the \[0, 1000\] range.

## \_AudioProcessor

```python
class _AudioProcessor()
```

Defined in: [src/strands/experimental/bidi/audio.py:58](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L58)

Owns microphone audio processing: the far-end reference buffer, the WebRTC processor, and resampling.

Applies noise suppression and automatic gain control, and — when echo cancellation is enabled — cancels the agent’s own speaker audio from the mic input using the played-back audio as a reference. A single instance is shared between the input and output streams of a `BidiAudioIO`.

The underlying pywebrtc-audio processor is not thread-safe; `process` is only ever called from the input path and never concurrently. The reference buffer is a thread-safe queue.

#### \_\_init\_\_

```python
def __init__(config: AudioProcessorConfig,
             max_ref_frames: int | None = None) -> None
```

Defined in: [src/strands/experimental/bidi/audio.py:80](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L80)

Initialize the processor.

**Arguments**:

-   `config` - Audio processing configuration.
-   `max_ref_frames` - Maximum number of reference frames to retain before dropping oldest. Defaults to `_DEFAULT_MAX_REF_FRAMES`.

#### configure

```python
def configure(input_rate: int, output_rate: int, num_channels: int) -> None
```

Defined in: [src/strands/experimental/bidi/audio.py:95](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L95)

Build the underlying WebRTC processor for the given audio format.

**Arguments**:

-   `input_rate` - Microphone sample rate in Hz. Must be a supported rate.
-   `output_rate` - Speaker sample rate in Hz (reference is resampled to input\_rate if different).
-   `num_channels` - Number of audio channels.

**Raises**:

-   `ImportError` - If pywebrtc-audio is not installed.
-   `ValueError` - If input\_rate is not supported by pywebrtc-audio.

#### record\_playback

```python
def record_playback(frame: bytes) -> None
```

Defined in: [src/strands/experimental/bidi/audio.py:139](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L139)

Record a played speaker frame as the far-end reference.

Called from the output stream callback at the moment audio exits the speaker — the correct temporal alignment point for echo cancellation. The frame is resampled to the input rate when the speaker and mic rates differ. Drops the oldest reference frame on overflow.

No-op when echo cancellation is disabled, since no reference signal is needed.

**Arguments**:

-   `frame` - PCM int16 speaker audio at the output sample rate.

#### process\_capture

```python
def process_capture(mic_data: bytes) -> bytes
```

Defined in: [src/strands/experimental/bidi/audio.py:163](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L163)

Process captured mic audio: noise suppression, gain control, and echo cancellation if enabled.

**Arguments**:

-   `mic_data` - PCM int16 microphone audio.

**Returns**:

Cleaned PCM int16 audio of the same length. Empty input is returned unchanged (the WebRTC processor rejects empty frames, and `_BidiAudioBuffer.stop` emits an empty shutdown sentinel).

#### clear\_reference

```python
def clear_reference() -> None
```

Defined in: [src/strands/experimental/bidi/audio.py:183](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/audio.py#L183)

Drain the reference buffer. Called on interruption to preserve time alignment.

The AEC filter itself is intentionally left converged — the acoustic path is unchanged by a barge-in.