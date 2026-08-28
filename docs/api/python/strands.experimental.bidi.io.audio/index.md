Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent, the output buffer is cleared to stop playback.

Audio configuration is provided by the model via agent.model.config\[“audio”\].

Optional microphone audio processing (acoustic echo cancellation, noise suppression, and automatic gain control) is enabled by passing an `AudioProcessorConfig` (from `strands.experimental.bidi.audio`) to `BidiAudioIO`. It requires pywebrtc-audio (pip install strands-agents\[bidi-aec\]); see that module for the processing implementation.

## \_BidiAudioBuffer

```python
class _BidiAudioBuffer()
```

Defined in: [src/strands/experimental/bidi/io/audio.py:51](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L51)

Buffer chunks of audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(size: int | None = None)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:57](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L57)

Initialize buffer settings.

**Arguments**:

-   `size` - Size of the buffer (default: unbounded).

#### start

```python
def start() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:65](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L65)

Setup buffer.

#### stop

```python
def stop() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L70)

Tear down buffer.

#### put

```python
def put(chunk: bytes) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:88](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L88)

Put data chunk into buffer.

If full, removes the oldest chunk.

#### get

```python
def get(byte_count: int | None = None) -> bytes
```

Defined in: [src/strands/experimental/bidi/io/audio.py:103](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L103)

Get the number of bytes specified from the buffer.

**Arguments**:

-   `byte_count` - Number of bytes to get from buffer.
    
    -   If the number of bytes specified is not available, the return is padded with silence.
    -   If the number of bytes is not specified, get the first chunk put in the buffer.

**Returns**:

Specified number of bytes.

#### clear

```python
def clear() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:133](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L133)

Clear the buffer.

## \_BidiAudioInput

```python
class _BidiAudioInput(BidiInput)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:142](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L142)

Handle audio input from user.

**Attributes**:

-   `_audio` - PyAudio instance for audio system access.
-   `_stream` - Audio input stream.
-   `_buffer` - Buffer for sharing audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(config: dict[str, Any],
             processor: "_AudioProcessor | None" = None) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:158](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L158)

Extract configs.

**Arguments**:

-   `config` - Audio device configuration.
-   `processor` - Shared audio processor for echo cancellation, or None to disable processing.

#### start

```python
async def start(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:187](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L187)

Start input stream.

**Arguments**:

-   `agent` - The BidiAgent instance, providing access to model configuration.

**Raises**:

-   `ValueError` - If audio processing is enabled but the input rate is unsupported.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:232](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L232)

Stop input stream.

#### \_\_call\_\_

```python
async def __call__() -> BidiAudioInputEvent
```

Defined in: [src/strands/experimental/bidi/io/audio.py:245](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L245)

Read audio from input stream, applying echo cancellation if enabled.

## \_BidiAudioOutput

```python
class _BidiAudioOutput(BidiOutput)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:269](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L269)

Handle audio output from bidi agent.

**Attributes**:

-   `_audio` - PyAudio instance for audio system access.
-   `_stream` - Audio output stream.
-   `_buffer` - Buffer for sharing audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(config: dict[str, Any],
             processor: "_AudioProcessor | None" = None) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:285](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L285)

Extract configs.

**Arguments**:

-   `config` - Audio device configuration.
-   `processor` - Shared audio processor — output records played frames as the AEC reference.

#### start

```python
async def start(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:303](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L303)

Start output stream.

**Arguments**:

-   `agent` - The BidiAgent instance, providing access to model configuration.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:337](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L337)

Stop output stream.

#### \_\_call\_\_

```python
async def __call__(event: BidiOutputEvent) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:350](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L350)

Send audio to output stream.

## BidiAudioIO

```python
class BidiAudioIO()
```

Defined in: [src/strands/experimental/bidi/io/audio.py:383](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L383)

Send and receive audio data from devices using PyAudio.

Reads microphone audio via `input()` and plays agent audio via `output()`. On interruption, the playback buffer is cleared to stop the agent mid-response.

When an `AudioProcessorConfig` is passed as `processor`, the microphone signal gets noise suppression and automatic gain control, and (when echo cancellation is enabled) the agent’s speaker output is used as a reference to cancel echo from the mic input, preventing the model from hearing its own voice. The same processor instance is shared between the input and output channels this factory produces, so echo cancellation only works when both `input()` and `output()` come from the *same* `BidiAudioIO` instance.

Audio processing requires pywebrtc-audio (`pip install strands-agents[bidi-aec]`) and a microphone sample rate of 16000, 32000, or 48000 Hz (set via the model’s audio config).

Device audio requires PyAudio. Install the PortAudio system library, then install `strands-agents[bidi-pyaudio]`.

**Example**:

```python
from strands.experimental.bidi.audio import AudioProcessorConfig
from strands.experimental.bidi.io import BidiAudioIO

# Plain mic/speaker, no processing (a headset is recommended to avoid echo):
audio_io = BidiAudioIO()
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Full processing: echo cancellation, noise suppression, and auto gain control:
audio_io = BidiAudioIO(processor=AudioProcessorConfig())
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Noise suppression and auto gain control without echo cancellation (e.g. headset users):
audio_io = BidiAudioIO(processor=AudioProcessorConfig(echo_cancellation=False))
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Processing on a specific input device:
audio_io = BidiAudioIO(processor=AudioProcessorConfig(), input_device_index=1)
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
```

#### \_\_init\_\_

```python
def __init__(**config: Any) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:425](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L425)

Initialize audio devices.

**Arguments**:

-   `**config` - Optional configuration:
    
    -   processor (AudioProcessorConfig): Enable microphone audio processing (noise suppression, automatic gain control, and optionally echo cancellation) by passing an `AudioProcessorConfig` (from `strands.experimental.bidi.audio`). Requires pywebrtc-audio (pip install strands-agents\[bidi-aec\]). Defaults to None (processing disabled).
    -   input\_buffer\_size (int): Maximum input buffer size (default: None). Ignored when echo cancellation is on — the mic buffer is bound to the reference horizon so the two stay aligned.
    -   input\_device\_index (int): Specific input device (default: None = system default)
    -   input\_frames\_per\_buffer (int): Input buffer size (default: 512, ignored when processing is on)
    -   output\_buffer\_size (int): Maximum output buffer size (default: None)
    -   output\_device\_index (int): Specific output device (default: None = system default)
    -   output\_frames\_per\_buffer (int): Output buffer size (default: 512, ignored when processing is on)

**Raises**:

-   `ImportError` - If a processor config is set but pywebrtc-audio is not installed.

#### input

```python
def input() -> _BidiAudioInput
```

Defined in: [src/strands/experimental/bidi/io/audio.py:457](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L457)

Return audio processing BidiInput.

#### output

```python
def output() -> _BidiAudioOutput
```

Defined in: [src/strands/experimental/bidi/io/audio.py:461](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L461)

Return audio processing BidiOutput.