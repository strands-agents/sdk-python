Send and receive audio data from devices.

Reads user audio from input device and sends agent audio to output device using PyAudio. If a user interrupts the agent, the output buffer is cleared to stop playback.

Audio configuration is provided by models that implement `AudioCapable`.

Optional microphone audio processing (acoustic echo cancellation, noise suppression, and automatic gain control) is enabled by passing `audio_processor=True` or a `BidiAudioProcessorConfig` to `BidiAudioIO`. It requires pywebrtc-audio (pip install strands-agents\[bidi-aec\]).

## BidiAudioProcessorConfig

```python
class BidiAudioProcessorConfig(TypedDict)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:38](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L38)

Configure microphone audio processing.

**Attributes**:

-   `echo_cancellation` - Cancel the agent’s own speaker audio from the mic input.
-   `stream_delay_ms` - Playback-to-capture delay hint in milliseconds for AEC. A value of 0 lets AEC3 auto-estimate the delay. Only set a non-zero value if echo cancellation is measurably failing on hardware with large or fixed playback-to-capture latency, such as Bluetooth.

## BidiAudioIOConfig

```python
class BidiAudioIOConfig(TypedDict)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:52](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L52)

Configure bidirectional audio input and output.

## \_BidiAudioBuffer

```python
class _BidiAudioBuffer()
```

Defined in: [src/strands/experimental/bidi/io/audio.py:64](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L64)

Buffer chunks of audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(size: int | None = None)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:70](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L70)

Initialize buffer settings.

**Arguments**:

-   `size` - Size of the buffer (default: unbounded).

#### start

```python
def start() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:78](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L78)

Setup buffer.

#### stop

```python
def stop() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:83](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L83)

Tear down buffer.

#### put

```python
def put(chunk: bytes) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:97](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L97)

Put data chunk into buffer.

If full, removes the oldest chunk.

#### get

```python
def get(byte_count: int | None = None) -> bytes
```

Defined in: [src/strands/experimental/bidi/io/audio.py:112](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L112)

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

Defined in: [src/strands/experimental/bidi/io/audio.py:142](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L142)

Clear the buffer.

## \_BidiAudioInput

```python
class _BidiAudioInput(BidiInput)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:152](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L152)

Handle audio input from user.

**Attributes**:

-   `_audio` - PyAudio instance for audio system access.
-   `_stream` - Audio input stream.
-   `_buffer` - Buffer for sharing audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(config: BidiAudioIOConfig, *,
             audio_processor: "_BidiAudioProcessor | None") -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:168](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L168)

Initialize input settings.

**Arguments**:

-   `config` - Audio device configuration.
-   `audio_processor` - Shared microphone audio processor.

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

Defined in: [src/strands/experimental/bidi/io/audio.py:229](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L229)

Stop input stream.

#### \_\_call\_\_

```python
async def __call__() -> BidiAudioInputEvent
```

Defined in: [src/strands/experimental/bidi/io/audio.py:242](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L242)

Read audio from input stream, applying echo cancellation if enabled.

## \_BidiAudioOutput

```python
class _BidiAudioOutput(BidiOutput)
```

Defined in: [src/strands/experimental/bidi/io/audio.py:266](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L266)

Handle audio output from bidi agent.

**Attributes**:

-   `_audio` - PyAudio instance for audio system access.
-   `_stream` - Audio output stream.
-   `_buffer` - Buffer for sharing audio data between agent and PyAudio.

#### \_\_init\_\_

```python
def __init__(config: BidiAudioIOConfig, *,
             audio_processor: "_BidiAudioProcessor | None") -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:282](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L282)

Initialize output settings.

**Arguments**:

-   `config` - Audio device configuration.
-   `audio_processor` - Shared audio processor that receives played audio for echo cancellation.

#### start

```python
async def start(agent: "BidiAgent") -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:301](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L301)

Start output stream.

**Arguments**:

-   `agent` - The BidiAgent instance, providing access to model configuration.

#### stop

```python
async def stop() -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:333](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L333)

Stop output stream.

#### \_\_call\_\_

```python
async def __call__(event: BidiOutputEvent) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:346](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L346)

Send audio to output stream.

## BidiAudioIO

```python
class BidiAudioIO()
```

Defined in: [src/strands/experimental/bidi/io/audio.py:379](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L379)

Send and receive audio data from devices using PyAudio.

Reads microphone audio via `input()` and plays agent audio via `output()`. On interruption, the playback buffer is cleared to stop the agent mid-response.

When `audio_processor=True` or a `BidiAudioProcessorConfig` is passed, the microphone signal gets audio processing and, when echo cancellation is enabled, the agent’s speaker output is used as a reference to cancel echo from the mic input. A shared processor coordinates the input and output channels, so echo cancellation only works when both come from the *same* `BidiAudioIO` instance.

Audio processing requires pywebrtc-audio (`pip install strands-agents[bidi-aec]`) and a microphone sample rate of 16000, 32000, or 48000 Hz (set via the model’s audio config).

Device audio requires PyAudio. Install the PortAudio system library, then install `strands-agents[bidi-pyaudio]`.

**Example**:

```python
from strands.experimental.bidi import BidiAudioProcessorConfig
from strands.experimental.bidi.io import BidiAudioIO

# Plain mic/speaker, no processing (a headset is recommended to avoid echo):
audio_io = BidiAudioIO()
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Full processing with defaults: echo cancellation, noise suppression, and auto gain control:
audio_io = BidiAudioIO(audio_processor=True)
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Noise suppression and auto gain control without echo cancellation (e.g. headset users):
audio_io = BidiAudioIO(audio_processor=BidiAudioProcessorConfig(echo_cancellation=False))
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])

# Processing on a specific input device:
audio_io = BidiAudioIO(audio_processor=BidiAudioProcessorConfig(), input_device_index=1)
await agent.run(inputs=[audio_io.input()], outputs=[audio_io.output()])
```

#### \_\_init\_\_

```python
def __init__(**config: Unpack[BidiAudioIOConfig]) -> None
```

Defined in: [src/strands/experimental/bidi/io/audio.py:422](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L422)

Initialize audio devices.

**Arguments**:

-   `**config` - Optional configuration:
    
    -   audio\_processor (bool | BidiAudioProcessorConfig): Set to True to enable microphone audio processing with defaults, or supply a configuration for custom options. False and None disable processing.
    -   input\_buffer\_size (int): Maximum input buffer size (default: None). Must be between 1 and 100 when echo cancellation is on; defaults to 100 so the mic and reference buffers remain aligned.
    -   input\_device\_index (int): Specific input device (default: None = system default)
    -   input\_frames\_per\_buffer (int): Input buffer size (default: 512). Must not be provided when echo cancellation is on because it is calculated from the model’s input rate.
    -   output\_buffer\_size (int): Maximum output buffer size (default: None)
    -   output\_device\_index (int): Specific output device (default: None = system default)
    -   output\_frames\_per\_buffer (int): Output buffer size (default: 512). Must not be provided when echo cancellation is on because it is calculated from the model’s output rate.

**Raises**:

-   `ImportError` - If audio processing is configured but its optional dependencies are unavailable.
-   `ValueError` - If the configuration is invalid.

#### input

```python
def input() -> _BidiAudioInput
```

Defined in: [src/strands/experimental/bidi/io/audio.py:518](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L518)

Return audio processing BidiInput.

#### output

```python
def output() -> _BidiAudioOutput
```

Defined in: [src/strands/experimental/bidi/io/audio.py:525](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/experimental/bidi/io/audio.py#L525)

Return audio processing BidiOutput.