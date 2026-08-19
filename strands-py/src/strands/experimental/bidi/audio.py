"""Microphone audio processing for bidirectional streaming.

Provides acoustic echo cancellation, noise suppression, and automatic gain control for microphone input,
backed by pywebrtc-audio (pip install strands-agents[bidi-aec]). Pass an ``AudioProcessorConfig`` to
``BidiAudioIO`` to enable it. When echo cancellation is enabled, the agent's speaker output is used as a
reference signal to cancel echo from the microphone input so the model does not hear its own voice.
"""

import logging
import queue
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

__all__ = ["AudioProcessorConfig"]

logger = logging.getLogger(__name__)

_MAX_STREAM_DELAY_MS = 1000
"""Upper bound for the stream delay hint; values beyond ~1s are meaningless for AEC alignment."""


@dataclass
class AudioProcessorConfig:
    """Configuration for microphone audio processing.

    Passing an instance of this config to ``BidiAudioIO`` enables microphone processing via WebRTC. Requires
    pywebrtc-audio (pip install strands-agents[bidi-aec]).

    Processing always applies noise suppression and automatic gain control to the microphone signal. Echo
    cancellation additionally removes the agent's own speaker audio from the mic input; it needs the speaker
    signal as a reference and therefore only works when the same ``BidiAudioIO`` produces both ``input()`` and
    ``output()``. Disable it for setups with no acoustic echo (for example a headset) while still benefiting
    from noise suppression and gain control.

    Attributes:
        echo_cancellation: Cancel the agent's own speaker audio from the mic input.
        stream_delay_ms: Playback-to-capture delay hint in ms for AEC. 0 lets AEC3 auto-estimate the delay.
            Advanced tuning knob; only set a non-zero value if echo cancellation is measurably failing on
            hardware with large or fixed playback-to-capture latency (e.g. Bluetooth).
    """

    echo_cancellation: bool = True
    stream_delay_ms: int = 0

    def __post_init__(self) -> None:
        """Validate configuration values.

        Raises:
            ValueError: If stream_delay_ms is out of the [0, 1000] range.
        """
        if not 0 <= self.stream_delay_ms <= _MAX_STREAM_DELAY_MS:
            raise ValueError(f"stream_delay_ms=<{self.stream_delay_ms}> | must be between 0 and {_MAX_STREAM_DELAY_MS}")


class _AudioProcessor:
    """Owns microphone audio processing: the far-end reference buffer, the WebRTC processor, and resampling.

    Applies noise suppression and automatic gain control, and — when echo cancellation is enabled — cancels
    the agent's own speaker audio from the mic input using the played-back audio as a reference. A single
    instance is shared between the input and output streams of a ``BidiAudioIO``.

    The underlying pywebrtc-audio processor is not thread-safe; ``process`` is only ever called from the input
    path and never concurrently. The reference buffer is a thread-safe queue.
    """

    _SUPPORTED_SAMPLE_RATES = (16000, 32000, 48000)
    """Sample rates supported by pywebrtc-audio's AudioProcessor."""

    _FRAME_DURATION_MS = 10
    """WebRTC audio processing operates on 10ms frames."""

    _DEFAULT_MAX_REF_FRAMES = 100
    """Default reference/mic buffer bound in frames (~1s at 10ms/frame). Both buffers share this bound so that
    under a sustained input stall they evict oldest data in lockstep, keeping the newest mic frame paired with
    the newest reference frame instead of letting the pairing drift apart."""

    def __init__(self, config: AudioProcessorConfig, max_ref_frames: int | None = None) -> None:
        """Initialize the processor.

        Args:
            config: Audio processing configuration.
            max_ref_frames: Maximum number of reference frames to retain before dropping oldest.
                Defaults to ``_DEFAULT_MAX_REF_FRAMES``.
        """
        self._config = config
        self._max_ref_frames = max_ref_frames if max_ref_frames is not None else self._DEFAULT_MAX_REF_FRAMES
        self._ref_buffer: queue.Queue[bytes] = queue.Queue(maxsize=self._max_ref_frames)
        self._processor: Any = None
        self._input_rate: int | None = None
        self._output_rate: int | None = None

    def configure(self, input_rate: int, output_rate: int, num_channels: int) -> None:
        """Build the underlying WebRTC processor for the given audio format.

        Args:
            input_rate: Microphone sample rate in Hz. Must be a supported rate.
            output_rate: Speaker sample rate in Hz (reference is resampled to input_rate if different).
            num_channels: Number of audio channels.

        Raises:
            ImportError: If pywebrtc-audio is not installed.
            ValueError: If input_rate is not supported by pywebrtc-audio.
        """
        if input_rate not in self._SUPPORTED_SAMPLE_RATES:
            raise ValueError(
                f"input_rate=<{input_rate}> | audio processing supports sample rates "
                f"{self._SUPPORTED_SAMPLE_RATES}. Configure the model's audio input_rate accordingly."
            )

        from pywebrtc_audio import AudioProcessor

        # Discard any reference frames left over from a previous session so a restart never pairs stale
        # far-end audio with fresh mic input. Safe without a lock: configure() runs before the PyAudio
        # stream and processing threads exist.
        self._drain(self._ref_buffer)

        self._input_rate = input_rate
        self._output_rate = output_rate
        # Noise suppression and automatic gain control are always on when processing is enabled; disabling
        # them is an expert-only tuning case not exposed on the public config.
        self._processor = AudioProcessor(
            sample_rate=input_rate,
            num_channels=num_channels,
            echo_cancellation=self._config.echo_cancellation,
            noise_suppression=True,
            auto_gain_control=True,
            stream_delay_ms=self._config.stream_delay_ms,
        )
        logger.debug(
            "input_rate=<%d>, output_rate=<%d>, channels=<%d> | audio processor started",
            input_rate,
            output_rate,
            num_channels,
        )

    def record_playback(self, frame: bytes) -> None:
        """Record a played speaker frame as the far-end reference.

        Called from the output stream callback at the moment audio exits the speaker — the correct temporal
        alignment point for echo cancellation. The frame is resampled to the input rate when the speaker and
        mic rates differ. Drops the oldest reference frame on overflow.

        No-op when echo cancellation is disabled, since no reference signal is needed.

        Args:
            frame: PCM int16 speaker audio at the output sample rate.
        """
        if not self._config.echo_cancellation:
            return

        frame = self._resample_reference(frame)
        if self._ref_buffer.full():
            logger.debug("ref_buffer_full=<True> | echo reference overflow, dropping oldest frame")
            try:
                self._ref_buffer.get_nowait()
            except queue.Empty:
                pass
        self._ref_buffer.put_nowait(frame)

    def process_capture(self, mic_data: bytes) -> bytes:
        """Process captured mic audio: noise suppression, gain control, and echo cancellation if enabled.

        Args:
            mic_data: PCM int16 microphone audio.

        Returns:
            Cleaned PCM int16 audio of the same length. Empty input is returned unchanged (the WebRTC
            processor rejects empty frames, and ``_BidiAudioBuffer.stop`` emits an empty shutdown sentinel).
        """
        if not mic_data:
            return mic_data

        near = np.frombuffer(mic_data, dtype=np.int16)
        # With echo cancellation off there is no reference signal; pass far=None (noise suppression and
        # auto gain control still run on the mic signal alone).
        far = self._take_reference(near) if self._config.echo_cancellation else None
        cleaned: npt.NDArray[np.int16] = self._processor.process(near, far)
        return cleaned.astype(np.int16).tobytes()

    def clear_reference(self) -> None:
        """Drain the reference buffer. Called on interruption to preserve time alignment.

        The AEC filter itself is intentionally left converged — the acoustic path is unchanged by a barge-in.
        """
        self._drain(self._ref_buffer)

    def _take_reference(self, mic: npt.NDArray[np.int16]) -> npt.NDArray[np.int16]:
        """Pull a reference frame aligned to the mic frame.

        The WebRTC AudioProcessor requires a non-null far-end frame of the same length as the near-end
        (mic) frame when echo cancellation is enabled. Any shortfall — including an empty reference buffer
        because the speaker is silent — is filled with zeros (silence).

        Args:
            mic: int16 numpy array of mic samples for the current frame.

        Returns:
            int16 numpy array of reference samples, same length as ``mic``.
        """
        byte_count = mic.nbytes
        data = bytearray()
        while len(data) < byte_count:
            try:
                data.extend(self._ref_buffer.get_nowait())
            except queue.Empty:
                break

        if len(data) < byte_count:
            data.extend(b"\x00" * (byte_count - len(data)))

        return np.frombuffer(bytes(data[:byte_count]), dtype=np.int16)

    def _resample_reference(self, data: bytes) -> bytes:
        """Resample speaker audio to the input rate via linear interpolation.

        Returns data unchanged when rates match.
        """
        if self._output_rate is None or self._input_rate is None or self._output_rate == self._input_rate:
            return data

        samples = np.frombuffer(data, dtype=np.int16)
        if len(samples) == 0:
            return data

        ratio = self._input_rate / self._output_rate
        new_length = max(int(round(len(samples) * ratio)), 1)
        positions = np.linspace(0, len(samples) - 1, new_length)
        resampled = np.interp(positions, np.arange(len(samples)), samples.astype(np.float32))
        return resampled.astype(np.int16).tobytes()

    @staticmethod
    def _drain(buffer: "queue.Queue[Any]") -> None:
        """Remove all items from a queue without blocking."""
        while True:
            try:
                buffer.get_nowait()
            except queue.Empty:
                break
