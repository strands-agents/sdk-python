"""Shared helpers for CAMB.AI Strands tools.

Provides async client management, task polling, audio format detection,
and audio file utilities.
"""

from __future__ import annotations

import asyncio
import json
import logging
import struct
import tempfile
from os import getenv
from typing import Any

logger = logging.getLogger(__name__)


class CambHelpers:
    """Shared helper utilities for CAMB.AI tools."""

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_poll_attempts: int = 60,
        poll_interval: float = 2.0,
    ) -> None:
        self._api_key = api_key or getenv("CAMB_API_KEY")
        if not self._api_key:
            raise ValueError(
                "CAMB_API_KEY not set. Please set the CAMB_API_KEY environment variable "
                "or pass api_key to CambAIToolProvider."
            )
        self._timeout = timeout
        self._max_poll_attempts = max_poll_attempts
        self._poll_interval = poll_interval
        self._client: Any = None

    def _get_client(self) -> Any:
        """Return a lazily-initialized ``AsyncCambAI`` client."""
        if self._client is None:
            try:
                from camb.client import AsyncCambAI
            except ImportError as e:
                raise ImportError(
                    "The 'camb' package is required. Install it with: "
                    "pip install 'strands-agents[camb]'"
                ) from e
            logger.debug("Initializing AsyncCambAI client")
            self._client = AsyncCambAI(api_key=self._api_key, timeout=self._timeout)
        return self._client

    async def _poll_async(self, get_status_fn: Any, task_id: Any) -> Any:
        """Poll a camb.ai async task until completion."""
        logger.debug("Polling task %s (max %d attempts)", task_id, self._max_poll_attempts)
        for attempt in range(self._max_poll_attempts):
            try:
                status = await get_status_fn(task_id)
            except Exception:
                logger.warning("Poll attempt %d for task %s failed", attempt + 1, task_id, exc_info=True)
                await asyncio.sleep(self._poll_interval)
                continue
            if hasattr(status, "status"):
                val = status.status
                if val in ("completed", "SUCCESS", "complete"):
                    logger.debug("Task %s completed on attempt %d", task_id, attempt + 1)
                    return status
                if val in ("failed", "FAILED", "error", "ERROR", "TIMEOUT", "PAYMENT_REQUIRED"):
                    reason = getattr(status, "exception_reason", "") or getattr(status, "error", "Unknown error")
                    logger.error("Task %s failed with status %s: %s", task_id, val, reason)
                    raise RuntimeError(f"CAMB.AI task failed: {val}. {reason}")
                logger.debug("Task %s status: %s (attempt %d)", task_id, val, attempt + 1)
            else:
                logger.warning("Task %s returned unexpected status object: %r", task_id, status)
            await asyncio.sleep(self._poll_interval)
        raise TimeoutError(
            f"CAMB.AI task {task_id} did not complete within "
            f"{self._max_poll_attempts * self._poll_interval}s"
        )

    @staticmethod
    def _detect_audio_format(data: bytes, content_type: str = "") -> str:
        """Detect audio format from raw bytes or content-type header."""
        if data.startswith(b"RIFF"):
            return "wav"
        if data.startswith((b"\xff\xfb", b"\xff\xfa", b"ID3")):
            return "mp3"
        if data.startswith(b"fLaC"):
            return "flac"
        if data.startswith(b"OggS"):
            return "ogg"
        ct = content_type.lower()
        content_type_map = [
            ("wav", "wav"), ("wave", "wav"), ("mpeg", "mp3"),
            ("mp3", "mp3"), ("flac", "flac"), ("ogg", "ogg"),
        ]
        for key, fmt in content_type_map:
            if key in ct:
                return fmt
        return "pcm"

    @staticmethod
    def _add_wav_header(pcm_data: bytes) -> bytes:
        """Wrap raw PCM data with a WAV header (24kHz, mono, 16-bit)."""
        sr, ch, bps = 24000, 1, 16
        byte_rate = sr * ch * bps // 8
        block_align = ch * bps // 8
        data_size = len(pcm_data)
        header = struct.pack(
            "<4sI4s4sIHHIIHH4sI",
            b"RIFF", 36 + data_size, b"WAVE", b"fmt ", 16, 1,
            ch, sr, byte_rate, block_align, bps, b"data", data_size,
        )
        return header + pcm_data

    @staticmethod
    def _save_audio(data: bytes, suffix: str = ".wav") -> str:
        """Save audio bytes to a temp file and return the path."""
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(data)
            return f.name

    @staticmethod
    def _extract_translation(result: Any) -> str:
        """Extract translated text from an SDK result."""
        if isinstance(result, str):
            return result
        # Prefer .text attribute (SDK response objects)
        if hasattr(result, "text"):
            return str(result.text)
        # Handle iterable/streaming results (generators, lists of chunks)
        if hasattr(result, "__iter__") and not isinstance(result, bytes):
            parts: list[str] = []
            for chunk in result:
                if hasattr(chunk, "text"):
                    parts.append(chunk.text)
                elif isinstance(chunk, str):
                    parts.append(chunk)
            return "".join(parts)
        return str(result)

    @staticmethod
    def _format_transcription(transcription: Any) -> str:
        """Format a transcription result as a JSON string."""
        out: dict[str, Any] = {
            "text": getattr(transcription, "text", ""),
            "segments": [],
        }
        # Handle both .segments (newer SDK) and .transcript (older SDK)
        raw_segments = getattr(transcription, "segments", None)
        if not raw_segments:
            raw_segments = getattr(transcription, "transcript", None)
        if raw_segments:
            for seg in raw_segments:
                out["segments"].append({
                    "start": getattr(seg, "start", 0),
                    "end": getattr(seg, "end", 0),
                    "text": getattr(seg, "text", ""),
                    "speaker": getattr(seg, "speaker", None),
                })
            out["speakers"] = list({s["speaker"] for s in out["segments"] if s.get("speaker")})
        return json.dumps(out, indent=2)

    @staticmethod
    def _format_voices(voices: Any) -> str:
        """Format a list of voice objects as a JSON string."""
        out: list[dict[str, Any]] = []
        for v in voices:
            if isinstance(v, dict):
                name = v.get("voice_name", v.get("name", "Unknown"))
                out.append({"id": v.get("id"), "voice_name": name})
            else:
                vid = getattr(v, "id", None)
                name = getattr(
                    v, "voice_name", getattr(v, "name", "Unknown"),
                )
                out.append({"id": vid, "voice_name": name})
        return json.dumps(out, indent=2)

    @staticmethod
    def _format_separation(sep: Any) -> str:
        """Format an audio separation result as a JSON string."""
        out: dict[str, Any] = {
            "foreground_audio_url": getattr(sep, "foreground_audio_url", None),
            "background_audio_url": getattr(sep, "background_audio_url", None),
        }
        return json.dumps(out, indent=2)
