"""Integration test for all 9 CAMB AI tools in Strands SDK.

Runs each tool against the live camb.ai API. Requires CAMB_API_KEY and
CAMB_AUDIO_SAMPLE environment variables (loaded from .env at repo root).

Usage::

    source .venv/bin/activate
    python examples/test_camb_tools.py
"""

from __future__ import annotations

import asyncio
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from dotenv import load_dotenv

# Load .env from repo root
load_dotenv(Path(__file__).resolve().parents[1] / ".env")

import httpx  # noqa: E402

from strands.tools.camb import CambAIToolProvider  # noqa: E402

API_KEY = os.environ.get("CAMB_API_KEY")
if not API_KEY:
    raise RuntimeError("Set CAMB_API_KEY environment variable to run examples")

AUDIO_SAMPLE = os.environ.get("CAMB_AUDIO_SAMPLE")
if not AUDIO_SAMPLE:
    raise RuntimeError(
        "Set CAMB_AUDIO_SAMPLE environment variable to a local audio file path"
    )


def play(path: str) -> None:
    """Play an audio file with afplay (macOS)."""
    if sys.platform == "darwin":
        print(f"  Playing: {path}")
        subprocess.run(["afplay", path], check=False)
    else:
        print(f"  Audio file at: {path} (afplay not available)")


def play_url(url: str, label: str = "") -> None:
    """Download and play an audio URL."""
    resp = httpx.get(url)
    with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
        f.write(resp.content)
        path = f.name
    if label:
        print(f"  {label}")
    play(path)


# Build the provider
provider = CambAIToolProvider(api_key=API_KEY)


async def _load_tools() -> dict:
    tools = await provider.load_tools()
    return {t.tool_name: t for t in tools}


async def _call(tools: dict, name: str, **kwargs) -> str:
    """Call a tool by name and return its string result."""
    tool_fn = tools[name]
    return await tool_fn._tool_func(**kwargs)


# ------------------------------------------------------------------
# Tests
# ------------------------------------------------------------------


async def test_tts(tools: dict) -> None:
    """1. Text-to-Speech: convert text to audio."""
    result = await _call(tools, "camb_tts", text="Hello from CAMB AI and Strands SDK!")
    data = json.loads(result)
    print(f"  Result: {data}")
    assert data.get("status") == "success"
    play(data["file_path"])


async def test_translation(tools: dict) -> None:
    """2. Translation: translate text between languages."""
    result = await _call(
        tools,
        "camb_translate",
        text="Hello, how are you?",
        source_language=1,
        target_language=2,
    )
    data = json.loads(result)
    print(f"  Result: {data}")
    assert data.get("status") == "success"
    assert len(data.get("translated_text", "")) > 0


async def test_voice_list(tools: dict) -> None:
    """3. Voice List: list available voices."""
    result = await _call(tools, "camb_list_voices")
    print(f"  Voices (first 200 chars): {result[:200]}")
    assert "id" in result


async def test_transcription(tools: dict) -> None:
    """4. Transcription: transcribe audio from local file."""
    result = await _call(
        tools, "camb_transcribe", language=1, audio_file_path=AUDIO_SAMPLE
    )
    print(f"  Transcription (first 300 chars): {result[:300]}")
    assert "text" in result


async def test_translated_tts(tools: dict) -> None:
    """5. Translated TTS: translate and speak in one step."""
    result = await _call(
        tools,
        "camb_translated_tts",
        text="Hello, how are you?",
        source_language=1,
        target_language=2,
    )
    data = json.loads(result)
    print(f"  Result: {data}")
    assert data.get("status") == "success"
    play(data["file_path"])


async def test_text_to_sound(tools: dict) -> None:
    """6. Text-to-Sound: generate audio from a description."""
    result = await _call(
        tools,
        "camb_text_to_sound",
        prompt="gentle rain on a rooftop",
        duration=5.0,
        audio_type="sound",
    )
    data = json.loads(result)
    print(f"  Result: {data}")
    assert data.get("status") == "success"
    play(data["file_path"])


async def test_voice_clone(tools: dict) -> None:
    """7. Voice Clone: clone a voice from an audio sample."""
    result = await _call(
        tools,
        "camb_clone_voice",
        voice_name="test_clone_strands",
        audio_file_path=AUDIO_SAMPLE,
        gender=2,
    )
    data = json.loads(result)
    print(f"  Result: {data}")
    assert data.get("status") == "created"
    # Speak with the cloned voice
    voice_id = data["voice_id"]
    print(f"  Speaking with cloned voice (id: {voice_id})...")
    tts_result = await _call(
        tools,
        "camb_tts",
        text="Hello! This is the cloned voice from CAMB AI and Strands.",
        voice_id=voice_id,
    )
    tts_data = json.loads(tts_result)
    play(tts_data["file_path"])


async def test_audio_separation(tools: dict) -> None:
    """8. Audio Separation: separate vocals from background."""
    result = await _call(
        tools, "camb_audio_separation", audio_file_path=AUDIO_SAMPLE
    )
    print(f"  Result: {result}")
    data = json.loads(result)
    assert data.get("foreground_audio_url") or data.get("background_audio_url")


async def test_voice_from_description(tools: dict) -> None:
    """9. Voice from Description: generate a voice from text description."""
    result = await _call(
        tools,
        "camb_voice_from_description",
        text=(
            "Hello, this is a comprehensive test of the voice generation "
            "feature from CAMB AI. We are testing whether we can create a "
            "new synthetic voice from just a text description alone."
        ),
        voice_description=(
            "A warm, friendly female voice with a slight British accent, "
            "aged around 30, professional tone suitable for narration "
            "and audiobooks, clear enunciation with a calm demeanor"
        ),
    )
    print(f"  Result (first 200 chars): {result[:200]}")
    assert "previews" in result
    data = json.loads(result)
    for i, url in enumerate(data.get("previews", [])):
        play_url(url, label=f"Preview {i + 1}")


async def main() -> None:
    tools = await _load_tools()

    tests = [
        test_tts,
        test_translation,
        test_voice_list,
        test_transcription,
        test_translated_tts,
        test_text_to_sound,
        test_voice_clone,
        test_audio_separation,
        test_voice_from_description,
    ]
    for t in tests:
        print(f"\n--- {t.__doc__} ---")
        try:
            await t(tools)
            print("  PASSED")
        except Exception as e:
            print(f"  FAILED: {e}")


if __name__ == "__main__":
    asyncio.run(main())
