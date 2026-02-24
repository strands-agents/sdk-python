"""CAMB.AI tool provider and tool functions for Strands Agents.

Provides 9 audio/speech tools powered by camb.ai as a ``ToolProvider``.

Usage::

    from strands import Agent
    from strands.tools.camb import CambAIToolProvider

    provider = CambAIToolProvider(api_key="your-api-key")
    agent = Agent(tools=[provider])
    response = agent("Convert 'Hello world' to speech")
"""

from __future__ import annotations

import json
import logging
import tempfile
from collections.abc import Sequence
from typing import TYPE_CHECKING, Any

from ..decorator import tool
from ..tool_provider import ToolProvider
from ._helpers import CambHelpers

if TYPE_CHECKING:
    from ...types.tools import AgentTool

logger = logging.getLogger(__name__)


class CambAIToolProvider(ToolProvider):
    """Tool provider that exposes camb.ai audio/speech services as Strands tools.

    Each enabled service is returned as a decorated tool function that agents
    can call. The underlying ``camb`` SDK is imported lazily so that installing
    the extra is only required when the toolkit is actually used.

    Args:
        api_key: camb.ai API key. Falls back to ``CAMB_API_KEY`` env var.
        timeout: Request timeout in seconds.
        max_poll_attempts: Maximum number of polling attempts for async tasks.
        poll_interval: Seconds between polling attempts.
        enable_tts: Enable the text-to-speech tool.
        enable_translation: Enable the translation tool.
        enable_transcription: Enable the transcription tool.
        enable_translated_tts: Enable the translated TTS tool.
        enable_voice_clone: Enable the voice cloning tool.
        enable_voice_list: Enable the voice listing tool.
        enable_text_to_sound: Enable the text-to-sound tool.
        enable_audio_separation: Enable the audio separation tool.
        enable_voice_from_description: Enable the voice-from-description tool.
    """

    def __init__(
        self,
        api_key: str | None = None,
        timeout: float = 60.0,
        max_poll_attempts: int = 60,
        poll_interval: float = 2.0,
        enable_tts: bool = True,
        enable_translation: bool = True,
        enable_transcription: bool = True,
        enable_translated_tts: bool = True,
        enable_voice_clone: bool = True,
        enable_voice_list: bool = True,
        enable_text_to_sound: bool = True,
        enable_audio_separation: bool = True,
        enable_voice_from_description: bool = True,
    ) -> None:
        """Initialize CambAIToolProvider with configuration and enabled tools."""
        self._helpers = CambHelpers(
            api_key=api_key,
            timeout=timeout,
            max_poll_attempts=max_poll_attempts,
            poll_interval=poll_interval,
        )
        self._enable_tts = enable_tts
        self._enable_translation = enable_translation
        self._enable_transcription = enable_transcription
        self._enable_translated_tts = enable_translated_tts
        self._enable_voice_clone = enable_voice_clone
        self._enable_voice_list = enable_voice_list
        self._enable_text_to_sound = enable_text_to_sound
        self._enable_audio_separation = enable_audio_separation
        self._enable_voice_from_description = enable_voice_from_description
        self._consumers: set[Any] = set()

    async def load_tools(self, **kwargs: Any) -> Sequence[AgentTool]:
        """Load and return the enabled CAMB.AI tools."""
        tools: list[AgentTool] = []
        h = self._helpers

        if self._enable_tts:
            tools.append(self._make_tts_tool(h))
        if self._enable_translation:
            tools.append(self._make_translate_tool(h))
        if self._enable_transcription:
            tools.append(self._make_transcribe_tool(h))
        if self._enable_translated_tts:
            tools.append(self._make_translated_tts_tool(h))
        if self._enable_voice_clone:
            tools.append(self._make_clone_voice_tool(h))
        if self._enable_voice_list:
            tools.append(self._make_list_voices_tool(h))
        if self._enable_text_to_sound:
            tools.append(self._make_text_to_sound_tool(h))
        if self._enable_audio_separation:
            tools.append(self._make_audio_separation_tool(h))
        if self._enable_voice_from_description:
            tools.append(self._make_voice_from_description_tool(h))

        return tools

    def add_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Register a consumer of these tools."""
        self._consumers.add(consumer_id)

    def remove_consumer(self, consumer_id: Any, **kwargs: Any) -> None:
        """Unregister a consumer (idempotent)."""
        self._consumers.discard(consumer_id)

    # ------------------------------------------------------------------
    # Tool builders
    # ------------------------------------------------------------------

    @staticmethod
    def _make_tts_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_tts(
            text: str,
            language: str = "en-us",
            voice_id: int = 147320,
            speech_model: str = "mars-flash",
            speed: float | None = None,
            user_instructions: str | None = None,
        ) -> str:
            """Convert text to speech using camb.ai.

            Supports 140+ languages and multiple voice models. The audio is
            saved to a temporary file and the file path is returned.

            Args:
                text: Text to convert to speech (3-3000 characters).
                language: BCP-47 language code (e.g. 'en-us', 'fr-fr').
                voice_id: Voice ID. Use camb_list_voices to find voices.
                speech_model: Model: 'mars-flash', 'mars-pro', 'mars-instruct'.
                speed: Optional speech speed multiplier.
                user_instructions: Instructions for mars-instruct model.
            """
            from camb import (
                StreamTtsOutputConfiguration,
                StreamTtsVoiceSettings,
            )

            client = h._get_client()
            kwargs: dict[str, Any] = {
                "text": text,
                "language": language,
                "voice_id": voice_id,
                "speech_model": speech_model,
                "output_configuration": StreamTtsOutputConfiguration(format="wav"),
            }
            if speed is not None:
                kwargs["voice_settings"] = StreamTtsVoiceSettings(speed=speed)
            if user_instructions and speech_model == "mars-instruct":
                kwargs["user_instructions"] = user_instructions

            chunks: list[bytes] = []
            async for chunk in client.text_to_speech.tts(**kwargs):
                chunks.append(chunk)

            audio_data = b"".join(chunks)
            if not audio_data:
                return json.dumps({"error": "TTS returned no audio data"})

            path = h._save_audio(audio_data, ".wav")
            return json.dumps({"status": "success", "file_path": path})

        return camb_tts

    @staticmethod
    def _make_translate_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_translate(
            text: str,
            source_language: int,
            target_language: int,
            formality: int | None = None,
        ) -> str:
            """Translate text between 140+ languages using camb.ai.

            Provide integer language codes: 1=English, 2=Spanish, 3=French,
            4=German, 5=Italian, 6=Portuguese, 7=Dutch, 8=Russian,
            9=Japanese, 10=Korean, 11=Chinese.

            Args:
                text: Text to translate.
                source_language: Source language code (integer).
                target_language: Target language code (integer).
                formality: Optional formality level: 1=formal, 2=informal.
            """
            from camb.core.api_error import ApiError

            client = h._get_client()
            kwargs: dict[str, Any] = {
                "text": text,
                "source_language": source_language,
                "target_language": target_language,
            }
            if formality:
                kwargs["formality"] = formality

            try:
                result = await client.translation.translation_stream(**kwargs)
                translated = h._extract_translation(result)
                return json.dumps({"status": "success", "translated_text": translated})
            except ApiError as e:
                if e.status_code == 200 and e.body:
                    return json.dumps({"status": "success", "translated_text": str(e.body)})
                raise

        return camb_translate

    @staticmethod
    def _make_transcribe_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_transcribe(
            language: int,
            audio_url: str | None = None,
            audio_file_path: str | None = None,
        ) -> str:
            """Transcribe audio to text with speaker identification using camb.ai.

            Supports audio URLs or local file paths. Returns JSON with full
            transcription text, timed segments, and speaker labels.

            Args:
                language: Language code (integer). 1=English, 2=Spanish, etc.
                audio_url: URL of the audio file to transcribe.
                audio_file_path: Local file path to the audio file.
            """
            logger.debug("camb_transcribe: language=%s url=%s file=%s", language, audio_url, audio_file_path)
            client = h._get_client()
            kwargs: dict[str, Any] = {"language": language}

            if audio_url:
                try:
                    import httpx
                except ImportError:
                    return json.dumps({"error": "The 'httpx' package is required for URL-based operations."})

                try:
                    async with httpx.AsyncClient() as http:
                        resp = await http.get(audio_url)
                        resp.raise_for_status()
                except Exception as e:
                    logger.error("Failed to download audio from %s: %s", audio_url, e)
                    return json.dumps({"error": f"Failed to download audio: {e}"})
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as f:
                    kwargs["media_file"] = f
                    result = await client.transcription.create_transcription(**kwargs)
            elif audio_file_path:
                try:
                    f = open(audio_file_path, "rb")
                except (FileNotFoundError, OSError) as e:
                    logger.error("Failed to open audio file %s: %s", audio_file_path, e)
                    return json.dumps({"error": f"File not found: {audio_file_path}"})
                with f:
                    kwargs["media_file"] = f
                    result = await client.transcription.create_transcription(**kwargs)
            else:
                return json.dumps({"error": "Provide either audio_url or audio_file_path"})

            task_id = result.task_id
            status = await h._poll_async(
                client.transcription.get_transcription_task_status, task_id
            )
            transcription = await client.transcription.get_transcription_result(status.run_id)
            return h._format_transcription(transcription)

        return camb_transcribe

    @staticmethod
    def _make_translated_tts_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_translated_tts(
            text: str,
            source_language: int,
            target_language: int,
            voice_id: int = 147320,
            formality: int | None = None,
        ) -> str:
            """Translate text and convert to speech in one step using camb.ai.

            Returns the file path to the generated audio file.

            Args:
                text: Text to translate and speak.
                source_language: Source language code (integer).
                target_language: Target language code (integer).
                voice_id: Voice ID for TTS output.
                formality: Optional formality: 1=formal, 2=informal.
            """
            try:
                import httpx
            except ImportError:
                return json.dumps({"error": "The 'httpx' package is required for URL-based operations."})

            logger.debug("camb_translated_tts called for source=%s target=%s", source_language, target_language)
            client = h._get_client()
            kwargs: dict[str, Any] = {
                "text": text,
                "voice_id": voice_id,
                "source_language": source_language,
                "target_language": target_language,
            }
            if formality:
                kwargs["formality"] = formality

            result = await client.translated_tts.create_translated_tts(**kwargs)
            status = await h._poll_async(
                client.translated_tts.get_translated_tts_task_status, result.task_id
            )

            run_id = getattr(status, "run_id", None)
            if not run_id:
                logger.error("Translated TTS task returned no run_id")
                return json.dumps({"error": "Translated TTS task returned no run_id"})

            content_type = ""
            audio_data = b""
            url = f"https://client.camb.ai/apis/tts-result/{run_id}"
            async with httpx.AsyncClient() as http:
                resp = await http.get(url, headers={"x-api-key": h._api_key or ""})
                if resp.status_code == 200:
                    audio_data = resp.content
                    content_type = resp.headers.get("content-type", "")

            if not audio_data:
                logger.error("Translated TTS returned no audio data for run_id=%s", run_id)
                return json.dumps({"error": "Translated TTS returned no audio data"})

            fmt = h._detect_audio_format(audio_data, content_type)
            if fmt == "pcm":
                audio_data = h._add_wav_header(audio_data)
                fmt = "wav"

            ext_map = {"wav": ".wav", "mp3": ".mp3", "flac": ".flac", "ogg": ".ogg"}
            path = h._save_audio(audio_data, ext_map.get(fmt, ".wav"))
            return json.dumps({"status": "success", "file_path": path})

        return camb_translated_tts

    @staticmethod
    def _make_clone_voice_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_clone_voice(
            voice_name: str,
            audio_file_path: str,
            gender: int = 0,
            description: str | None = None,
            age: int | None = None,
            language: int | None = None,
        ) -> str:
            """Clone a voice from an audio sample using camb.ai.

            Creates a custom voice from a 2+ second audio sample that can be
            used with camb_tts and camb_translated_tts.

            Args:
                voice_name: Name for the new cloned voice.
                audio_file_path: Path to audio file (minimum 2 seconds).
                gender: Gender: 0=Not Specified, 1=Male, 2=Female, 9=N/A.
                description: Optional description of the voice.
                age: Optional age of the voice.
                language: Optional language code for the voice.
            """
            logger.debug("camb_clone_voice called with name=%s, file=%s", voice_name, audio_file_path)
            client = h._get_client()
            try:
                f = open(audio_file_path, "rb")
            except (FileNotFoundError, OSError) as e:
                logger.error("Failed to open audio file %s: %s", audio_file_path, e)
                return json.dumps({"error": f"File not found: {audio_file_path}"})
            with f:
                kwargs: dict[str, Any] = {
                    "voice_name": voice_name,
                    "gender": gender,
                    "file": f,
                }
                if description:
                    kwargs["description"] = description
                if age:
                    kwargs["age"] = age
                if language:
                    kwargs["language"] = language
                result = await client.voice_cloning.create_custom_voice(**kwargs)

            return json.dumps({
                "voice_id": getattr(result, "voice_id", getattr(result, "id", None)),
                "voice_name": voice_name,
                "status": "created",
            }, indent=2)

        return camb_clone_voice

    @staticmethod
    def _make_list_voices_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_list_voices() -> str:
            """List all available voices from camb.ai.

            Returns voice IDs and names. Use the voice ID with camb_tts or
            camb_translated_tts.
            """
            client = h._get_client()
            voices = await client.voice_cloning.list_voices()
            return h._format_voices(voices)

        return camb_list_voices

    @staticmethod
    def _make_text_to_sound_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_text_to_sound(
            prompt: str,
            duration: float | None = None,
            audio_type: str | None = None,
        ) -> str:
            """Generate sounds, music, or soundscapes from text descriptions using camb.ai.

            Describe the sound or music you want and the tool will generate it.
            Returns the file path to the generated audio file.

            Args:
                prompt: Description of the sound or music to generate.
                duration: Optional duration in seconds.
                audio_type: Optional type: 'music' or 'sound'.
            """
            client = h._get_client()
            kwargs: dict[str, Any] = {"prompt": prompt}
            if duration:
                kwargs["duration"] = duration
            if audio_type:
                kwargs["audio_type"] = audio_type

            result = await client.text_to_audio.create_text_to_audio(**kwargs)
            status = await h._poll_async(
                client.text_to_audio.get_text_to_audio_status, result.task_id
            )

            chunks: list[bytes] = []
            async for chunk in client.text_to_audio.get_text_to_audio_result(status.run_id):
                chunks.append(chunk)

            audio_data = b"".join(chunks)
            if not audio_data:
                return json.dumps({"error": "Text-to-sound returned no audio data"})

            path = h._save_audio(audio_data, ".wav")
            return json.dumps({"status": "success", "file_path": path})

        return camb_text_to_sound

    @staticmethod
    def _make_audio_separation_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_audio_separation(
            audio_url: str | None = None,
            audio_file_path: str | None = None,
        ) -> str:
            """Separate vocals/speech from background audio using camb.ai.

            Provide either an audio URL or a local file path. Returns JSON with
            URLs to the separated vocals and background audio files.

            Args:
                audio_url: URL of the audio file to separate.
                audio_file_path: Local file path to the audio file.
            """
            logger.debug("camb_audio_separation called with url=%s, file=%s", audio_url, audio_file_path)
            client = h._get_client()
            kwargs: dict[str, Any] = {}

            if audio_url:
                try:
                    import httpx
                except ImportError:
                    return json.dumps({"error": "The 'httpx' package is required for URL-based operations."})

                try:
                    async with httpx.AsyncClient() as http:
                        resp = await http.get(audio_url)
                        resp.raise_for_status()
                except Exception as e:
                    logger.error("Failed to download audio from %s: %s", audio_url, e)
                    return json.dumps({"error": f"Failed to download audio: {e}"})
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
                    tmp.write(resp.content)
                    tmp_path = tmp.name
                with open(tmp_path, "rb") as f:
                    kwargs["media_file"] = f
                    result = await client.audio_separation.create_audio_separation(**kwargs)
            elif audio_file_path:
                try:
                    f = open(audio_file_path, "rb")
                except (FileNotFoundError, OSError) as e:
                    logger.error("Failed to open audio file %s: %s", audio_file_path, e)
                    return json.dumps({"error": f"File not found: {audio_file_path}"})
                with f:
                    kwargs["media_file"] = f
                    result = await client.audio_separation.create_audio_separation(**kwargs)
            else:
                return json.dumps({"error": "Provide either audio_url or audio_file_path"})

            status = await h._poll_async(
                client.audio_separation.get_audio_separation_status, result.task_id
            )
            sep = await client.audio_separation.get_audio_separation_run_info(status.run_id)
            return h._format_separation(sep)

        return camb_audio_separation

    @staticmethod
    def _make_voice_from_description_tool(h: CambHelpers) -> AgentTool:
        @tool
        async def camb_voice_from_description(
            text: str,
            voice_description: str,
        ) -> str:
            """Generate a synthetic voice from a text description using camb.ai.

            Creates a new voice based on a detailed description and generates
            preview audio samples. Returns JSON with preview URLs.

            Args:
                text: Sample text the generated voice should speak.
                voice_description: Detailed description of the desired voice.
            """
            client = h._get_client()
            result = await client.text_to_voice.create_text_to_voice(
                text=text,
                voice_description=voice_description,
            )
            status = await h._poll_async(
                client.text_to_voice.get_text_to_voice_status,
                result.task_id,
            )
            voice_result = await client.text_to_voice.get_text_to_voice_result(
                status.run_id,
            )
            out = {
                "previews": getattr(voice_result, "previews", []),
                "status": "completed",
            }
            return json.dumps(out, indent=2)

        return camb_voice_from_description
