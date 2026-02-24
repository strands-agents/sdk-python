"""Tests for CAMB.AI Strands tools."""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from strands.tools.camb._helpers import CambHelpers
from strands.tools.camb.camb_tools import CambAIToolProvider


@pytest.fixture
def mock_api_key():
    return "test-api-key-12345"


@pytest.fixture
def helpers(mock_api_key):
    return CambHelpers(api_key=mock_api_key)


@pytest.fixture
def provider(mock_api_key):
    return CambAIToolProvider(api_key=mock_api_key)


class TestCambHelpers:
    def test_init_with_api_key(self, mock_api_key):
        h = CambHelpers(api_key=mock_api_key)
        assert h._api_key == mock_api_key

    def test_init_without_api_key_raises(self):
        with patch.dict("os.environ", {}, clear=True):
            with pytest.raises(ValueError, match="CAMB_API_KEY"):
                CambHelpers(api_key=None)

    def test_init_from_env(self):
        with patch.dict("os.environ", {"CAMB_API_KEY": "env-key"}):
            h = CambHelpers()
            assert h._api_key == "env-key"

    def test_detect_audio_format_wav(self, helpers):
        assert helpers._detect_audio_format(b"RIFF" + b"\x00" * 100) == "wav"

    def test_detect_audio_format_mp3(self, helpers):
        assert helpers._detect_audio_format(b"\xff\xfb" + b"\x00" * 100) == "mp3"

    def test_detect_audio_format_mp3_id3(self, helpers):
        assert helpers._detect_audio_format(b"ID3" + b"\x00" * 100) == "mp3"

    def test_detect_audio_format_flac(self, helpers):
        assert helpers._detect_audio_format(b"fLaC" + b"\x00" * 100) == "flac"

    def test_detect_audio_format_ogg(self, helpers):
        assert helpers._detect_audio_format(b"OggS" + b"\x00" * 100) == "ogg"

    def test_detect_audio_format_pcm(self, helpers):
        assert helpers._detect_audio_format(b"\x00" * 100) == "pcm"

    def test_detect_audio_format_content_type(self, helpers):
        assert helpers._detect_audio_format(b"\x00", "audio/mpeg") == "mp3"
        assert helpers._detect_audio_format(b"\x00", "audio/wav") == "wav"

    def test_add_wav_header(self, helpers):
        pcm = b"\x00\x01" * 100
        wav = helpers._add_wav_header(pcm)
        assert wav.startswith(b"RIFF")
        assert b"WAVE" in wav[:12]
        assert wav.endswith(pcm)

    def test_save_audio(self, helpers, tmp_path):
        data = b"fake audio data"
        path = helpers._save_audio(data, ".wav")
        assert path.endswith(".wav")
        with open(path, "rb") as f:
            assert f.read() == data

    def test_extract_translation_string(self, helpers):
        assert helpers._extract_translation("hello") == "hello"

    def test_extract_translation_object(self, helpers):
        """MagicMock with .text should return .text, not enter iterable branch."""
        obj = MagicMock()
        obj.text = "translated"
        assert helpers._extract_translation(obj) == "translated"

    def test_extract_translation_iterable(self, helpers):
        """Iterable of chunk objects with .text should be joined."""
        chunk1 = MagicMock(spec=[])
        chunk1.text = "hello "
        chunk2 = MagicMock(spec=[])
        chunk2.text = "world"
        assert helpers._extract_translation([chunk1, chunk2]) == "hello world"

    def test_extract_translation_plain_object(self, helpers):
        """Object without .text or __iter__ falls through to str()."""
        obj = object()
        result = helpers._extract_translation(obj)
        assert isinstance(result, str)

    def test_format_transcription_with_segments(self, helpers):
        """Handle SDK response with .segments attribute."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.5
        seg.text = "hello"
        seg.speaker = "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hello"
        transcription.segments = [seg]
        # Remove .transcript so .segments is used
        del transcription.transcript
        result = json.loads(helpers._format_transcription(transcription))
        assert result["text"] == "hello"
        assert len(result["segments"]) == 1
        assert result["segments"][0]["speaker"] == "SPEAKER_0"
        assert "SPEAKER_0" in result["speakers"]

    def test_format_transcription_speakers_list(self, helpers):
        """Speakers list should contain unique speaker IDs."""
        seg1 = MagicMock()
        seg1.start, seg1.end, seg1.text, seg1.speaker = 0.0, 1.0, "hi", "SPEAKER_0"
        seg2 = MagicMock()
        seg2.start, seg2.end, seg2.text, seg2.speaker = 1.0, 2.0, "hello", "SPEAKER_1"
        seg3 = MagicMock()
        seg3.start, seg3.end, seg3.text, seg3.speaker = 2.0, 3.0, "bye", "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hi hello bye"
        transcription.segments = [seg1, seg2, seg3]
        del transcription.transcript
        result = json.loads(helpers._format_transcription(transcription))
        assert sorted(result["speakers"]) == ["SPEAKER_0", "SPEAKER_1"]

    def test_format_transcription_with_transcript(self, helpers):
        """Handle SDK response with .transcript attribute (fallback)."""
        seg = MagicMock()
        seg.start = 0.0
        seg.end = 1.5
        seg.text = "hello"
        seg.speaker = "SPEAKER_0"
        transcription = MagicMock()
        transcription.text = "hello"
        transcription.transcript = [seg]
        # Remove .segments so .transcript fallback is used
        del transcription.segments
        result = json.loads(helpers._format_transcription(transcription))
        assert result["text"] == "hello"
        assert len(result["segments"]) == 1

    def test_format_voices(self, helpers):
        voices = [{"id": 1, "voice_name": "Alice"}, {"id": 2, "voice_name": "Bob"}]
        result = json.loads(helpers._format_voices(voices))
        assert len(result) == 2
        assert result[0]["id"] == 1
        assert result[0]["voice_name"] == "Alice"

    def test_format_voices_objects(self, helpers):
        v = MagicMock()
        v.id = 42
        v.voice_name = "TestVoice"
        result = json.loads(helpers._format_voices([v]))
        assert result[0]["id"] == 42
        assert result[0]["voice_name"] == "TestVoice"

    def test_format_separation(self, helpers):
        sep = MagicMock()
        sep.foreground_audio_url = "http://fg.wav"
        sep.background_audio_url = "http://bg.wav"
        result = json.loads(helpers._format_separation(sep))
        assert result["foreground_audio_url"] == "http://fg.wav"
        assert result["background_audio_url"] == "http://bg.wav"


class TestCambAIToolProvider:
    def test_init(self, provider):
        assert provider._helpers._api_key == "test-api-key-12345"

    @pytest.mark.asyncio
    async def test_load_tools_all_enabled(self, provider):
        tools = await provider.load_tools()
        assert len(tools) == 9

    @pytest.mark.asyncio
    async def test_load_tools_selective(self, mock_api_key):
        provider = CambAIToolProvider(
            api_key=mock_api_key,
            enable_tts=True,
            enable_translation=True,
            enable_transcription=False,
            enable_translated_tts=False,
            enable_voice_clone=False,
            enable_voice_list=False,
            enable_text_to_sound=False,
            enable_audio_separation=False,
            enable_voice_from_description=False,
        )
        tools = await provider.load_tools()
        assert len(tools) == 2

    def test_add_remove_consumer(self, provider):
        provider.add_consumer("agent-1")
        assert "agent-1" in provider._consumers
        provider.remove_consumer("agent-1")
        assert "agent-1" not in provider._consumers

    def test_remove_consumer_idempotent(self, provider):
        provider.remove_consumer("nonexistent")  # Should not raise

    @pytest.mark.asyncio
    async def test_tts_tool(self, provider):
        tools = await provider.load_tools()
        tts_tool = tools[0]
        assert tts_tool.tool_name == "camb_tts"

    @pytest.mark.asyncio
    async def test_translate_tool(self, provider):
        tools = await provider.load_tools()
        translate_tool = tools[1]
        assert translate_tool.tool_name == "camb_translate"

    @pytest.mark.asyncio
    async def test_list_voices_tool(self, provider):
        tools = await provider.load_tools()
        voice_list = [t for t in tools if t.tool_name == "camb_list_voices"]
        assert len(voice_list) == 1

    @pytest.mark.asyncio
    async def test_voice_from_description_tool(self, provider):
        tools = await provider.load_tools()
        vfd = [t for t in tools if t.tool_name == "camb_voice_from_description"]
        assert len(vfd) == 1

    @pytest.mark.asyncio
    async def test_all_tool_names(self, provider):
        tools = await provider.load_tools()
        names = {t.tool_name for t in tools}
        expected = {
            "camb_tts",
            "camb_translate",
            "camb_transcribe",
            "camb_translated_tts",
            "camb_clone_voice",
            "camb_list_voices",
            "camb_text_to_sound",
            "camb_audio_separation",
            "camb_voice_from_description",
        }
        assert names == expected


class TestCambHelpersPollAsync:
    @pytest.mark.asyncio
    async def test_poll_success(self, helpers):
        status_result = MagicMock()
        status_result.status = "SUCCESS"
        status_result.run_id = "run-123"
        get_status = AsyncMock(return_value=status_result)

        result = await helpers._poll_async(get_status, "task-1")
        assert result.run_id == "run-123"

    @pytest.mark.asyncio
    async def test_poll_failure(self, helpers):
        helpers._max_poll_attempts = 1
        status_result = MagicMock()
        status_result.status = "FAILED"
        status_result.error = "Something went wrong"
        get_status = AsyncMock(return_value=status_result)

        with pytest.raises(RuntimeError, match="CAMB.AI task failed"):
            await helpers._poll_async(get_status, "task-1")

    @pytest.mark.asyncio
    async def test_poll_timeout(self, helpers):
        helpers._max_poll_attempts = 2
        helpers._poll_interval = 0.01
        status_result = MagicMock()
        status_result.status = "PENDING"
        get_status = AsyncMock(return_value=status_result)

        with pytest.raises(TimeoutError, match="did not complete"):
            await helpers._poll_async(get_status, "task-1")

    @pytest.mark.asyncio
    async def test_poll_transient_error_then_success(self, helpers):
        """Transient API errors should be retried, not propagated."""
        helpers._max_poll_attempts = 3
        helpers._poll_interval = 0.01
        success_status = MagicMock()
        success_status.status = "SUCCESS"
        success_status.run_id = "run-ok"
        get_status = AsyncMock(side_effect=[ConnectionError("network"), success_status])

        result = await helpers._poll_async(get_status, "task-1")
        assert result.run_id == "run-ok"
        assert get_status.call_count == 2

    @pytest.mark.asyncio
    async def test_poll_all_transient_errors_timeout(self, helpers):
        """If every poll attempt fails with transient errors, should timeout."""
        helpers._max_poll_attempts = 2
        helpers._poll_interval = 0.01
        get_status = AsyncMock(side_effect=ConnectionError("network"))

        with pytest.raises(TimeoutError, match="did not complete"):
            await helpers._poll_async(get_status, "task-1")


class TestTranscribeErrorHandling:
    @pytest.mark.asyncio
    async def test_transcribe_file_not_found(self, provider):
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]
        result = await transcribe(language=1, audio_file_path="/nonexistent/file.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/file.wav" in body["error"]

    @pytest.mark.asyncio
    async def test_transcribe_url_download_failure(self, provider):
        tools = await provider.load_tools()
        transcribe = [t for t in tools if t.tool_name == "camb_transcribe"][0]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("404 Not Found")

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await transcribe(language=1, audio_url="https://example.com/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "Failed to download audio" in body["error"]


class TestCloneVoiceErrorHandling:
    @pytest.mark.asyncio
    async def test_clone_voice_file_not_found(self, provider):
        tools = await provider.load_tools()
        clone = [t for t in tools if t.tool_name == "camb_clone_voice"][0]
        result = await clone(voice_name="test", audio_file_path="/nonexistent/voice.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/voice.wav" in body["error"]


class TestAudioSeparationErrorHandling:
    @pytest.mark.asyncio
    async def test_audio_separation_file_not_found(self, provider):
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]
        result = await sep(audio_file_path="/nonexistent/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "/nonexistent/audio.wav" in body["error"]

    @pytest.mark.asyncio
    async def test_audio_separation_url_download_failure(self, provider):
        tools = await provider.load_tools()
        sep = [t for t in tools if t.tool_name == "camb_audio_separation"][0]

        mock_resp = MagicMock()
        mock_resp.raise_for_status.side_effect = Exception("Server error")

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            result = await sep(audio_url="https://example.com/audio.wav")
        body = json.loads(result)
        assert "error" in body
        assert "Failed to download audio" in body["error"]


class TestTranslatedTtsErrorHandling:
    @pytest.mark.asyncio
    async def test_translated_tts_no_run_id(self, provider):
        """Should return error when run_id is None."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.task_id = "task-1"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_result)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = None  # No run_id
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        mock_httpx = MagicMock()

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(
                    text="hello",
                    source_language=1,
                    target_language=2,
                )
        body = json.loads(result)
        assert "error" in body
        assert "no run_id" in body["error"]

    @pytest.mark.asyncio
    async def test_translated_tts_empty_audio(self, provider):
        """Should return error when audio data is empty."""
        tools = await provider.load_tools()
        tts = [t for t in tools if t.tool_name == "camb_translated_tts"][0]

        mock_client = MagicMock()
        mock_result = MagicMock()
        mock_result.task_id = "task-1"
        mock_client.translated_tts.create_translated_tts = AsyncMock(return_value=mock_result)

        mock_status = MagicMock()
        mock_status.status = "SUCCESS"
        mock_status.run_id = "run-123"
        mock_client.translated_tts.get_translated_tts_task_status = AsyncMock(return_value=mock_status)

        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.content = b""  # Empty audio
        mock_resp.headers = {"content-type": "audio/wav"}

        mock_http = AsyncMock()
        mock_http.get = AsyncMock(return_value=mock_resp)
        mock_http.__aenter__ = AsyncMock(return_value=mock_http)
        mock_http.__aexit__ = AsyncMock(return_value=False)

        mock_httpx = MagicMock()
        mock_httpx.AsyncClient.return_value = mock_http

        with patch.dict("sys.modules", {"httpx": mock_httpx}):
            with patch.object(provider._helpers, "_get_client", return_value=mock_client):
                result = await tts(
                    text="hello",
                    source_language=1,
                    target_language=2,
                )
        body = json.loads(result)
        assert "error" in body
        assert "no audio data" in body["error"]
