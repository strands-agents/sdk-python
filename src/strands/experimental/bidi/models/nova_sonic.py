"""Nova Sonic bidirectional model provider for real-time streaming conversations.

Implements the BidiModel interface for Amazon's Nova Sonic, handling the
complex event sequencing and audio processing required by Nova Sonic's
InvokeModelWithBidirectionalStream protocol.

Nova Sonic specifics:

- Hierarchical event sequences: connectionStart → promptStart → content streaming
- Base64-encoded audio format with hex encoding
- Tool execution with content containers and identifier tracking
- 8-minute connection limits with proper cleanup sequences
- Interruption detection through stopReason events
"""

import asyncio
import base64
import json
import logging
import uuid
from typing import Any, AsyncGenerator, cast

import boto3
from aws_sdk_bedrock_runtime.client import BedrockRuntimeClient, InvokeModelWithBidirectionalStreamOperationInput
from aws_sdk_bedrock_runtime.config import Config, HTTPAuthSchemeResolver, SigV4AuthScheme
from aws_sdk_bedrock_runtime.models import (
    BidirectionalInputPayloadPart,
    InvokeModelWithBidirectionalStreamInputChunk,
    ModelTimeoutException,
    ValidationException,
)
from smithy_aws_core.identity.static import StaticCredentialsResolver
from smithy_core.aio.eventstream import DuplexEventStream
from smithy_core.shapes import ShapeID

from ....types._events import ToolResultEvent, ToolUseStreamEvent
from ....types.content import Messages
from ....types.tools import ToolResult, ToolSpec, ToolUse
from .._async import stop_all
from ..types.events import (
    AudioChannel,
    AudioSampleRate,
    BidiAudioInputEvent,
    BidiAudioStreamEvent,
    BidiConnectionStartEvent,
    BidiInputEvent,
    BidiInterruptionEvent,
    BidiOutputEvent,
    BidiResponseCompleteEvent,
    BidiResponseStartEvent,
    BidiTextInputEvent,
    BidiTranscriptStreamEvent,
    BidiUsageEvent,
)
from ..types.model import AudioConfig
from .model import BidiModel, BidiModelTimeoutError

logger = logging.getLogger(__name__)

_NOVA_INFERENCE_CONFIG_KEYS = {
    "max_tokens": "maxTokens",
    "temperature": "temperature",
    "top_p": "topP",
}

NOVA_AUDIO_INPUT_CONFIG = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 16000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "audioType": "SPEECH",
    "encoding": "base64",
}

NOVA_AUDIO_OUTPUT_CONFIG = {
    "mediaType": "audio/lpcm",
    "sampleRateHertz": 16000,
    "sampleSizeBits": 16,
    "channelCount": 1,
    "voiceId": "matthew",
    "encoding": "base64",
    "audioType": "SPEECH",
}

NOVA_TEXT_CONFIG = {"mediaType": "text/plain"}
NOVA_TOOL_CONFIG = {"mediaType": "application/json"}


class BidiNovaSonicModel(BidiModel):
    """Nova Sonic implementation for bidirectional streaming.

    Combines model configuration and connection state in a single class.
    Manages Nova Sonic's complex event sequencing, audio format conversion, and
    tool execution patterns while providing the standard BidiModel interface.

    Attributes:
        _stream: open bedrock stream to nova sonic.
    """

    _stream: DuplexEventStream

    def __init__(
        self,
        model_id: str = "amazon.nova-sonic-v1:0",
        provider_config: dict[str, Any] | None = None,
        client_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize Nova Sonic bidirectional model.

        Args:
            model_id: Model identifier (default: amazon.nova-sonic-v1:0)
            provider_config: Model behavior (audio, inference settings)
            client_config: AWS authentication (boto_session OR region, not both)
            **kwargs: Reserved for future parameters.
        """
        # Store model ID
        self.model_id = model_id

        # Resolve client config with defaults
        self._client_config = self._resolve_client_config(client_config or {})

        # Resolve provider config with defaults
        self.config = self._resolve_provider_config(provider_config or {})

        # Store session and region for later use
        self._session = self._client_config["boto_session"]
        self.region = self._client_config["region"]

        # Track API-provided identifiers
        self._connection_id: str | None = None
        self._audio_content_name: str | None = None
        self._current_completion_id: str | None = None

        # Indicates if model is done generating transcript
        self._generation_stage: str | None = None

        # Ensure certain events are sent in sequence when required
        self._send_lock = asyncio.Lock()

        logger.debug("model_id=<%s> | nova sonic model initialized", model_id)

    def _resolve_client_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Resolve AWS client config (creates boto session if needed)."""
        if "boto_session" in config and "region" in config:
            raise ValueError("Cannot specify both 'boto_session' and 'region' in client_config")

        resolved = config.copy()

        # Create boto session if not provided
        if "boto_session" not in resolved:
            resolved["boto_session"] = boto3.Session()

        # Resolve region from session or use default
        if "region" not in resolved:
            resolved["region"] = resolved["boto_session"].region_name or "us-east-1"

        return resolved

    def _resolve_provider_config(self, config: dict[str, Any]) -> dict[str, Any]:
        """Merge user config with defaults (user takes precedence)."""
        default_audio: AudioConfig = {
            "input_rate": cast(AudioSampleRate, NOVA_AUDIO_INPUT_CONFIG["sampleRateHertz"]),
            "output_rate": cast(AudioSampleRate, NOVA_AUDIO_OUTPUT_CONFIG["sampleRateHertz"]),
            "channels": cast(AudioChannel, NOVA_AUDIO_INPUT_CONFIG["channelCount"]),
            "format": "pcm",
            "voice": cast(str, NOVA_AUDIO_OUTPUT_CONFIG["voiceId"]),
        }

        resolved = {
            "audio": {
                **default_audio,
                **config.get("audio", {}),
            },
            "inference": config.get("inference", {}),
        }
        return resolved

    async def start(
        self,
        system_prompt: str | None = None,
        tools: list[ToolSpec] | None = None,
        messages: Messages | None = None,
        **kwargs: Any,
    ) -> None:
        """Establish bidirectional connection to Nova Sonic.

        Args:
            system_prompt: System instructions for the model.
            tools: List of tools available to the model.
            messages: Conversation history to initialize with.
            **kwargs: Additional configuration options.

        Raises:
            RuntimeError: If user calls start again without first stopping.
        """
        if self._connection_id:
            raise RuntimeError("model already started | call stop before starting again")

        logger.debug("nova connection starting")

        self._connection_id = str(uuid.uuid4())

        # Get credentials from boto3 session (full credential chain)
        credentials = self._session.get_credentials()

        if not credentials:
            raise ValueError(
                "no AWS credentials found. configure credentials via environment variables, "
                "credential files, IAM roles, or SSO."
            )

        # Use static resolver with credentials configured as properties
        resolver = StaticCredentialsResolver()

        config = Config(
            endpoint_uri=f"https://bedrock-runtime.{self.region}.amazonaws.com",
            region=self.region,
            aws_credentials_identity_resolver=resolver,
            auth_scheme_resolver=HTTPAuthSchemeResolver(),
            auth_schemes={ShapeID("aws.auth#sigv4"): SigV4AuthScheme(service="bedrock")},
            # Configure static credentials as properties
            aws_access_key_id=credentials.access_key,
            aws_secret_access_key=credentials.secret_key,
            aws_session_token=credentials.token,
        )

        self.client = BedrockRuntimeClient(config=config)
        logger.debug("region=<%s> | nova sonic client initialized", self.region)

        client = BedrockRuntimeClient(config=config)
        self._stream = await client.invoke_model_with_bidirectional_stream(
            InvokeModelWithBidirectionalStreamOperationInput(model_id=self.model_id)
        )
        logger.debug("region=<%s> | nova sonic client initialized", self.region)

        init_events = self._build_initialization_events(system_prompt, tools, messages)
        logger.debug("event_count=<%d> | sending nova sonic initialization events", len(init_events))
        await self._send_nova_events(init_events)

        logger.info("connection_id=<%s> | nova sonic connection established", self._connection_id)

    def _build_initialization_events(
        self, system_prompt: str | None, tools: list[ToolSpec] | None, messages: Messages | None
    ) -> list[str]:
        """Build the sequence of initialization events."""
        tools = tools or []
        events = [
            self._get_connection_start_event(),
            self._get_prompt_start_event(tools),
            *self._get_system_prompt_events(system_prompt),
        ]

        # Add conversation history if provided
        if messages:
            events.extend(self._get_message_history_events(messages))
            logger.debug("message_count=<%d> | conversation history added to initialization", len(messages))

        return events

    def _log_event_type(self, nova_event: dict[str, Any]) -> None:
        """Log specific Nova Sonic event types for debugging."""
        if "usageEvent" in nova_event:
            logger.debug("usage=<%s> | nova usage event received", nova_event["usageEvent"])
        elif "textOutput" in nova_event:
            logger.debug("nova text output received")
        elif "toolUse" in nova_event:
            tool_use = nova_event["toolUse"]
            logger.debug(
                "tool_name=<%s>, tool_use_id=<%s> | nova tool use received",
                tool_use["toolName"],
                tool_use["toolUseId"],
            )
        elif "audioOutput" in nova_event:
            audio_content = nova_event["audioOutput"]["content"]
            audio_bytes = base64.b64decode(audio_content)
            logger.debug("audio_bytes=<%d> | nova audio output received", len(audio_bytes))

    async def receive(self) -> AsyncGenerator[BidiOutputEvent, None]:
        """Receive Nova Sonic events and convert to provider-agnostic format.

        Raises:
            RuntimeError: If start has not been called.
        """
        if not self._connection_id:
            raise RuntimeError("model not started | call start before receiving")

        logger.debug("nova event stream starting")
        yield BidiConnectionStartEvent(connection_id=self._connection_id, model=self.model_id)

        _, output = await self._stream.await_output()
        while True:
            try:
                event_data = await output.receive()

            except ValidationException as error:
                if "InternalErrorCode=531" in error.message:
                    # nova also times out if user is silent for 175 seconds
                    raise BidiModelTimeoutError(error.message) from error
                raise

            except ModelTimeoutException as error:
                raise BidiModelTimeoutError(error.message) from error

            if not event_data:
                continue

            nova_event = json.loads(event_data.value.bytes_.decode("utf-8"))["event"]
            self._log_event_type(nova_event)

            model_event = self._convert_nova_event(nova_event)
            if model_event:
                yield model_event

    async def send(self, content: BidiInputEvent | ToolResultEvent) -> None:
        """Unified send method for all content types. Sends the given content to Nova Sonic.

        Dispatches to appropriate internal handler based on content type.

        Args:
            content: Input event.

        Raises:
            ValueError: If content type not supported (e.g., image content).
        """
        if not self._connection_id:
            raise RuntimeError("model not started | call start before sending")

        if isinstance(content, BidiTextInputEvent):
            await self._send_text_content(content.text)
        elif isinstance(content, BidiAudioInputEvent):
            await self._send_audio_content(content)
        elif isinstance(content, ToolResultEvent):
            tool_result = content.get("tool_result")
            if tool_result:
                await self._send_tool_result(tool_result)
        else:
            raise ValueError(f"content_type={type(content)} | content not supported")

    async def _start_audio_connection(self) -> None:
        """Internal: Start audio input connection (call once before sending audio chunks)."""
        logger.debug("nova audio connection starting")
        self._audio_content_name = str(uuid.uuid4())

        # Build audio input configuration from config
        audio_input_config = {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": self.config["audio"]["input_rate"],
            "sampleSizeBits": 16,
            "channelCount": self.config["audio"]["channels"],
            "audioType": "SPEECH",
            "encoding": "base64",
        }

        audio_content_start = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._connection_id,
                        "contentName": self._audio_content_name,
                        "type": "AUDIO",
                        "interactive": True,
                        "role": "USER",
                        "audioInputConfiguration": audio_input_config,
                    }
                }
            }
        )

        await self._send_nova_events([audio_content_start])

    async def _send_audio_content(self, audio_input: BidiAudioInputEvent) -> None:
        """Internal: Send audio using Nova Sonic protocol-specific format."""
        # Start audio connection if not already active
        if not self._audio_content_name:
            await self._start_audio_connection()

        # Audio is already base64 encoded in the event
        # Send audio input event
        audio_event = json.dumps(
            {
                "event": {
                    "audioInput": {
                        "promptName": self._connection_id,
                        "contentName": self._audio_content_name,
                        "content": audio_input.audio,
                    }
                }
            }
        )

        await self._send_nova_events([audio_event])

    async def _end_audio_input(self) -> None:
        """Internal: End current audio input connection to trigger Nova Sonic processing."""
        if not self._audio_content_name:
            return

        logger.debug("nova audio connection ending")

        audio_content_end = json.dumps(
            {"event": {"contentEnd": {"promptName": self._connection_id, "contentName": self._audio_content_name}}}
        )

        await self._send_nova_events([audio_content_end])
        self._audio_content_name = None

    async def _send_text_content(self, text: str) -> None:
        """Internal: Send text content using Nova Sonic format."""
        content_name = str(uuid.uuid4())
        events = [
            self._get_text_content_start_event(content_name),
            self._get_text_input_event(content_name, text),
            self._get_content_end_event(content_name),
        ]
        await self._send_nova_events(events)

    async def _send_tool_result(self, tool_result: ToolResult) -> None:
        """Internal: Send tool result using Nova Sonic toolResult format."""
        tool_use_id = tool_result["toolUseId"]

        logger.debug("tool_use_id=<%s> | sending nova tool result", tool_use_id)

        # Validate content types and preserve structure
        content = tool_result.get("content", [])

        # Validate all content types are supported
        for block in content:
            if "text" not in block and "json" not in block:
                # Unsupported content type - raise error
                raise ValueError(
                    f"tool_use_id=<{tool_use_id}>, content_types=<{list(block.keys())}> | "
                    f"Content type not supported by Nova Sonic"
                )

        # Optimize for single content item - unwrap the array
        if len(content) == 1:
            result_data = cast(dict[str, Any], content[0])
        else:
            # Multiple items - send as array
            result_data = {"content": content}

        content_name = str(uuid.uuid4())
        events = [
            self._get_tool_content_start_event(content_name, tool_use_id),
            self._get_tool_result_event(content_name, result_data),
            self._get_content_end_event(content_name),
        ]
        await self._send_nova_events(events)

    async def stop(self) -> None:
        """Close Nova Sonic connection with proper cleanup sequence."""
        logger.debug("nova connection cleanup starting")

        async def stop_events() -> None:
            if not self._connection_id:
                return

            await self._end_audio_input()
            cleanup_events = [self._get_prompt_end_event(), self._get_connection_end_event()]
            await self._send_nova_events(cleanup_events)

        async def stop_stream() -> None:
            if not hasattr(self, "_stream"):
                return

            await self._stream.close()

        async def stop_connection() -> None:
            self._connection_id = None

        await stop_all(stop_events, stop_stream, stop_connection)

        logger.debug("nova connection closed")

    def _convert_nova_event(self, nova_event: dict[str, Any]) -> BidiOutputEvent | None:
        """Convert Nova Sonic events to TypedEvent format."""
        # Handle completion start - track completionId
        if "completionStart" in nova_event:
            completion_data = nova_event["completionStart"]
            self._current_completion_id = completion_data.get("completionId")
            logger.debug("completion_id=<%s> | nova completion started", self._current_completion_id)
            return None

        # Handle completion end
        if "completionEnd" in nova_event:
            completion_data = nova_event["completionEnd"]
            completion_id = completion_data.get("completionId", self._current_completion_id)
            stop_reason = completion_data.get("stopReason", "END_TURN")

            event = BidiResponseCompleteEvent(
                response_id=completion_id or str(uuid.uuid4()),  # Fallback to UUID if missing
                stop_reason="interrupted" if stop_reason == "INTERRUPTED" else "complete",
            )

            # Clear completion tracking
            self._current_completion_id = None
            return event

        # Handle audio output
        if "audioOutput" in nova_event:
            # Audio is already base64 string from Nova Sonic
            audio_content = nova_event["audioOutput"]["content"]
            return BidiAudioStreamEvent(
                audio=audio_content,
                format="pcm",
                sample_rate=cast(AudioSampleRate, self.config["audio"]["output_rate"]),
                channels=cast(AudioChannel, self.config["audio"]["channels"]),
            )

        # Handle text output (transcripts)
        elif "textOutput" in nova_event:
            text_output = nova_event["textOutput"]
            text_content = text_output["content"]
            # Check for Nova Sonic interruption pattern
            if '{ "interrupted" : true }' in text_content:
                logger.debug("nova interruption detected in text output")
                return BidiInterruptionEvent(reason="user_speech")

            return BidiTranscriptStreamEvent(
                delta={"text": text_content},
                text=text_content,
                role=text_output["role"].lower(),
                is_final=self._generation_stage == "FINAL",
                current_transcript=text_content,
            )

        # Handle tool use
        if "toolUse" in nova_event:
            tool_use = nova_event["toolUse"]
            tool_use_event: ToolUse = {
                "toolUseId": tool_use["toolUseId"],
                "name": tool_use["toolName"],
                "input": json.loads(tool_use["content"]),
            }
            # Return ToolUseStreamEvent - cast to dict for type compatibility
            return ToolUseStreamEvent(delta={"toolUse": tool_use_event}, current_tool_use=dict(tool_use_event))

        # Handle interruption
        if nova_event.get("stopReason") == "INTERRUPTED":
            logger.debug("nova interruption detected via stop reason")
            return BidiInterruptionEvent(reason="user_speech")

        # Handle usage events - convert to multimodal usage format
        if "usageEvent" in nova_event:
            usage_data = nova_event["usageEvent"]
            total_input = usage_data.get("totalInputTokens", 0)
            total_output = usage_data.get("totalOutputTokens", 0)

            return BidiUsageEvent(
                input_tokens=total_input,
                output_tokens=total_output,
                total_tokens=usage_data.get("totalTokens", total_input + total_output),
            )

        # Handle content start events (emit response start)
        if "contentStart" in nova_event:
            content_data = nova_event["contentStart"]
            if content_data["type"] == "TEXT":
                self._generation_stage = json.loads(content_data["additionalModelFields"])["generationStage"]

            # Emit response start event using API-provided completionId
            # completionId should already be tracked from completionStart event
            return BidiResponseStartEvent(
                response_id=self._current_completion_id or str(uuid.uuid4())  # Fallback to UUID if missing
            )

        if "contentEnd" in nova_event:
            self._generation_stage = None

        # Ignore all other events
        return None

    def _get_connection_start_event(self) -> str:
        """Generate Nova Sonic connection start event."""
        inference_config = {_NOVA_INFERENCE_CONFIG_KEYS[key]: value for key, value in self.config["inference"].items()}
        return json.dumps({"event": {"sessionStart": {"inferenceConfiguration": inference_config}}})

    def _get_prompt_start_event(self, tools: list[ToolSpec]) -> str:
        """Generate Nova Sonic prompt start event with tool configuration."""
        # Build audio output configuration from config
        audio_output_config = {
            "mediaType": "audio/lpcm",
            "sampleRateHertz": self.config["audio"]["output_rate"],
            "sampleSizeBits": 16,
            "channelCount": self.config["audio"]["channels"],
            "voiceId": self.config["audio"].get("voice", "matthew"),
            "encoding": "base64",
            "audioType": "SPEECH",
        }

        prompt_start_event: dict[str, Any] = {
            "event": {
                "promptStart": {
                    "promptName": self._connection_id,
                    "textOutputConfiguration": NOVA_TEXT_CONFIG,
                    "audioOutputConfiguration": audio_output_config,
                }
            }
        }

        if tools:
            tool_config = self._build_tool_configuration(tools)
            prompt_start_event["event"]["promptStart"]["toolUseOutputConfiguration"] = NOVA_TOOL_CONFIG
            prompt_start_event["event"]["promptStart"]["toolConfiguration"] = {"tools": tool_config}

        return json.dumps(prompt_start_event)

    def _build_tool_configuration(self, tools: list[ToolSpec]) -> list[dict[str, Any]]:
        """Build tool configuration from tool specs."""
        tool_config: list[dict[str, Any]] = []
        for tool in tools:
            input_schema = (
                {"json": json.dumps(tool["inputSchema"]["json"])}
                if "json" in tool["inputSchema"]
                else {"json": json.dumps(tool["inputSchema"])}
            )

            tool_config.append(
                {"toolSpec": {"name": tool["name"], "description": tool["description"], "inputSchema": input_schema}}
            )
        return tool_config

    def _get_system_prompt_events(self, system_prompt: str | None) -> list[str]:
        """Generate system prompt events."""
        content_name = str(uuid.uuid4())
        return [
            self._get_text_content_start_event(content_name, "SYSTEM"),
            self._get_text_input_event(content_name, system_prompt or ""),
            self._get_content_end_event(content_name),
        ]

    def _get_message_history_events(self, messages: Messages) -> list[str]:
        """Generate conversation history events from agent messages.

        Converts agent message history to Nova Sonic format following the
        contentStart/textInput/contentEnd pattern for each message.

        Args:
            messages: List of conversation messages with role and content.

        Returns:
            List of JSON event strings for Nova Sonic.
        """
        events = []

        for message in messages:
            role = message["role"].upper()  # Convert to ASSISTANT or USER
            content_blocks = message.get("content", [])

            # Extract text content from content blocks
            text_parts = []
            for block in content_blocks:
                if "text" in block:
                    text_parts.append(block["text"])

            # Combine all text parts
            if text_parts:
                combined_text = "\n".join(text_parts)
                content_name = str(uuid.uuid4())

                # Add contentStart, textInput, and contentEnd events
                events.extend(
                    [
                        self._get_text_content_start_event(content_name, role),
                        self._get_text_input_event(content_name, combined_text),
                        self._get_content_end_event(content_name),
                    ]
                )

        return events

    def _get_text_content_start_event(self, content_name: str, role: str = "USER") -> str:
        """Generate text content start event."""
        return json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._connection_id,
                        "contentName": content_name,
                        "type": "TEXT",
                        "role": role,
                        "interactive": True,
                        "textInputConfiguration": NOVA_TEXT_CONFIG,
                    }
                }
            }
        )

    def _get_tool_content_start_event(self, content_name: str, tool_use_id: str) -> str:
        """Generate tool content start event."""
        return json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._connection_id,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": NOVA_TEXT_CONFIG,
                        },
                    }
                }
            }
        )

    def _get_text_input_event(self, content_name: str, text: str) -> str:
        """Generate text input event."""
        return json.dumps(
            {"event": {"textInput": {"promptName": self._connection_id, "contentName": content_name, "content": text}}}
        )

    def _get_tool_result_event(self, content_name: str, result: dict[str, Any]) -> str:
        """Generate tool result event."""
        return json.dumps(
            {
                "event": {
                    "toolResult": {
                        "promptName": self._connection_id,
                        "contentName": content_name,
                        "content": json.dumps(result),
                    }
                }
            }
        )

    def _get_content_end_event(self, content_name: str) -> str:
        """Generate content end event."""
        return json.dumps({"event": {"contentEnd": {"promptName": self._connection_id, "contentName": content_name}}})

    def _get_prompt_end_event(self) -> str:
        """Generate prompt end event."""
        return json.dumps({"event": {"promptEnd": {"promptName": self._connection_id}}})

    def _get_connection_end_event(self) -> str:
        """Generate connection end event."""
        return json.dumps({"event": {"connectionEnd": {}}})

    async def _send_nova_events(self, events: list[str]) -> None:
        """Send event JSON string to Nova Sonic stream.

        A lock is used to send events in sequence when required (e.g., tool result start, content, and end).

        Args:
            events: Jsonified events.
        """
        async with self._send_lock:
            for event in events:
                bytes_data = event.encode("utf-8")
                chunk = InvokeModelWithBidirectionalStreamInputChunk(
                    value=BidirectionalInputPayloadPart(bytes_=bytes_data)
                )
                await self._stream.input_stream.send(chunk)
                logger.debug("nova sonic event sent successfully")
