"""AWS Bedrock model provider.

Supports two transports:

- Converse API (default) via the ``bedrock-runtime`` endpoint. Used when ``openai_endpoint``
  is not set on the config. Provides guardrails, prompt caching, and the full Converse feature
  set.
- OpenAI-compatible endpoint (``bedrock-mantle``) when ``openai_endpoint`` is provided on the
  config. Routes through the OpenAI Python SDK to the Responses or Chat Completions API.
  Unlocks features such as server-side stateful conversations, Responses API reasoning, and
  built-in tools.

Generic inference parameters (``temperature``, ``top_p``, ``max_tokens``) apply to both
transports and live at the top level of ``BedrockConfig``. The ``openai_endpoint`` path
always streams (the OpenAI SDK's Responses and Chat Completions surfaces do not offer a
non-streaming mode), and ``stop_sequences`` is forwarded to Chat Completions as ``stop``
but is not accepted by the Responses API. Converse-only fields (``guardrail_*``,
``cache_*``, ``service_tier``, ``additional_args``, etc.) may not be combined with
``openai_endpoint``; the same applies to ``streaming=False`` and to ``stop_sequences``
when ``api="responses"``. All of these raise at init time rather than silently no-op.

Docs:

- Bedrock overview: https://aws.amazon.com/bedrock/
- OpenAI-compatible endpoints: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html
"""

import asyncio
import json
import logging
import os
import warnings
from collections.abc import AsyncGenerator, Callable, Iterable, ValuesView
from typing import TYPE_CHECKING, Any, Literal, TypedDict, TypeVar, cast

import boto3
from botocore.config import Config as BotocoreConfig
from botocore.exceptions import ClientError
from pydantic import BaseModel
from typing_extensions import Unpack, override

from strands.types.media import S3Location, SourceLocation

from .._exception_notes import add_exception_note
from ..event_loop import streaming
from ..tools import convert_pydantic_to_tool_spec
from ..tools._tool_helpers import noop_tool
from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.exceptions import (
    ContextWindowOverflowException,
    ModelThrottledException,
    ProviderTokenCountError,
)
from ..types.streaming import CitationsDelta, StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import BaseModelConfig, CacheConfig, Model

if TYPE_CHECKING:
    from .openai import OpenAIModel
    from .openai_responses import OpenAIResponsesModel

logger = logging.getLogger(__name__)

# See: `BedrockModel._get_default_model_with_warning` for why we need both
DEFAULT_BEDROCK_MODEL_ID = "global.anthropic.claude-sonnet-4-6"
_DEFAULT_BEDROCK_MODEL_ID = "{}.anthropic.claude-sonnet-4-6"
DEFAULT_BEDROCK_REGION = "us-west-2"

BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES = [
    "Input is too long for requested model",
    "input length and `max_tokens` exceed context limit",
    "too many total text bytes",
    "prompt is too long",
]

# Models that should include tool result status (include_tool_result_status = True)
_MODELS_INCLUDE_STATUS = [
    "anthropic.claude",
]

T = TypeVar("T", bound=BaseModel)

DEFAULT_READ_TIMEOUT = 120

# Bedrock OpenAI-compatible endpoint (Mantle). See:
# https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html
_BEDROCK_MANTLE_BASE_URL_TEMPLATE = "https://bedrock-mantle.{region}.api.aws/v1"

# Config fields that only apply to the Converse transport. Setting any of these together with
# ``openai_endpoint`` would silently no-op, so we reject the combination at init time.
#
# ``include_tool_result_status`` is intentionally excluded: it is always auto-defaulted by
# ``__init__`` and only affects Converse-side tool-result serialization. It has no effect on
# the Mantle path either way, so requiring users to clear it would be a pure footgun.
_CONVERSE_ONLY_CONFIG_KEYS: frozenset[str] = frozenset(
    {
        "additional_args",
        "additional_request_fields",
        "additional_response_field_paths",
        "cache_prompt",
        "cache_config",
        "cache_tools",
        "guardrail_id",
        "guardrail_trace",
        "guardrail_version",
        "guardrail_stream_processing_mode",
        "guardrail_redact_input",
        "guardrail_redact_input_message",
        "guardrail_redact_output",
        "guardrail_redact_output_message",
        "guardrail_latest_message",
        "service_tier",
    }
)


class OpenAIEndpointConfig(TypedDict, total=False):
    """Configuration for routing a :class:`BedrockModel` through the OpenAI-compatible endpoint.

    When this config is present on :class:`BedrockModel.BedrockConfig`, requests are sent
    through the Bedrock Mantle endpoint (``bedrock-mantle.<region>.api.aws``) using the OpenAI
    Python SDK instead of the Converse API. This unlocks features that are specific to the
    OpenAI-compatible surface, such as the Responses API's server-side stateful conversations
    and reasoning controls.

    Generic inference parameters (``temperature``, ``top_p``, ``max_tokens``,
    ``stop_sequences``, ``streaming``) continue to live on :class:`BedrockModel.BedrockConfig`
    and are forwarded to the underlying OpenAI model.

    Attributes:
        api: Which OpenAI API surface to use. ``"responses"`` maps to the Responses API and
            ``"chat_completions"`` maps to the Chat Completions API. Required.
        api_key: Bedrock API key to send as the bearer token. The OpenAI SDK is only the
            transport here; this is a Bedrock-issued key, not an OpenAI account key. The
            AWS docs recommend setting the ``OPENAI_API_KEY`` environment variable to your
            Bedrock API key, which is the OpenAI SDK's default env var. When ``api_key`` is
            omitted, the underlying SDK picks up ``OPENAI_API_KEY`` automatically.
        stateful: Enable server-side conversation state management. Responses API only.
        params: Extra parameters forwarded to the OpenAI SDK ``params`` dict. Use this for
            Responses-only options such as ``reasoning``.
        client_args: Extra arguments merged into the OpenAI client constructor. Use this
            to plug in a custom ``http_client`` (for example, to sign requests with AWS
            SigV4) or to override timeouts. ``api_key`` and ``base_url`` set here override
            the values derived from ``api_key`` and ``region``. When providing a custom
            ``http_client``, the caller owns its lifecycle: the OpenAI SDK's per-request
            context manager will call ``aclose()`` on whatever client it is given, so a
            long-lived injected client should either override ``aclose()`` to a no-op or
            be constructed fresh per request.
    """

    api: Literal["responses", "chat_completions"]
    api_key: str | None
    stateful: bool | None
    params: dict[str, Any] | None
    client_args: dict[str, Any] | None


class BedrockModel(Model):
    """AWS Bedrock model provider implementation.

    The implementation handles Bedrock-specific features such as:

    - Tool configuration for function calling
    - Guardrails integration
    - Caching points for system prompts and tools
    - Streaming responses
    - Context window overflow detection
    """

    class BedrockConfig(BaseModelConfig, total=False):
        """Configuration options for Bedrock models.

        Attributes:
            additional_args: Any additional arguments to include in the request
            additional_request_fields: Additional fields to include in the Bedrock request
            additional_response_field_paths: Additional response field paths to extract
            cache_prompt: Cache point type for the system prompt (deprecated, use cache_config)
            cache_config: Configuration for prompt caching. Use CacheConfig(strategy="auto") for automatic caching.
            cache_tools: Cache point type for tools
            guardrail_id: ID of the guardrail to apply
            guardrail_trace: Guardrail trace mode. Defaults to enabled.
            guardrail_version: Version of the guardrail to apply
            guardrail_stream_processing_mode: The guardrail processing mode
            guardrail_redact_input: Flag to redact input if a guardrail is triggered. Defaults to True.
            guardrail_redact_input_message: If a Bedrock Input guardrail triggers, replace the input with this message.
            guardrail_redact_output: Flag to redact output if guardrail is triggered. Defaults to False.
            guardrail_redact_output_message: If a Bedrock Output guardrail triggers, replace output with this message.
            guardrail_latest_message: Flag to send only the lastest user message to guardrails.
                Defaults to False.
            max_tokens: Maximum number of tokens to generate in the response
            model_id: The Bedrock model ID (e.g., "global.anthropic.claude-sonnet-4-6")
            include_tool_result_status: Flag to include status field in tool results.
                True includes status, False removes status, "auto" determines based on model_id. Defaults to "auto".
            openai_endpoint: When set, route requests through Bedrock's OpenAI-compatible endpoint
                (``bedrock-mantle``) using the OpenAI Python SDK instead of the Converse API.
                See :class:`OpenAIEndpointConfig`. Raises at init time when combined with
                Converse-only fields (``guardrail_*``, ``cache_*``, ``service_tier``,
                ``additional_args``, etc.), with ``streaming=False``, or with
                ``stop_sequences`` when ``api="responses"``.
            service_tier: Service tier for the request, controlling the trade-off between latency and cost.
                Valid values: "default" (standard), "priority" (faster, premium), "flex" (cheaper, slower).
                Please check https://docs.aws.amazon.com/bedrock/latest/userguide/service-tiers-inference.html for
                supported service tiers, models, and regions
            stop_sequences: List of sequences that will stop generation when encountered
            streaming: Flag to enable/disable streaming. Defaults to True.
            temperature: Controls randomness in generation (higher = more random)
            top_p: Controls diversity via nucleus sampling (alternative to temperature)
        """

        additional_args: dict[str, Any] | None
        additional_request_fields: dict[str, Any] | None
        additional_response_field_paths: list[str] | None
        cache_prompt: str | None
        cache_config: CacheConfig | None
        cache_tools: str | None
        guardrail_id: str | None
        guardrail_trace: Literal["enabled", "disabled", "enabled_full"] | None
        guardrail_stream_processing_mode: Literal["sync", "async"] | None
        guardrail_version: str | None
        guardrail_redact_input: bool | None
        guardrail_redact_input_message: str | None
        guardrail_redact_output: bool | None
        guardrail_redact_output_message: str | None
        guardrail_latest_message: bool | None
        max_tokens: int | None
        model_id: str
        include_tool_result_status: Literal["auto"] | bool | None
        service_tier: str | None
        stop_sequences: list[str] | None
        streaming: bool | None
        temperature: float | None
        top_p: float | None
        openai_endpoint: OpenAIEndpointConfig | None

    def __init__(
        self,
        *,
        boto_session: boto3.Session | None = None,
        boto_client_config: BotocoreConfig | None = None,
        region_name: str | None = None,
        endpoint_url: str | None = None,
        **model_config: Unpack[BedrockConfig],
    ):
        """Initialize provider instance.

        Args:
            boto_session: Boto Session to use when calling the Bedrock Model.
            boto_client_config: Configuration to use when creating the Bedrock-Runtime Boto Client.
            region_name: AWS region to use for the Bedrock service.
                Defaults to the AWS_REGION environment variable if set, or "us-west-2" if not set.
            endpoint_url: Custom endpoint URL for VPC endpoints (PrivateLink)
            **model_config: Configuration options for the Bedrock model.
        """
        if region_name and boto_session:
            raise ValueError("Cannot specify both `region_name` and `boto_session`.")

        session = boto_session or boto3.Session()
        resolved_region = region_name or session.region_name or os.environ.get("AWS_REGION") or DEFAULT_BEDROCK_REGION
        self._resolved_region = resolved_region
        self.config = BedrockModel.BedrockConfig(
            model_id=BedrockModel._get_default_model_with_warning(resolved_region, model_config),
            include_tool_result_status="auto",
        )
        self.update_config(**model_config)

        logger.debug("config=<%s> | initializing", self.config)

        # When ``openai_endpoint`` is configured, requests are routed through the Bedrock Mantle
        # OpenAI-compatible endpoint via the OpenAI Python SDK. Skip the boto client since it is
        # not used on that path.
        self._openai_delegate: OpenAIModel | OpenAIResponsesModel | None = None
        endpoint_config = self.config.get("openai_endpoint")
        if endpoint_config is not None:
            self._validate_openai_endpoint_config()
            self._openai_delegate = self._build_openai_delegate()
            self.client = None
            logger.debug(
                "region=<%s>, api=<%s> | bedrock openai-compatible delegate created",
                resolved_region,
                endpoint_config["api"],
            )
            return

        # Add strands-agents to the request user agent
        if boto_client_config:
            existing_user_agent = getattr(boto_client_config, "user_agent_extra", None)

            # Append 'strands-agents' to existing user_agent_extra or set it if not present
            if existing_user_agent:
                new_user_agent = f"{existing_user_agent} strands-agents"
            else:
                new_user_agent = "strands-agents"

            client_config = boto_client_config.merge(BotocoreConfig(user_agent_extra=new_user_agent))
        else:
            client_config = BotocoreConfig(user_agent_extra="strands-agents", read_timeout=DEFAULT_READ_TIMEOUT)

        self.client = session.client(
            service_name="bedrock-runtime",
            config=client_config,
            endpoint_url=endpoint_url,
            region_name=resolved_region,
        )

        logger.debug("region=<%s> | bedrock client created", self.client.meta.region_name)

    def _validate_openai_endpoint_config(self) -> None:
        """Validate ``openai_endpoint`` against the rest of :class:`BedrockConfig`.

        Runs the full set of Mantle-path checks:

        1. ``api`` must be one of ``"responses"`` or ``"chat_completions"``.
        2. No Converse-only fields may be set alongside ``openai_endpoint``.
        3. ``streaming=False`` is rejected; both OpenAI delegate APIs always stream.
        4. ``stop_sequences`` is rejected when ``api="responses"``; the Responses API
           does not accept stop sequences. It is forwarded as ``stop`` for Chat Completions.
        5. ``stateful=True`` is Responses-only.

        Each failure raises rather than silently no-opping so misconfigurations surface
        at init time.

        Raises:
            ValueError: When any of the above checks fail.
        """
        endpoint = cast(OpenAIEndpointConfig, self.config["openai_endpoint"])
        api = endpoint.get("api")
        if api not in ("responses", "chat_completions"):
            raise ValueError(f'openai_endpoint requires "api" to be "responses" or "chat_completions", got {api!r}')

        conflicting = sorted(k for k in _CONVERSE_ONLY_CONFIG_KEYS if self.config.get(k) is not None)
        if conflicting:
            raise ValueError(
                "openai_endpoint cannot be combined with Converse-only config fields: "
                f"{conflicting}. Remove these fields or drop openai_endpoint to use the Converse API."
            )

        # Both OpenAI delegates always stream; ``streaming=False`` would be silently
        # overridden on the Mantle path, which is the class of bug this validator exists
        # to prevent.
        if self.config.get("streaming") is False:
            raise ValueError(
                "openai_endpoint does not support streaming=False. The OpenAI SDK's Responses "
                "and Chat Completions surfaces always stream; remove streaming=False or drop "
                "openai_endpoint to use the Converse API."
            )

        # The Responses API does not accept stop sequences. Chat Completions forwards them
        # as ``stop``, so this only applies on the Responses path.
        if api == "responses" and self.config.get("stop_sequences") is not None:
            raise ValueError(
                'openai_endpoint with api="responses" does not accept stop_sequences. '
                'Remove stop_sequences or use api="chat_completions".'
            )

        # ``stateful`` is only meaningful for the Responses API.
        if endpoint.get("stateful") and api != "responses":
            raise ValueError(f'openai_endpoint.stateful is only supported when api="responses". Got api={api!r}.')

    def _build_openai_delegate(self) -> "OpenAIModel | OpenAIResponsesModel":
        """Construct the OpenAI-compatible delegate for the Mantle endpoint.

        Forwards generic inference params from :class:`BedrockConfig` into the OpenAI SDK
        ``params`` dict, translating names where the OpenAI and Bedrock conventions differ.

        Returns:
            An :class:`OpenAIResponsesModel` or :class:`OpenAIModel` configured to talk to
            ``bedrock-mantle.<region>.api.aws``.
        """
        endpoint = cast(OpenAIEndpointConfig, self.config["openai_endpoint"])
        api = endpoint["api"]

        # The Mantle base URL is fully determined by region; AWS owns the endpoint list:
        # https://docs.aws.amazon.com/bedrock/latest/userguide/inference-openai.html
        base_url = _BEDROCK_MANTLE_BASE_URL_TEMPLATE.format(region=self._resolved_region)

        params: dict[str, Any] = dict(endpoint.get("params") or {})
        # Forward generic inference params from BedrockConfig. Translate naming where the
        # Responses API differs from Chat Completions (max_tokens -> max_output_tokens).
        max_tokens = self.config.get("max_tokens")
        if max_tokens is not None:
            params.setdefault("max_output_tokens" if api == "responses" else "max_tokens", max_tokens)
        temperature = self.config.get("temperature")
        if temperature is not None:
            params.setdefault("temperature", temperature)
        top_p = self.config.get("top_p")
        if top_p is not None:
            params.setdefault("top_p", top_p)
        stop_sequences = self.config.get("stop_sequences")
        if stop_sequences is not None:
            # Reaching this branch implies api == "chat_completions"; the validator
            # rejects stop_sequences on the Responses path so no fallback is needed.
            params.setdefault("stop", stop_sequences)

        # The OpenAI SDK's ``api_key`` parameter is just the bearer token it sends in the
        # Authorization header; when pointed at bedrock-mantle this is a Bedrock-issued key.
        # The AWS docs recommend setting OPENAI_API_KEY to the Bedrock API key so existing
        # OpenAI SDK code works unchanged. If the user does not pass ``api_key`` here, the
        # OpenAI SDK will read OPENAI_API_KEY from the environment on its own.
        client_args: dict[str, Any] = {"base_url": base_url}
        if api_key := endpoint.get("api_key"):
            client_args["api_key"] = api_key
        # User-supplied client_args win over derived defaults. This is the escape hatch for
        # plumbing a signed httpx client (SigV4), custom timeouts, etc.
        if extra_client_args := endpoint.get("client_args"):
            client_args.update(extra_client_args)

        if api == "responses":
            from .openai_responses import OpenAIResponsesModel

            stateful = bool(endpoint.get("stateful"))
            return OpenAIResponsesModel(
                client_args=client_args,
                model_id=self.config["model_id"],
                params=params,
                stateful=stateful,
            )

        from .openai import OpenAIModel

        return OpenAIModel(
            client_args=client_args,
            model_id=self.config["model_id"],
            params=params,
        )

    @property
    def _cache_strategy(self) -> str | None:
        """The cache strategy for this model based on its model ID.

        Returns the appropriate cache strategy name, or None if automatic caching is not supported for this model.
        """
        model_id = self.config.get("model_id", "").lower()
        if "claude" in model_id or "anthropic" in model_id:
            return "anthropic"
        return None

    @property
    @override
    def stateful(self) -> bool:
        """Whether the model manages conversation state server-side.

        Delegates to the underlying OpenAI-compatible model when ``openai_endpoint`` is configured,
        otherwise returns False (the Converse API is always stateless).
        """
        if self._openai_delegate is not None:
            return self._openai_delegate.stateful
        return False

    @override
    def update_config(self, **model_config: Unpack[BedrockConfig]) -> None:  # type: ignore
        """Update the Bedrock Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        validate_config_keys(model_config, self.BedrockConfig)
        self.config.update(model_config)

        # If the delegate is already built and the caller changed anything the delegate depends on,
        # rebuild it so subsequent calls pick up the new config. Skipped during __init__ where
        # ``_openai_delegate`` is not yet set.
        if getattr(self, "_openai_delegate", None) is not None:
            self._validate_openai_endpoint_config()
            self._openai_delegate = self._build_openai_delegate()

    @override
    def get_config(self) -> BedrockConfig:
        """Get the current Bedrock Model configuration.

        Returns:
            The Bedrock model configuration.
        """
        return self.config

    def _format_request(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> dict[str, Any]:
        """Format a Bedrock converse stream request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.

        Returns:
            A Bedrock converse stream request.
        """
        if not tool_specs:
            has_tool_content = any(
                any("toolUse" in block or "toolResult" in block for block in msg.get("content", [])) for msg in messages
            )
            if has_tool_content:
                tool_specs = [noop_tool.tool_spec]

        # Use system_prompt_content directly (copy for mutability)
        system_blocks: list[SystemContentBlock] = system_prompt_content.copy() if system_prompt_content else []
        # Add cache point if configured (backwards compatibility)
        if cache_prompt := self.config.get("cache_prompt"):
            warnings.warn(
                "cache_prompt is deprecated. Use SystemContentBlock with cachePoint instead.", UserWarning, stacklevel=3
            )
            system_blocks.append({"cachePoint": {"type": cache_prompt}})

        return {
            "modelId": self.config["model_id"],
            "messages": self._format_bedrock_messages(messages),
            "system": system_blocks,
            **({"serviceTier": {"type": self.config["service_tier"]}} if self.config.get("service_tier") else {}),
            **(
                {
                    "toolConfig": {
                        "tools": [
                            *[
                                {
                                    "toolSpec": {
                                        "name": tool_spec["name"],
                                        "description": tool_spec["description"],
                                        "inputSchema": tool_spec["inputSchema"],
                                    }
                                }
                                for tool_spec in tool_specs
                            ],
                            *(
                                [{"cachePoint": {"type": self.config["cache_tools"]}}]
                                if self.config.get("cache_tools")
                                else []
                            ),
                        ],
                        **({"toolChoice": tool_choice if tool_choice else {"auto": {}}}),
                    }
                }
                if tool_specs
                else {}
            ),
            **(self._get_additional_request_fields(tool_choice)),
            **(
                {"additionalModelResponseFieldPaths": self.config["additional_response_field_paths"]}
                if self.config.get("additional_response_field_paths")
                else {}
            ),
            **(
                {
                    "guardrailConfig": {
                        "guardrailIdentifier": self.config["guardrail_id"],
                        "guardrailVersion": self.config["guardrail_version"],
                        "trace": self.config.get("guardrail_trace", "enabled"),
                        **(
                            {"streamProcessingMode": self.config.get("guardrail_stream_processing_mode")}
                            if self.config.get("guardrail_stream_processing_mode")
                            else {}
                        ),
                    }
                }
                if self.config.get("guardrail_id") and self.config.get("guardrail_version")
                else {}
            ),
            "inferenceConfig": {
                key: value
                for key, value in [
                    ("maxTokens", self.config.get("max_tokens")),
                    ("temperature", self.config.get("temperature")),
                    ("topP", self.config.get("top_p")),
                    ("stopSequences", self.config.get("stop_sequences")),
                ]
                if value is not None
            },
            **(
                self.config["additional_args"]
                if "additional_args" in self.config and self.config["additional_args"] is not None
                else {}
            ),
        }

    def _get_additional_request_fields(self, tool_choice: ToolChoice | None) -> dict[str, Any]:
        """Get additional request fields, removing thinking if tool_choice forces tool use.

        Bedrock's API does not allow thinking mode when tool_choice forces tool use.
        When forcing a tool (e.g., for structured_output retry), we temporarily disable thinking.

        Args:
            tool_choice: The tool choice configuration.

        Returns:
            A dict containing additionalModelRequestFields if configured, or empty dict.
        """
        additional_fields = self.config.get("additional_request_fields")
        if not additional_fields:
            return {}

        # Check if tool_choice is forcing tool use ("any" or specific "tool")
        is_forcing_tool = tool_choice is not None and ("any" in tool_choice or "tool" in tool_choice)

        if is_forcing_tool and "thinking" in additional_fields:
            # Create a copy without the thinking key
            fields_without_thinking = {k: v for k, v in additional_fields.items() if k != "thinking"}
            if fields_without_thinking:
                return {"additionalModelRequestFields": fields_without_thinking}
            return {}

        return {"additionalModelRequestFields": additional_fields}

    def _inject_cache_point(self, messages: list[dict[str, Any]]) -> None:
        """Inject a cache point at the end of the last user message.

        Args:
            messages: List of messages to inject cache point into (modified in place).
        """
        if not messages:
            return

        last_user_idx: int | None = None
        for msg_idx, msg in enumerate(messages):
            content = msg.get("content", [])
            for block_idx, block in reversed(list(enumerate(content))):
                if "cachePoint" in block:
                    del content[block_idx]
                    logger.warning(
                        "msg_idx=<%s>, block_idx=<%s> | stripped existing cache point (auto mode manages cache points)",
                        msg_idx,
                        block_idx,
                    )
            if msg.get("role") == "user":
                last_user_idx = msg_idx

        if last_user_idx is not None and messages[last_user_idx].get("content"):
            messages[last_user_idx]["content"].append({"cachePoint": {"type": "default"}})
            logger.debug("msg_idx=<%s> | added cache point to last user message", last_user_idx)

    def _find_last_user_text_message_index(self, messages: Messages) -> int | None:
        """Find the index of the last user message containing text or image content.

        This is used for guardrail_latest_message to ensure that guardContent wrapping
        targets the correct message even when toolResult messages follow.

        Args:
            messages: List of messages to search

        Returns:
            Index of the last user message with text/image content, or None if not found
        """
        for idx, msg in reversed(list(enumerate(messages))):
            if msg["role"] == "user" and any("text" in cb or "image" in cb for cb in msg.get("content", [])):
                return idx
        return None

    def _format_bedrock_messages(self, messages: Messages) -> list[dict[str, Any]]:
        """Format messages for Bedrock API compatibility.

        This function ensures messages conform to Bedrock's expected format by:
        - Filtering out SDK_UNKNOWN_MEMBER content blocks
        - Eagerly filtering content blocks to only include Bedrock-supported fields
        - Ensuring all message content blocks are properly formatted for the Bedrock API
        - Optionally wrapping the last user message in guardrailConverseContent blocks
        - Injecting cache points when cache_config is set with strategy="auto"

        Args:
            messages: List of messages to format

        Returns:
            Messages formatted for Bedrock API compatibility

        Note:
            Unlike other APIs that ignore unknown fields, Bedrock only accepts a strict
            subset of fields for each content block type and throws validation exceptions
            when presented with unexpected fields. Therefore, we must eagerly filter all
            content blocks to remove any additional fields before sending to Bedrock.
            https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ContentBlock.html
        """
        cleaned_messages: list[dict[str, Any]] = []

        filtered_unknown_members = False
        dropped_deepseek_reasoning_content = False

        # Pre-compute the index of the last user message containing text or image content.
        # This ensures guardContent wrapping is maintained across tool execution cycles, where
        # the final message in the list is a toolResult (role=user) rather than text/image content.
        last_user_text_idx = None
        if self.config.get("guardrail_latest_message", False):
            last_user_text_idx = self._find_last_user_text_message_index(messages)

        for idx, message in enumerate(messages):
            cleaned_content: list[dict[str, Any]] = []

            for content_block in message["content"]:
                # Filter out SDK_UNKNOWN_MEMBER content blocks
                if "SDK_UNKNOWN_MEMBER" in content_block:
                    filtered_unknown_members = True
                    continue

                # DeepSeek models have issues with reasoningContent
                # TODO: Replace with systematic model configuration registry (https://github.com/strands-agents/sdk-python/issues/780)
                if "deepseek" in self.config["model_id"].lower() and "reasoningContent" in content_block:
                    dropped_deepseek_reasoning_content = True
                    continue

                # Format content blocks for Bedrock API compatibility
                formatted_content = self._format_request_message_content(content_block)
                if formatted_content is None:
                    continue

                # Wrap text or image content in guardContent if this is the last user text/image message
                if idx == last_user_text_idx and ("text" in formatted_content or "image" in formatted_content):
                    if "text" in formatted_content:
                        formatted_content = {"guardContent": {"text": {"text": formatted_content["text"]}}}
                    elif "image" in formatted_content:
                        formatted_content = {"guardContent": {"image": formatted_content["image"]}}

                cleaned_content.append(formatted_content)

            # Create new message with cleaned content (skip if empty)
            if cleaned_content:
                cleaned_messages.append({"content": cleaned_content, "role": message["role"]})

        if filtered_unknown_members:
            logger.warning(
                "Filtered out SDK_UNKNOWN_MEMBER content blocks from messages, consider upgrading boto3 version"
            )
        if dropped_deepseek_reasoning_content:
            logger.debug(
                "Filtered DeepSeek reasoningContent content blocks from messages - https://api-docs.deepseek.com/guides/reasoning_model#multi-round-conversation"
            )

        # Inject cache point into cleaned_messages (not original messages) if cache_config is set
        cache_config = self.config.get("cache_config")
        if cache_config:
            strategy: str | None = cache_config.strategy
            if strategy == "auto":
                strategy = self._cache_strategy
                if not strategy:
                    logger.warning(
                        "model_id=<%s> | cache_config is enabled but this model does not support automatic caching",
                        self.config.get("model_id"),
                    )
            if strategy == "anthropic":
                self._inject_cache_point(cleaned_messages)

        return cleaned_messages

    def _should_include_tool_result_status(self) -> bool:
        """Determine whether to include tool result status based on current config."""
        include_status = self.config.get("include_tool_result_status", "auto")

        if include_status is True:
            return True
        elif include_status is False:
            return False
        else:  # "auto"
            return any(model in self.config["model_id"] for model in _MODELS_INCLUDE_STATUS)

    def _handle_location(self, location: SourceLocation) -> dict[str, Any] | None:
        """Convert location content block to Bedrock format if its an S3Location."""
        if location["type"] == "s3":
            s3_location = cast(S3Location, location)
            formatted_document_s3: dict[str, Any] = {"uri": s3_location["uri"]}
            if "bucketOwner" in s3_location:
                formatted_document_s3["bucketOwner"] = s3_location["bucketOwner"]
            return {"s3Location": formatted_document_s3}
        else:
            logger.warning("Non s3 location sources are not supported by Bedrock | skipping content block")
            return None

    def _format_request_message_content(self, content: ContentBlock) -> dict[str, Any] | None:
        """Format a Bedrock content block.

        Bedrock strictly validates content blocks and throws exceptions for unknown fields.
        This function extracts only the fields that Bedrock supports for each content type.

        Args:
            content: Content block to format.

        Returns:
            Bedrock formatted content block.

        Raises:
            TypeError: If the content block type is not supported by Bedrock.
        """
        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CachePointBlock.html
        if "cachePoint" in content:
            cache_point = content["cachePoint"]
            result: dict[str, Any] = {"type": cache_point["type"]}
            if "ttl" in cache_point:
                result["ttl"] = cache_point["ttl"]
            return {"cachePoint": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_DocumentBlock.html
        if "document" in content:
            document = content["document"]
            result = {}

            # Handle required fields (all optional due to total=False)
            if "name" in document:
                result["name"] = document["name"]
            if "format" in document:
                result["format"] = document["format"]

            # Handle source - supports bytes or location
            if "source" in document:
                source = document["source"]
                formatted_document_source: dict[str, Any] | None
                if "location" in source:
                    formatted_document_source = self._handle_location(source["location"])
                    if formatted_document_source is None:
                        return None
                elif "bytes" in source:
                    formatted_document_source = {"bytes": source["bytes"]}
                result["source"] = formatted_document_source

            # Handle optional fields
            if "citations" in document and document["citations"] is not None:
                result["citations"] = {"enabled": document["citations"]["enabled"]}
            if "context" in document:
                result["context"] = document["context"]

            return {"document": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_GuardrailConverseContentBlock.html
        if "guardContent" in content:
            guard = content["guardContent"]
            guard_text = guard["text"]
            result = {"text": {"text": guard_text["text"], "qualifiers": guard_text["qualifiers"]}}
            return {"guardContent": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ImageBlock.html
        if "image" in content:
            image = content["image"]
            source = image["source"]
            formatted_image_source: dict[str, Any] | None
            if "location" in source:
                formatted_image_source = self._handle_location(source["location"])
                if formatted_image_source is None:
                    return None
            elif "bytes" in source:
                formatted_image_source = {"bytes": source["bytes"]}
            result = {"format": image["format"], "source": formatted_image_source}
            return {"image": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ReasoningContentBlock.html
        if "reasoningContent" in content:
            reasoning = content["reasoningContent"]
            result = {}

            if "reasoningText" in reasoning:
                reasoning_text = reasoning["reasoningText"]
                result["reasoningText"] = {}
                if "text" in reasoning_text:
                    result["reasoningText"]["text"] = reasoning_text["text"]
                # Only include signature if truthy (avoid empty strings)
                if reasoning_text.get("signature"):
                    result["reasoningText"]["signature"] = reasoning_text["signature"]

            if "redactedContent" in reasoning:
                result["redactedContent"] = reasoning["redactedContent"]

            return {"reasoningContent": result}

        # Pass through text and other simple content types
        if "text" in content:
            return {"text": content["text"]}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolResultBlock.html
        if "toolResult" in content:
            tool_result = content["toolResult"]
            # Normalize empty toolResult content arrays.
            # Some model providers (e.g., Nemotron) reject toolResult blocks with
            # content: [] via the Converse API, while others (e.g., Claude) accept
            # them. Replace empty content with a minimal text block to ensure
            # cross-model compatibility. This follows the same pattern as the
            # TypeScript SDK's _formatMessages in bedrock.ts.
            tool_result_content_list = tool_result.get("content") or [{"text": ""}]
            formatted_content: list[dict[str, Any]] = []
            for tool_result_content in tool_result_content_list:
                if "json" in tool_result_content:
                    # Handle json field since not in ContentBlock but valid in ToolResultContent
                    formatted_content.append({"json": tool_result_content["json"]})
                else:
                    formatted_message_content = self._format_request_message_content(
                        cast(ContentBlock, tool_result_content)
                    )
                    if formatted_message_content is None:
                        continue
                    formatted_content.append(formatted_message_content)

            result = {
                "content": formatted_content,
                "toolUseId": tool_result["toolUseId"],
            }
            if "status" in tool_result and self._should_include_tool_result_status():
                result["status"] = tool_result["status"]
            return {"toolResult": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_ToolUseBlock.html
        if "toolUse" in content:
            tool_use = content["toolUse"]
            return {
                "toolUse": {
                    "input": tool_use["input"],
                    "name": tool_use["name"],
                    "toolUseId": tool_use["toolUseId"],
                }
            }

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_VideoBlock.html
        if "video" in content:
            video = content["video"]
            source = video["source"]
            formatted_video_source: dict[str, Any] | None
            if "location" in source:
                formatted_video_source = self._handle_location(source["location"])
                if formatted_video_source is None:
                    return None
            elif "bytes" in source:
                formatted_video_source = {"bytes": source["bytes"]}
            result = {"format": video["format"], "source": formatted_video_source}
            return {"video": result}

        # https://docs.aws.amazon.com/bedrock/latest/APIReference/API_runtime_CitationsContentBlock.html
        if "citationsContent" in content:
            citations = content["citationsContent"]
            result = {}

            if "citations" in citations:
                result["citations"] = []
                for citation in citations["citations"]:
                    filtered_citation: dict[str, Any] = {}
                    if "location" in citation:
                        filtered_citation["location"] = citation["location"]
                    if "sourceContent" in citation:
                        filtered_source_content: list[dict[str, Any]] = []
                        for source_content in citation["sourceContent"]:
                            if "text" in source_content:
                                filtered_source_content.append({"text": source_content["text"]})
                        if filtered_source_content:
                            filtered_citation["sourceContent"] = filtered_source_content
                    if "title" in citation:
                        filtered_citation["title"] = citation["title"]
                    result["citations"].append(filtered_citation)

            if "content" in citations:
                filtered_content: list[dict[str, Any]] = []
                for generated_content in citations["content"]:
                    if "text" in generated_content:
                        filtered_content.append({"text": generated_content["text"]})
                if filtered_content:
                    result["content"] = filtered_content

            return {"citationsContent": result}

        raise TypeError(f"content_type=<{next(iter(content))}> | unsupported type")

    def _has_blocked_guardrail(self, guardrail_data: dict[str, Any]) -> bool:
        """Check if guardrail data contains any blocked policies.

        Args:
            guardrail_data: Guardrail data from trace information.

        Returns:
            True if any blocked guardrail is detected, False otherwise.
        """
        input_assessment = guardrail_data.get("inputAssessment", {})
        output_assessments = guardrail_data.get("outputAssessments", {})

        # Check input assessments
        if any(self._find_detected_and_blocked_policy(assessment) for assessment in input_assessment.values()):
            return True

        # Check output assessments
        if any(self._find_detected_and_blocked_policy(assessment) for assessment in output_assessments.values()):
            return True

        return False

    def _generate_redaction_events(self) -> list[StreamEvent]:
        """Generate redaction events based on configuration.

        Returns:
            List of redaction events to yield.
        """
        events: list[StreamEvent] = []

        if self.config.get("guardrail_redact_input", True):
            logger.debug("Redacting user input due to guardrail.")
            events.append(
                {
                    "redactContent": {
                        "redactUserContentMessage": self.config.get(
                            "guardrail_redact_input_message", "[User input redacted.]"
                        )
                    }
                }
            )

        if self.config.get("guardrail_redact_output", False):
            logger.debug("Redacting assistant output due to guardrail.")
            events.append(
                {
                    "redactContent": {
                        "redactAssistantContentMessage": self.config.get(
                            "guardrail_redact_output_message",
                            "[Assistant output redacted.]",
                        )
                    }
                }
            )

        return events

    @override
    async def count_tokens(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
    ) -> int:
        """Count tokens using Bedrock's native CountTokens API.

        Uses the same message format as the Converse API to get accurate token counts
        directly from the Bedrock service.

        Args:
            messages: List of message objects to count tokens for.
            tool_specs: List of tool specifications to include in the count.
            system_prompt: Plain string system prompt. Ignored if system_prompt_content is provided.
            system_prompt_content: Structured system prompt content blocks.

        Returns:
            Total input token count.
        """
        # The openai_endpoint path has no Bedrock Converse client and no equivalent native
        # count endpoint, so fall back to the base ``Model.count_tokens`` estimation
        # (tiktoken when available, heuristic otherwise).
        if self._openai_delegate is not None:
            return await super().count_tokens(messages, tool_specs, system_prompt, system_prompt_content)

        try:
            # The openai_endpoint early-return above guarantees ``self.client`` exists here.
            assert self.client is not None, "Bedrock Converse client is unavailable"
            if system_prompt and system_prompt_content is None:
                system_prompt_content = [{"text": system_prompt}]

            request = self._format_request(messages, tool_specs, system_prompt_content)
            converse_input: dict[str, Any] = {}
            if "messages" in request:
                converse_input["messages"] = request["messages"]
            if "system" in request:
                converse_input["system"] = request["system"]
            if "toolConfig" in request:
                converse_input["toolConfig"] = request["toolConfig"]

            response = await asyncio.to_thread(
                self.client.count_tokens,
                modelId=self.config["model_id"],
                input={"converse": converse_input},
            )
            input_tokens = response.get("inputTokens")
            if input_tokens is None:
                raise ProviderTokenCountError("Bedrock count_tokens returned None for inputTokens")
            total_tokens: int = input_tokens

            logger.debug("model_id=<%s>, total_tokens=<%d> | native token count", self.config["model_id"], total_tokens)
            return total_tokens
        except Exception as e:
            logger.debug(
                "model_id=<%s>, error=<%s> | native token counting failed, falling back to estimation",
                self.config["model_id"],
                e,
            )
            return await super().count_tokens(messages, tool_specs, system_prompt, system_prompt_content)

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt: str | None = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        """Stream conversation with the Bedrock model.

        This method calls either the Bedrock converse_stream API or the converse API
        based on the streaming parameter in the configuration.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.
            tool_choice: Selection strategy for tool invocation.
            system_prompt_content: System prompt content blocks to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """
        if self._openai_delegate is not None:
            # The OpenAI delegates accept ``system_prompt`` (a plain string) but not
            # ``system_prompt_content``. Collapse structured text blocks into a single
            # string so the delegate receives the same content; non-text blocks (cache
            # points, etc.) are Converse-only and are dropped here rather than silently
            # forwarded as unsupported payload.
            delegate_system_prompt = system_prompt
            if delegate_system_prompt is None and system_prompt_content:
                delegate_system_prompt = (
                    "\n\n".join(block["text"] for block in system_prompt_content if "text" in block) or None
                )
            async for delegate_event in self._openai_delegate.stream(
                messages,
                tool_specs,
                delegate_system_prompt,
                tool_choice=tool_choice,
                **kwargs,
            ):
                yield delegate_event
            return

        def callback(event: StreamEvent | None = None) -> None:
            loop.call_soon_threadsafe(queue.put_nowait, event)
            if event is None:
                return

        loop = asyncio.get_event_loop()
        queue: asyncio.Queue[StreamEvent | None] = asyncio.Queue()

        # Handle backward compatibility: if system_prompt is provided but system_prompt_content is None
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        thread = asyncio.to_thread(self._stream, callback, messages, tool_specs, system_prompt_content, tool_choice)
        task = asyncio.create_task(thread)

        while True:
            event = await queue.get()
            if event is None:
                break

            yield event

        await task

    def _stream(
        self,
        callback: Callable[..., None],
        messages: Messages,
        tool_specs: list[ToolSpec] | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        tool_choice: ToolChoice | None = None,
    ) -> None:
        """Stream conversation with the Bedrock model.

        This method operates in a separate thread to avoid blocking the async event loop with the call to
        Bedrock's converse_stream.

        Args:
            callback: Function to send events to the main thread.
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt_content: System prompt content blocks to provide context to the model.
            tool_choice: Selection strategy for tool invocation.

        Raises:
            ContextWindowOverflowException: If the input exceeds the model's context window.
            ModelThrottledException: If the model service is throttling requests.
        """
        # Converse-only path; the Mantle delegate handles streaming directly in ``stream``.
        assert self.client is not None, "Bedrock Converse client is unavailable"
        try:
            logger.debug("formatting request")
            request = self._format_request(messages, tool_specs, system_prompt_content, tool_choice)
            logger.debug("request=<%s>", request)

            logger.debug("invoking model")
            streaming = self.config.get("streaming", True)

            logger.debug("got response from model")
            if streaming:
                response = self.client.converse_stream(**request)
                for chunk in response["stream"]:
                    if (
                        "metadata" in chunk
                        and "trace" in chunk["metadata"]
                        and "guardrail" in chunk["metadata"]["trace"]
                    ):
                        guardrail_data = chunk["metadata"]["trace"]["guardrail"]
                        if self._has_blocked_guardrail(guardrail_data):
                            for event in self._generate_redaction_events():
                                callback(event)

                    callback(chunk)

            else:
                response = self.client.converse(**request)
                for event in self._convert_non_streaming_to_streaming(response):
                    callback(event)

                if (
                    "trace" in response
                    and "guardrail" in response["trace"]
                    and self._has_blocked_guardrail(response["trace"]["guardrail"])
                ):
                    for event in self._generate_redaction_events():
                        callback(event)

        except ClientError as e:
            error_message = str(e)

            if (
                e.response["Error"]["Code"] == "ThrottlingException"
                or e.response["Error"]["Code"] == "throttlingException"
            ):
                raise ModelThrottledException(error_message) from e

            if any(overflow_message in error_message for overflow_message in BEDROCK_CONTEXT_WINDOW_OVERFLOW_MESSAGES):
                logger.warning("bedrock threw context window overflow error")
                raise ContextWindowOverflowException(e) from e

            region = self.client.meta.region_name

            # Aid in debugging by adding more information
            add_exception_note(e, f"└ Bedrock region: {region}")
            add_exception_note(e, f"└ Model id: {self.config.get('model_id')}")

            if (
                e.response["Error"]["Code"] == "AccessDeniedException"
                and "You don't have access to the model" in error_message
            ):
                add_exception_note(
                    e,
                    "└ For more information see "
                    "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#model-access-issue",
                )

            if (
                e.response["Error"]["Code"] == "ValidationException"
                and "with on-demand throughput isn’t supported" in error_message
            ):
                add_exception_note(
                    e,
                    "└ For more information see "
                    "https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/#on-demand-throughput-isnt-supported",
                )

            raise e

        finally:
            callback()
            logger.debug("finished streaming response from model")

    def _convert_non_streaming_to_streaming(self, response: dict[str, Any]) -> Iterable[StreamEvent]:
        """Convert a non-streaming response to the streaming format.

        Args:
            response: The non-streaming response from the Bedrock model.

        Returns:
            An iterable of response events in the streaming format.
        """
        # Yield messageStart event
        yield {"messageStart": {"role": response["output"]["message"]["role"]}}

        # Process content blocks
        for content in cast(list[ContentBlock], response["output"]["message"]["content"]):
            # Yield contentBlockStart event if needed
            if "toolUse" in content:
                yield {
                    "contentBlockStart": {
                        "start": {
                            "toolUse": {
                                "toolUseId": content["toolUse"]["toolUseId"],
                                "name": content["toolUse"]["name"],
                            }
                        },
                    }
                }

                # For tool use, we need to yield the input as a delta
                input_value = json.dumps(content["toolUse"]["input"])

                yield {"contentBlockDelta": {"delta": {"toolUse": {"input": input_value}}}}
            elif "text" in content:
                # Then yield the text as a delta
                yield {
                    "contentBlockDelta": {
                        "delta": {"text": content["text"]},
                    }
                }
            elif "reasoningContent" in content:
                # Then yield the reasoning content as a delta
                yield {
                    "contentBlockDelta": {
                        "delta": {"reasoningContent": {"text": content["reasoningContent"]["reasoningText"]["text"]}}
                    }
                }

                if "signature" in content["reasoningContent"]["reasoningText"]:
                    yield {
                        "contentBlockDelta": {
                            "delta": {
                                "reasoningContent": {
                                    "signature": content["reasoningContent"]["reasoningText"]["signature"]
                                }
                            }
                        }
                    }
            elif "citationsContent" in content:
                # For non-streaming citations, emit text and metadata deltas in sequence
                # to match streaming behavior where they flow naturally
                if "content" in content["citationsContent"]:
                    text_content = "".join([content["text"] for content in content["citationsContent"]["content"]])
                    yield {
                        "contentBlockDelta": {"delta": {"text": text_content}},
                    }

                for citation in content["citationsContent"]["citations"]:
                    # Emit citation metadata, only including fields that are present
                    # Nova grounding may omit title/sourceContent
                    citation_metadata: CitationsDelta = {}
                    if "title" in citation:
                        citation_metadata["title"] = citation["title"]
                    if "location" in citation:
                        citation_metadata["location"] = citation["location"]
                    if "sourceContent" in citation:
                        citation_metadata["sourceContent"] = citation["sourceContent"]
                    yield {"contentBlockDelta": {"delta": {"citation": citation_metadata}}}

            # Yield contentBlockStop event
            yield {"contentBlockStop": {}}

        # Yield messageStop event
        yield {
            "messageStop": {
                "stopReason": response["stopReason"],
                "additionalModelResponseFields": response.get("additionalModelResponseFields"),
            }
        }

        # Yield metadata event
        if "usage" in response or "metrics" in response or "trace" in response:
            metadata: StreamEvent = {"metadata": {}}
            if "usage" in response:
                metadata["metadata"]["usage"] = response["usage"]
            if "metrics" in response:
                metadata["metadata"]["metrics"] = response["metrics"]
            if "trace" in response:
                metadata["metadata"]["trace"] = response["trace"]
            yield metadata

    def _find_detected_and_blocked_policy(self, input: Any) -> bool:
        """Recursively checks if the assessment contains a detected and blocked guardrail.

        Args:
            input: The assessment to check.

        Returns:
            True if the input contains a detected and blocked guardrail, False otherwise.

        """
        # Check if input is a dictionary
        if isinstance(input, dict):
            # Check if current dictionary has action: BLOCKED and detected: true
            if input.get("action") == "BLOCKED" and input.get("detected") and isinstance(input.get("detected"), bool):
                return True

            # Otherwise, recursively check all values in the dictionary
            return self._find_detected_and_blocked_policy(input.values())

        elif isinstance(input, (list, ValuesView)):
            # Handle case where input is a list or dict_values
            return any(self._find_detected_and_blocked_policy(item) for item in input)
        # Otherwise return False
        return False

    @override
    async def structured_output(
        self,
        output_model: type[T],
        prompt: Messages,
        system_prompt: str | None = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, T | Any], None]:
        """Get structured output from the model.

        Args:
            output_model: The output model to use for the agent.
            prompt: The prompt messages to use for the agent.
            system_prompt: System prompt to provide context to the model.
            **kwargs: Additional keyword arguments for future extensibility.

        Yields:
            Model events with the last being the structured output.
        """
        if self._openai_delegate is not None:
            async for delegate_event in self._openai_delegate.structured_output(
                output_model, prompt, system_prompt, **kwargs
            ):
                yield delegate_event
            return

        tool_spec = convert_pydantic_to_tool_spec(output_model)

        response = self.stream(
            messages=prompt,
            tool_specs=[tool_spec],
            system_prompt=system_prompt,
            tool_choice=cast(ToolChoice, {"any": {}}),
            **kwargs,
        )
        async for event in streaming.process_stream(response):
            yield event

        stop_reason, messages, _, _ = event["stop"]

        if stop_reason != "tool_use":
            raise ValueError(f'Model returned stop_reason: {stop_reason} instead of "tool_use".')

        content = messages["content"]
        output_response: dict[str, Any] | None = None
        for block in content:
            # if the tool use name doesn't match the tool spec name, skip, and if the block is not a tool use, skip.
            # if the tool use name never matches, raise an error.
            if block.get("toolUse") and block["toolUse"]["name"] == tool_spec["name"]:
                output_response = block["toolUse"]["input"]
            else:
                continue

        if output_response is None:
            raise ValueError("No valid tool use or tool use input was found in the Bedrock response.")

        yield {"output": output_model(**output_response)}

    @staticmethod
    def _get_default_model_with_warning(region_name: str, model_config: BedrockConfig | None = None) -> str:
        """Get the default Bedrock modelId based on region.

        If the region is not **known** to support inference then we show a helpful warning
        that compliments the exception that Bedrock will throw.
        If the customer provided a model_id in their config or they overrode the `DEFAULT_BEDROCK_MODEL_ID`
        then we should not process further.

        Args:
            region_name (str): region for bedrock model
            model_config (Optional[dict[str, Any]]): Model Config that caller passes in on init
        """
        model_config = model_config or {}
        if model_config.get("model_id"):
            return model_config["model_id"]

        if DEFAULT_BEDROCK_MODEL_ID != _DEFAULT_BEDROCK_MODEL_ID.format("us"):
            return DEFAULT_BEDROCK_MODEL_ID

        prefix_inference_map = {"ap": "apac"}  # some inference endpoints can be a bit different than the region prefix

        prefix = "-".join(region_name.split("-")[:-2]).lower()  # handles `us-east-1` or `us-gov-east-1`
        if prefix not in {"us", "eu", "ap", "us-gov"}:
            warnings.warn(
                f"""
            ================== WARNING ==================

                This region {region_name} does not support
                our default inference endpoint: {_DEFAULT_BEDROCK_MODEL_ID.format(prefix)}.
                Update the agent to pass in a 'model_id' like so:
                ```
                Agent(..., model='valid_model_id', ...)
                ````
                Documentation: https://docs.aws.amazon.com/bedrock/latest/userguide/inference-profiles-support.html

            ==================================================
            """,
                stacklevel=2,
            )

        default_model_id = _DEFAULT_BEDROCK_MODEL_ID.format(prefix_inference_map.get(prefix, prefix))
        warnings.warn(
            f"You're using default model '{default_model_id}', which is subject to change. "
            "Specify a model explicitly to pin the model target.",
            stacklevel=2,
        )
        return default_model_id
