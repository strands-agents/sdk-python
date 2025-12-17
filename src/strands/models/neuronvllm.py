import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, Type, TypeVar, Union, cast

from openai import AsyncOpenAI
from pydantic import BaseModel
from typing_extensions import TypedDict, Unpack, override

from ..types.content import ContentBlock, Messages, SystemContentBlock
from ..types.streaming import StopReason, StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys, warn_on_tool_choice_not_supported
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class NeuronVLLMModel(Model):
    """Neuron-vLLM model provider implementation."""

    class NeuronVLLMConfig(TypedDict, total=False):
        model_id: str
        max_model_len: Optional[int]
        max_num_seqs: Optional[int]
        tensor_parallel_size: Optional[int]
        block_size: Optional[int]
        enable_prefix_caching: Optional[bool]
        neuron_config: Optional[Dict[str, Any]]
        device: Optional[str]
        temperature: Optional[float]
        top_p: Optional[float]
        max_tokens: Optional[int]
        stop_sequences: Optional[List[str]]
        additional_args: Optional[Dict[str, Any]]
        openai_api_key: Optional[str]
        openai_api_base: Optional[str]

    def __init__(self, config: NeuronVLLMConfig):
        validate_config_keys(config, self.NeuronVLLMConfig)
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not config.get("model_id"):
            raise ValueError("model_id is required")
        self._validate_hardware()
        self.logger.info(f"Initializing NeuronVLLMModel with model: {config['model_id']}")

    def _validate_hardware(self) -> None:
        try:
            import torch_neuronx  # type: ignore
            self.logger.info("Neuron hardware validation passed")
        except ImportError:
            self.logger.warning("Neuron libraries not available - running in compatibility mode")

    @override
    def update_config(self, **model_config: Unpack[NeuronVLLMConfig]) -> None:
        validate_config_keys(model_config, self.NeuronVLLMConfig)
        self.config.update(model_config)

    @override
    def get_config(self) -> NeuronVLLMConfig:
        return self.config

    def _format_request_message_contents(self, role: str, content: ContentBlock) -> list[dict[str, Any]]:
        if "text" in content:
            return [{"role": role, "content": content["text"]}]
        if "image" in content:
            return [{"role": role, "images": [content["image"]["source"]["bytes"]]}]
        if "toolUse" in content:
            return [{"role": role, "tool_calls": [{"function": {"name": content["toolUse"]["toolUseId"], "arguments": content["toolUse"]["input"]}}]}]
        if "toolResult" in content:
            return [
                formatted
                for tool_result in content["toolResult"]["content"]
                for formatted in self._format_request_message_contents(
                    "tool",
                    {"text": json.dumps(tool_result["json"])} if "json" in tool_result else cast(ContentBlock, tool_result),
                )
            ]
        raise TypeError(f"Unsupported content type: {next(iter(content))}")

    def _format_request_messages(self, messages: Messages, system_prompt: Optional[str] = None) -> list[dict[str, Any]]:
        system_message = [{"role": "system", "content": system_prompt}] if system_prompt else []
        return system_message + [
            formatted_message
            for message in messages
            for content in message["content"]
            for formatted_message in self._format_request_message_contents(message["role"], content)
        ]

    def format_request(self, messages: Messages, tool_specs: Optional[List[ToolSpec]] = None, system_prompt: Optional[str] = None, stream: bool = True) -> dict[str, Any]:
        """Return a dictionary suitable for OpenAI Async client."""
        request: dict[str, Any] = {
            "messages": self._format_request_messages(messages, system_prompt),
            "model": self.config["model_id"],
            "temperature": self.config.get("temperature"),
            "top_p": self.config.get("top_p"),
            "max_tokens": self.config.get("max_tokens"),
            "stop": self.config.get("stop_sequences"),
            "stream": stream,
        }
        if tool_specs:
            request["functions"] = [
                {
                    "name": t["name"],
                    "description": t["description"],
                    "parameters": t["inputSchema"]["json"],
                }
                for t in tool_specs
            ]
        if self.config.get("additional_args"):
            request.update(self.config["additional_args"])
        return request

    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Convert raw events into StreamEvent."""
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}
            case "content_start":
                if event["data_type"] == "text":
                    return {"contentBlockStart": {"start": {}}}
                tool_name = event["data"].function.name
                return {"contentBlockStart": {"start": {"toolUse": {"name": tool_name, "toolUseId": tool_name}}}}
            case "content_delta":
                if event["data_type"] == "text":
                    return {"contentBlockDelta": {"delta": {"text": event["data"]}}}
                tool_arguments = event["data"].function.arguments
                return {"contentBlockDelta": {"delta": {"toolUse": {"input": json.dumps(tool_arguments)}}}}
            case "content_stop":
                return {"contentBlockStop": {}}
            case "message_stop":
                reason: StopReason = "tool_use" if event["data"] == "tool_use" else "end_turn"
                return {"messageStop": {"stopReason": reason}}
            case "metadata":
                return {"metadata": {"usage": {}, "metrics": {}}}
            case _:
                raise RuntimeError(f"Unknown chunk_type: {event['chunk_type']}")

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: Optional[List[SystemContentBlock]] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[StreamEvent, None]:
        warn_on_tool_choice_not_supported(tool_choice)

        request = self.format_request(messages, tool_specs, system_prompt, stream=True)
        client = AsyncOpenAI(
            api_key=self.config.get("openai_api_key", "EMPTY"),
            base_url=self.config.get("openai_api_base", "http://localhost:8084/v1"),
        )

        tool_requested = False
        finish_reason: str | None = None

        yield self.format_chunk({"chunk_type": "message_start"})
        yield self.format_chunk({"chunk_type": "content_start", "data_type": "text"})

        stream_response = await client.chat.completions.create(**request)
        async for chunk in stream_response:
            choice = chunk.choices[0]
            delta = choice.delta

            if delta.content:
                yield self.format_chunk({"chunk_type": "content_delta", "data_type": "text", "data": delta.content})

            if delta.tool_calls:
                for tool_call in delta.tool_calls:
                    yield self.format_chunk({"chunk_type": "content_start", "data_type": "tool", "data": tool_call})
                    yield self.format_chunk({"chunk_type": "content_delta", "data_type": "tool", "data": tool_call})
                    yield self.format_chunk({"chunk_type": "content_stop", "data_type": "tool", "data": tool_call})
                    tool_requested = True

            if choice.finish_reason:
                finish_reason = choice.finish_reason

        yield self.format_chunk({"chunk_type": "content_stop", "data_type": "text"})
        yield self.format_chunk({"chunk_type": "message_stop", "data": "tool_use" if tool_requested else finish_reason})

    @override
    async def structured_output(
        self,
        output_model: Type[T],
        prompt: Messages,
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        tool_spec = ToolSpec(
            name=output_model.__name__,
            description=f"Return a {output_model.__name__}",
            input_schema=output_model.model_json_schema(),
        )
        request = self.format_request(messages=prompt, tool_specs=[tool_spec], system_prompt=system_prompt, stream=False)
        request["tool_choice"] = {"type": "function", "function": {"name": tool_spec.name}}

        client = AsyncOpenAI(
            api_key=self.config.get("openai_api_key", "EMPTY"),
            base_url=self.config.get("openai_api_base", "http://localhost:8084/v1"),
        )
        response = await client.chat.completions.create(**request)

        message = response.choices[0].message
        if not message.tool_calls:
            raise ValueError("Expected structured output via tool call")

        tool_call = message.tool_calls[0]
        output = output_model.model_validate_json(tool_call.function.arguments)
        yield {"output": output}
