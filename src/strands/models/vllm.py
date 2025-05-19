"""vLLM model provider.

- Docs: https://github.com/vllm-project/vllm
"""

import json
import logging
from typing import Any, Iterable, Optional

import requests
from typing_extensions import TypedDict, Unpack, override

from ..types.content import Messages
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)


class VLLMModel(Model):
    """vLLM model provider implementation.

    Assumes OpenAI-compatible vLLM server at `http://<host>/v1/completions`.

    The implementation handles vLLM-specific features such as:

    - Local model invocation
    - Streaming responses
    - Tool/function calling
    """

    class VLLMConfig(TypedDict, total=False):
        """Configuration parameters for vLLM models.

        Attributes:
            additional_args: Any additional arguments to include in the request.
            max_tokens: Maximum number of tokens to generate in the response.
            model_id: vLLM model ID (e.g., "meta-llama/Llama-3.2-3B,microsoft/Phi-3-mini-128k-instruct").
            options: Additional model parameters (e.g., top_k).
            temperature: Controls randomness in generation (higher = more random).
            top_p: Controls diversity via nucleus sampling (alternative to temperature).
        """

        model_id: str
        temperature: Optional[float]
        top_p: Optional[float]
        max_tokens: Optional[int]
        stop_sequences: Optional[list[str]]
        additional_args: Optional[dict[str, Any]]

    def __init__(self, host: str, **model_config: Unpack[VLLMConfig]) -> None:
        """Initialize provider instance.

        Args:
            host: The address of the vLLM server hosting the model.
            **model_config: Configuration options for the vLLM model.
        """
        self.config = VLLMModel.VLLMConfig(**model_config)
        self.host = host.rstrip("/")
        logger.debug("Initializing vLLM provider with config: %s", self.config)

    @override
    def update_config(self, **model_config: Unpack[VLLMConfig]) -> None:
        """Update the vLLM Model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> VLLMConfig:
        """Get the vLLM model configuration.

        Returns:
            The vLLM model configuration.
        """
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> dict[str, Any]:
        """Format an vLLM chat streaming request.

        Args:
            messages: List of message objects to be processed by the model.
            tool_specs: List of tool specifications to make available to the model.
            system_prompt: System prompt to provide context to the model.

        Returns:
            An vLLM chat streaming request.
        """

        # Concatenate messages to form a prompt string
        prompt_parts = [
            f"{msg['role']}: {content['text']}" for msg in messages for content in msg["content"] if "text" in content
        ]
        if system_prompt:
            prompt_parts.insert(0, f"system: {system_prompt}")
        prompt = "\n".join(prompt_parts) + "\nassistant:"

        payload = {
            "model": self.config["model_id"],
            "prompt": prompt,
            "temperature": self.config.get("temperature", 0.7),
            "top_p": self.config.get("top_p", 1.0),
            "max_tokens": self.config.get("max_tokens", 128),
            "stop": self.config.get("stop_sequences"),
            "stream": False,  # Disable streaming
        }

        if self.config.get("additional_args"):
            payload.update(self.config["additional_args"])

        return payload

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format the vLLM response events into standardized message chunks.

        Args:
            event: A response event from the vLLM model.

        Returns:
            The formatted chunk.

        """
        choice = event.get("choices", [{}])[0]

        if "text" in choice:
            return {"contentBlockDelta": {"delta": {"text": choice["text"]}}}

        if "finish_reason" in choice:
            return {"messageStop": {"stopReason": choice["finish_reason"] or "end_turn"}}

        return {}

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[dict[str, Any]]:
        """Send the request to the vLLM model and get the streaming response.

        This method calls the /v1/completions endpoint and returns the stream of response events.

        Args:
            request: The formatted request to send to the vLLM model.

        Returns:
            An iterable of response events from the vLLM model.
        """
        headers = {"Content-Type": "application/json"}
        url = f"{self.host}/v1/completions"
        request["stream"] = True  # Enable streaming

        full_output = ""

        try:
            with requests.post(url, headers=headers, data=json.dumps(request), stream=True) as response:
                if response.status_code != 200:
                    logger.error("vLLM server error: %d - %s", response.status_code, response.text)
                    raise Exception(f"Request failed: {response.status_code} - {response.text}")

                yield {"chunk_type": "message_start"}
                yield {"chunk_type": "content_start", "data_type": "text"}

                for line in response.iter_lines(decode_unicode=True):
                    if not line:
                        continue

                    if line.startswith("data: "):
                        line = line[len("data: ") :]

                    if line.strip() == "[DONE]":
                        break

                    try:
                        data = json.loads(line)
                        choice = data.get("choices", [{}])[0]
                        text = choice.get("text", "")
                        finish_reason = choice.get("finish_reason")

                        if text:
                            full_output += text
                            print(text, end="", flush=True)  # Stream to stdout without newline
                            yield {
                                "chunk_type": "content_delta",
                                "data_type": "text",
                                "data": text,
                            }

                        if finish_reason:
                            yield {"chunk_type": "content_stop", "data_type": "text"}
                            yield {"chunk_type": "message_stop", "data": finish_reason}
                            break

                    except json.JSONDecodeError:
                        logger.warning("Failed to decode streamed line: %s", line)

                else:
                    yield {"chunk_type": "content_stop", "data_type": "text"}
                    yield {"chunk_type": "message_stop", "data": "end_turn"}

        except requests.RequestException as e:
            logger.error("Request to vLLM failed: %s", str(e))
            raise Exception("Failed to reach vLLM server") from e
