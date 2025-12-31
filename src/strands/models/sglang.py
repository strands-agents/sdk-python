"""SGLang model provider (native API).

This provider integrates with the SGLang Runtime **native** HTTP APIs, primarily:
- `/generate` for text generation (supports SSE streaming)
- `/tokenize` for tokenizing a prompt (optional; used for token-out prompt ids)

Docs:
- https://docs.sglang.io/basic_usage/native_api.html

Notes:
-----
`/generate` is completion-style: it accepts a single prompt (or input token IDs) and returns a single completion.
Strands uses a message-based interface, so this provider serializes text-only conversations into a single prompt.
Tool calling is not supported via `/generate`.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncGenerator, AsyncIterable, Optional, Type, TypedDict, TypeVar, Union, cast

import httpx
from pydantic import BaseModel
from typing_extensions import Unpack, override

from ..types.content import Messages, SystemContentBlock
from ..types.event_loop import Metrics, Usage
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.streaming import StreamEvent
from ..types.tools import ToolChoice, ToolSpec
from ._validation import validate_config_keys
from .model import Model

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class SGLangModel(Model):
    """SGLang native `/generate` provider with token-in/out helpers."""

    class SGLangConfig(TypedDict, total=False):
        """Configuration options for SGLang native API models."""

        base_url: str
        model_id: Optional[str]
        params: Optional[dict[str, Any]]  # default sampling params (merged into sampling_params)
        timeout: Optional[Union[float, tuple[float, float]]]

    def __init__(
        self,
        *,
        return_token_ids: bool = False,
        **model_config: Unpack[SGLangConfig],
    ) -> None:
        """Create an SGLang model client."""
        validate_config_keys(model_config, self.SGLangConfig)

        base_url = str(model_config.get("base_url") or "http://localhost:30000").rstrip("/")
        timeout = model_config.get("timeout")
        if isinstance(timeout, tuple):
            timeout_obj = httpx.Timeout(connect=timeout[0], read=timeout[1])
        else:
            timeout_obj = httpx.Timeout(timeout or 30.0)

        self.client = httpx.AsyncClient(base_url=base_url, timeout=timeout_obj)
        self.config = dict(model_config)
        self.config["base_url"] = base_url
        self._return_token_ids_default = bool(return_token_ids)

        logger.debug("config=<%s> | initializing", self.config)

    @override
    def update_config(self, **model_config: Unpack[SGLangConfig]) -> None:  # type: ignore[override]
        validate_config_keys(model_config, self.SGLangConfig)
        if "base_url" in model_config and model_config["base_url"]:
            # Preserve base_url canonicalization
            self.config["base_url"] = str(model_config["base_url"]).rstrip("/")
        self.config.update(model_config)

    @override
    def get_config(self) -> SGLangConfig:
        return cast(SGLangModel.SGLangConfig, self.config)

    def _messages_to_prompt(
        self,
        messages: Messages,
        system_prompt: Optional[str],
        *,
        system_prompt_content: Optional[list[SystemContentBlock]] = None,
    ) -> str:
        # Only support text content blocks. Tools and multimodal content are not supported via /generate.
        def text_from_blocks(role: str, blocks: list[dict[str, Any]]) -> str:
            parts: list[str] = []
            for block in blocks:
                if "text" in block:
                    parts.append(str(block["text"]))
                else:
                    raise TypeError(f"SGLangModel only supports text content blocks. got role={role} block={block}")
            return "".join(parts)

        # Back-compat: if system_prompt is provided but system_prompt_content is None.
        if system_prompt and system_prompt_content is None:
            system_prompt_content = [{"text": system_prompt}]

        lines: list[str] = []
        for block in system_prompt_content or []:
            if "text" in block:
                lines.append(f"system: {block['text']}")

        for msg in messages:
            role = msg.get("role", "user")
            content = msg.get("content", [])
            # Reject tool/multimodal blocks early
            if any(k in b for b in content for k in ("toolUse", "toolResult", "image", "document", "reasoningContent")):
                raise TypeError("SGLangModel /generate does not support tools or multimodal message blocks.")
            text = text_from_blocks(str(role), cast(list[dict[str, Any]], content))
            if text.strip():
                lines.append(f"{role}: {text}")

        # Add a final assistant prefix to make the completion shape stable.
        lines.append("assistant:")
        return "\n".join(lines).strip() + "\n"

    async def _tokenize(self, prompt: str) -> list[int]:
        model_id = self.get_config().get("model_id")
        payload: dict[str, Any] = {
            "prompt": prompt,
            "add_special_tokens": False,
        }
        if model_id:
            payload["model"] = model_id

        resp = await self.client.post("/tokenize", json=payload)
        resp.raise_for_status()
        data = resp.json()
        tokens = data.get("tokens")
        if not isinstance(tokens, list) or not all(isinstance(x, int) for x in tokens):
            raise ValueError(f"Unexpected /tokenize response: {data}")
        return cast(list[int], tokens)

    def _build_generate_payload(
        self,
        *,
        prompt: Optional[str],
        prompt_token_ids: Optional[list[int]],
        sampling_params: dict[str, Any],
        stream: bool,
    ) -> dict[str, Any]:
        model_id = self.get_config().get("model_id")
        payload: dict[str, Any] = {"stream": stream}

        if model_id:
            payload["model"] = model_id

        if prompt_token_ids is not None:
            payload["input_ids"] = prompt_token_ids
        else:
            payload["text"] = prompt or ""

        if sampling_params:
            payload["sampling_params"] = sampling_params

        return payload

    @override
    async def stream(
        self,
        messages: Messages,
        tool_specs: Optional[list[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
        *,
        tool_choice: ToolChoice | None = None,
        system_prompt_content: list[SystemContentBlock] | None = None,
        **kwargs: Any,
    ) -> AsyncIterable[StreamEvent]:
        if tool_specs is not None or tool_choice is not None:
            raise TypeError("SGLangModel /generate does not support tool_specs/tool_choice.")

        return_token_ids = bool(kwargs.pop("return_token_ids", self._return_token_ids_default))
        prompt_token_ids = kwargs.pop("prompt_token_ids", None)
        if prompt_token_ids is not None:
            if (
                not isinstance(prompt_token_ids, list)
                or not prompt_token_ids
                or not all(isinstance(x, int) for x in prompt_token_ids)
            ):
                raise TypeError("prompt_token_ids must be a non-empty list[int].")
            prompt_token_ids = cast(list[int], prompt_token_ids)

        sampling_params: dict[str, Any] = {}
        cfg_params = self.get_config().get("params")
        if isinstance(cfg_params, dict):
            sampling_params.update(cfg_params)

        if "sampling_params" in kwargs:
            sp = kwargs.pop("sampling_params")
            if sp is not None:
                if not isinstance(sp, dict):
                    raise TypeError("sampling_params must be a dict when provided.")
                sampling_params.update(cast(dict[str, Any], sp))

        sampling_params.update(kwargs)

        prompt_text: str | None = None
        prompt_token_ids_out: list[int] | None = None
        if prompt_token_ids is None:
            prompt_text = self._messages_to_prompt(messages, system_prompt, system_prompt_content=system_prompt_content)
            if return_token_ids:
                try:
                    prompt_token_ids_out = await self._tokenize(prompt_text)
                except httpx.HTTPStatusError as e:
                    if e.response.status_code == 429:
                        raise ModelThrottledException(str(e)) from e
                    raise

        payload = self._build_generate_payload(
            prompt=prompt_text,
            prompt_token_ids=prompt_token_ids,
            sampling_params=sampling_params,
            stream=True,
        )

        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}

        prev_text = ""
        last_output_ids: list[int] = []
        last_meta: dict[str, Any] | None = None

        try:
            async with self.client.stream("POST", "/generate", json=payload) as resp:
                resp.raise_for_status()

                async for line in resp.aiter_lines():
                    if not line:
                        continue
                    if not line.startswith("data:"):
                        continue
                    data_content = line[len("data:") :].strip()
                    if data_content == "[DONE]":
                        break
                    try:
                        event = json.loads(data_content)
                    except json.JSONDecodeError:
                        continue

                    new_text = event.get("text")
                    if isinstance(new_text, str):
                        if new_text.startswith(prev_text):
                            delta = new_text[len(prev_text) :]
                        else:
                            delta = new_text
                        prev_text = new_text
                        if delta:
                            yield {"contentBlockDelta": {"delta": {"text": delta}}}

                    output_ids = event.get("output_ids")
                    if isinstance(output_ids, list) and all(isinstance(x, int) for x in output_ids):
                        last_output_ids = cast(list[int], output_ids)

                    meta = event.get("meta_info")
                    if isinstance(meta, dict):
                        last_meta = cast(dict[str, Any], meta)

        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            if status == 400:
                raise ContextWindowOverflowException(str(e)) from e
            if status in (429, 503):
                raise ModelThrottledException(str(e)) from e
            raise

        yield {"contentBlockStop": {}}

        additional: dict[str, Any] = {}
        if prompt_token_ids is not None:
            additional["prompt_token_ids"] = prompt_token_ids
        elif prompt_token_ids_out is not None:
            additional["prompt_token_ids"] = prompt_token_ids_out
        if last_output_ids:
            additional["token_ids"] = last_output_ids

        stop_reason: str = "end_turn"
        if last_meta and isinstance(last_meta.get("finish_reason"), dict):
            fr = cast(dict[str, Any], last_meta.get("finish_reason"))
            if fr.get("type") == "length":
                stop_reason = "max_tokens"

        yield {"messageStop": {"stopReason": cast(Any, stop_reason), "additionalModelResponseFields": additional}}

        if last_meta:
            usage: Usage = {
                "inputTokens": int(last_meta.get("prompt_tokens") or 0),
                "outputTokens": int(last_meta.get("completion_tokens") or 0),
                "totalTokens": int((last_meta.get("prompt_tokens") or 0) + (last_meta.get("completion_tokens") or 0)),
            }
            latency_ms = int(float(last_meta.get("e2e_latency") or 0.0) * 1000)
            metrics: Metrics = {"latencyMs": latency_ms}
            yield {"metadata": {"usage": usage, "metrics": metrics}}

    @override
    async def structured_output(
        self, output_model: Type[T], prompt: Messages, system_prompt: Optional[str] = None, **kwargs: Any
    ) -> AsyncGenerator[dict[str, Union[T, Any]], None]:
        instruction = (
            "Return ONLY valid JSON that matches the schema. Do not include any extra keys or prose.\n"
            f"Schema: {output_model.model_json_schema()}\n"
        )
        prompt2: Messages = [
            {"role": "user", "content": [{"text": instruction}]},
            *prompt,
        ]

        text = ""
        async for event in self.stream(
            prompt2,
            tool_specs=None,
            system_prompt=system_prompt,
            system_prompt_content=kwargs.pop("system_prompt_content", None),
            **kwargs,
        ):
            if "contentBlockDelta" in event:
                delta = event["contentBlockDelta"]["delta"]
                if "text" in delta:
                    text += delta["text"]

        try:
            yield {"output": output_model.model_validate_json(text.strip())}
        except Exception as e:
            raise ValueError(f"Failed to parse structured output JSON: {e}") from e
