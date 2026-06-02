"""Internal marshalling helpers shared by the SDK shim layer.

These translate user-friendly inputs into the wire shape WIT records expect:
variant-arm wrapping for typed unions, ``ContentBlock`` arm wrapping for raw
payloads, JSON encoding of arbitrary Python objects, and reflection helpers
used by the ``@tool`` decorator. None of this is part of the public API.
"""

from __future__ import annotations

import json
import types as _stdtypes
import typing
from dataclasses import asdict, is_dataclass
from typing import Any

from wasmtime.component import VariantCase as _WitVariantCase

from strands import types

# --- Variant-arm wrapping ----------------------------------------------------

CONTENT_ARM_BY_TYPE: dict[type, type] = {
    types.TextBlock: types.ContentBlock.Text,
    types.JsonBlock: types.ContentBlock.Json,
    types.ToolUseBlock: types.ContentBlock.ToolUse,
    types.ToolResultBlock: types.ContentBlock.ToolResult,
    types.ReasoningBlock: types.ContentBlock.Reasoning,
    types.CachePointBlock: types.ContentBlock.CachePoint,
    types.ImageBlock: types.ContentBlock.Image,
    types.VideoBlock: types.ContentBlock.Video,
    types.DocumentBlock: types.ContentBlock.Document,
    types.CitationsBlock: types.ContentBlock.Citations,
    types.InterruptResponseBlock: types.ContentBlock.InterruptResponse,
}

MODEL_ARM_BY_TYPE: dict[type, type] = {
    types.BedrockModel: types.ModelConfig.Bedrock,
    types.AnthropicModel: types.ModelConfig.Anthropic,
    types.OpenaiModel: types.ModelConfig.Openai,
    types.GoogleModel: types.ModelConfig.Gemini,
    types.CustomModel: types.ModelConfig.Custom,
}

CM_ARM_BY_TYPE: dict[type, type] = {
    types.SlidingWindowConversationManager: types.ConversationManagerConfig.SlidingWindow,
    types.SummarizingConversationManager: types.ConversationManagerConfig.Summarizing,
}

VENDED_TOOL_ARM_BY_TYPE: dict[type, type] = {
    types.BashTool: types.VendedTool.Bash,
    types.FileEditorTool: types.VendedTool.FileEditor,
    types.HttpRequestTool: types.VendedTool.HttpRequest,
    types.NotebookTool: types.VendedTool.Notebook,
}

VENDED_PLUGIN_ARM_BY_TYPE: dict[type, type] = {
    types.AgentSkills: types.VendedPlugin.Skills,
    types.ContextOffloader: types.VendedPlugin.ContextOffloader,
}


def wrap(value: Any, arm_table: dict[type, type]) -> Any:
    """Wrap ``value`` in the variant arm whose payload type matches its MRO.

    Walks the MRO so SDK ergonomic subclasses (``BedrockModel`` extends
    ``types.BedrockModel``) hit the same arm as the raw bindgen type. Returns
    the value unchanged if it's already an arm (so passing a fully-constructed
    ``ModelConfig.Bedrock(...)`` is idempotent) or doesn't match anything.
    """
    if value is None or isinstance(value, _WitVariantCase):
        return value
    for cls in type(value).__mro__:
        arm = arm_table.get(cls)
        if arm is not None:
            return arm(value)
    return value


# --- Content-block / prompt coercion ----------------------------------------


def as_content_block(item: Any) -> types.ContentBlock:
    """Wrap any accepted content shape as a ``ContentBlock`` variant arm."""
    if isinstance(item, str):
        return types.ContentBlock.Text(types.TextBlock(text=item))
    for block_type, arm in CONTENT_ARM_BY_TYPE.items():
        if isinstance(item, block_type):
            return arm(item)
    assert isinstance(item, types.ContentBlock)
    return item


def coerce_prompt(value: Any) -> str | list[types.ContentBlock]:
    """Coerce a string or iterable of content blocks to a ``prompt-input``."""
    if isinstance(value, str):
        return value
    if isinstance(value, types.Message):
        raise TypeError(
            "stream_async/invoke take content blocks as input, not Messages. "
            "Pass conversation history via Agent(messages=[...]) instead."
        )
    return [as_content_block(c) for c in value]


def coerce_tool_choice(value: Any) -> types.ToolChoice | None:
    if value is None:
        return None
    if isinstance(value, str):
        return types.ToolChoice.Named(value)
    return value


# --- JSON helpers -----------------------------------------------------------


def extras_to_json(extras: dict[str, Any] | None) -> str | None:
    return json.dumps(extras) if extras else None


def json_default(obj: Any) -> Any:
    if is_dataclass(obj) and not isinstance(obj, type):
        return asdict(obj)
    if hasattr(obj, "__dict__"):
        return {k: v for k, v in obj.__dict__.items() if not k.startswith("_")}
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")


# --- Tool reflection --------------------------------------------------------


def py_type_to_schema(py_type: Any) -> dict[str, Any]:
    origin = typing.get_origin(py_type)

    # Strip Annotated[T, ...] -- only the runtime type matters for the schema.
    if origin is typing.Annotated:
        return py_type_to_schema(typing.get_args(py_type)[0])

    # Optional[T] / Union[T, None] / T | None: emit T's schema and mark nullable.
    if origin is typing.Union or origin is _stdtypes.UnionType:
        args = typing.get_args(py_type)
        non_none = [a for a in args if a is not type(None)]
        nullable = len(non_none) != len(args)
        if len(non_none) == 1:
            schema = py_type_to_schema(non_none[0])
            if nullable:
                schema = {**schema, "nullable": True}
            return schema
        return {}  # heterogeneous union -- caller should supply input_schema

    if py_type is str:
        return {"type": "string"}
    if py_type is int:
        return {"type": "integer"}
    if py_type is float:
        return {"type": "number"}
    if py_type is bool:
        return {"type": "boolean"}
    if origin is list:
        args = typing.get_args(py_type)
        return {"type": "array", "items": py_type_to_schema(args[0]) if args else {}}
    if origin is dict:
        return {"type": "object"}
    if origin is typing.Literal:
        return {"enum": list(typing.get_args(py_type))}
    return {}
