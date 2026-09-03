"""Truncate reduction method.

Replaces content with a preview (head, tail, or head-tail).
"""

from __future__ import annotations

import json
import math
from typing import Literal

from typing_extensions import TypedDict

from ...types.content import ContentBlock
from ...types.tools import ToolResult, ToolResultContent

TRUNCATED_PREFIX = "[Truncated:"

DEFAULT_PREVIEW_TOKENS = 1000
CHARS_PER_TOKEN = 4


class TruncateConfig(TypedDict, total=False):
    """Configuration for the truncate method.

    Attributes:
        preview_tokens: Number of tokens to keep as preview text. Defaults to 1,000.
        preview: Which portion of the text to keep as preview. Defaults to "head_tail".
    """

    preview_tokens: int
    preview: Literal["head", "tail", "head_tail"]


def build_preview(full_text: str, block_count: int, config: TruncateConfig | None = None) -> str:
    """Build a preview of the given text (head, tail, or head-tail depending on config).

    Args:
        full_text: The full text to preview.
        block_count: Number of original blocks being previewed (for metadata header).
        config: Truncation configuration.

    Returns:
        The preview string with truncation metadata header, or the original text if within budget.
    """
    config = config or {}
    raw_preview_tokens = config.get("preview_tokens", DEFAULT_PREVIEW_TOKENS)
    preview_tokens = (
        max(0, int(raw_preview_tokens)) if isinstance(raw_preview_tokens, (int, float)) else DEFAULT_PREVIEW_TOKENS
    )
    preview_chars = preview_tokens * CHARS_PER_TOKEN
    preview_mode = config.get("preview", "head_tail")
    total_chars = len(full_text)

    if total_chars <= preview_chars:
        return full_text

    head_share = {"head": 1.0, "tail": 0.0, "head_tail": 0.6}[preview_mode]
    head_chars = int(preview_chars * head_share)
    tail_chars = preview_chars - head_chars
    head = full_text[:head_chars]
    tail = full_text[-tail_chars:] if tail_chars > 0 else ""
    elided = total_chars - head_chars - tail_chars
    marker = f"[... {elided:,} chars elided ...]"
    preview = "\n\n".join(part for part in [head, marker, tail] if part)

    blocks_word = "block" if block_count == 1 else "blocks"
    approx_tokens = math.ceil(total_chars / CHARS_PER_TOKEN)
    result = f"{TRUNCATED_PREFIX} {block_count} {blocks_word}, ~{approx_tokens:,} tokens]\n\n{preview}"

    if len(result) >= total_chars:
        return full_text

    return result


def truncate_tool_result(block: ToolResult, config: TruncateConfig | None = None) -> ToolResult:
    """Create a replacement ToolResult with textual content truncated.

    Text and JSON blocks are serialized into a preview. Opaque blocks (images, documents)
    are preserved untouched.

    Args:
        block: The original tool result block.
        config: Truncation configuration.

    Returns:
        A new ToolResult with truncated content, or the original if no truncation needed.
    """
    textual: list[str] = []
    opaque: list[ToolResultContent] = []

    for content in block["content"]:
        if "text" in content:
            textual.append(content["text"])
        elif "json" in content:
            textual.append(json.dumps(content["json"]))
        else:
            opaque.append(content)

    if not textual:
        return block

    full_text = "\n".join(textual)
    preview = build_preview(full_text, len(textual), config)

    if preview == full_text:
        return block

    new_content: list[ToolResultContent] = [{"text": preview}, *opaque]
    return ToolResult(
        toolUseId=block["toolUseId"],
        status=block["status"],
        content=new_content,
    )


def truncate_text_block(block: ContentBlock, config: TruncateConfig | None = None) -> ContentBlock:
    """Create a replacement text ContentBlock containing the truncated preview.

    Args:
        block: The original text content block (must have a "text" key).
        config: Truncation configuration.

    Returns:
        A new ContentBlock with truncated text, or the original if no truncation needed.
    """
    text = block["text"]
    preview = build_preview(text, 1, config)
    if preview == text:
        return block
    return ContentBlock(text=preview)
