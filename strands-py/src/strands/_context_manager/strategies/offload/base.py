"""Base offload strategy and shared infrastructure."""

from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Literal

from typing_extensions import TypedDict

from ....types.content import ContentBlock, Message, Messages
from ....types.tools import ToolUse
from ...types import ContextState

if TYPE_CHECKING:
    from ....agent.agent import Agent

logger = logging.getLogger(__name__)

OffloadTarget = Literal["*", "tool_results", "tool_result_errors", "assistant_text", "user_text"] | list[str]
"""Target for offload operations."""


class OffloadConditions(TypedDict, total=False):
    """Conditions that determine when an offload strategy fires.

    Attributes:
        threshold: Token threshold above which individual blocks are offloaded.
        utilization: Context utilization ratio (0-1+) above which the strategy fires.
        preserve_recent: Number of most recent matching messages to leave untouched.
    """

    threshold: int
    utilization: float
    preserve_recent: int


# --- Block type helpers ---


def _is_tool_result_block(block: ContentBlock) -> bool:
    return "toolResult" in block


def _is_tool_use_block(block: ContentBlock) -> bool:
    return "toolUse" in block


def _is_text_block(block: ContentBlock) -> bool:
    return "text" in block and "toolResult" not in block and "toolUse" not in block


# --- Shared helpers ---


def _finite_or_none(value: int | float | None) -> int | float | None:
    if isinstance(value, (int, float)) and math.isfinite(value):
        return max(0, value)
    return None


def build_tool_name_map(messages: Messages) -> dict[str, str]:
    """Build a toolUseId -> toolName map from all assistant messages.

    Args:
        messages: The conversation messages.

    Returns:
        Mapping from tool use ID to tool name.
    """
    name_map: dict[str, str] = {}
    for message in messages:
        if message["role"] != "assistant":
            continue
        for block in message["content"]:
            if "toolUse" in block:
                tool_use: ToolUse = block["toolUse"]
                name_map[tool_use["toolUseId"]] = tool_use["name"]
    return name_map


def tool_matches_target(
    block: ContentBlock,
    target: OffloadTarget,
    tool_name_map: dict[str, str],
    tool_include_filter: set[str] | None,
    tool_exclude_filter: set[str] | None,
) -> bool:
    """Check if a tool result block matches the given target.

    Args:
        block: A content block containing a toolResult.
        target: The offload target.
        tool_name_map: Mapping of tool use IDs to tool names.
        tool_include_filter: Set of tool names to include (if any).
        tool_exclude_filter: Set of tool names to exclude (if any).

    Returns:
        True if the block matches the target.
    """
    tool_result = block["toolResult"]
    if target == "*":
        return True
    if target == "tool_results":
        return tool_result["status"] == "success"
    if target == "tool_result_errors":
        return tool_result["status"] == "error"

    tool_name = tool_name_map.get(tool_result["toolUseId"])
    if not tool_name:
        return False

    if tool_exclude_filter:
        return tool_name not in tool_exclude_filter
    if tool_include_filter:
        return tool_name in tool_include_filter

    return False


def target_matches_message(target: OffloadTarget | None, message: Message) -> bool:
    """Check if a message matches a text-level target.

    Args:
        target: The offload target.
        message: The message to check.

    Returns:
        True if the message matches.
    """
    if target is None or target == "*":
        return True
    if target == "assistant_text":
        return message["role"] == "assistant" and any(_is_text_block(b) for b in message["content"])
    if target == "user_text":
        return message["role"] == "user" and any(_is_text_block(b) for b in message["content"])
    return False


def message_matches_target(
    message: Message,
    target: OffloadTarget | None,
    tool_name_map: dict[str, str],
    tool_include_filter: set[str] | None,
    tool_exclude_filter: set[str] | None,
) -> bool:
    """Check if a message matches the target (text-level or tool result).

    Args:
        message: The message to check.
        target: The offload target.
        tool_name_map: Tool use ID to name mapping.
        tool_include_filter: Tool name include filter.
        tool_exclude_filter: Tool name exclude filter.

    Returns:
        True if the message matches.
    """
    if target_matches_message(target, message):
        return True
    if target is None:
        return False

    if message["role"] != "user":
        return False
    for block in message["content"]:
        if _is_tool_result_block(block):
            if tool_matches_target(block, target, tool_name_map, tool_include_filter, tool_exclude_filter):
                return True
    return False


def get_oldest_matches(
    messages: Messages,
    target: OffloadTarget | None,
    count: int,
    tool_name_map: dict[str, str],
    tool_include_filter: set[str] | None,
    tool_exclude_filter: set[str] | None,
) -> list[Message]:
    """Return target-matching messages excluding the N most recent matches.

    Args:
        messages: All messages.
        target: The offload target.
        count: Number of most recent matches to exclude.
        tool_name_map: Tool use ID to name mapping.
        tool_include_filter: Tool name include filter.
        tool_exclude_filter: Tool name exclude filter.

    Returns:
        Oldest matching messages (excluding the most recent `count`).
    """
    matching = [
        msg
        for msg in messages
        if message_matches_target(msg, target, tool_name_map, tool_include_filter, tool_exclude_filter)
    ]
    if count >= len(matching):
        return []
    return matching[:-count]


def collect_removable_with_pair(messages: Messages, index: int) -> list[Message]:
    """Collect a message and its paired partner for safe removal.

    If removing a message would orphan a tool-use/tool-result pair, includes the partner.
    Skips messages[0] (head-pin).

    Args:
        messages: All messages.
        index: Index of the message to remove.

    Returns:
        List of messages that should be removed together.
    """
    if index <= 0 or index >= len(messages):
        return []

    message = messages[index]
    result: list[Message] = [message]

    has_tool_result = any(_is_tool_result_block(b) for b in message["content"])
    if has_tool_result and index > 0:
        prev = messages[index - 1]
        if any(_is_tool_use_block(b) for b in prev["content"]):
            if index - 1 > 0:
                result.append(prev)
            else:
                return []

    has_tool_use = any(_is_tool_use_block(b) for b in message["content"])
    if has_tool_use and index < len(messages) - 1:
        next_msg = messages[index + 1]
        if any(_is_tool_result_block(b) for b in next_msg["content"]):
            result.append(next_msg)

    return result


def splice_with_pairs(messages: Messages, to_remove: list[Message]) -> tuple[int, int]:
    """Remove messages from the list, respecting tool-use/tool-result pairs.

    Args:
        messages: The message list (mutated in place).
        to_remove: Messages to remove.

    Returns:
        Tuple of (number of messages removed, lowest index that was removed).
    """
    identity_map = {id(msg): idx for idx, msg in enumerate(messages)}
    to_splice: set[int] = set()
    for message in to_remove:
        index = identity_map.get(id(message))
        if index is None:
            continue
        for removable in collect_removable_with_pair(messages, index):
            removable_index = identity_map.get(id(removable))
            if removable_index is not None:
                to_splice.add(removable_index)

    removed = 0
    lowest_index = len(messages)
    for index in sorted(to_splice, reverse=True):
        if index < lowest_index:
            lowest_index = index
        messages.pop(index)
        removed += 1

    return removed, lowest_index


def repair_alternation(messages: Messages) -> None:
    """Merge consecutive same-role messages to restore user/assistant alternation.

    Mutates the messages list in place.

    Args:
        messages: The message list to repair.
    """
    write_index = 0
    for read_index in range(len(messages)):
        current = messages[read_index]
        if write_index > 0 and messages[write_index - 1]["role"] == current["role"]:
            prev = messages[write_index - 1]
            messages[write_index - 1] = Message(
                role=prev["role"],
                content=[*prev["content"], *current["content"]],
            )
        else:
            messages[write_index] = current
            write_index += 1
    del messages[write_index:]


def resolve_tool_filter(target: OffloadTarget | None) -> tuple[set[str] | None, set[str] | None]:
    """Parse a string list target into include/exclude filter sets.

    Entries must be prefixed with ``tool::`` (e.g. ``'tool::bash'``).
    An additional ``!`` prefix excludes (e.g. ``'!tool::bash'``).

    Args:
        target: The offload target.

    Returns:
        Tuple of (include_filter, exclude_filter).
    """
    if not isinstance(target, list):
        return None, None

    tool_prefix = "tool::"
    includes: list[str] = []
    excludes: list[str] = []

    for entry in target:
        if entry.startswith("!"):
            name = entry[1:]
            excludes.append(name[len(tool_prefix):] if name.startswith(tool_prefix) else name)
        else:
            includes.append(entry[len(tool_prefix):] if entry.startswith(tool_prefix) else entry)

    if excludes and includes:
        logger.warning(
            "includes=<%s>, excludes=<%s> | tool filter contains both, excludes will be ignored",
            includes,
            excludes,
        )
        return set(includes), None
    if excludes:
        return None, set(excludes)
    if includes:
        return set(includes), None

    return None, None


# --- Base strategy class ---


class BaseOffloadStrategy(ABC):
    """Shared offload logic: target routing, eager hooks, preserveRecent."""

    _target: OffloadTarget | None
    _threshold: int | None
    _utilization_threshold: float | None
    _preserve_recent: int
    _removal_ratio: float = 0.3
    _include_filter: set[str] | None
    _exclude_filter: set[str] | None

    def __init__(self, target: OffloadTarget | None = None, conditions: OffloadConditions | None = None) -> None:
        if isinstance(target, list) and len(target) == 0:
            raise ValueError("Empty array target matches nothing — provide at least one target")

        self._target = target
        conditions = conditions or {}
        threshold = _finite_or_none(conditions.get("threshold"))
        self._threshold = int(threshold) if threshold is not None else None
        util = _finite_or_none(conditions.get("utilization"))
        self._utilization_threshold = float(util) if util is not None else None
        preserve = _finite_or_none(conditions.get("preserve_recent"))
        self._preserve_recent = int(preserve) if preserve is not None else 0

        self._include_filter, self._exclude_filter = resolve_tool_filter(target)

    @property
    def _is_message_level(self) -> bool:
        return self._utilization_threshold is not None

    def init(self, agent: Agent) -> None:
        """Register eager hooks if this is a per-block strategy without preserveRecent.

        Args:
            agent: The agent to register hooks on.
        """
        from ....hooks.events import MessageAddedEvent

        if self._is_message_level:
            return
        if self._preserve_recent > 0:
            return

        async def _eager_hook(event: MessageAddedEvent) -> None:
            messages = event.agent.messages
            tool_name_map = build_tool_name_map(messages)
            await self._transform_blocks(event.message, messages, tool_name_map, event.agent)

        agent.hooks.add_callback(MessageAddedEvent, _eager_hook)

    async def apply(self, context: ContextState) -> bool:
        """Apply the strategy to the context.

        Args:
            context: Current context state.

        Returns:
            True if the strategy made changes.
        """
        if self._is_message_level:
            if context.utilization < self._utilization_threshold:  # type: ignore[operator]
                return False
            return await self._apply_per_message(context)

        return await self._apply_per_block(context)

    async def _apply_per_block(self, context: ContextState) -> bool:
        """Per-block execution: walk each message, transform individual blocks above threshold."""
        messages = context.messages
        agent = context.agent
        tool_name_map = build_tool_name_map(messages)
        if self._preserve_recent > 0:
            eligible = get_oldest_matches(
                messages, self._target, self._preserve_recent, tool_name_map, self._include_filter, self._exclude_filter
            )
        else:
            eligible = list(messages)

        acted = False
        for message in eligible:
            if await self._transform_blocks(message, messages, tool_name_map, agent):
                acted = True

        return acted

    async def _apply_per_message(self, context: ContextState) -> bool:
        """Message-level execution: remove oldest 30% of eligible messages with pair safety."""
        messages = context.messages
        if len(messages) <= 1:
            return False

        eligible = await self._get_eligible_messages(context)
        if not eligible:
            return False

        target_removal = max(1, int(len(eligible) * self._removal_ratio))
        to_remove = eligible[:target_removal]

        removed, lowest_index = splice_with_pairs(messages, to_remove)
        if removed == 0:
            return False

        marker = self._make_removal_marker(removed)
        if marker:
            insert_index = max(1, min(lowest_index, len(messages)))
            messages.insert(insert_index, Message(role="user", content=[ContentBlock(text=marker)]))

        repair_alternation(messages)
        return True

    def _make_removal_marker(self, count: int) -> str | None:
        """Override to insert a marker when messages are removed. Return None for no marker."""
        return None

    def _block_matches_target(
        self, block: ContentBlock, message: Message, tool_name_map: dict[str, str]
    ) -> bool:
        """Check whether a block is eligible for offload given target and filters."""
        if _is_tool_use_block(block):
            return False
        if _is_text_block(block):
            return target_matches_message(self._target, message)
        if _is_tool_result_block(block):
            return self._target is None or tool_matches_target(
                block, self._target, tool_name_map, self._include_filter, self._exclude_filter
            )
        return self._target is None or self._target == "*"

    async def _transform_blocks(
        self,
        message: Message,
        messages: Messages,
        tool_name_map: dict[str, str],
        agent: Agent,
    ) -> bool:
        """Process eligible blocks in a message."""
        effective_threshold = self._threshold or 0
        acted = False
        content = message["content"]
        for block_index in range(len(content)):
            block = content[block_index]
            if not self._block_matches_target(block, message, tool_name_map):
                continue

            tokens = await agent.model.count_tokens([Message(role=message["role"], content=[block])])
            if tokens <= effective_threshold:
                continue

            if _is_text_block(block) or _is_tool_result_block(block):
                replacement = await self._replace_block(block, tokens, message, agent)
            else:
                block_type = next(
                    (media_type for media_type in ("image", "document", "audio", "video") if media_type in block),
                    "media",
                )
                replacement = ContentBlock(text=f"[Offloaded: {block_type} block, ~{tokens:,} tokens]")
            if replacement is not None and replacement is not block:
                content[block_index] = replacement
                acted = True

        return acted

    async def _get_eligible_messages(self, context: ContextState) -> list[Message]:
        """Collect eligible messages for message-level operations."""
        messages = context.messages
        agent = context.agent
        tool_name_map = build_tool_name_map(messages)

        head_id = id(messages[0]) if messages else None
        if self._preserve_recent > 0:
            candidates = [
                msg
                for msg in get_oldest_matches(
                    messages,
                    self._target,
                    self._preserve_recent,
                    tool_name_map,
                    self._include_filter,
                    self._exclude_filter,
                )
                if id(msg) != head_id
            ]
        else:
            candidates = [
                msg
                for idx, msg in enumerate(messages)
                if idx > 0
                and message_matches_target(msg, self._target, tool_name_map, self._include_filter, self._exclude_filter)
            ]

        if self._threshold is None:
            return candidates

        eligible: list[Message] = []
        for message in candidates:
            has_oversize = False
            for block in message["content"]:
                if not self._block_matches_target(block, message, tool_name_map):
                    continue
                tokens = await agent.model.count_tokens([Message(role=message["role"], content=[block])])
                if tokens > self._threshold:
                    has_oversize = True
                    break
            if has_oversize:
                eligible.append(message)
        return eligible

    @abstractmethod
    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
    ) -> ContentBlock | None:
        """Transform a block. Return the replacement, or None to skip."""
        ...


# --- Emergency truncation strategy ---


class EmergencyTruncateStrategy(BaseOffloadStrategy):
    """Last-resort strategy that drops the oldest 20% of messages when still overflowing."""

    _removal_ratio: float = 0.2

    @property
    def name(self) -> str:
        """Strategy name."""
        return "offload:emergency-truncate"

    def __init__(self) -> None:
        super().__init__("*")

    def init(self, agent: Agent) -> None:
        """No eager hooks for emergency truncation."""

    async def apply(self, context: ContextState) -> bool:
        """Fire only when utilization >= 1.0 and messages > 3.

        Args:
            context: Current context state.

        Returns:
            True if messages were removed.
        """
        if len(context.messages) <= 3:
            return False
        tokens = await context.agent.model.count_tokens(context.messages)
        utilization = context.agent.model.estimate_utilization(tokens)
        if utilization < 1.0:
            return False
        state = ContextState(messages=context.messages, agent=context.agent, utilization=utilization)
        return await self._apply_per_message(state)

    async def _replace_block(
        self,
        block: ContentBlock,
        tokens: int,
        message: Message,
        agent: Agent,
    ) -> ContentBlock | None:
        return None
