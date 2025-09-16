"""Conversation manager that applies message mapping/transformation functions.

This module provides a simple, flexible approach to conversation management through
composable message mappers. Instead of complex strategy hierarchies, users provide
callable functions or classes that map messages to transformed versions or None.
"""

import copy
import logging
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from typing_extensions import Protocol, override

from ...agent.conversation_manager.conversation_manager import ConversationManager
from ...types.content import Message, Messages
from ...types.tools import ToolResult, ToolResultContent

if TYPE_CHECKING:
    from ...agent.agent import Agent

logger = logging.getLogger(__name__)


class MessageMapper(Protocol):
    """Protocol for message mapping/transformation functions.

    A MessageMapper is any callable that takes a message and its context,
    and returns either a transformed message or None (to remove it).

    This protocol enables both simple lambda functions and complex class-based
    mappers to be used interchangeably.

    Important: Mappers should be stateless with respect to conversation history.
    While configuration parameters (like thresholds or templates) are acceptable,
    mappers should not accumulate state across invocations since mapper-specific
    state is not persisted by the conversation manager.

    Example:
        # Simple lambda mapper
        remove_old = lambda msg, idx, msgs: None if idx < 5 else msg

        # Class-based mapper with configuration (stateless)
        class CustomMapper:
            def __init__(self, threshold: int):
                self.threshold = threshold  # Configuration only, not accumulated state

            def __call__(self, message, index, messages):
                # transformation logic using self.threshold
                return transformed_message
    """

    def __call__(self, message: Message, index: int, messages: Messages) -> Optional[Message]:
        """Transform a message.

        Args:
            message: The message to transform
            index: The index of this message in the conversation
            messages: The full conversation history (read-only)

        Returns:
            Transformed message, or None to remove the message entirely
        """
        ...


class LargeToolResultMapper:
    """Maps messages by compressing large tool results while preserving structure.

    This mapper identifies tool results that exceed a token threshold and compresses
    them using simple heuristics (text truncation, JSON summarization) while
    maintaining essential information about tool execution status.

    This mapper is stateless - it only uses configuration parameters and does not
    accumulate state across invocations.

    Example:
        mapper = LargeToolResultMapper(
            max_tokens=50_000,
            truncate_at=500,
            compression_template="[Compressed: {original_size} -> {compressed_size} tokens]"
        )

        manager = MappingConversationManager(
            mapper=mapper,
            preserve_first=1,
            preserve_last=2
        )
    """

    def __init__(
        self,
        max_tokens: int = 50_000,
        truncate_at: int = 500,
        compression_template: str = (
            "[Tool result compressed: {original_size} tokens -> {compressed_size} tokens. Original status: {status}]"
        ),
    ):
        """Initialize the large tool result mapper.

        Args:
            max_tokens: Maximum tokens allowed in tool results before compression
            truncate_at: Character length threshold for truncating text content
            compression_template: Template string for compression messages.
                Available placeholders: {original_size}, {compressed_size}, {status}
        """
        self.max_tokens = max_tokens
        self.truncate_at = truncate_at
        self.compression_template = compression_template

    def __call__(self, message: Message, index: int, messages: Messages) -> Optional[Message]:
        """Transform message by compressing large tool results.

        Args:
            message: The message to potentially compress
            index: Index of this message in the conversation
            messages: Full conversation history

        Returns:
            Message with compressed tool results, or original if no compression needed
        """
        # Check if message has tool results that need compression
        has_large_result = False
        for content in message.get("content", []):
            if "toolResult" in content:
                result_size = self._estimate_tool_result_tokens(content["toolResult"])
                if result_size > self.max_tokens:
                    has_large_result = True
                    break

        if not has_large_result:
            return message

        # Create a deep copy and compress tool results
        compressed_message = copy.deepcopy(message)

        for content in compressed_message.get("content", []):
            if "toolResult" in content:
                tool_result = content["toolResult"]
                original_size = self._estimate_tool_result_tokens(tool_result)

                if original_size > self.max_tokens:
                    compressed_result = self._compress_tool_result(tool_result)
                    content["toolResult"] = compressed_result

                    compressed_size = self._estimate_tool_result_tokens(compressed_result)
                    logger.info(
                        "Compressed tool result at index %d: %d -> %d tokens",
                        index,
                        original_size,
                        compressed_size,
                    )

        return compressed_message

    def _estimate_tool_result_tokens(self, tool_result: ToolResult) -> int:
        """Estimate token count for a tool result.

        Uses a simple heuristic: ~4 characters per token on average.

        Args:
            tool_result: The tool result to estimate

        Returns:
            Estimated token count
        """
        total_tokens = 0

        for content_item in tool_result.get("content", []):
            if "text" in content_item:
                char_count = len(content_item["text"])
                total_tokens += int(char_count / 4)
            elif "json" in content_item:
                json_str = str(content_item["json"])
                char_count = len(json_str)
                total_tokens += int(char_count / 4)
            elif "document" in content_item:
                total_tokens += len(content_item["document"]["source"]["bytes"])
            elif "image" in content_item:
                total_tokens += len(content_item["image"]["source"]["bytes"])

        return total_tokens

    def _compress_tool_result(self, tool_result: ToolResult) -> ToolResult:
        """Apply compression to tool result.

        Compression strategies:
        - Text: Truncate long text content
        - JSON: Summarize large JSON objects/arrays
        - Other: Keep as-is

        Args:
            tool_result: The tool result to compress

        Returns:
            Compressed tool result
        """
        original_size = self._estimate_tool_result_tokens(tool_result)
        compressed_content: List[ToolResultContent] = []

        for content_item in tool_result.get("content", []):
            if "text" in content_item:
                text = content_item["text"]
                if len(text) > self.truncate_at:
                    compressed_text = text[: self.truncate_at] + f"... [truncated from {len(text)} chars]"
                    compressed_content.append({"text": compressed_text})
                else:
                    compressed_content.append(content_item)

            elif "json" in content_item:
                json_data = content_item["json"]
                json_str = str(json_data)

                if len(json_str) > 500:
                    if isinstance(json_data, dict):
                        compressed_json = {
                            "_compressed": True,
                            "_type": "dict",
                            "_original_keys": len(json_data.keys()),
                            "_size": len(json_str),
                        }
                        # Include small values as samples
                        for idx, (key, value) in enumerate(json_data.items()):
                            if idx >= 3:  # Limit to first 3 items
                                break
                            value_str = str(value)
                            if len(value_str) < 100:
                                compressed_json[key] = value

                        compressed_content.append({"json": compressed_json})

                    elif isinstance(json_data, list):
                        sample_list: List[Any] = []
                        # Include small items as samples
                        for idx, item in enumerate(json_data):
                            if idx >= 3:  # Limit to first 3 items
                                break
                            if len(str(item)) < 100:
                                sample_list.append(item)

                        compressed_json = {
                            "_compressed": True,
                            "_type": "list",
                            "_length": len(json_data),
                            "_size": len(json_str),
                            "_sample": sample_list,
                        }

                        compressed_content.append({"json": compressed_json})
                    else:
                        compressed_content.append(content_item)
                else:
                    compressed_content.append(content_item)

            else:
                # Keep other content types (documents, images) as-is
                compressed_content.append(content_item)

        # Calculate compressed size for reporting
        compressed_size = self._estimate_tool_result_tokens(
            ToolResult(
                content=compressed_content,
                status=tool_result["status"],
                toolUseId=tool_result["toolUseId"],
            )
        )

        # Prepend compression note
        compression_note = self.compression_template.format(
            original_size=original_size,
            compressed_size=compressed_size,
            status=tool_result["status"],
        )

        final_content: List[ToolResultContent] = [{"text": compression_note}, *compressed_content]

        return ToolResult(
            content=final_content,
            status=tool_result["status"],
            toolUseId=tool_result["toolUseId"],
        )


class MappingConversationManager(ConversationManager):
    """Conversation manager that applies message mapping functions.

    This manager provides a simple, composable approach to conversation management.
    Instead of inheritance hierarchies, users provide callable mapper functions that
    transform or remove messages. Mappers are applied to messages in the "prunable"
    range (excluding preserved initial and recent messages).

    Example:
        # Using built-in mapper
        manager = MappingConversationManager(
            mapper=LargeToolResultMapper(max_tokens=100_000),
            preserve_first=1,
            preserve_last=2
        )

        # Using lambda for simple cases
        manager = MappingConversationManager(
            mapper=lambda msg, idx, msgs: None if should_remove(msg) else msg
        )

        # Using custom mapper class
        manager = MappingConversationManager(
            mapper=CustomMapper(config_value=42)
        )
    """

    def __init__(
        self,
        mapper: MessageMapper,
        preserve_first: int = 1,
        preserve_last: int = 2,
    ):
        """Initialize the mapping conversation manager.

        Args:
            mapper: Message mapper function to apply. Mappers should be stateless with
                respect to conversation history. While configuration parameters (like
                thresholds) are acceptable, mappers should not accumulate state across
                invocations since mapper-specific state is not persisted by the
                conversation manager.
            preserve_first: Number of initial messages to never map/remove
            preserve_last: Number of recent messages to never map/remove
        """
        super().__init__()
        self.mapper = mapper
        self.preserve_first = preserve_first
        self.preserve_last = preserve_last

    @override
    def apply_management(self, agent: "Agent", **kwargs: Any) -> None:
        """Apply message mapping if there are prunable messages.

        Args:
            agent: The agent whose conversation will be managed
            **kwargs: Additional keyword arguments for extensibility
        """
        original_count = len(agent.messages)
        if not self._can_apply_mappers(agent):
            logger.debug(
                "Too few messages to map safely: %d messages, %d preserved",
                original_count,
                self.preserve_first + self.preserve_last,
            )
            return

        self.reduce_context(agent, **kwargs)

    @override
    def reduce_context(
        self,
        agent: "Agent",
        e: Optional[Exception] = None,
        **kwargs: Any,  # noqa: ARG002
    ) -> None:
        """Reduce context by applying message mappers.

        Applies all configured mappers to messages in the prunable range
        (excluding preserved initial and recent messages).

        Args:
            agent: The agent whose conversation will be reduced
            e: The exception that triggered reduction, if any
            **kwargs: Additional keyword arguments for extensibility
        """
        original_count = len(agent.messages)
        if not self._can_apply_mappers(agent):
            logger.warning(
                "Too few messages to map safely: %d messages, %d preserved",
                original_count,
                self.preserve_first + self.preserve_last,
            )
            return

        mapped_messages, removed_count = self._apply_mapper(agent.messages)
        agent.messages[:] = mapped_messages
        self.removed_message_count += removed_count

        logger.info(
            "Mapping completed: %d -> %d messages (%d removed)",
            original_count,
            len(mapped_messages),
            removed_count,
        )

    def _can_apply_mappers(self, agent: "Agent") -> bool:
        """Check if there are enough messages to safely apply mappers.

        Args:
            agent: The agent to check

        Returns:
            True if there are messages in the prunable range
        """
        original_count = len(agent.messages)
        total_preserved = self.preserve_first + self.preserve_last
        return original_count > total_preserved

    def _apply_mapper(self, messages: Messages) -> tuple[Messages, int]:
        """Apply mapper to prunable messages.

        Args:
            messages: The messages to map

        Returns:
            Tuple of (mapped messages, count of removed messages)
        """
        min_mappable_index = self.preserve_first
        max_mappable_index = len(messages) - self.preserve_last

        mapped_messages = []
        removed_count = 0

        for i, message in enumerate(messages):
            # Preserve initial messages
            if i < min_mappable_index:
                mapped_messages.append(message)
                continue

            # Preserve recent messages
            if i >= max_mappable_index:
                mapped_messages.append(message)
                continue

            # Apply mapper to prunable messages
            current_message = self.mapper(message, i, messages)

            if current_message is not None:
                mapped_messages.append(current_message)
            else:
                removed_count += 1

        return mapped_messages, removed_count

    @override
    def get_state(self) -> Dict[str, Any]:
        """Get current state for session persistence.

        Returns:
            Dictionary containing manager state
        """
        return super().get_state()

    @override
    def restore_from_session(self, state: Dict[str, Any]) -> Optional[List[Message]]:
        """Restore manager state from session.

        Args:
            state: State dictionary to restore from

        Returns:
            None (no messages to prepend)
        """
        super().restore_from_session(state)
        return None
