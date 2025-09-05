"""Strategy for pruning large tool results while preserving tool use context."""

import copy
import logging
from typing import TYPE_CHECKING, Optional

from typing_extensions import override

from ....types.content import Message
from ....types.tools import ToolResult, ToolResultContent
from .. import MessageContext, PruningStrategy

if TYPE_CHECKING:
    from ....agent.agent import Agent

logger = logging.getLogger(__name__)


class LargeToolResultPruningStrategy(PruningStrategy):
    """Prune large tool results while preserving tool use context.

    This strategy identifies tool results that exceed a specified token threshold
    and compresses them while maintaining the essential information about the
    tool execution and its status.
    """

    def __init__(
        self,
        max_tool_result_tokens: int = 50_000,
        compression_template: str = (
            "[Tool result compressed: {original_size} tokens -> {compressed_size} tokens. Original status: {status}]"
        ),
        enable_llm_compression: bool = False,
    ):
        """Initialize the tool result pruning strategy.

        Args:
            max_tool_result_tokens: Maximum tokens allowed in tool results before compression
            compression_template: Template for compression messages
            enable_llm_compression: Whether to use LLM for intelligent compression (future feature)
        """
        self.max_tool_result_tokens = max_tool_result_tokens
        self.compression_template = compression_template
        self.enable_llm_compression = enable_llm_compression

    @override
    def get_strategy_name(self) -> str:
        """Get the name of this pruning strategy.

        Returns:
            The strategy name
        """
        return "LargeToolResultPruningStrategy"

    @override
    def should_prune_message(self, message: Message, context: MessageContext) -> bool:
        """Check if message contains large tool results that should be pruned.

        Args:
            message: The message to evaluate
            context: Context information about the message

        Returns:
            True if the message contains tool results exceeding the size threshold
        """
        # Check if message has tool results
        has_tool_result = context.get("has_tool_result", False)

        if not has_tool_result:
            return False

        for content in message.get("content", []):
            if "toolResult" in content and content["toolResult"]:
                result_size = self._estimate_tool_result_tokens(content["toolResult"])
                if result_size > self.max_tool_result_tokens:
                    logger.debug("Tool result size %d exceeds threshold %d", result_size, self.max_tool_result_tokens)
                    return True
        return False

    @override
    def prune_message(self, message: Message, agent: "Agent") -> Optional[Message]:
        """Compress large tool results while preserving structure.

        Args:
            message: The message to prune
            agent: The agent instance for context

        Returns:
            The message with compressed tool results
        """
        pruned_message = self._deep_copy_message(message)

        for content in pruned_message.get("content", []):
            if "toolResult" in content:
                tool_result = content["toolResult"]
                original_size = self._estimate_tool_result_tokens(tool_result)

                if original_size > self.max_tool_result_tokens:
                    if self.enable_llm_compression:
                        compressed_result = self._llm_compress_tool_result(tool_result, agent)
                    else:
                        compressed_result = self._simple_compress_tool_result(tool_result)

                    content["toolResult"] = compressed_result

                    compressed_size = self._estimate_tool_result_tokens(compressed_result)
                    logger.info("Compressed tool result: %d -> %d tokens", original_size, compressed_size)

        return pruned_message

    def _estimate_tool_result_tokens(self, tool_result: ToolResult) -> int:
        """Estimate token count for a tool result.

        Args:
            tool_result: The tool result to estimate

        Returns:
            Estimated token count
        """
        total_tokens = 0

        for content_item in tool_result.get("content", []):
            if "text" in content_item:
                text = content_item["text"]
                # estimation: ~4 characters per token on average
                char_count = len(text)
                total_tokens += int(char_count / 4)
            elif "json" in content_item:
                # JSON content estimation
                json_str = str(content_item["json"])
                char_count = len(json_str)
                total_tokens += int(char_count / 4)
            elif "document" in content_item:
                # Document content estimation (simplified)
                total_tokens += len(content_item["document"]["source"]["bytes"])
            elif "image" in content_item:
                # Image content estimation (simplified)
                total_tokens += len(content_item["image"]["source"]["bytes"])

        return total_tokens

    def _simple_compress_tool_result(self, tool_result: ToolResult) -> ToolResult:
        """Apply simple compression to tool result.

        Args:
            tool_result: The tool result to compress

        Returns:
            Compressed tool result
        """
        original_size = self._estimate_tool_result_tokens(tool_result)

        # Create compressed version
        compressed_content: list[ToolResultContent] = []

        for content_item in tool_result.get("content", []):
            if "text" in content_item:
                text = content_item["text"]
                if len(text) > 500:  # Truncate long text
                    compressed_text = text[:500] + f"... [truncated from {len(text)} chars]"
                    compressed_content.append({"text": compressed_text})
                else:
                    compressed_content.append(content_item)
            elif "json" in content_item:
                # Simplify JSON content
                json_data = content_item["json"]
                if isinstance(json_data, dict) and len(str(json_data)) > 500:
                    compressed_json = {
                        "_compressed": True,
                        "_n_original_keys": len(list(json_data.keys())),
                        "_size": len(str(json_data)),
                        "_type": "dict",
                    }
                    # Include a few sample values if they're small
                    for idx, (key, value) in enumerate(json_data.items()):
                        value_str = str(value)
                        if len(value_str) < 100:
                            compressed_json[key] = json_data[key]
                        if idx >= 100:
                            break
                elif isinstance(json_data, list) and len(str(json_data)) > 500:
                    compressed_json = {
                        "_compressed": True,
                        "_length": len(json_data),
                        "_size": len(str(json_data)),
                        "_type": "list",
                    }
                    samples = []
                    for idx, item in enumerate(json_data):
                        if len(str(item)) < 100:
                            samples.append(item)
                        if idx >= 100:
                            break
                    compressed_json["_sample"] = samples
                    compressed_content.append({"json": compressed_json})
                else:
                    compressed_content.append(content_item)
            else:
                # Keep other content types as-is for now
                compressed_content.append(content_item)

        # Calculate compressed size
        compressed_size = self._estimate_tool_result_tokens(
            ToolResult(
                content=compressed_content,
                status=tool_result["status"],
                toolUseId=tool_result["toolUseId"],
            )
        )

        # Create compression note
        compression_note = self.compression_template.format(
            original_size=original_size,
            compressed_size=compressed_size,
            status=tool_result["status"],
        )

        # Create the compressed tool result first
        compressed_content = [{"text": compression_note}, *compressed_content]
        compressed_tool_result = ToolResult(
            content=compressed_content,
            status=tool_result["status"],
            toolUseId=tool_result["toolUseId"],
        )

        return compressed_tool_result

    def _llm_compress_tool_result(self, tool_result: ToolResult, agent: "Agent") -> ToolResult:
        """Use LLM to intelligently compress tool result.

        This is a placeholder for future LLM-based compression functionality.

        Args:
            tool_result: The tool result to compress
            agent: The agent instance for LLM access

        Returns:
            Compressed tool result
        """
        raise NotImplementedError("LLM-based tool result compression not implemented yet!")

    def _deep_copy_message(self, message: Message) -> Message:
        """Create a deep copy of a message.

        Args:
            message: The message to copy

        Returns:
            Deep copy of the message
        """
        return copy.deepcopy(message)
