"""Internal reasoner lifecycle manager.

NOT exposed to developers. Handles low-level Nova Sonic event protocol
for streaming and barge-in cancellation.

This class is instantiated by BidiNovaSonicModel when a reasoner is configured.
"""

import asyncio
import json
import logging
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..models.nova_sonic import BidiNovaSonicModel
    from .config import ReasonerConfig

logger = logging.getLogger(__name__)


class _ReasonerManager:
    """Manages reasoner invocations within BidiNovaSonicModel.

    Responsibilities:
    - Receives reasoner toolUse events from Nova Sonic
    - Cancels previous reasoning on barge-in (user interruption)
    - Opens content containers (contentStart)
    - Streams reasoner output as toolResult events
    - Closes content containers (contentEnd)
    - Handles errors with fallback responses
    """

    def __init__(self, model: "BidiNovaSonicModel", config: "ReasonerConfig") -> None:
        self._model = model
        self._config = config
        self._active_task: asyncio.Task | None = None
        self._active_content_name: str | None = None
        self._active_tool_use_id: str | None = None
        self._content_opened: bool = False
        self._data_sent: bool = False
        self._content_closed: bool = False

    async def handle_invocation(self, tool_use: dict[str, Any]) -> None:
        """Handle a new reasoner invocation.

        Args:
            tool_use: The toolUse event data containing toolUseId and content.
        """
        messages = self._extract_messages(tool_use)
        if not self._has_user_message(messages):
            logger.debug("tool_use_id=<%s> | skipping — no user message", tool_use.get("toolUseId", "?")[:8])
            return

        # Cancel previous invocation (barge-in)
        await self._cancel_active()

        tool_use_id = tool_use["toolUseId"]
        content_name = str(uuid.uuid4())

        self._active_tool_use_id = tool_use_id
        self._active_content_name = content_name
        self._content_opened = False
        self._data_sent = False
        self._content_closed = False

        logger.debug("tool_use_id=<%s> | starting reasoning", tool_use_id[:8])
        self._active_task = asyncio.create_task(self._run(messages, tool_use_id, content_name))
        self._active_task.add_done_callback(self._on_task_done)

    async def _cancel_active(self) -> None:
        """Cancel the active reasoning task and clean up resources."""
        if self._active_task and not self._active_task.done():
            self._active_task.cancel()
            try:
                await self._active_task
            except asyncio.CancelledError:
                pass

        if self._active_content_name and self._content_opened and self._data_sent and not self._content_closed:
            try:
                await self._send_content_end(self._active_content_name)
                self._content_closed = True
            except Exception as e:
                logger.debug("error closing content on cancel: %s", e)

        if self._config.on_interrupted:
            try:
                self._config.on_interrupted()
            except Exception as e:
                logger.debug("on_interrupted callback error: %s", e)

        self._active_task = None
        self._active_content_name = None
        self._active_tool_use_id = None
        self._content_opened = False
        self._data_sent = False
        self._content_closed = False

    async def _run(self, messages: list[dict[str, Any]], tool_use_id: str, content_name: str) -> None:
        """Run reasoning and stream results back to Nova Sonic."""
        try:
            await self._send_tool_content_start(content_name, tool_use_id)
            self._content_opened = True

            reasoner = self._config.reasoner
            async for sentence in reasoner.reason(messages):
                if not sentence:
                    continue
                payload = json.dumps({"text": sentence, "type": "TEXT"})
                await self._send_tool_result(content_name, payload)
                self._data_sent = True

            if not self._data_sent:
                fallback = json.dumps({"text": self._config.fallback_message, "type": "TEXT"})
                await self._send_tool_result(content_name, fallback)
                self._data_sent = True

            await self._send_content_end(content_name)
            self._content_closed = True

        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.error("reasoner error: %s", e)
            try:
                if not self._content_opened:
                    await self._send_tool_content_start(content_name, tool_use_id)
                    self._content_opened = True
                error_payload = json.dumps({"text": "Sorry, something went wrong.", "type": "TEXT"})
                await self._send_tool_result(content_name, error_payload)
                self._data_sent = True
                if not self._content_closed:
                    await self._send_content_end(content_name)
                    self._content_closed = True
            except Exception:
                pass

    async def shutdown(self) -> None:
        """Shutdown the manager, cancelling any active task."""
        await self._cancel_active()

    async def _send_tool_content_start(self, content_name: str, tool_use_id: str) -> None:
        event = json.dumps(
            {
                "event": {
                    "contentStart": {
                        "promptName": self._model._connection_id,
                        "contentName": content_name,
                        "interactive": False,
                        "type": "TOOL",
                        "role": "TOOL",
                        "toolResultInputConfiguration": {
                            "toolUseId": tool_use_id,
                            "type": "TEXT",
                            "textInputConfiguration": {"mediaType": "text/plain"},
                        },
                    }
                }
            }
        )
        await self._model._send_nova_events([event])

    async def _send_tool_result(self, content_name: str, content: str) -> None:
        event = json.dumps(
            {
                "event": {
                    "toolResult": {
                        "promptName": self._model._connection_id,
                        "contentName": content_name,
                        "content": content,
                    }
                }
            }
        )
        await self._model._send_nova_events([event])

    async def _send_content_end(self, content_name: str) -> None:
        event = json.dumps(
            {
                "event": {
                    "contentEnd": {
                        "promptName": self._model._connection_id,
                        "contentName": content_name,
                    }
                }
            }
        )
        await self._model._send_nova_events([event])

    def _extract_messages(self, tool_use: dict[str, Any]) -> list[dict[str, Any]]:
        try:
            content_str = tool_use.get("content", "{}")
            if isinstance(content_str, str):
                content = json.loads(content_str)
            else:
                content = content_str
            return content.get("messages", [])
        except (json.JSONDecodeError, AttributeError):
            logger.debug("failed to extract messages from content")
            return []

    def _has_user_message(self, messages: list[dict[str, Any]]) -> bool:
        return any(
            m.get("role", "").upper() == "USER" and any(p.get("text", "").strip() for p in m.get("content", []))
            for m in messages
        )

    def _on_task_done(self, task: asyncio.Task) -> None:
        if task.done() and not task.cancelled() and task.exception():
            logger.debug("reasoning task failed: %s", task.exception())
