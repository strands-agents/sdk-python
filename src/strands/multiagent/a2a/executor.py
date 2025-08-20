"""Strands Agent executor for the A2A protocol.

This module provides the StrandsA2AExecutor class, which adapts a Strands Agent
to be used as an executor in the A2A protocol. It handles the execution of agent
requests and the conversion of Strands Agent streamed responses to A2A events.

The A2A AgentExecutor ensures clients receive responses for synchronous and
streamed requests to the A2AServer.
"""

import json
import logging
import mimetypes
from typing import Any, Literal

from a2a.server.agent_execution import AgentExecutor, RequestContext
from a2a.server.events import EventQueue
from a2a.server.tasks import TaskUpdater
from a2a.types import DataPart, FilePart, InternalError, Part, TaskState, TextPart, UnsupportedOperationError
from a2a.utils import new_agent_text_message, new_task
from a2a.utils.errors import ServerError

from ...agent.agent import Agent as SAAgent
from ...agent.agent import AgentResult as SAAgentResult
from ...types.content import ContentBlock
from ...types.media import (
    DocumentContent,
    DocumentSource,
    ImageContent,
    ImageSource,
    VideoContent,
    VideoSource,
)

logger = logging.getLogger(__name__)


class StrandsA2AExecutor(AgentExecutor):
    """Executor that adapts a Strands Agent to the A2A protocol.

    This executor uses streaming mode to handle the execution of agent requests
    and converts Strands Agent responses to A2A protocol events.
    """

    # Default formats for each file type when MIME type is unavailable or unrecognized
    DEFAULT_FORMATS = {"document": "txt", "image": "png", "video": "mp4", "unknown": "txt"}

    # Handle special cases where format differs from extension
    FORMAT_MAPPINGS = {"jpg": "jpeg", "htm": "html", "3gp": "three_gp", "3gpp": "three_gp", "3g2": "three_gp"}

    def __init__(self, agent: SAAgent):
        """Initialize a StrandsA2AExecutor.

        Args:
            agent: The Strands Agent instance to adapt to the A2A protocol.
        """
        self.agent = agent

    async def execute(
        self,
        context: RequestContext,
        event_queue: EventQueue,
    ) -> None:
        """Execute a request using the Strands Agent and send the response as A2A events.

        This method executes the user's input using the Strands Agent in streaming mode
        and converts the agent's response to A2A events.

        Args:
            context: The A2A request context, containing the user's input and task metadata.
            event_queue: The A2A event queue used to send response events back to the client.

        Raises:
            ServerError: If an error occurs during agent execution
        """
        task = context.current_task
        if not task:
            task = new_task(context.message)  # type: ignore
            await event_queue.enqueue_event(task)

        updater = TaskUpdater(event_queue, task.id, task.context_id)

        try:
            await self._execute_streaming(context, updater)
        except Exception as e:
            raise ServerError(error=InternalError()) from e

    async def _execute_streaming(self, context: RequestContext, updater: TaskUpdater) -> None:
        """Execute request in streaming mode.

        Streams the agent's response in real-time, sending incremental updates
        as they become available from the agent.

        Args:
            context: The A2A request context, containing the user's input and other metadata.
            updater: The task updater for managing task state and sending updates.
        """
        # Convert A2A message parts to Strands ContentBlocks
        if context.message and hasattr(context.message, "parts"):
            content_blocks = self._convert_a2a_parts_to_content_blocks(context.message.parts)
            if not content_blocks:
                raise ValueError("No content blocks available")
        else:
            raise ValueError("No content blocks available")

        try:
            async for event in self.agent.stream_async(content_blocks):
                await self._handle_streaming_event(event, updater)
        except Exception:
            logger.exception("Error in streaming execution")
            raise

    async def _handle_streaming_event(self, event: dict[str, Any], updater: TaskUpdater) -> None:
        """Handle a single streaming event from the Strands Agent.

        Processes streaming events from the agent, converting data chunks to A2A
        task updates and handling the final result when streaming is complete.

        Args:
            event: The streaming event from the agent, containing either 'data' for
                incremental content or 'result' for the final response.
            updater: The task updater for managing task state and sending updates.
        """
        logger.debug("Streaming event: %s", event)
        if "data" in event:
            if text_content := event["data"]:
                await updater.update_status(
                    TaskState.working,
                    new_agent_text_message(
                        text_content,
                        updater.context_id,
                        updater.task_id,
                    ),
                )
        elif "result" in event:
            await self._handle_agent_result(event["result"], updater)

    async def _handle_agent_result(self, result: SAAgentResult | None, updater: TaskUpdater) -> None:
        """Handle the final result from the Strands Agent.

        Processes the agent's final result, extracts text content from the response,
        and adds it as an artifact to the task before marking the task as complete.

        Args:
            result: The agent result object containing the final response, or None if no result.
            updater: The task updater for managing task state and adding the final artifact.
        """
        if final_content := str(result):
            await updater.add_artifact(
                [Part(root=TextPart(text=final_content))],
                name="agent_response",
            )
        await updater.complete()

    async def cancel(self, context: RequestContext, event_queue: EventQueue) -> None:
        """Cancel an ongoing execution.

        This method is called when a request cancellation is requested. Currently,
        cancellation is not supported by the Strands Agent executor, so this method
        always raises an UnsupportedOperationError.

        Args:
            context: The A2A request context.
            event_queue: The A2A event queue.

        Raises:
            ServerError: Always raised with an UnsupportedOperationError, as cancellation
                is not currently supported.
        """
        logger.warning("Cancellation requested but not supported")
        raise ServerError(error=UnsupportedOperationError())

    def _get_file_type_from_mime_type(self, mime_type: str | None) -> Literal["document", "image", "video", "unknown"]:
        """Classify file type based on MIME type.

        Args:
            mime_type: The MIME type of the file

        Returns:
            The classified file type
        """
        if not mime_type:
            return "unknown"

        mime_type = mime_type.lower()

        if mime_type.startswith("image/"):
            return "image"
        elif mime_type.startswith("video/"):
            return "video"
        elif (
            mime_type.startswith("text/")
            or mime_type.startswith("application/")
            or mime_type in ["application/pdf", "application/json", "application/xml"]
        ):
            return "document"
        else:
            return "unknown"

    def _get_file_format_from_mime_type(self, mime_type: str | None, file_type: str) -> str:
        """Extract file format from MIME type using Python's mimetypes library.

        Args:
            mime_type: The MIME type of the file
            file_type: The classified file type (image, video, document, txt)

        Returns:
            The file format string
        """
        if not mime_type:
            return self.DEFAULT_FORMATS.get(file_type, "txt")

        mime_type = mime_type.lower()

        # Extract subtype from MIME type and check existing format mappings
        if "/" in mime_type:
            subtype = mime_type.split("/")[-1]
            if subtype in self.FORMAT_MAPPINGS:
                return self.FORMAT_MAPPINGS[subtype]

        # Use mimetypes library to find extensions for the MIME type
        extensions = mimetypes.guess_all_extensions(mime_type)

        if extensions:
            extension = extensions[0][1:]  # Remove the leading dot
            return self.FORMAT_MAPPINGS.get(extension, extension)

        # Fallback to defaults for unknown MIME types
        return self.DEFAULT_FORMATS.get(file_type, "txt")

    def _strip_file_extension(self, file_name: str) -> str:
        """Strip the file extension from a file name.

        Args:
            file_name: The original file name with extension

        Returns:
            The file name without extension
        """
        if "." in file_name:
            return file_name.rsplit(".", 1)[0]
        return file_name

    def _convert_a2a_parts_to_content_blocks(self, parts: list[Part]) -> list[ContentBlock]:
        """Convert A2A message parts to Strands ContentBlocks.

        Args:
            parts: List of A2A Part objects

        Returns:
            List of Strands ContentBlock objects
        """
        content_blocks: list[ContentBlock] = []

        for part in parts:
            try:
                part_root = part.root

                if isinstance(part_root, TextPart):
                    # Handle TextPart
                    content_blocks.append(ContentBlock(text=part_root.text))

                elif isinstance(part_root, FilePart):
                    # Handle FilePart
                    file_obj = part_root.file
                    mime_type = getattr(file_obj, "mime_type", None)
                    raw_file_name = getattr(file_obj, "name", "FileNameNotProvided")
                    file_name = self._strip_file_extension(raw_file_name)
                    file_type = self._get_file_type_from_mime_type(mime_type)
                    file_format = self._get_file_format_from_mime_type(mime_type, file_type)

                    # Handle FileWithBytes vs FileWithUri
                    bytes_data = getattr(file_obj, "bytes", None)
                    uri_data = getattr(file_obj, "uri", None)

                    if bytes_data:
                        if file_type == "image":
                            content_blocks.append(
                                ContentBlock(
                                    image=ImageContent(
                                        format=file_format,  # type: ignore
                                        source=ImageSource(bytes=bytes_data),
                                    )
                                )
                            )
                        elif file_type == "video":
                            content_blocks.append(
                                ContentBlock(
                                    video=VideoContent(
                                        format=file_format,  # type: ignore
                                        source=VideoSource(bytes=bytes_data),
                                    )
                                )
                            )
                        else:  # document or unknown
                            content_blocks.append(
                                ContentBlock(
                                    document=DocumentContent(
                                        format=file_format,  # type: ignore
                                        name=file_name,
                                        source=DocumentSource(bytes=bytes_data),
                                    )
                                )
                            )
                    # Handle FileWithUri
                    elif uri_data:
                        # For URI files, create a text representation since Strands ContentBlocks expect bytes
                        content_blocks.append(
                            ContentBlock(
                                text="[File: %s (%s)] - Referenced file at: %s" % (file_name, mime_type, uri_data)
                            )
                        )
                elif isinstance(part_root, DataPart):
                    # Handle DataPart - convert structured data to JSON text
                    try:
                        data_text = json.dumps(part_root.data, indent=2)
                        content_blocks.append(ContentBlock(text="[Structured Data]\n%s" % data_text))
                    except Exception:
                        logger.exception("Failed to serialize data part")
            except Exception:
                logger.exception("Error processing part")

        return content_blocks
