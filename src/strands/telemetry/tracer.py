"""OpenTelemetry integration.

This module provides tracing capabilities using OpenTelemetry,
enabling trace data to be sent to OTLP endpoints.
"""

import json
import logging
import os
from datetime import date, datetime, timezone
from typing import Any, Dict, Mapping, Optional, cast

import opentelemetry.trace as trace_api
from opentelemetry.instrumentation.threading import ThreadingInstrumentor
from opentelemetry.trace import Span, StatusCode

from ..agent.agent_result import AgentResult
from ..types.content import ContentBlock, Message, Messages
from ..types.interrupt import InterruptResponseContent
from ..types.multiagent import MultiAgentInput
from ..types.streaming import Metrics, StopReason, Usage
from ..types.tools import ToolResult, ToolUse
from ..types.traces import Attributes, AttributeValue

logger = logging.getLogger(__name__)


class JSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles non-serializable types."""

    def encode(self, obj: Any) -> str:
        """Recursively encode objects, preserving structure and only replacing unserializable values.

        Args:
            obj: The object to encode

        Returns:
            JSON string representation of the object
        """
        # Process the object to handle non-serializable values
        processed_obj = self._process_value(obj)
        # Use the parent class to encode the processed object
        return super().encode(processed_obj)

    def _process_value(self, value: Any) -> Any:
        """Process any value, handling containers recursively.

        Args:
            value: The value to process

        Returns:
            Processed value with unserializable parts replaced
        """
        # Handle datetime objects directly
        if isinstance(value, (datetime, date)):
            return value.isoformat()

        # Handle dictionaries
        elif isinstance(value, dict):
            return {k: self._process_value(v) for k, v in value.items()}

        # Handle lists
        elif isinstance(value, list):
            return [self._process_value(item) for item in value]

        # Handle all other values
        else:
            try:
                # Test if the value is JSON serializable
                json.dumps(value)
                return value
            except (TypeError, OverflowError, ValueError):
                return "<replaced>"


class Tracer:
    """Handles OpenTelemetry tracing.

    This class provides a simple interface for creating and managing traces,
    with support for sending to OTLP endpoints.

    When the OTEL_EXPORTER_OTLP_ENDPOINT environment variable is set, traces
    are sent to the OTLP endpoint.

    Both attributes are controlled by including "gen_ai_latest_experimental" or "gen_ai_tool_definitions",
    respectively, in the OTEL_SEMCONV_STABILITY_OPT_IN environment variable.
    """

    def __init__(self) -> None:
        """Initialize the tracer."""
        self.service_name = __name__
        self.tracer_provider: Optional[trace_api.TracerProvider] = None
        self.tracer_provider = trace_api.get_tracer_provider()
        self.tracer = self.tracer_provider.get_tracer(self.service_name)
        ThreadingInstrumentor().instrument()

        # Read OTEL_SEMCONV_STABILITY_OPT_IN environment variable
        opt_in_values = self._parse_semconv_opt_in()
        ## To-do: should not set below attributes directly, use env var instead
        self.use_latest_genai_conventions = "gen_ai_latest_experimental" in opt_in_values
        self._include_tool_definitions = "gen_ai_tool_definitions" in opt_in_values

    def _parse_semconv_opt_in(self) -> set[str]:
        """Parse the OTEL_SEMCONV_STABILITY_OPT_IN environment variable.

        Returns:
            A set of opt-in values from the environment variable.
        """
        opt_in_env = os.getenv("OTEL_SEMCONV_STABILITY_OPT_IN", "")
        return {value.strip() for value in opt_in_env.split(",")}

    def _start_span(
        self,
        span_name: str,
        parent_span: Optional[Span] = None,
        attributes: Optional[Dict[str, AttributeValue]] = None,
        span_kind: trace_api.SpanKind = trace_api.SpanKind.INTERNAL,
    ) -> Span:
        """Generic helper method to start a span with common attributes.

        Args:
            span_name: Name of the span to create
            parent_span: Optional parent span to link this span to
            attributes: Dictionary of attributes to set on the span
            span_kind: enum of OptenTelemetry SpanKind

        Returns:
            The created span, or None if tracing is not enabled
        """
        if not parent_span:
            parent_span = trace_api.get_current_span()

        context = None
        if parent_span and parent_span.is_recording() and parent_span != trace_api.INVALID_SPAN:
            context = trace_api.set_span_in_context(parent_span)

        span = self.tracer.start_span(name=span_name, context=context, kind=span_kind)

        # Set start time as a common attribute
        span.set_attribute("gen_ai.event.start_time", datetime.now(timezone.utc).isoformat())

        # Add all provided attributes
        if attributes:
            self._set_attributes(span, attributes)

        return span

    def _set_attributes(self, span: Span, attributes: Dict[str, AttributeValue]) -> None:
        """Set attributes on a span, handling different value types appropriately.

        Args:
            span: The span to set attributes on
            attributes: Dictionary of attributes to set
        """
        if not span:
            return

        for key, value in attributes.items():
            span.set_attribute(key, value)

    def _add_optional_usage_and_metrics_attributes(
        self, attributes: Dict[str, AttributeValue], usage: Usage, metrics: Metrics
    ) -> None:
        """Add optional usage and metrics attributes if they have values.

        Args:
            attributes: Dictionary to add attributes to
            usage: Token usage information from the model call
            metrics: Metrics from the model call
        """
        if "cacheReadInputTokens" in usage:
            attributes["gen_ai.usage.cache_read_input_tokens"] = usage["cacheReadInputTokens"]

        if "cacheWriteInputTokens" in usage:
            attributes["gen_ai.usage.cache_write_input_tokens"] = usage["cacheWriteInputTokens"]

        if metrics.get("timeToFirstByteMs", 0) > 0:
            attributes["gen_ai.server.time_to_first_token"] = metrics["timeToFirstByteMs"]

        if metrics.get("latencyMs", 0) > 0:
            attributes["gen_ai.server.request.duration"] = metrics["latencyMs"]

    def _end_span(
        self,
        span: Span,
        attributes: Optional[Dict[str, AttributeValue]] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """Generic helper method to end a span.

        Args:
            span: The span to end
            attributes: Optional attributes to set before ending the span
            error: Optional exception if an error occurred
        """
        if not span:
            return

        try:
            # Set end time as a common attribute
            span.set_attribute("gen_ai.event.end_time", datetime.now(timezone.utc).isoformat())

            # Add any additional attributes
            if attributes:
                self._set_attributes(span, attributes)

            # Handle error if present
            if error:
                span.set_status(StatusCode.ERROR, str(error))
                span.record_exception(error)
            else:
                span.set_status(StatusCode.OK)
        except Exception as e:
            logger.warning("error=<%s> | error while ending span", e, exc_info=True)
        finally:
            span.end()
            # Force flush to ensure spans are exported
            if self.tracer_provider and hasattr(self.tracer_provider, "force_flush"):
                try:
                    self.tracer_provider.force_flush()
                except Exception as e:
                    logger.warning("error=<%s> | failed to force flush tracer provider", e)

    def end_span_with_error(self, span: Span, error_message: str, exception: Optional[Exception] = None) -> None:
        """End a span with error status.

        Args:
            span: The span to end.
            error_message: Error message to set in the span status.
            exception: Optional exception to record in the span.
        """
        if not span:
            return

        error = exception or Exception(error_message)
        self._end_span(span, error=error)

    def _add_event(self, span: Optional[Span], event_name: str, event_attributes: Attributes) -> None:
        """Add an event with attributes to a span.

        Args:
            span: The span to add the event to
            event_name: Name of the event
            event_attributes: Dictionary of attributes to set on the event
        """
        if not span:
            return

        span.add_event(event_name, attributes=event_attributes)

    def _get_event_name_for_message(self, message: Message) -> str:
        """Determine the appropriate OpenTelemetry event name for a message.

        According to OpenTelemetry semantic conventions v1.36.0, messages containing tool results
        should be labeled as 'gen_ai.tool.message' regardless of their role field.
        This ensures proper categorization of tool responses in traces.

        Note: The GenAI namespace is experimental and may change in future versions.

        Reference: https://github.com/open-telemetry/semantic-conventions/blob/v1.36.0/docs/gen-ai/gen-ai-events.md#event-gen_aitoolmessage

        Args:
            message: The message to determine the event name for

        Returns:
            The OpenTelemetry event name (e.g., 'gen_ai.user.message', 'gen_ai.tool.message')
        """
        # Check if the message contains a tool result
        for content_block in message.get("content", []):
            if "toolResult" in content_block:
                return "gen_ai.tool.message"

        return f"gen_ai.{message['role']}.message"

    def start_model_invoke_span(
        self,
        messages: Messages,
        parent_span: Optional[Span] = None,
        model_id: Optional[str] = None,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        **kwargs: Any,
    ) -> Span:
        """Start a new span for a model invocation.

        Args:
            messages: Messages being sent to the model.
            parent_span: Optional parent span to link this span to.
            model_id: Optional identifier for the model being invoked.
            custom_trace_attributes: Optional mapping of custom trace attributes to include in the span.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = self._get_common_attributes(operation_name="chat")

        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)

        if model_id:
            attributes["gen_ai.request.model"] = model_id

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        span = self._start_span("chat", parent_span, attributes=attributes, span_kind=trace_api.SpanKind.INTERNAL)
        self._add_event_messages(span, messages)

        return span

    def end_model_invoke_span(
        self,
        span: Span,
        message: Message,
        usage: Usage,
        metrics: Metrics,
        stop_reason: StopReason,
        error: Optional[Exception] = None,
    ) -> None:
        """End a model invocation span with results and metrics.

        Args:
            span: The span to end.
            message: The message response from the model.
            usage: Token usage information from the model call.
            metrics: Metrics from the model call.
            stop_reason (StopReason): The reason the model stopped generating.
            error: Optional exception if the model call failed.
        """
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.usage.prompt_tokens": usage["inputTokens"],
            "gen_ai.usage.input_tokens": usage["inputTokens"],
            "gen_ai.usage.completion_tokens": usage["outputTokens"],
            "gen_ai.usage.output_tokens": usage["outputTokens"],
            "gen_ai.usage.total_tokens": usage["totalTokens"],
        }

        # Add optional attributes if they have values
        self._add_optional_usage_and_metrics_attributes(attributes, usage, metrics)

        if self.use_latest_genai_conventions:
            self._add_event(
                span,
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.output.messages": serialize(
                        [
                            {
                                "role": message["role"],
                                "parts": self._map_content_blocks_to_otel_parts(message["content"]),
                                "finish_reason": str(stop_reason),
                            }
                        ]
                    ),
                },
            )
        else:
            self._add_event(
                span,
                "gen_ai.choice",
                event_attributes={"finish_reason": str(stop_reason), "message": serialize(message["content"])},
            )

        self._end_span(span, attributes, error)

    def start_tool_call_span(
        self,
        tool: ToolUse,
        parent_span: Optional[Span] = None,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        **kwargs: Any,
    ) -> Span:
        """Start a new span for a tool call.

        Args:
            tool: The tool being used.
            parent_span: Optional parent span to link this span to.
            custom_trace_attributes: Optional mapping of custom trace attributes to include in the span.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = self._get_common_attributes(operation_name="execute_tool")
        attributes.update(
            {
                "gen_ai.tool.name": tool["name"],
                "gen_ai.tool.call.id": tool["toolUseId"],
            }
        )

        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)
        # Add additional kwargs as attributes
        attributes.update(kwargs)

        span_name = f"execute_tool {tool['name']}"
        span = self._start_span(span_name, parent_span, attributes=attributes, span_kind=trace_api.SpanKind.INTERNAL)

        if self.use_latest_genai_conventions:
            self._add_event(
                span,
                "gen_ai.client.inference.operation.details",
                {
                    "gen_ai.input.messages": serialize(
                        [
                            {
                                "role": "tool",
                                "parts": [
                                    {
                                        "type": "tool_call",
                                        "name": tool["name"],
                                        "id": tool["toolUseId"],
                                        "arguments": tool["input"],
                                    }
                                ],
                            }
                        ]
                    )
                },
            )
        else:
            self._add_event(
                span,
                "gen_ai.tool.message",
                event_attributes={
                    "role": "tool",
                    "content": serialize(tool["input"]),
                    "id": tool["toolUseId"],
                },
            )

        return span

    def end_tool_call_span(
        self, span: Span, tool_result: Optional[ToolResult], error: Optional[Exception] = None
    ) -> None:
        """End a tool call span with results.

        Args:
            span: The span to end.
            tool_result: The result from the tool execution.
            error: Optional exception if the tool call failed.
        """
        attributes: Dict[str, AttributeValue] = {}
        if tool_result is not None:
            status = tool_result.get("status")
            status_str = str(status) if status is not None else ""

            attributes.update(
                {
                    "gen_ai.tool.status": status_str,
                }
            )

            if self.use_latest_genai_conventions:
                self._add_event(
                    span,
                    "gen_ai.client.inference.operation.details",
                    {
                        "gen_ai.output.messages": serialize(
                            [
                                {
                                    "role": "tool",
                                    "parts": [
                                        {
                                            "type": "tool_call_response",
                                            "id": tool_result.get("toolUseId", ""),
                                            "response": tool_result.get("content"),
                                        }
                                    ],
                                }
                            ]
                        )
                    },
                )
            else:
                self._add_event(
                    span,
                    "gen_ai.choice",
                    event_attributes={
                        "message": serialize(tool_result.get("content")),
                        "id": tool_result.get("toolUseId", ""),
                    },
                )

        self._end_span(span, attributes, error)

    def start_event_loop_cycle_span(
        self,
        invocation_state: Any,
        messages: Messages,
        parent_span: Optional[Span] = None,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        **kwargs: Any,
    ) -> Optional[Span]:
        """Start a new span for an event loop cycle.

        Args:
            invocation_state: Arguments for the event loop cycle.
            parent_span: Optional parent span to link this span to.
            messages:  Messages being processed in this cycle.
            custom_trace_attributes: Optional mapping of custom trace attributes to include in the span.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        event_loop_cycle_id = str(invocation_state.get("event_loop_cycle_id"))
        parent_span = parent_span if parent_span else invocation_state.get("event_loop_parent_span")

        attributes: Dict[str, AttributeValue] = {
            "event_loop.cycle_id": event_loop_cycle_id,
        }

        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)

        if "event_loop_parent_cycle_id" in invocation_state:
            attributes["event_loop.parent_cycle_id"] = str(invocation_state["event_loop_parent_cycle_id"])

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        span_name = "execute_event_loop_cycle"
        span = self._start_span(span_name, parent_span, attributes)
        self._add_event_messages(span, messages)

        return span

    def end_event_loop_cycle_span(
        self,
        span: Span,
        message: Message,
        tool_result_message: Optional[Message] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """End an event loop cycle span with results.

        Args:
            span: The span to end.
            message: The message response from this cycle.
            tool_result_message: Optional tool result message if a tool was called.
            error: Optional exception if the cycle failed.
        """
        attributes: Dict[str, AttributeValue] = {}
        event_attributes: Dict[str, AttributeValue] = {"message": serialize(message["content"])}

        if tool_result_message:
            event_attributes["tool.result"] = serialize(tool_result_message["content"])

            if self.use_latest_genai_conventions:
                self._add_event(
                    span,
                    "gen_ai.client.inference.operation.details",
                    {
                        "gen_ai.output.messages": serialize(
                            [
                                {
                                    "role": tool_result_message["role"],
                                    "parts": self._map_content_blocks_to_otel_parts(tool_result_message["content"]),
                                }
                            ]
                        )
                    },
                )
            else:
                self._add_event(span, "gen_ai.choice", event_attributes=event_attributes)
        self._end_span(span, attributes, error)

    def start_agent_span(
        self,
        messages: Messages,
        agent_name: str,
        model_id: Optional[str] = None,
        tools: Optional[list] = None,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        tools_config: Optional[dict] = None,
        **kwargs: Any,
    ) -> Span:
        """Start a new span for an agent invocation.

        Args:
            messages: List of messages being sent to the agent.
            agent_name: Name of the agent.
            model_id: Optional model identifier.
            tools: Optional list of tools being used.
            custom_trace_attributes: Optional mapping of custom trace attributes to include in the span.
            tools_config: Optional dictionary of tool configurations.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = self._get_common_attributes(operation_name="invoke_agent")
        attributes.update(
            {
                "gen_ai.agent.name": agent_name,
            }
        )

        if model_id:
            attributes["gen_ai.request.model"] = model_id

        if tools:
            attributes["gen_ai.agent.tools"] = serialize(tools)

        if self._include_tool_definitions and tools_config:
            try:
                tool_definitions = self._construct_tool_definitions(tools_config)
                attributes["gen_ai.tool.definitions"] = serialize(tool_definitions)
            except Exception:
                # A failure in telemetry should not crash the agent
                logger.warning("failed to attach tool metadata to agent span", exc_info=True)

        # Add custom trace attributes if provided
        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        span = self._start_span(
            f"invoke_agent {agent_name}", attributes=attributes, span_kind=trace_api.SpanKind.INTERNAL
        )
        self._add_event_messages(span, messages)

        return span

    def end_agent_span(
        self,
        span: Span,
        response: Optional[AgentResult] = None,
        error: Optional[Exception] = None,
    ) -> None:
        """End an agent span with results and metrics.

        Args:
            span: The span to end.
            response: The response from the agent.
            error: Any error that occurred.
        """
        attributes: Dict[str, AttributeValue] = {}

        if response:
            if self.use_latest_genai_conventions:
                self._add_event(
                    span,
                    "gen_ai.client.inference.operation.details",
                    {
                        "gen_ai.output.messages": serialize(
                            [
                                {
                                    "role": "assistant",
                                    "parts": [{"type": "text", "content": str(response)}],
                                    "finish_reason": str(response.stop_reason),
                                }
                            ]
                        )
                    },
                )
            else:
                self._add_event(
                    span,
                    "gen_ai.choice",
                    event_attributes={"message": str(response), "finish_reason": str(response.stop_reason)},
                )

            if hasattr(response, "metrics") and hasattr(response.metrics, "accumulated_usage"):
                if "langfuse" in os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "") or "langfuse" in os.getenv(
                    "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT", ""
                ):
                    attributes.update({"langfuse.observation.type": "span"})
                accumulated_usage = response.metrics.accumulated_usage
                attributes.update(
                    {
                        "gen_ai.usage.prompt_tokens": accumulated_usage["inputTokens"],
                        "gen_ai.usage.completion_tokens": accumulated_usage["outputTokens"],
                        "gen_ai.usage.input_tokens": accumulated_usage["inputTokens"],
                        "gen_ai.usage.output_tokens": accumulated_usage["outputTokens"],
                        "gen_ai.usage.total_tokens": accumulated_usage["totalTokens"],
                        "gen_ai.usage.cache_read_input_tokens": accumulated_usage.get("cacheReadInputTokens", 0),
                        "gen_ai.usage.cache_write_input_tokens": accumulated_usage.get("cacheWriteInputTokens", 0),
                    }
                )

        self._end_span(span, attributes, error)

    def _construct_tool_definitions(self, tools_config: dict) -> list[dict[str, Any]]:
        """Constructs a list of tool definitions from the provided tools_config."""
        return [
            {
                "name": name,
                "description": spec.get("description"),
                "inputSchema": spec.get("inputSchema"),
                "outputSchema": spec.get("outputSchema"),
            }
            for name, spec in tools_config.items()
        ]

    def start_multiagent_span(
        self,
        task: MultiAgentInput,
        instance: str,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
    ) -> Span:
        """Start a new span for swarm invocation."""
        operation = f"invoke_{instance}"
        attributes: Dict[str, AttributeValue] = self._get_common_attributes(operation)
        attributes.update(
            {
                "gen_ai.agent.name": instance,
            }
        )

        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)

        span = self._start_span(operation, attributes=attributes, span_kind=trace_api.SpanKind.CLIENT)

        if self.use_latest_genai_conventions:
            parts: list[dict[str, Any]] = []
            if isinstance(task, list):
                parts = self._map_content_blocks_to_otel_parts(task)
            else:
                parts = [{"type": "text", "content": task}]
            self._add_event(
                span,
                "gen_ai.client.inference.operation.details",
                {"gen_ai.input.messages": serialize([{"role": "user", "parts": parts}])},
            )
        else:
            self._add_event(
                span,
                "gen_ai.user.message",
                event_attributes={"content": serialize(task) if isinstance(task, list) else task},
            )

        return span

    def end_swarm_span(
        self,
        span: Span,
        result: Optional[str] = None,
    ) -> None:
        """End a swarm span with results."""
        if result:
            if self.use_latest_genai_conventions:
                self._add_event(
                    span,
                    "gen_ai.client.inference.operation.details",
                    {
                        "gen_ai.output.messages": serialize(
                            [
                                {
                                    "role": "assistant",
                                    "parts": [{"type": "text", "content": result}],
                                }
                            ]
                        )
                    },
                )
            else:
                self._add_event(
                    span,
                    "gen_ai.choice",
                    event_attributes={"message": result},
                )

    def _get_common_attributes(
        self,
        operation_name: str,
    ) -> Dict[str, AttributeValue]:
        """Returns a dictionary of common attributes based on the convention version used.

        Args:
            operation_name: The name of the operation.

        Returns:
            A dictionary of attributes following the appropriate GenAI conventions.
        """
        common_attributes = {"gen_ai.operation.name": operation_name}
        if self.use_latest_genai_conventions:
            common_attributes.update(
                {
                    "gen_ai.provider.name": "strands-agents",
                }
            )
        else:
            common_attributes.update(
                {
                    "gen_ai.system": "strands-agents",
                }
            )
        return dict(common_attributes)

    def _add_event_messages(self, span: Span, messages: Messages) -> None:
        """Adds messages as event to the provided span based on the current GenAI conventions.

        Args:
            span: The span to which events will be added.
            messages: List of messages being sent to the agent.
        """
        if self.use_latest_genai_conventions:
            input_messages: list = []
            for message in messages:
                input_messages.append(
                    {"role": message["role"], "parts": self._map_content_blocks_to_otel_parts(message["content"])}
                )
            self._add_event(
                span, "gen_ai.client.inference.operation.details", {"gen_ai.input.messages": serialize(input_messages)}
            )
        else:
            for message in messages:
                self._add_event(
                    span,
                    self._get_event_name_for_message(message),
                    {"content": serialize(message["content"])},
                )

    def _map_content_blocks_to_otel_parts(
        self, content_blocks: list[ContentBlock] | list[InterruptResponseContent]
    ) -> list[dict[str, Any]]:
        """Map content blocks to OpenTelemetry parts format."""
        parts: list[dict[str, Any]] = []

        for block in cast(list[dict[str, Any]], content_blocks):
            if "interruptResponse" in block:
                interrupt_response = block["interruptResponse"]
                parts.append(
                    {
                        "type": "interrupt_response",
                        "id": interrupt_response["interruptId"],
                        "response": interrupt_response["response"],
                    },
                )
            elif "text" in block:
                # Standard TextPart
                parts.append({"type": "text", "content": block["text"]})
            elif "toolUse" in block:
                # Standard ToolCallRequestPart
                tool_use = block["toolUse"]
                parts.append(
                    {
                        "type": "tool_call",
                        "name": tool_use["name"],
                        "id": tool_use["toolUseId"],
                        "arguments": tool_use["input"],
                    }
                )
            elif "toolResult" in block:
                # Standard ToolCallResponsePart
                tool_result = block["toolResult"]
                parts.append(
                    {
                        "type": "tool_call_response",
                        "id": tool_result["toolUseId"],
                        "response": tool_result["content"],
                    }
                )
            else:
                # For all other ContentBlock types, use the key as type and value as content
                for key, value in block.items():
                    parts.append({"type": key, "content": value})
        return parts


# Singleton instance for global access
_tracer_instance = None


def get_tracer() -> Tracer:
    """Get or create the global tracer.

    Returns:
        The global tracer instance.
    """
    global _tracer_instance

    if not _tracer_instance:
        _tracer_instance = Tracer()

    return _tracer_instance


def serialize(obj: Any) -> str:
    """Serialize an object to JSON with consistent settings.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, ensure_ascii=False, cls=JSONEncoder)
