"""OpenTelemetry integration.

This module provides tracing capabilities using OpenTelemetry,
enabling trace data to be sent to OTLP endpoints.
"""

import json
import logging
from datetime import date, datetime, timezone
from typing import Any, Dict, Mapping, Optional

import opentelemetry.trace as trace_api
from opentelemetry.trace import Span, StatusCode

from ..agent.agent_result import AgentResult
from ..types.content import Message, Messages
from ..types.streaming import Usage
from ..types.tools import ToolResult, ToolUse
from ..types.traces import AttributeValue

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

    When the STRANDS_OTEL_ENABLE_CONSOLE_EXPORT environment variable is set,
    traces are printed to the console.
    """

    def __init__(
        self,
        service_name: str = "strands-agents",
    ):
        """Initialize the tracer.

        Args:
            service_name: Name of the service for OpenTelemetry.
        """
        self.service_name = service_name
        self.tracer_provider: Optional[trace_api.TracerProvider] = None
        self.tracer: Optional[trace_api.Tracer] = None

        self.tracer_provider = trace_api.get_tracer_provider()
        self.tracer = self.tracer_provider.get_tracer(self.service_name)

    def _start_span(
        self,
        span_name: str,
        parent_span: Optional[Span] = None,
        attributes: Optional[Dict[str, AttributeValue]] = None,
    ) -> Optional[Span]:
        """Generic helper method to start a span with common attributes.

        Args:
            span_name: Name of the span to create
            parent_span: Optional parent span to link this span to
            attributes: Dictionary of attributes to set on the span

        Returns:
            The created span, or None if tracing is not enabled
        """
        if self.tracer is None:
            return None

        context = trace_api.set_span_in_context(parent_span) if parent_span else None
        span = self.tracer.start_span(name=span_name, context=context)

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

    def start_model_invoke_span(
        self,
        parent_span: Optional[Span] = None,
        agent_name: str = "Strands Agent",
        messages: Optional[Messages] = None,
        model_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Optional[Span]:
        """Start a new span for a model invocation.

        Args:
            parent_span: Optional parent span to link this span to.
            agent_name: Name of the agent making the model call.
            messages: Optional messages being sent to the model.
            model_id: Optional identifier for the model being invoked.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.system": "strands-agents",
            "agent.name": agent_name,
            "gen_ai.agent.name": agent_name,
            "gen_ai.prompt": serialize(messages),
        }

        if model_id:
            attributes["gen_ai.request.model"] = model_id

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        return self._start_span("Model invoke", parent_span, attributes)

    def end_model_invoke_span(
        self, span: Span, message: Message, usage: Usage, error: Optional[Exception] = None
    ) -> None:
        """End a model invocation span with results and metrics.

        Args:
            span: The span to end.
            message: The message response from the model.
            usage: Token usage information from the model call.
            error: Optional exception if the model call failed.
        """
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.completion": serialize(message["content"]),
            "gen_ai.usage.prompt_tokens": usage["inputTokens"],
            "gen_ai.usage.completion_tokens": usage["outputTokens"],
            "gen_ai.usage.total_tokens": usage["totalTokens"],
        }

        self._end_span(span, attributes, error)

    def start_tool_call_span(self, tool: ToolUse, parent_span: Optional[Span] = None, **kwargs: Any) -> Optional[Span]:
        """Start a new span for a tool call.

        Args:
            tool: The tool being used.
            parent_span: Optional parent span to link this span to.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.prompt": serialize(tool),
            "tool.name": tool["name"],
            "tool.id": tool["toolUseId"],
            "tool.parameters": serialize(tool["input"]),
        }

        # Add additional kwargs as attributes
        attributes.update(kwargs)

        span_name = f"Tool: {tool['name']}"
        return self._start_span(span_name, parent_span, attributes)

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

            tool_result_content_json = serialize(tool_result.get("content"))
            attributes.update(
                {
                    "tool.result": tool_result_content_json,
                    "gen_ai.completion": tool_result_content_json,
                    "tool.status": status_str,
                }
            )

        self._end_span(span, attributes, error)

    def start_event_loop_cycle_span(
        self,
        event_loop_kwargs: Any,
        parent_span: Optional[Span] = None,
        messages: Optional[Messages] = None,
        **kwargs: Any,
    ) -> Optional[Span]:
        """Start a new span for an event loop cycle.

        Args:
            event_loop_kwargs: Arguments for the event loop cycle.
            parent_span: Optional parent span to link this span to.
            messages: Optional messages being processed in this cycle.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        event_loop_cycle_id = str(event_loop_kwargs.get("event_loop_cycle_id"))
        parent_span = parent_span if parent_span else event_loop_kwargs.get("event_loop_parent_span")

        attributes: Dict[str, AttributeValue] = {
            "gen_ai.prompt": serialize(messages),
            "event_loop.cycle_id": event_loop_cycle_id,
        }

        if "event_loop_parent_cycle_id" in event_loop_kwargs:
            attributes["event_loop.parent_cycle_id"] = str(event_loop_kwargs["event_loop_parent_cycle_id"])

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        span_name = f"Cycle {event_loop_cycle_id}"
        return self._start_span(span_name, parent_span, attributes)

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
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.completion": serialize(message["content"]),
        }

        if tool_result_message:
            attributes["tool.result"] = serialize(tool_result_message["content"])

        self._end_span(span, attributes, error)

    def start_agent_span(
        self,
        prompt: str,
        agent_name: str = "Strands Agent",
        model_id: Optional[str] = None,
        tools: Optional[list] = None,
        custom_trace_attributes: Optional[Mapping[str, AttributeValue]] = None,
        **kwargs: Any,
    ) -> Optional[Span]:
        """Start a new span for an agent invocation.

        Args:
            prompt: The user prompt being sent to the agent.
            agent_name: Name of the agent.
            model_id: Optional model identifier.
            tools: Optional list of tools being used.
            custom_trace_attributes: Optional mapping of custom trace attributes to include in the span.
            **kwargs: Additional attributes to add to the span.

        Returns:
            The created span, or None if tracing is not enabled.
        """
        attributes: Dict[str, AttributeValue] = {
            "gen_ai.system": "strands-agents",
            "agent.name": agent_name,
            "gen_ai.agent.name": agent_name,
            "gen_ai.prompt": prompt,
        }

        if model_id:
            attributes["gen_ai.request.model"] = model_id

        if tools:
            tools_json = serialize(tools)
            attributes["agent.tools"] = tools_json
            attributes["gen_ai.agent.tools"] = tools_json

        # Add custom trace attributes if provided
        if custom_trace_attributes:
            attributes.update(custom_trace_attributes)

        # Add additional kwargs as attributes
        attributes.update({k: v for k, v in kwargs.items() if isinstance(v, (str, int, float, bool))})

        return self._start_span(agent_name, attributes=attributes)

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
            metrics: Metrics data to add to the span.
        """
        attributes: Dict[str, AttributeValue] = {}

        if response:
            attributes.update(
                {
                    "gen_ai.completion": str(response),
                }
            )

            if hasattr(response, "metrics") and hasattr(response.metrics, "accumulated_usage"):
                accumulated_usage = response.metrics.accumulated_usage
                attributes.update(
                    {
                        "gen_ai.usage.prompt_tokens": accumulated_usage["inputTokens"],
                        "gen_ai.usage.completion_tokens": accumulated_usage["outputTokens"],
                        "gen_ai.usage.total_tokens": accumulated_usage["totalTokens"],
                    }
                )

        self._end_span(span, attributes, error)


# Singleton instance for global access
_tracer_instance = None


def get_tracer(
    service_name: str = "strands-agents",
) -> Tracer:
    """Get or create the global tracer.

    Args:
        service_name: Name of the service for OpenTelemetry.

    Returns:
        The global tracer instance.
    """
    global _tracer_instance

    if not _tracer_instance:
        _tracer_instance = Tracer(
            service_name=service_name,
        )

    return _tracer_instance


def serialize(obj: Any) -> str:
    """Serialize an object to JSON with consistent settings.

    Args:
        obj: The object to serialize

    Returns:
        JSON string representation of the object
    """
    return json.dumps(obj, ensure_ascii=False, cls=JSONEncoder)
