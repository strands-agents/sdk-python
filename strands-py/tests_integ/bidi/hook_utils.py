"""Shared utilities for testing BidiAgent hooks."""

from strands.experimental.hooks.events import (
    BidiAfterInvocationEvent,
    BidiAfterToolCallEvent,
    BidiAgentInitializedEvent,
    BidiBeforeInvocationEvent,
    BidiBeforeToolCallEvent,
    BidiInterruptionEvent,
    BidiMessageAddedEvent,
)
from strands.hooks import HookProvider


class HookEventCollector(HookProvider):
    """Hook provider that collects all emitted events for testing."""

    def __init__(self):
        self.events = []

    def register_hooks(self, registry):
        registry.add_callback(BidiAgentInitializedEvent, self.on_initialized)
        registry.add_callback(BidiBeforeInvocationEvent, self.on_before_invocation)
        registry.add_callback(BidiAfterInvocationEvent, self.on_after_invocation)
        registry.add_callback(BidiBeforeToolCallEvent, self.on_before_tool_call)
        registry.add_callback(BidiAfterToolCallEvent, self.on_after_tool_call)
        registry.add_callback(BidiMessageAddedEvent, self.on_message_added)
        registry.add_callback(BidiInterruptionEvent, self.on_interruption)

    def on_initialized(self, event: BidiAgentInitializedEvent):
        self.events.append(("initialized", event))

    def on_before_invocation(self, event: BidiBeforeInvocationEvent):
        self.events.append(("before_invocation", event))

    def on_after_invocation(self, event: BidiAfterInvocationEvent):
        self.events.append(("after_invocation", event))

    def on_before_tool_call(self, event: BidiBeforeToolCallEvent):
        self.events.append(("before_tool_call", event))

    def on_after_tool_call(self, event: BidiAfterToolCallEvent):
        self.events.append(("after_tool_call", event))

    def on_message_added(self, event: BidiMessageAddedEvent):
        self.events.append(("message_added", event))

    def on_interruption(self, event: BidiInterruptionEvent):
        self.events.append(("interruption", event))

    def get_event_types(self):
        """Get list of event type names in order."""
        return [event_type for event_type, _ in self.events]

    def get_events_by_type(self, event_type):
        """Get all events of a specific type."""
        return [event for et, event in self.events if et == event_type]

    def get_tool_calls(self):
        """Get list of tool names that were called."""
        before_calls = self.get_events_by_type("before_tool_call")
        return [event.tool_use["name"] for event in before_calls]

    def verify_tool_execution(self):
        """Verify that tool execution hooks were properly paired."""
        before_calls = self.get_events_by_type("before_tool_call")
        after_calls = self.get_events_by_type("after_tool_call")

        assert len(before_calls) == len(after_calls), "Before and after tool call hooks should be paired"

        before_tools = [event.tool_use["name"] for event in before_calls]
        after_tools = [event.tool_use["name"] for event in after_calls]

        assert before_tools == after_tools, "Tool call order should match between before and after hooks"

        return before_tools
