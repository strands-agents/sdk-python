from typing import Iterator, Literal, Tuple, Type

from strands import Agent
from strands.experimental.hooks import (
    AfterModelInvocationEvent,
    AfterToolInvocationEvent,
    BeforeModelInvocationEvent,
    BeforeToolInvocationEvent,
)
from strands.hooks import (
    AfterInvocationEvent,
    AgentInitializedEvent,
    BeforeInvocationEvent,
    HookEvent,
    HookProvider,
    HookRegistry,
    MessageAddedEvent,
)


class MockHookProvider(HookProvider):
    def __init__(self, event_types: list[Type] | Literal["all"]):
        if event_types == "all":
            event_types = [
                AgentInitializedEvent,
                BeforeInvocationEvent,
                AfterInvocationEvent,
                AfterToolInvocationEvent,
                BeforeToolInvocationEvent,
                BeforeModelInvocationEvent,
                AfterModelInvocationEvent,
                MessageAddedEvent,
            ]

        self.events_received = []
        self.events_types = event_types

    @property
    def event_types_received(self):
        return [type(event) for event in self.events_received]

    def get_events(self) -> Tuple[int, Iterator[HookEvent]]:
        return len(self.events_received), iter(self.events_received)

    def register_hooks(self, registry: HookRegistry) -> None:
        for event_type in self.events_types:
            registry.add_callback(event_type, self.add_event)

    def add_event(self, event: HookEvent) -> None:
        self.events_received.append(event)

    def extract_for(self, agent: Agent) -> "MockHookProvider":
        """Extracts a hook provider for the given agent, including the events that were fired for that agent.

        Convenience method when sharing a hook provider between multiple agents."""
        child_provider = MockHookProvider(self.events_types)
        child_provider.events_received = [event for event in self.events_received if event.agent == agent]
        return child_provider
