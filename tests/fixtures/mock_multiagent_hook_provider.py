from typing import Iterator, Literal, Tuple, Type

from strands.experimental.multiagent_hooks import (
    AfterMultiAgentInvocationEvent,
    AfterNodeInvocationEvent,
    BeforeNodeInvocationEvent,
    MultiagentInitializedEvent,
)
from strands.hooks import (
    HookEvent,
    HookProvider,
    HookRegistry,
)


class MockMultiAgentHookProvider(HookProvider):
    def __init__(self, event_types: list[Type] | Literal["all"]):
        if event_types == "all":
            event_types = [
                MultiagentInitializedEvent,
                BeforeNodeInvocationEvent,
                AfterNodeInvocationEvent,
                AfterMultiAgentInvocationEvent,
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
