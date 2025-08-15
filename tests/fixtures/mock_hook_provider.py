from typing import Iterator, Tuple, Type

from strands.hooks import HookEvent, HookProvider, HookRegistry


class MockHookProvider(HookProvider):
    def __init__(self, event_types: list[Type]):
        self.events_received = []
        self.events_types = event_types

    def get_events(self) -> Tuple[int, Iterator[HookEvent]]:
        return len(self.events_received), iter(self.events_received)

    def register_hooks(self, registry: HookRegistry) -> None:
        for event_type in self.events_types:
            registry.add_callback(event_type, self.add_event)

    def add_event(self, event: HookEvent) -> None:
        self.events_received.append(event)
