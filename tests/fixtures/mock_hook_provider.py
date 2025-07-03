from typing import Type

from strands.experimental.hooks import HookEvent, HookProvider, HookRegistry


class MockHookProvider(HookProvider):
    def __init__(self, event_types: list[Type]):
        self.events_received = []
        self.events_types = event_types

    def register_hooks(self, registry: HookRegistry) -> None:
        for event_type in self.events_types:
            registry.add_callback(event_type, self._add_event)

    def _add_event(self, event: HookEvent) -> None:
        self.events_received.append(event)
