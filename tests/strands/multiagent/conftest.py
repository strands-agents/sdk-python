import pytest

from strands.experimental.hooks.multiagent import BeforeNodeCallEvent
from strands.hooks import HookProvider


@pytest.fixture
def interrupt_hook():
    class Hook(HookProvider):
        def register_hooks(self, registry):
            registry.add_callback(BeforeNodeCallEvent, self.interrupt)

        def interrupt(self, event):
            return event.interrupt("test_name", reason="test_reason")

    return Hook()
