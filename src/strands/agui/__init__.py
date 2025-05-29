"""Strands AG-UI integration package.

This package provides integration between Strands agents and AG-UI protocol
compatible frontends, including state management and event streaming.
"""

from .bridge import (
    AGUIEventType,
    StrandsAGUIBridge,
    StrandsAGUIEndpoint,
    create_strands_agui_setup,
)
from .state_tools import (
    StrandsStateManager,
    emit_ui_update,
    get_agent_state,
    get_state_manager,
    set_agent_state,
    setup_agent_state_management,
    update_agent_state,
)

__all__ = [
    "AGUIEventType",
    "StrandsAGUIBridge",
    "StrandsAGUIEndpoint",
    "create_strands_agui_setup",
    "StrandsStateManager",
    "emit_ui_update",
    "get_agent_state",
    "get_state_manager",
    "set_agent_state",
    "setup_agent_state_management",
    "update_agent_state",
]
