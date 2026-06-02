"""Public type surface for the Strands SDK.

Re-exports the wire types from :mod:`strands._generated` (machine-written by
``wasmtime.component.bindgen``). Users import wire types from here, not from
``_generated`` directly.

To remove a type from the public surface, add its name to ``_DROPPED``. The
underlying generated module stays untouched.

The ``*Input`` aliases below name the unions accepted at the Agent
boundary, where we auto-coerce a bare payload into its variant arm
(e.g. ``BedrockModel(...)`` becomes ``ModelConfig.Bedrock(...)``).
Everywhere else the caller wraps explicitly.
"""

from __future__ import annotations

from strands._generated import *  # noqa: F401, F403
from strands._generated import __all__ as _generated_all
from strands._generated.strands_agent.conversation import (
    ConversationManagerConfig,
    SlidingWindowConversationManager,
    SummarizingConversationManager,
)
from strands._generated.strands_agent.models import (
    AnthropicModel,
    BedrockModel,
    CustomModel,
    GoogleModel,
    ModelConfig,
    OpenaiModel,
)
from strands._generated.strands_agent.vended import (
    AgentSkills,
    BashTool,
    ContextOffloader,
    FileEditorTool,
    HttpRequestTool,
    NotebookTool,
    VendedPlugin,
    VendedTool,
)

ModelInput = ModelConfig | BedrockModel | AnthropicModel | OpenaiModel | GoogleModel | CustomModel
ConversationManagerInput = ConversationManagerConfig | SlidingWindowConversationManager | SummarizingConversationManager
VendedToolInput = VendedTool | BashTool | FileEditorTool | HttpRequestTool | NotebookTool
VendedPluginInput = VendedPlugin | AgentSkills | ContextOffloader

_DROPPED: set[str] = {
    "Datetime",
    "Duration",
    "Error",
    "InputStream",
    "Instant",
    "OutputStream",
    "Pollable",
}

__all__ = sorted(  # pyright: ignore[reportUnsupportedDunderAll]
    (set(_generated_all) - _DROPPED)
    | {
        "ConversationManagerInput",
        "ModelInput",
        "VendedPluginInput",
        "VendedToolInput",
    }
)
