"""Primitive types for the memory extraction subsystem.

This module defines the building blocks that automatic extraction is composed
from: the :class:`ExtractionResult` an extractor produces, the
:class:`Extractor` contract that turns messages into entries, the
:class:`MemoryMessageFilter` that prunes content blocks before extraction, the
:class:`ExtractionTrigger` that decides *when* extraction runs, and the
:class:`ExtractionConfig` that ties them together on a store.

These are Python ports of the TypeScript ``memory/extraction/types.ts``
interfaces. All public field and method names use ``snake_case`` per the
requirements naming convention mapping (e.g. ``defaultModel`` ->
``default_model``).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, Protocol, runtime_checkable

from ...models.model import Model
from ...types.content import Message

if TYPE_CHECKING:
    # Imported lazily to avoid a circular import: ``ExtractionTriggerContext``
    # only references ``Agent`` in annotations, so a ``TYPE_CHECKING`` import
    # (with a string annotation) is sufficient.
    from ...agent.agent import Agent

# Metadata mapping for an extracted entry. Modeled as ``dict[str, Any]`` because
# entry metadata holds arbitrary JSON values (scores, ids, timestamps, etc.).
Metadata = dict[str, Any]

# Content-block kinds that a ``MemoryMessageFilter`` can exclude before messages
# reach an ``Extractor`` (or the no-extractor ``add_messages`` passthrough).
#
# Mirrors the keys of ``strands.types.content.ContentBlock``: each key is a block
# kind (``{"toolUse": ...}`` -> ``"toolUse"``, ``{"text": ...}`` -> ``"text"``).
# These are the filterable kinds; the coordinator's ``_block_kind`` resolves a
# block to one of these strings at runtime.
MemoryContentBlockType = Literal[
    "text",
    "toolUse",
    "toolResult",
    "image",
    "document",
    "reasoningContent",
    "video",
    "guardContent",
    "citationsContent",
    "cachePoint",
]


@dataclass
class ExtractionResult:
    """A discrete entry produced by an :class:`Extractor`.

    Ready to be written to a store via its ``add``.

    Attributes:
        content: The textual content of the entry.
        metadata: Optional metadata to associate with the entry.
    """

    content: str
    metadata: Metadata | None = None


@dataclass
class ExtractorContext:
    """Context passed to :meth:`Extractor.extract`.

    Lets the manager hand an extractor a fallback model without the extractor
    having to be wired to the agent directly. :class:`ModelExtractor` uses its
    own configured model when set, else :attr:`default_model`.

    Attributes:
        default_model: The agent's model, supplied so an extractor can default
            to it.
    """

    default_model: Model | None = None


@runtime_checkable
class Extractor(Protocol):
    """Transforms conversation messages into discrete, searchable entries.

    Implementations distill raw turns into facts worth remembering. Optional on
    a store's :class:`ExtractionConfig`: when absent, the manager passes messages
    straight to the store's ``add_messages`` (the no-extractor passthrough),
    which is the right path for backends that extract server-side.
    """

    async def extract(self, messages: list[Message], context: ExtractorContext | None = None) -> list[ExtractionResult]:
        """Extract entries from a batch of messages.

        Args:
            messages: The filtered messages to extract from.
            context: Optional context (e.g. a fallback model).

        Returns:
            The entries to write to the store.
        """
        ...


@dataclass
class MemoryMessageFilter:
    """Filters content blocks out of messages before extraction.

    Blocks whose kind is in :attr:`exclude` are stripped; a message left with no
    content is dropped entirely. Defaults to excluding tool traffic (``toolUse``
    / ``toolResult``), which is rarely useful as long-term memory and adds noise.

    Attributes:
        exclude: Content block kinds to strip before extraction.
    """

    exclude: list[MemoryContentBlockType]


# Default filter: drop tool-call traffic, keep everything else (text, reasoning,
# media).
DEFAULT_MEMORY_MESSAGE_FILTER = MemoryMessageFilter(exclude=["toolUse", "toolResult"])


@dataclass
class ExtractionTriggerContext:
    """Context handed to :meth:`ExtractionTrigger.attach`.

    Lets a trigger wire itself into the agent lifecycle and signal when
    extraction should run for its store.

    Attributes:
        agent: The agent the trigger attaches its hooks to.
        fire: Save this store's unsaved messages now. Runs in the background and
            returns immediately, so calling it from a hook never blocks the
            agent. To await completion, see ``MemoryManager.flush``.
    """

    agent: Agent
    fire: Callable[[], None]


class ExtractionTrigger(ABC):
    """Controls when a store's :class:`ExtractionConfig` runs.

    A trigger is a self-attaching value object: :meth:`attach` wires whatever
    agent hooks the trigger needs and calls :attr:`ExtractionTriggerContext.fire`
    when extraction should happen. Subclass this for custom triggering logic.

    A trigger must eventually fire for its store's buffered messages to be
    written: the high-water-mark dedup means skipped turns are still picked up on
    the *next* fire, but a trigger that never fires never extracts (and its
    messages stay buffered for the session). For a guaranteed final write at a
    boundary, the caller uses ``MemoryManager.flush``, which force-completes
    regardless of triggers.

    Attributes:
        name: Stable identifier for this trigger kind, used in logging.
    """

    name: str

    @abstractmethod
    def attach(self, context: ExtractionTriggerContext) -> None:
        """Wire this trigger into the agent lifecycle.

        Called once per store during ``MemoryManager`` initialization. Register
        hooks on ``context.agent`` and call ``context.fire()`` when extraction
        should run.

        Args:
            context: The agent to attach to and the fire callback bound to this
                trigger's store.
        """
        ...


@dataclass
class ExtractionConfig:
    """Per-store automatic-extraction configuration.

    Lives on a store (via ``MemoryStoreConfig``) so different stores can extract
    on different schedules and in different styles. :attr:`trigger` decides
    *when*; :attr:`extractor` decides *how* (omit it to pass raw messages
    straight to the store); :attr:`filter` prunes content blocks first.

    Attributes:
        trigger: When to run extraction. A single trigger or a list; an empty
            list is rejected at construction. Multiple triggers compose
            (extraction runs whenever any of them fires).
        extractor: How to turn messages into entries. When set, the store must
            implement ``add`` (entries are written to it). When omitted, the
            manager hands the filtered messages straight to the store's
            ``add_messages`` (which the store must then implement) -- so backends
            that extract server-side need no client-side extractor.
        filter: Content blocks to strip before extraction. Defaults to
            :data:`DEFAULT_MEMORY_MESSAGE_FILTER` (excludes ``toolUse`` /
            ``toolResult``). For use cases that value distilling over the *full*
            turn, pass ``MemoryMessageFilter(exclude=[])`` so tool blocks reach
            ``add_messages``.
    """

    trigger: ExtractionTrigger | list[ExtractionTrigger]
    extractor: Extractor | None = None
    filter: MemoryMessageFilter | None = None
