"""A :class:`~strands.memory.types.MemoryStore` backed by Graphiti.

`Graphiti <https://github.com/getzep/graphiti>`_ (by Zep, Apache-2.0) is a
temporal knowledge-graph library purpose-built for agent memory. It ingests
discrete *episodes* and extracts a bi-temporal graph of entities and
relationships server-side, then answers hybrid (semantic + keyword + graph)
queries over them.

This store maps the :class:`~strands.memory.types.MemoryStore` contract onto
Graphiti's client:

- :meth:`GraphitiMemoryStore.search` -> ``Graphiti.search`` (hybrid retrieval),
  returning one :class:`~strands.memory.types.MemoryEntry` per retrieved
  ``EntityEdge``. The edge's ``fact`` becomes the entry content, and its
  identifiers and bi-temporal timestamps are surfaced in ``metadata``.
- :meth:`GraphitiMemoryStore.add` -> ``Graphiti.add_episode`` (a ``text``
  episode) for single-fact writes from the ``add_memory`` tool, programmatic
  ``MemoryManager.add``, or a client-side extractor.
- :meth:`GraphitiMemoryStore.add_messages` -> ``Graphiti.add_episode`` (a
  ``message`` episode) for server-side extraction: the manager hands a raw
  message batch straight to Graphiti, which extracts it with no extra model
  call.

``graphiti-core`` is an optional dependency. It is imported lazily so importing
this module never requires it; a missing install raises a clear
:class:`ImportError` only when a write is attempted. The store accepts an
already-configured ``Graphiti`` client, so the choice of graph backend (Neo4j,
FalkorDB, Kuzu, ...) is the caller's and no database is needed to construct or
unit-test the store.

Example:
    ```python
    from graphiti_core import Graphiti
    from strands import Agent
    from strands.memory import MemoryManager
    from strands.memory.stores import GraphitiMemoryStore

    client = Graphiti("bolt://localhost:7687", "neo4j", "password")
    await client.build_indices_and_constraints()

    store = GraphitiMemoryStore(client=client, name="graph", writable=True)
    agent = Agent(memory_manager=MemoryManager(stores=[store]))
    ```
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any

from ..types import AddMessagesContext, MemoryEntry, Metadata, SearchOptions

if TYPE_CHECKING:
    from ...types.content import Message
    from ..extraction.types import ExtractionConfig

logger = logging.getLogger(__name__)

# Graphiti's own default page size for ``search``; used when neither the caller nor the store
# specifies one.
DEFAULT_MAX_SEARCH_RESULTS = 10

# Shown when ``graphiti-core`` is not installed but a write is attempted.
_GRAPHITI_IMPORT_ERROR = (
    "GraphitiMemoryStore requires the 'graphiti-core' package, which is not installed. "
    "Install it with: pip install 'strands-agents[graphiti]' (or: pip install graphiti-core). "
    "Graphiti pairs with a graph backend such as Neo4j, FalkorDB, or Kuzu."
)

# Metadata keys that parameterize the Graphiti episode on a write rather than being stored as
# opaque payload (``add_episode`` takes no free-form metadata map).
_EPISODE_NAME_KEY = "name"
_SOURCE_DESCRIPTION_KEY = "source_description"
_REFERENCE_TIME_KEY = "reference_time"
_GROUP_ID_KEY = "group_id"


def _resolve_episode_type(source: str) -> Any:
    """Resolve a Graphiti ``EpisodeType`` member by name, importing ``graphiti-core`` lazily.

    Args:
        source: The ``EpisodeType`` member name (e.g. ``"text"`` or ``"message"``).

    Returns:
        The matching ``EpisodeType`` member.

    Raises:
        ImportError: If ``graphiti-core`` is not installed.
    """
    try:
        from graphiti_core.nodes import EpisodeType
    except ImportError as error:
        raise ImportError(_GRAPHITI_IMPORT_ERROR) from error
    return EpisodeType[source]


def _isoformat(value: datetime | None) -> str | None:
    """Render a datetime as an ISO 8601 string for JSON-compatible metadata, or ``None``."""
    return value.isoformat() if value is not None else None


class GraphitiMemoryStore:
    """A :class:`~strands.memory.types.MemoryStore` backed by a Graphiti temporal knowledge graph.

    Wraps an already-configured ``graphiti_core.Graphiti`` client. Searches map to Graphiti's
    hybrid retrieval and writes map to ``add_episode``. The store is namespaced by an optional
    ``group_id`` (Graphiti's tenancy primitive): it scopes every search and stamps every write,
    so one client can back many isolated stores.

    Attributes:
        name: Unique identifier for this store, used to target it in tools.
        description: Human-readable description; included in tool descriptions.
        max_search_results: Default maximum results per search.
        writable: Whether this store accepts writes.
        extraction: Resolved automatic-extraction configuration, or ``None``/``False`` when off.
    """

    name: str
    description: str | None
    max_search_results: int | None
    writable: bool
    extraction: ExtractionConfig | bool | None

    def __init__(
        self,
        *,
        client: Any,
        name: str,
        description: str | None = None,
        max_search_results: int | None = None,
        writable: bool = False,
        extraction: ExtractionConfig | bool | None = None,
        group_id: str | None = None,
        source_description: str = "strands-agent-memory",
    ) -> None:
        """Initialize the store.

        Args:
            client: A configured ``graphiti_core.Graphiti`` instance. The store never constructs
                one itself, so the graph backend (Neo4j, FalkorDB, Kuzu, ...) and any LLM/embedder
                clients are the caller's to wire up. Typed as ``Any`` so importing this module does
                not require ``graphiti-core``.
            name: Unique identifier for this store, used to target it in tools.
            description: Human-readable description; included in tool descriptions.
            max_search_results: Default maximum results per search. Must be at least 1 when set.
            writable: Whether this store accepts writes. A writable store exposes ``add`` and
                ``add_messages``.
            extraction: Automatic-extraction configuration for this writable store. Because the
                store implements ``add_messages``, the default (``True``) extracts server-side:
                Graphiti receives the raw messages and extracts them with no extra model call.
            group_id: Graphiti group identifier used to namespace this store. Scopes every search
                and is stamped on every write, isolating this store's episodes from others sharing
                the same client. Omit for Graphiti's default group.
            source_description: Default ``source_description`` stamped on episodes written by this
                store. A per-write ``metadata['source_description']`` overrides it.

        Raises:
            ValueError: If ``max_search_results`` is set below 1.
        """
        if max_search_results is not None and max_search_results < 1:
            raise ValueError("GraphitiMemoryStore: max_search_results must be at least 1")

        self.name = name
        self.description = description
        self.max_search_results = max_search_results
        self.writable = writable
        self.extraction = extraction

        self._client = client
        # An empty group id is meaningless to Graphiti (it would scope to a nonexistent group on
        # search and pin episodes to a "" group on write), so normalize it to "unscoped".
        self._group_id = group_id or None
        self._source_description = source_description

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Search the graph for facts matching the query, ordered by relevance.

        Delegates to ``Graphiti.search`` (hybrid semantic + keyword + graph retrieval) and maps
        each returned ``EntityEdge`` to a :class:`~strands.memory.types.MemoryEntry`. The edge's
        ``fact`` becomes the entry content; its identifiers and bi-temporal timestamps are surfaced
        in ``metadata`` so callers can attribute and weigh each fact:

        - ``uuid``, ``name``, ``group_id``, ``source_node_uuid``, ``target_node_uuid``,
          ``episodes`` — graph identifiers.
        - ``created_at`` / ``expired_at`` — *transaction time*: when the fact was recorded and,
          if superseded, retracted.
        - ``valid_at`` / ``invalid_at`` — *valid time*: when the fact became and ceased to be true
          in the world.

        Only present (non-``None``) keys are included. Timestamps are rendered as ISO 8601 strings.

        Args:
            query: What to search for.
            options: Optional search configuration. ``max_search_results`` caps the results,
                falling back to the store's ``max_search_results`` and then
                :data:`DEFAULT_MAX_SEARCH_RESULTS`.

        Returns:
            Matching memory entries ordered by relevance.

        Raises:
            ValueError: If ``options['max_search_results']`` is set below 1.
        """
        caller_max = options.get("max_search_results") if options is not None else None
        if caller_max is not None and caller_max < 1:
            raise ValueError("GraphitiMemoryStore: max_search_results must be at least 1")

        limit = (
            caller_max
            if caller_max is not None
            else self.max_search_results
            if self.max_search_results is not None
            else DEFAULT_MAX_SEARCH_RESULTS
        )
        group_ids = [self._group_id] if self._group_id is not None else None

        logger.debug(
            "store=<%s>, group_id=<%s>, num_results=<%s> | searching graphiti",
            self.name,
            self._group_id,
            limit,
        )

        edges = await self._client.search(query, group_ids=group_ids, num_results=limit)

        entries = [MemoryEntry(content=edge.fact, metadata=self._edge_metadata(edge)) for edge in edges]
        logger.debug("store=<%s>, results=<%d> | graphiti search complete", self.name, len(entries))
        return entries

    async def add(self, content: str, metadata: Metadata | None = None) -> Any:
        """Add a single fact to the graph as a ``text`` episode.

        Wraps ``Graphiti.add_episode``, which extracts entities and relationships from ``content``
        server-side. Used by the ``add_memory`` tool, programmatic ``MemoryManager.add``, and any
        client-side extractor. ``add_episode`` is idempotent on a stable ``metadata['uuid']``, so
        the at-least-once extraction path tolerates duplicate writes when one is supplied.

        Args:
            content: The fact to store. Must not be empty.
            metadata: Optional episode parameters. Recognized keys: ``name`` (episode name),
                ``source_description`` (overrides the store default), ``reference_time`` (a
                ``datetime`` or ISO 8601 string; defaults to now in UTC), ``group_id`` (overrides
                the store's namespace), and ``uuid`` (a stable id for idempotent writes). Other
                keys are ignored, since ``add_episode`` takes no free-form metadata map.

        Returns:
            The Graphiti ``AddEpisodeResults`` (store-specific; not consumed by the manager).

        Raises:
            ImportError: If ``graphiti-core`` is not installed.
            ValueError: If the store is not writable or ``content`` is empty.
        """
        return await self._add_episode(content, source="text", metadata=metadata)

    async def add_messages(self, messages: list[Message], context: AddMessagesContext | None = None) -> Any:
        """Ingest a batch of conversation messages as a single ``message`` episode.

        The server-side extraction sink: the manager hands the filtered batch here and Graphiti
        extracts it into the graph with no extra model call. Messages are serialized to a
        ``role: text`` transcript preserving turn order, then written via ``Graphiti.add_episode``
        as a ``message`` episode.

        Args:
            messages: The conversation messages to ingest.
            context: Manager-supplied context (currently unused).

        Returns:
            The Graphiti ``AddEpisodeResults`` (store-specific; not consumed by the manager), or
            ``None`` when ``messages`` has no text to ingest.

        Raises:
            ImportError: If ``graphiti-core`` is not installed.
            ValueError: If the store is not writable.
        """
        transcript = self._serialize_messages(messages)
        if not transcript:
            logger.debug("store=<%s> | no text in message batch | skipping graphiti write", self.name)
            return None
        return await self._add_episode(transcript, source="message", metadata=None)

    async def _add_episode(self, body: str, *, source: str, metadata: Metadata | None) -> Any:
        """Write one episode to Graphiti, resolving episode parameters from ``metadata``.

        Args:
            body: The episode body to ingest.
            source: The ``EpisodeType`` member name (``"text"`` or ``"message"``).
            metadata: Optional episode parameters; see :meth:`add`.

        Returns:
            The Graphiti ``AddEpisodeResults``.

        Raises:
            ImportError: If ``graphiti-core`` is not installed.
            ValueError: If the store is not writable or ``body`` is empty.
        """
        if not self.writable:
            raise ValueError(f"GraphitiMemoryStore: store '{self.name}' is not writable")
        if not body.strip():
            raise ValueError("GraphitiMemoryStore: content must not be empty")

        metadata = metadata or {}
        episode_type = _resolve_episode_type(source)
        name = str(metadata.get(_EPISODE_NAME_KEY, f"{self.name} {source} episode"))
        source_description = str(metadata.get(_SOURCE_DESCRIPTION_KEY, self._source_description))
        reference_time = self._resolve_reference_time(metadata.get(_REFERENCE_TIME_KEY))
        # Truthiness, not ``is not None``: an empty group id is meaningless to Graphiti, so fall
        # back to the store's namespace rather than pinning the episode to a "" group.
        group_id = metadata.get(_GROUP_ID_KEY) or self._group_id
        uuid = metadata.get("uuid")

        kwargs: dict[str, Any] = {
            "name": name,
            "episode_body": body,
            "source_description": source_description,
            "reference_time": reference_time,
            "source": episode_type,
        }
        if group_id is not None:
            kwargs["group_id"] = group_id
        if uuid is not None:
            kwargs["uuid"] = uuid

        logger.debug(
            "store=<%s>, group_id=<%s>, source=<%s> | adding graphiti episode",
            self.name,
            group_id,
            source,
        )
        return await self._client.add_episode(**kwargs)

    def _edge_metadata(self, edge: Any) -> Metadata:
        """Build a JSON-compatible metadata mapping from a Graphiti ``EntityEdge``.

        Surfaces graph identifiers and the bi-temporal timestamps, dropping ``None`` values and
        rendering datetimes as ISO 8601 strings.
        """
        candidates: Metadata = {
            "uuid": getattr(edge, "uuid", None),
            "name": getattr(edge, "name", None),
            "group_id": getattr(edge, "group_id", None),
            "source_node_uuid": getattr(edge, "source_node_uuid", None),
            "target_node_uuid": getattr(edge, "target_node_uuid", None),
            "episodes": getattr(edge, "episodes", None),
            "created_at": _isoformat(getattr(edge, "created_at", None)),
            "expired_at": _isoformat(getattr(edge, "expired_at", None)),
            "valid_at": _isoformat(getattr(edge, "valid_at", None)),
            "invalid_at": _isoformat(getattr(edge, "invalid_at", None)),
        }
        return {key: value for key, value in candidates.items() if value is not None}

    def _resolve_reference_time(self, value: Any) -> datetime:
        """Resolve a reference time from metadata into a tz-aware UTC ``datetime``.

        Graphiti's bi-temporal layer works in timezone-aware UTC, so a naive datetime (a bare
        ``datetime`` or an ISO string without an offset) is treated as UTC and made aware; an aware
        value is converted to UTC. Defaults to now in UTC when unset.
        """
        if isinstance(value, datetime):
            resolved = value
        elif isinstance(value, str):
            resolved = datetime.fromisoformat(value)
        else:
            return datetime.now(timezone.utc)

        if resolved.tzinfo is None:
            return resolved.replace(tzinfo=timezone.utc)
        return resolved.astimezone(timezone.utc)

    @staticmethod
    def _serialize_messages(messages: list[Message]) -> str:
        """Serialize messages into a ``role: text`` transcript, preserving turn order.

        Only text content blocks are included; messages with no text are skipped.
        """
        lines: list[str] = []
        for message in messages:
            text = "\n".join(block["text"] for block in message["content"] if "text" in block).strip()
            if text:
                lines.append(f"{message['role']}: {text}")
        return "\n".join(lines)
