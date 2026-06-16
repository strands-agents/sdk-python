"""A :class:`~strands.memory.types.MemoryStore` backed by Amazon Bedrock Knowledge Bases.

Supports semantic search via ``Retrieve`` and document ingestion via
``IngestKnowledgeBaseDocuments`` for ``CUSTOM`` and ``S3`` data sources.
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import TYPE_CHECKING, Any, cast

import boto3
from typing_extensions import Unpack

from ...memory.types import MemoryEntry, MemoryStore, Metadata, SearchOptions
from .types import (
    BedrockKnowledgeBaseAddResult,
    BedrockKnowledgeBaseS3Config,
    BedrockKnowledgeBaseStoreConfig,
)

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent import AgentsforBedrockClient
    from mypy_boto3_bedrock_agent.type_defs import KnowledgeBaseDocumentTypeDef
    from mypy_boto3_bedrock_agent_runtime import AgentsforBedrockRuntimeClient
    from mypy_boto3_bedrock_agent_runtime.type_defs import KnowledgeBaseVectorSearchConfigurationTypeDef

logger = logging.getLogger(__name__)

DEFAULT_MAX_SEARCH_RESULTS = 10

# A Bedrock attribute value: ``{"type": ..., "stringValue"/"numberValue"/...}``. There is no Python
# SDK type for this (boto3 passes plain dicts), so it is modeled as ``dict[str, Any]``.
_AttributeValue = dict[str, Any]


def _new_id() -> str:
    """Return a fresh document/object identifier.

    The id is used solely as a unique handle; the store does not rely on any ordering of ids. This
    returns a UUID v4 (Python's stdlib offers v1/v3/v4/v5 but no time-ordered v7). Defined at module
    level so tests can monkeypatch it to a fixed value.
    """
    return str(uuid.uuid4())


def _to_attribute_value(value: Any) -> _AttributeValue | None:
    """Convert a caller metadata value into a Bedrock attribute value, or ``None`` if unsupported."""
    # ``bool`` must be checked before ``int``: in Python ``bool`` is a subclass of ``int``, so a bare
    # ``isinstance(value, int)`` would map ``True``/``False`` to NUMBER.
    if isinstance(value, bool):
        return {"type": "BOOLEAN", "booleanValue": value}
    if isinstance(value, str):
        return {"type": "STRING", "stringValue": value}
    if isinstance(value, (int, float)):
        return {"type": "NUMBER", "numberValue": value}
    if isinstance(value, list) and len(value) > 0 and all(isinstance(item, str) for item in value):
        return {"type": "STRING_LIST", "stringListValue": value}
    return None


class BedrockKnowledgeBaseStore(MemoryStore):
    """A :class:`~strands.memory.types.MemoryStore` backed by Amazon Bedrock Knowledge Bases.

    Supports semantic search via ``Retrieve`` and document ingestion via
    ``IngestKnowledgeBaseDocuments`` for ``CUSTOM`` and ``S3`` data sources.

    Example:
        ```python
        from strands.vended_memory_stores.bedrock_knowledge_base import BedrockKnowledgeBaseStore

        store = BedrockKnowledgeBaseStore(
            config={
                "knowledge_base_id": "KB123",
                "data_source_type": "CUSTOM",
                "data_source_id": "DS456",
            },
            name="preferences",
            scope="user-abc",
            writable=True,
        )

        results = await store.search("what are my preferences?")
        result = await store.add("User prefers dark mode")
        ```
    """

    def __init__(self, **store_config: Unpack[BedrockKnowledgeBaseStoreConfig]) -> None:
        """Initialize the store.

        Args:
            **store_config: See :class:`BedrockKnowledgeBaseStoreConfig`.

        Raises:
            ValueError: If ``max_search_results`` is less than 1, or (when ``writable``) if the write
                configuration is invalid.
        """
        kb_config = store_config["config"]
        self.name = store_config["name"]
        self.description = store_config.get("description")
        max_search_results = store_config.get("max_search_results")
        if max_search_results is not None and max_search_results < 1:
            raise ValueError("BedrockKnowledgeBaseStore: max_search_results must be at least 1.")
        self.max_search_results = max_search_results
        self.writable = store_config.get("writable", False)
        self.extraction = store_config.get("extraction")

        self._config = kb_config
        self._agent_client: AgentsforBedrockClient | None = kb_config.get("agent_client")
        s3_config = kb_config.get("s3")
        self._s3_client: Any | None = s3_config.get("client") if s3_config else None
        self._s3_config = s3_config
        self._knowledge_base_id = kb_config["knowledge_base_id"]
        self._data_source_type = kb_config.get("data_source_type")
        self._data_source_id = kb_config.get("data_source_id")

        # The runtime client is built eagerly: search is the read path every store exercises. A
        # default client is only constructed when none was injected.
        self._runtime_client: AgentsforBedrockRuntimeClient = kb_config.get("runtime_client") or boto3.client(
            "bedrock-agent-runtime"
        )

        if self.writable:
            self._validate_write_config()

        self.scope = store_config.get("scope")
        self.scope_metadata_key = kb_config.get("scope_metadata_key") or "namespace"
        self.filter = store_config.get("filter")

    def _resolve_filter(self) -> dict[str, Any] | None:
        """Resolve the effective retrieval filter for a search.

        An explicit :attr:`filter` wins; otherwise one is derived from :attr:`scope` /
        :attr:`scope_metadata_key`.
        """
        if self.filter:
            return self.filter
        if self.scope:
            return {"equals": {"key": self.scope_metadata_key, "value": self.scope}}
        return None

    async def search(self, query: str, options: SearchOptions | None = None) -> list[MemoryEntry]:
        """Search the knowledge base for entries matching the query.

        Args:
            query: The search query text.
            options: Optional search configuration.

        Returns:
            Matching memory entries ordered by relevance. Each entry's ``metadata`` includes
            user-provided attributes plus two reserved synthetic keys: ``_relevance_score`` (number)
            and ``_source_location`` (Bedrock retrieval location object).

        Raises:
            ValueError: If ``options.max_search_results`` is less than 1.
        """
        caller_max = options.get("max_search_results") if options is not None else None
        if caller_max is not None and caller_max < 1:
            raise ValueError("BedrockKnowledgeBaseStore: max_search_results must be at least 1.")
        limit = caller_max or self.max_search_results or DEFAULT_MAX_SEARCH_RESULTS
        filter_ = self._resolve_filter()

        vector_search_configuration: dict[str, Any] = {"numberOfResults": limit}
        if filter_:
            vector_search_configuration["filter"] = filter_

        try:
            response = self._runtime_client.retrieve(
                knowledgeBaseId=self._knowledge_base_id,
                retrievalQuery={"text": query},
                retrievalConfiguration={
                    "vectorSearchConfiguration": cast(
                        "KnowledgeBaseVectorSearchConfigurationTypeDef", vector_search_configuration
                    )
                },
            )
        except Exception as error:
            logger.error(
                "store=<%s>, knowledge_base_id=<%s>, error=<%s> | knowledge base retrieve failed",
                self.name,
                self._knowledge_base_id,
                error,
            )
            raise

        entries: list[MemoryEntry] = []
        for result in response.get("retrievalResults") or []:
            metadata: Metadata = {}
            if result.get("metadata"):
                for key, value in result["metadata"].items():
                    metadata[key] = value
            if result.get("location"):
                metadata["_source_location"] = result["location"]
            if result.get("score") is not None:
                metadata["_relevance_score"] = result["score"]

            content = (result.get("content") or {}).get("text") or ""
            entries.append(MemoryEntry(content=content, metadata=metadata))

        return entries

    async def add(self, content: str, metadata: Metadata | None = None) -> BedrockKnowledgeBaseAddResult:
        """Ingest ``content`` (with optional ``metadata``) into the knowledge base.

        Only ``CUSTOM`` and ``S3`` data sources support this. Requires ``data_source_id`` (and, for
        ``S3``, an ``s3`` config).

        Args:
            content: The text content to ingest.
            metadata: Optional metadata attributes to attach to the document.

        Returns:
            The document identifier (UUID for ``CUSTOM``, ``s3://`` URI for ``S3``).

        Raises:
            ValueError: If the store is not writable or ``content`` is empty/whitespace.
        """
        if not self.writable:
            raise ValueError(
                "BedrockKnowledgeBaseStore: store is not writable. Set writable=True in config to enable add()."
            )
        if not content.strip():
            raise ValueError("BedrockKnowledgeBaseStore: content must not be empty.")
        data_source_id, data_source_type = self._validate_write_config()

        # S3 and CUSTOM data sources accept fundamentally different documents. S3 ingests objects, so
        # its document references objects uploaded to S3 first; CUSTOM ingests the text inline. Either
        # way the store mints the document's identifier (CUSTOM: a UUID; S3: the content object's URI)
        # and returns it, so the caller has a stable handle to the document Bedrock now tracks.
        if data_source_type == "S3":
            content_uri, sidecar_uri = self._upload_s3_objects(content, metadata)
            document = self._build_s3_document(content_uri, sidecar_uri)
            document_id = content_uri
        else:
            document_id = _new_id()
            document = self._build_custom_document(document_id, content, metadata)

        try:
            self._get_agent_client().ingest_knowledge_base_documents(
                knowledgeBaseId=self._knowledge_base_id,
                dataSourceId=data_source_id,
                documents=[cast("KnowledgeBaseDocumentTypeDef", document)],
            )
        except Exception as error:
            logger.error(
                "store=<%s>, knowledge_base_id=<%s>, data_source_id=<%s>, data_source_type=<%s>, error=<%s> "
                "| knowledge base document ingestion failed",
                self.name,
                self._knowledge_base_id,
                data_source_id,
                data_source_type,
                error,
            )
            raise

        return BedrockKnowledgeBaseAddResult(document_id=document_id)

    def _upload_s3_objects(self, content: str, metadata: Metadata | None) -> tuple[str, str | None]:
        """Upload the objects that back one S3 ingestion and return their ``s3://`` URIs.

        A single ``add`` produces up to two objects: the content (always written, as a ``.txt``
        object) and a ``<object-key>.metadata.json`` sidecar carrying scope/metadata, written only
        when there is any to attach. Returns ``(content_uri, sidecar_uri_or_None)``.

        Uploads are not transactional: if the sidecar upload (or subsequent ingestion) fails after the
        content object lands, the uploaded object(s) remain in the bucket un-ingested. They are inert;
        a later data-source sync may pick them up, or they can be cleaned up out of band.
        """
        s3 = self._s3_config
        assert s3 is not None  # guaranteed by _validate_write_config
        prefix = s3["prefix"] if s3["prefix"].endswith("/") else f"{s3['prefix']}/"
        key = f"{prefix}{_new_id()}.txt"

        content_uri = self._put_object(s3, key, content, "text/plain; charset=utf-8")

        attributes = self._build_s3_sidecar_attributes(metadata)
        if not attributes:
            return content_uri, None

        # The sidecar must sit beside the source object and be named ``<object-key>.metadata.json``.
        # ``separators=(",", ":")`` emits compact, no-whitespace JSON for the sidecar body.
        sidecar = json.dumps({"metadataAttributes": attributes}, separators=(",", ":"))
        sidecar_uri = self._put_object(s3, f"{key}.metadata.json", sidecar, "application/json")
        return content_uri, sidecar_uri

    def _put_object(self, s3: BedrockKnowledgeBaseS3Config, key: str, body: str, content_type: str) -> str:
        """Upload a single object to the configured bucket and return its ``s3://`` URI."""
        bucket = s3["bucket"]
        try:
            self._get_s3_client().put_object(Bucket=bucket, Key=key, Body=body, ContentType=content_type)
        except Exception as error:
            logger.error(
                "store=<%s>, uri=<s3://%s/%s>, error=<%s> | S3 upload failed before ingestion",
                self.name,
                bucket,
                key,
                error,
            )
            raise
        return f"s3://{bucket}/{key}"

    def _build_s3_document(self, content_uri: str, sidecar_uri: str | None) -> dict[str, Any]:
        """Build a document for an ``S3`` data source from the URIs produced by :meth:`_upload_s3_objects`.

        ``content_uri`` becomes the document's content (the object to index), and ``sidecar_uri``, when
        present, becomes its metadata (an ``S3_LOCATION`` pointing at the sidecar). With no
        scope/metadata there's no sidecar, so the document carries no metadata.
        """
        document: dict[str, Any] = {
            "content": {
                "dataSourceType": "S3",
                "s3": {"s3Location": {"uri": content_uri}},
            },
        }

        if sidecar_uri:
            document["metadata"] = {
                "type": "S3_LOCATION",
                "s3Location": {"uri": sidecar_uri},
            }

        return document

    def _resolve_attributes(self, metadata: Metadata | None) -> list[tuple[str, _AttributeValue]]:
        """Resolve scope and caller metadata into a flat list of ``(key, value)`` pairs.

        Handles collision detection and unsupported-type filtering. Shared by both CUSTOM (inline
        attributes) and S3 (sidecar) document builders.
        """
        attrs: list[tuple[str, _AttributeValue]] = []

        if self.scope:
            attrs.append((self.scope_metadata_key, {"type": "STRING", "stringValue": self.scope}))

        if metadata:
            for key, value in metadata.items():
                if self.scope and key == self.scope_metadata_key:
                    logger.warning(
                        "store=<%s>, key=<%s> | dropping metadata key that collides with scope_metadata_key",
                        self.name,
                        key,
                    )
                    continue
                attribute_value = _to_attribute_value(value)
                if attribute_value is not None:
                    attrs.append((key, attribute_value))
                else:
                    logger.debug(
                        "store=<%s>, key=<%s> | dropping metadata value of unsupported type",
                        self.name,
                        key,
                    )

        return attrs

    def _build_custom_document(self, document_id: str, content: str, metadata: Metadata | None) -> dict[str, Any]:
        """Build a document for a ``CUSTOM`` data source.

        The text is ingested inline, with the scope and any caller metadata attached as inline
        attributes for retrieval filtering. ``document_id`` becomes the document's
        ``customDocumentIdentifier``.
        """
        attrs = self._resolve_attributes(metadata)

        document: dict[str, Any] = {
            "content": {
                "dataSourceType": "CUSTOM",
                "custom": {
                    "customDocumentIdentifier": {"id": document_id},
                    "sourceType": "IN_LINE",
                    "inlineContent": {
                        "type": "TEXT",
                        "textContent": {"data": content},
                    },
                },
            },
        }

        if attrs:
            document["metadata"] = {
                "type": "IN_LINE_ATTRIBUTE",
                "inlineAttributes": [{"key": key, "value": value} for key, value in attrs],
            }

        return document

    def _build_s3_sidecar_attributes(self, metadata: Metadata | None) -> dict[str, dict[str, Any]]:
        """Build the ``metadataAttributes`` map for an S3 ``.metadata.json`` sidecar.

        ``includeForEmbedding`` is ``False`` so the attribute is stored for filtering only and does
        not influence the embedding (matching how inline attributes behave for ``CUSTOM`` documents).
        Returns an empty map when there's nothing to attach; :meth:`_upload_s3_objects` treats that as
        "no sidecar" and skips writing the second object.
        """
        attrs = self._resolve_attributes(metadata)
        attributes: dict[str, dict[str, Any]] = {}
        for key, value in attrs:
            attributes[key] = {"value": value, "includeForEmbedding": False}
        return attributes

    def _validate_write_config(self) -> tuple[str, str]:
        """Validate the write configuration and return ``(data_source_id, data_source_type)``.

        Raises:
            ValueError: If the data source type is not writable, ``data_source_id`` is missing, or an
                ``S3`` data source has no ``s3`` config.
        """
        if self._data_source_type not in ("CUSTOM", "S3"):
            raise ValueError(
                f"BedrockKnowledgeBaseStore: add requires data_source_type 'CUSTOM' or 'S3', but it is "
                f"'{self._data_source_type if self._data_source_type is not None else 'None'}'. "
                "'OTHER' backends are read-only."
            )
        if not self._data_source_id:
            raise ValueError(
                "BedrockKnowledgeBaseStore: data_source_id is required for write operations. "
                "Provide it in the config to enable add()."
            )
        if self._data_source_type == "S3" and not self._s3_config:
            raise ValueError(
                "BedrockKnowledgeBaseStore: s3 config is required when data_source_type is 'S3'. "
                "Provide bucket and prefix to enable add()."
            )
        return self._data_source_id, self._data_source_type

    def _get_s3_client(self) -> Any:
        """Return the S3 client, constructing a default one lazily on first use.

        A default client is built with no extra configuration. Callers needing a specific region,
        credentials, or endpoint build the client themselves and inject it via the ``s3`` config.
        """
        if self._s3_client is None:
            self._s3_client = boto3.client("s3")
        return self._s3_client

    def _get_agent_client(self) -> AgentsforBedrockClient:
        """Return the agent client, constructing a default one lazily on first use.

        A default client is built with no extra configuration. Callers needing a specific region,
        credentials, or endpoint build the client themselves and inject it via ``agent_client``.
        """
        if self._agent_client is None:
            self._agent_client = boto3.client("bedrock-agent")
        return self._agent_client
