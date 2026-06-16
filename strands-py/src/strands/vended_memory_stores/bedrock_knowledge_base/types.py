"""Configuration and result types for the Bedrock Knowledge Base memory store."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

from typing_extensions import Required, TypedDict

from ...memory.types import MemoryStoreConfig

if TYPE_CHECKING:
    from mypy_boto3_bedrock_agent import AgentsforBedrockClient
    from mypy_boto3_bedrock_agent_runtime import AgentsforBedrockRuntimeClient


class BedrockKnowledgeBaseS3Config(TypedDict, total=False):
    """S3 ingestion settings for :class:`BedrockKnowledgeBaseStore`, required when ``data_source_type`` is ``'S3'``.

    An S3 data source indexes objects from a bucket (there is no inline-text path), so ``add`` uploads
    to this bucket and then ingests directly (``IngestKnowledgeBaseDocuments``). Unlike a ``CUSTOM``
    document, an S3 document can't carry metadata inline, so a single ``add`` may write *two* objects:

    - the content, as a ``.txt`` object; and
    - when ``scope``/``metadata`` are present, a ``<object-key>.metadata.json`` *sidecar* beside it.
      This is Bedrock's out-of-band convention for attaching attributes to an S3 object: it is paired
      to the content by name and used for retrieval filtering. With no scope/metadata, no sidecar is
      written.

    Direct ingestion indexes whatever object you point it at, so ``add`` works with any
    ``bucket``/``prefix`` the credentials can write to, regardless of where the data source is
    configured to read. The ``bucket``/``prefix`` choice only governs durability across future
    data-source syncs (``StartIngestionJob``): a sync reconciles the index to match the data source's
    scanned location, so an object outside that location is treated as deleted and removed from the
    index. If syncs will run against this data source, upload to the bucket it reads from and a
    ``prefix`` within its inclusion prefixes so directly-ingested memories survive them; otherwise the
    location is free.

    Attributes:
        bucket: Bucket the content object and its optional ``.metadata.json`` sidecar are uploaded to
            before ingestion.
        prefix: Key prefix for uploaded objects (e.g. ``'memories/'``). A trailing slash is added when
            missing. Both the content object and its sidecar (when written) land under this prefix.
        client: Client used to upload objects. When omitted, a default S3 client is constructed using
            the default credential chain on first write.
    """

    bucket: Required[str]
    prefix: Required[str]
    client: Any


class BedrockKnowledgeBaseConfig(TypedDict, total=False):
    """Connection to a Bedrock Knowledge Base: the knowledge base, the data source, and its clients.

    This is the reusable half of a store's config: build one and pass it (as the ``config`` argument)
    to many :class:`BedrockKnowledgeBaseStore` instances that differ only by ``scope``.

    Attributes:
        knowledge_base_id: The Bedrock Knowledge Base identifier to query and ingest into.
        data_source_type: The type of data source backing this knowledge base, matching Bedrock's
            ``dataSourceType``. Only ``'CUSTOM'`` and ``'S3'`` data sources accept direct document
            ingestion, so only those can be written to. ``'CUSTOM'`` ingests its ``content`` as inline
            text, with scope/metadata attached as inline attributes; ``'S3'`` uploads to the configured
            ``s3`` bucket and ingests that object, writing scope/metadata as a separate
            ``.metadata.json`` sidecar (an S3 document cannot carry attributes inline); ``'OTHER'``
            backends (Confluence, SharePoint, Salesforce, Web, SQL/Redshift, etc.) sync from an
            external store or are query-only and so are read-only. Effective writability is
            ``writable and data_source_type in {'CUSTOM', 'S3'}``: a store is writable only when the
            caller opts in and the backend supports direct ingestion. When omitted, the store is
            read-only.
        data_source_id: Data source to ingest into when writing. Required for ``add`` to succeed.
        s3: S3 ingestion settings. Required when ``data_source_type`` is ``'S3'``; ignored otherwise.
        scope_metadata_key: Metadata attribute key used for scope-based filtering. Defaults to
            ``'namespace'``.
        runtime_client: Pre-constructed runtime client for ``Retrieve`` calls. When omitted, a default
            boto3 client is constructed from the default credential chain.
        agent_client: Pre-constructed agent client for ``IngestKnowledgeBaseDocuments`` calls. When
            omitted, a default boto3 client is constructed lazily on first write. To target a specific
            region/credentials/endpoint, build the client yourself and inject it here.
    """

    knowledge_base_id: Required[str]
    data_source_type: Literal["CUSTOM", "S3", "OTHER"]
    data_source_id: str
    s3: BedrockKnowledgeBaseS3Config
    scope_metadata_key: str
    runtime_client: AgentsforBedrockRuntimeClient
    agent_client: AgentsforBedrockClient


class BedrockKnowledgeBaseStoreConfig(MemoryStoreConfig, total=False):
    """Full configuration for a :class:`BedrockKnowledgeBaseStore`, passed as its constructor kwargs.

    Attributes:
        config: Connection to the knowledge base. Reuse one across stores that differ only by ``scope``.
        scope: Namespace isolating this store's documents; applied as a search filter and stamped on
            writes. Not a store-identity field, so it never affects ``MemoryManager`` routing.
        filter: Explicit search filter, overriding the scope-derived one. Affects ``search`` only;
            writes always scope by ``scope``.
    """

    config: Required[BedrockKnowledgeBaseConfig]
    scope: str
    filter: dict[str, Any]


@dataclass
class BedrockKnowledgeBaseAddResult:
    """Result returned by :meth:`BedrockKnowledgeBaseStore.add`.

    Attributes:
        document_id: ``CUSTOM``: the generated document id (UUID). ``S3``: the ``s3://`` URI of the
            uploaded content object.
    """

    document_id: str
