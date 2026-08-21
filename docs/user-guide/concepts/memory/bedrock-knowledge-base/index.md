`BedrockKnowledgeBaseStore` is a [`MemoryStore`](/docs/user-guide/concepts/memory/overview/index.md#stores) backed by [Amazon Bedrock Knowledge Bases](https://docs.aws.amazon.com/bedrock/latest/userguide/knowledge-base.html).

Connect the store to a knowledge base you have already set up. Configure your store with its knowledge base ID and data source (see [Data Source Types and Writability](#data-source-types-and-writability)). A store with only a knowledge base ID is read-only. The store reaches Bedrock with the standard AWS credential chain, the same as the [Amazon Bedrock model provider](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md).

Both managed and vector knowledge bases are supported. The store detects which type it is connected to and shapes its Retrieve request to match, so the same configuration works for either (see [Required IAM Permissions](#required-iam-permissions) for the permission this detection needs). Managed knowledge bases can have ACL-aware data sources; see [Access control lists](#access-control-lists) to configure per-document access.

(( tab "Python" ))
```python
from strands import Agent
from strands.memory import MemoryManager
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="docs",
    description="Company documentation and policies.",
    config={"knowledge_base_id": "KB123"},
)

agent = Agent(memory_manager=MemoryManager(stores=[store]))
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { Agent, BedrockModel } from '@strands-agents/sdk'
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'docs',
  description: 'Company documentation and policies.',
  config: { knowledgeBaseId: 'KB123' },
})

const agent = new Agent({
  model: new BedrockModel(),
  memoryManager: { stores: [store] },
})
```
(( /tab "TypeScript" ))

To enable writing memories through the [`add_memory` tool](/docs/user-guide/concepts/memory/overview/index.md#memory-tools) or [automatic extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction), mark the store writable and point it at a data source that accepts ingestion:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    description="User preferences and stable facts.",
    writable=True,
    config={
        "knowledge_base_id": "KB123",
        "data_source_type": "CUSTOM",
        "data_source_id": "DS456",
    },
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  description: 'User preferences and stable facts.',
  writable: true,
  config: {
    knowledgeBaseId: 'KB123',
    dataSourceType: 'CUSTOM',
    dataSourceId: 'DS456',
  },
})
```
(( /tab "TypeScript" ))

## Configuration

### Scope and namespaces

`scope` isolates documents within a single knowledge base. It is stamped on every write (under `scope_metadata_key``scopeMetadataKey`, default `'namespace'`) and applied as a metadata filter on search. Unlike `name`, scope is not a routing identity, so it never affects which store the `MemoryManager` targets. For per-tenant isolation, construct one store per scope; since they share a connection, this is cheap.

### Store config

The outer `BedrockKnowledgeBaseStoreConfig` carries the per-store identity and behavior, plus the [shared `MemoryStore` fields](/docs/user-guide/concepts/memory/overview/index.md#stores) (`name`, `description`, `max_search_results``maxSearchResults`, `writable`, `extraction`):

| Field | Purpose |
| --- | --- |
| `config` | The knowledge base connection (see below). Reuse one across stores that differ only by `scope`. |
| `scope` | Logical namespace isolating documents. Applied as a metadata filter on search and stamped on writes. |
| `filter` | Explicit retrieval filter; overrides the auto-generated scope filter on search. |
| `access_control_list``accessControlList` | Document-level ACL entries stamped on every write. Required when the data source has ACL awareness enabled. See [Access control lists](#access-control-lists). |
| `min_score``minScore` | Score floor a result must meet. Use for similarity-scored knowledge bases, where a higher score is a better match. See [Filtering by score](#filtering-by-score). |
| `max_score``maxScore` | Score ceiling a result must not exceed. Use for distance-scored knowledge bases, where a lower score is a better match. See [Filtering by score](#filtering-by-score). |

### Connection config

The inner `BedrockKnowledgeBaseConfig` is the reusable connection: which knowledge base, which data source, and the clients used to reach them:

| Field | Purpose |
| --- | --- |
| `knowledge_base_id``knowledgeBaseId` | The Bedrock Knowledge Base to query and ingest into. Required. |
| `knowledge_base_type``knowledgeBaseType` | The knowledge base type (`'MANAGED'`, `'VECTOR'`, `'KENDRA'`, or `'SQL'`). When provided, skips the `GetKnowledgeBase` call during initialization. When omitted, detected automatically. |
| `data_source_type``dataSourceType` | `'CUSTOM'`, `'S3'`, or `'OTHER'`. Governs whether and how the store can be written to. |
| `data_source_id``dataSourceId` | The data source to ingest into. Required for writes. |
| `s3` | S3 ingestion settings (`bucket`, `prefix`). Required when the data source type is `'S3'`. |
| `scope_metadata_key``scopeMetadataKey` | Metadata attribute key used for scope filtering. Defaults to `'namespace'`. |
| `runtime_client / agent_client``runtimeClient / agentClient` | Pre-constructed AWS clients. When omitted, default clients are constructed using the standard credential chain (the agent client lazily, on first write). |
| `region_name``regionName` | AWS region hint for the default clients the store constructs. When omitted, boto3 resolves the region itself. Ignored for any client you inject explicitly (`runtime_client` / `agent_client` / `s3.client`). |

Because the connection is a separate object, you build it once and vary only `name` and `scope` per store. This is the cheap way to give each tenant an isolated namespace over a single knowledge base:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

# Build the connection once, vary only name and scope per store.
connection = {
    "knowledge_base_id": "KB123",
    "data_source_type": "CUSTOM",
    "data_source_id": "DS456",
}

alice = BedrockKnowledgeBaseStore(name="alice", writable=True, scope="user-alice", config=connection)
bob = BedrockKnowledgeBaseStore(name="bob", writable=True, scope="user-bob", config=connection)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import {
  BedrockKnowledgeBaseStore,
  type BedrockKnowledgeBaseConfig,
} from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

// Build the connection once, vary only name and scope per store.
const connection: BedrockKnowledgeBaseConfig = {
  knowledgeBaseId: 'KB123',
  dataSourceType: 'CUSTOM',
  dataSourceId: 'DS456',
}

const alice = new BedrockKnowledgeBaseStore({
  name: 'alice',
  writable: true,
  scope: 'user-alice',
  config: connection,
})

const bob = new BedrockKnowledgeBaseStore({
  name: 'bob',
  writable: true,
  scope: 'user-bob',
  config: connection,
})
```
(( /tab "TypeScript" ))

## Data Source Types and Writability

Only data sources that accept direct ingestion can be written to. A store is writable only when you opt in *and* the backend supports ingestion: that is, the store is marked writable and the data source type is `'CUSTOM'` or `'S3'`.

| Data source type | Writable | How writes work |
| --- | --- | --- |
| `CUSTOM` | Yes | Ingests the content as inline text, with scope and metadata attached as inline attributes. |
| `S3` | Yes | Uploads the content to the configured `s3` bucket and ingests that object. Requires an `s3` config. |
| `OTHER` | No | External backends such as Confluence, SharePoint, Salesforce, Web, or SQL/Redshift that sync from their own source or are query-only. Read-only. |
| omitted | No | Read-only. |

A writable `CUSTOM` store needs a data source ID. A writable `S3` store needs a data source ID and an `s3` config:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    config={
        "knowledge_base_id": "KB123",
        "data_source_type": "S3",
        "data_source_id": "DS789",
        "s3": {"bucket": "my-agent-memories", "prefix": "memories/"},
    },
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  config: {
    knowledgeBaseId: 'KB123',
    dataSourceType: 'S3',
    dataSourceId: 'DS789',
    s3: { bucket: 'my-agent-memories', prefix: 'memories/' },
  },
})
```
(( /tab "TypeScript" ))

### How S3 writes work

An S3 document can’t carry metadata inline, so when a scope or metadata are present a single write produces two objects: the content as a `.txt` object, and a `<object-key>.metadata.json` sidecar beside it (Bedrock’s convention for attaching attributes to an S3 object). The uploads are not transactional, so if a later step fails, any already-uploaded objects remain in the bucket un-ingested, where a future sync may pick them up or you can clean them up out of band.

A write ingests its object directly, regardless of where the data source is configured to scan. A later data-source sync reconciles the index to the scanned location, so a memory written outside that location is treated as deleted and removed. If you run periodic syncs against this data source, upload to the bucket and prefix it scans (the `s3` config) so directly-ingested memories survive them.

## Search and Ingestion

`search` runs the Bedrock Retrieve API and returns entries ordered by relevance, while `add` ingests new content and returns its document id:

(( tab "Python" ))
```python
from strands.memory.types import SearchOptions
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    config={
        "knowledge_base_id": "KB123",
        "data_source_type": "CUSTOM",
        "data_source_id": "DS456",
    },
)

results = await store.search("what are my preferences?", SearchOptions(max_search_results=5))
for entry in results:
    print(entry.content, entry.metadata.get("_relevance_score"))

# add returns the new document's id (a UUID for CUSTOM, an s3:// URI for S3)
result = await store.add("User prefers aisle seats", {"category": "travel"})
print(result.document_id)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})

const results = await store.search('what are my preferences?', { maxSearchResults: 5 })
for (const entry of results) {
  console.log(entry.content, entry.metadata?._relevanceScore)
}

// add returns the new document's id (a UUID for CUSTOM, an s3:// URI for S3)
const { documentId } = await store.add('User prefers aisle seats', {
  category: 'travel',
})
```
(( /tab "TypeScript" ))

The search result cap defaults to `10` when neither the call nor the store sets one. This `10` default applies only to direct calls on the store. Reached through a `MemoryManager` tool or its search method, the manager supplies its own per-call cap instead; set `max_search_results``maxSearchResults` on the store to pin a value across both paths.

Each entry’s `metadata` carries the document’s own attributes plus two reserved synthetic keys: `_relevance_score``_relevanceScore` (the Bedrock relevance score) and `_source_location``_sourceLocation` (the retrieval location). When `scope` is set, search filters to that namespace automatically; an explicit `filter` overrides the scope-derived one. The asymmetry is deliberate: an explicit `filter` affects search only, while writes always scope by `scope`.

The document id from `add` is the generated UUID for a `CUSTOM` document, or the `s3://` URI of the uploaded object for `S3`. Writes require a data source ID (and an `s3` config for S3 data sources); a write against a missing or read-only configuration raises.

Ingestion is eventually consistent: a successful write does not mean the content is immediately searchable.

### Filtering by score

Set `min_score``minScore` or `max_score``maxScore` on the store to enforce score limits on returned results. For similarity-scored knowledge bases use `min_score``minScore` to keep results at or above the floor. For distance-scored knowledge bases use `max_score``maxScore` to keep results at or below the ceiling.

The score metric is a property of the vector store behind the knowledge base. Filtering is applied after retrieval, so a search may therefore return fewer entries than `max_search_results``maxSearchResults`, or none at all when nothing clears the bound. A result without a score is retained.

## Extraction

Enable [automatic extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction) on a writable store to capture memories from the conversation. By default it runs every 5 turns, distilling facts client-side with a `ModelExtractor` that uses the agent’s own model:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="preferences",
    writable=True,
    extraction=True,
    config={
        "knowledge_base_id": "KB123",
        "data_source_type": "CUSTOM",
        "data_source_id": "DS456",
    },
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'preferences',
  writable: true,
  extraction: true,
  config: { knowledgeBaseId: 'KB123', dataSourceType: 'CUSTOM', dataSourceId: 'DS456' },
})
```
(( /tab "TypeScript" ))

To change the cadence, swap the extractor, or extract server-side, see [Automatic Extraction](/docs/user-guide/concepts/memory/overview/index.md#automatic-extraction) on the Memory page.

## Required IAM Permissions

The credentials the store uses must allow the Bedrock operations it calls, plus S3 writes when using an S3 data source:

-   `bedrock:Retrieve` - for search.
-   `bedrock:GetKnowledgeBase` - called once at initialization to detect the knowledge base type, then memoized. Not required if you provide `knowledge_base_type``knowledgeBaseType` in the config.
-   `bedrock:IngestKnowledgeBaseDocuments` - for writes (`CUSTOM` and `S3`).
-   `s3:PutObject` - for writes to an `S3` data source, on the configured bucket and prefix.

Here is a sample IAM policy for a writable store backed by a `CUSTOM` data source:

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Effect": "Allow",
            "Action": [
                "bedrock:Retrieve",
                "bedrock:GetKnowledgeBase",
                "bedrock:IngestKnowledgeBaseDocuments"
            ],
            "Resource": "arn:aws:bedrock:*:*:knowledge-base/KB123"
        }
    ]
}
```

For an `S3` data source, add `s3:PutObject` on the bucket and prefix the store uploads to:

```json
{
    "Effect": "Allow",
    "Action": "s3:PutObject",
    "Resource": "arn:aws:s3:::my-agent-memories/memories/*"
}
```

## Access Control Lists

Managed knowledge base data sources can have [ACL awareness](https://docs.aws.amazon.com/bedrock/latest/userguide/kb-managed-acl.html) enabled, which requires every ingested document to carry an access control list specifying who can retrieve it. Set `access_control_list``accessControlList` on the store to supply entries for every write:

(( tab "Python" ))
```python
from strands.vended_memory_stores import BedrockKnowledgeBaseStore

store = BedrockKnowledgeBaseStore(
    name="policies",
    writable=True,
    scope="hr",
    access_control_list=[
        {"access": "ALLOW", "name": "alice@example.com", "type": "USER"},
    ],
    config={
        "knowledge_base_id": "KB123",
        "data_source_type": "CUSTOM",
        "data_source_id": "DS456",
    },
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
```typescript
import { BedrockKnowledgeBaseStore } from '@strands-agents/sdk/vended-memory-stores/bedrock-knowledge-base'

const store = new BedrockKnowledgeBaseStore({
  name: 'policies',
  writable: true,
  scope: 'hr',
  accessControlList: [{ access: 'ALLOW', name: 'alice@example.com', type: 'USER' }],
  config: {
    knowledgeBaseId: 'KB123',
    dataSourceType: 'CUSTOM',
    dataSourceId: 'DS456',
  },
})
```
(( /tab "TypeScript" ))

Each entry specifies `access` (`'ALLOW'` or `'DENY'`), `name` (the user’s email), and `type` (`'USER'`). Deny overrides allow. The entries apply to both `CUSTOM` and `S3` writes. For the `CUSTOM` path, an inline ACL requires at least one metadata attribute (a `scope` satisfies this); if none is present the store raises with guidance.

If the data source has ACL awareness enabled and no `access_control_list``accessControlList` is set, the store surfaces a clear error pointing at the field.

## Related

-   [Memory](/docs/user-guide/concepts/memory/overview/index.md) - the `MemoryManager` concept this store plugs into, including the tools, extraction, and injection it enables.
-   [Amazon Bedrock](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md) - credential and region setup shared with the Bedrock model provider.

## Related pages

- [Guardrails](/docs/user-guide/safety-security/guardrails/index.md) (2 shared tags)
- [Amazon Nova](/docs/user-guide/concepts/model-providers/amazon-nova/index.md) (2 shared tags)
- [Nova Sonic](/docs/user-guide/concepts/bidirectional-streaming/models/nova_sonic/index.md) (2 shared tags)
- [Deploying Strands Agents to Amazon Bedrock AgentCore Runtime](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/index.md) (2 shared tags)
- [Python Deployment to Amazon Bedrock AgentCore Runtime](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/index.md) (2 shared tags)
- [TypeScript Deployment to Amazon Bedrock AgentCore Runtime](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/typescript/index.md) (2 shared tags)
- [AgentCore Evaluation Dashboard Configuration](/docs/user-guide/evals-sdk/how-to/agentcore_evaluation_dashboard/index.md) (2 shared tags)
- [Memory](/docs/user-guide/concepts/memory/overview/index.md) (1 shared tag)
- [Test Memory Store](/docs/user-guide/concepts/memory/test-memory-store/index.md) (1 shared tag)
- [Amazon Bedrock](/docs/user-guide/concepts/model-providers/amazon-bedrock/index.md) (2 shared tags)


## Implementation

### Python

- [harness-sdk/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_memory_stores/bedrock_knowledge_base/store.py)

### TypeScript

- [harness-sdk/strands-ts/src/vended-memory-stores/bedrock-knowledge-base/store.ts](https://github.com/strands-agents/harness-sdk/blob/main/strands-ts/src/vended-memory-stores/bedrock-knowledge-base/store.ts)
