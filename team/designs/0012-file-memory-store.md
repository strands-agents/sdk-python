# Design Doc: File-Based Agent Memory

| Field  | Value          |
|--------|----------------|
| Status | Proposed       |
| Date   | June 12, 2026  |
| Issue  | TBD            |
| Scope  | TypeScript SDK |

---

## Context

The Strands SDK has no local memory store. The existing store implementation, `BedrockKnowledgeBaseStore`, requires provisioned AWS infrastructure (Bedrock Knowledge Base, credentials, optional S3). This is well-suited for production and enterprise deployments where teams already have AWS infrastructure, but developers who want to prototype, run integration tests, or experiment with agent memory in non-production environments have no option that works without standing up a managed service. `FileMemoryStore` fills this gap: a local store that requires zero external infrastructure, just a filesystem.

Separately, any long-lived memory system needs a maintenance mechanism. As memory accumulates over extended interactions, quality degrades — redundancy grows, contradictions go unresolved, and retrieval becomes less reliable. Managed stores like `BedrockKnowledgeBaseStore` handle this server-side — deduplication, indexing, and retrieval quality are responsibilities of the backend service. This works because the infrastructure runs outside the agent loop: it can process knowledge asynchronously, build embeddings, and serve semantic search without adding latency to agent sessions. A local store needs an equivalent offline step to prevent quality from degrading over time.

`FileMemoryStore` addresses both needs. It organizes knowledge as a structured file hierarchy that the agent can navigate directly, and exposes consolidation as an offline maintenance step — analogous to how managed backends process knowledge asynchronously. By running offline, this step can also build local indexes, enabling semantic search without a managed vector service. Because it operates through the unified `Storage` interface, the backend can be extended to git-based storage, S3, or any other persistence layer without changing the core memory model.

---

## Decision

This proposal introduces **`FileMemoryStore`**, which implements the `MemoryStore` interface (for `MemoryManager` L2, long-term memory). It handles knowledge: extracted facts, progressive disclosure, search, and consolidation.

For L1 (session persistence), the existing `ContextManager` already accepts a `Storage` instance and provides file operations (`put`, `get`, `list`, `delete`). No additional wrapper class is needed.

`FileMemoryStore` uses a `Storage` instance for its file operations. When none is provided, it defaults to `LocalFileStorage` at `~/.strands/`.

Both L1 and L2 can share the same `Storage` instance pointed at the same root directory, giving a unified, inspectable filesystem containing everything an agent has learned and experienced — without conflating L1 and L2 into a single construct.

The existing Strands API remains unchanged. `MemoryManager` still owns L1 → L2 extraction. What changes is the physical storage: instead of separate, disconnected backends for each layer, both write to the same file hierarchy — `ContextManager` writes to `context/` for L1 and `FileMemoryStore` writes to `knowledge/` for L2.

### File Hierarchy

`ContextManager` and `FileMemoryStore` can share the same `Storage` instance and root directory. They are isolated by path: `ContextManager` writes to `context/`, while `FileMemoryStore` writes to `knowledge/`. Consolidation metadata lives in `consolidation/`.

```
~/.strands/
├── context/                         # L1 - ContextManager writes here
│   ├── current.md
│   └── history/
│       ├── 2026-06-10-session-a.md
│       └── 2026-06-11-session-b.md
├── knowledge/                       # L2 - MemoryStore writes here (called by MemoryManager)
│   ├── system/                      # always loaded in full every turn
│   │   └── user-preferences.md
│   └── facts/                       # visible by name + description; loaded on demand
│       ├── testing-philosophy.md
│       └── project-context.md
└── consolidation/
    └── changelog.md                 # human-readable log of consolidation
```

---

## Progressive Disclosure

Not everything loads into context every turn. The agent retrieves relevant knowledge on demand by navigating the file hierarchy directly. LLMs are precise and accurate at scoped filesystem calls (listing directories, grepping for keywords, reading specific files), and progressive disclosure leverages this skill as the primary retrieval mechanism.

### Relationship to MemoryManager Retrieval

`MemoryManager` provides two retrieval mechanisms: automatic injection (searches stores every turn, injects results into model input) and the `search_memory` tool (agent-initiated). Both call `store.search()` on the store. Progressive disclosure is a *third*, independent retrieval path. The agent navigates the file hierarchy using tools (`read_memory_file`, `grep_memory`) registered by `FileMemoryStore` via `getTools()`.

These are not mutually exclusive, and the user controls which are active:

| Mechanism | Controlled by | How it works with `FileMemoryStore` |
|-----------|---------------|-------------------------------------|
| Injection | `MemoryManager` config (`injection: true`) | Calls `FileMemoryStore.search()` (keyword matching) → injects results as `<memory>` XML |
| `search_memory` tool | Agent-initiated | Same — calls `FileMemoryStore.search()` |
| Progressive disclosure | Agent-initiated | Agent sees file tree in system prompt, navigates with `read_memory_file`/`grep_memory` tools |

When `FileMemoryStore` is the only store, **progressive disclosure is the recommended primary retrieval path** — the agent's judgment over filenames and descriptions is a better retrieval engine than keyword matching for a filesystem store. `MemoryManager` injection is redundant in this case and can be disabled (`injection: false`). The `search_memory` tool remains available as a fallback — it searches inside file content, so it can surface relevant knowledge when filenames and descriptions alone aren't enough to identify the right file.

### How It Works

`FileMemoryStore` registers two things on the agent at initialization (via `getTools()` for the navigation tools, and via `renderContent` for context injection — both auto-wired by `MemoryManager`):

**1. The file tree (always in the system prompt)**

The full directory listing of `knowledge/` with each file's `description` frontmatter is injected into the model context every turn via `FileMemoryStore.renderContent()`, which `MemoryManager` wires as injection middleware during `initAgent`. This is independent of `MemoryManager`'s search-based injection. The agent always knows what knowledge exists without loading the content:

```
knowledge/
├── system/                          [loaded in full]
│   └── user-preferences.md         — "Core preferences: editor, language, testing style"
└── facts/
    ├── testing-philosophy.md       — "Integration-first, mock at boundaries"
    ├── deploy-process.md           — "Team's deployment pipeline and rollback procedures"
    └── project-architecture.md     — "Service boundaries and data flow"
```


### Context Loading

Files in `knowledge/system/` are always loaded in full into the system prompt. This is where core context lives (persona, key preferences, critical project facts). Everything outside `system/` is visible by filename + description only, loaded when the agent reads it.

**Who manages `system/`:** Developers seed `system/` at repo creation with anything the agent always needs (persona, core preferences). The consolidation agent promotes and demotes files during offline maintenance, analyzing cross-session patterns to move broadly relevant files into `system/` and overly specific ones out. The main agent never writes to `system/` during a session.

### Retrieval in Practice

The agent uses the file tree to judge relevance by filename and description, then loads specific files with `read_memory_file` or searches across files with `grep_memory` — both registered by `FileMemoryStore` via `getTools()`. For a targeted question, it reads a single matching file. For a broader query, it greps across `knowledge/`, then reads the best matches. See [Appendix A](#appendix-a-retrieval-worked-examples) for worked examples.

### File Format

Knowledge files are markdown with YAML frontmatter containing one field — `description`. The description is always visible in the file tree, letting the agent judge relevance without reading every file:

```markdown
---
description: "How the user approaches testing: integration-first, mock at boundaries"
---

- Prefers integration tests over unit tests for API layers
- Uses VS Code with vim keybindings
- Mocks external services at the HTTP boundary, not at the module level
```

---

### Architecture

#### FileMemoryStore

`FileMemoryStore` implements the `MemoryStore` interface (called by `MemoryManager`). It handles L2 — knowledge storage, progressive disclosure, search, and consolidation. It operates on `knowledge/` through a `Storage` instance for file operations.

```typescript
interface FileMemoryStoreConfig {
  // Required
  name: string;

  // Optional
  storage?: Storage;  // default: LocalFileStorage at ~/.strands/
  description?: string;
  maxSearchResults?: number;
  extraction?: ExtractionConfig;
  retrieval?: { maxTokens?: number }; // default: 2000
}

interface ConsolidateConfig {
  model: Model;
  operations?: ("deduplicate" | "resolveContradictions" | "deriveInsights" | "prune" | "reorganize")[];
  maxDirectories?: number;  // default: 8
}

class FileMemoryStore implements MemoryStore {
  constructor(config: FileMemoryStoreConfig)

  // --- MemoryStore (MemoryManager L2) ---
  async search(query: string, options?: SearchOptions): Promise<MemoryEntry[]>
  async add(content: string, metadata?: Record<string, JSONValue>): Promise<void>

  // --- Consolidation ---
  async consolidate(config: ConsolidateConfig): Promise<void>
}
```

### Method Behavior

#### FileMemoryStore

**`add(content, metadata?)`**

Writes a new markdown file to `knowledge/facts/` by default. No model call. Pass `metadata.path` to write to a custom location under `knowledge/` (e.g., `operations/debugging` → `knowledge/operations/debugging.md`).

- **Filename:** `metadata.path` if present (used as the target path), otherwise `metadata.title` if present, otherwise first few words of the content plus a timestamp (e.g., `testing-preferences.md` or `user-prefers-dark-mode-1718234.md`)
- **Frontmatter `description`:** `metadata.description` if present, otherwise first sentence of the content

The `metadata` fields come from the `ModelExtractor` when automatic extraction is configured — its system prompt instructs it to produce a title and description for each extracted fact (see [Appendix B](#appendix-b-extraction-configuration) for the configuration example). When the agent uses the `store_memory` tool instead (explicit write), no extractor is involved — `add()` receives raw content with no metadata and falls back to deriving both from the content. The `path` override is primarily for programmatic writes (setup scripts, external tools) that know exactly where content belongs.

**`search(query, options?)`**

Required by the `MemoryStore` interface. The default implementation performs keyword matching against filenames, `description` frontmatter, and file content, excluding `knowledge/system/` (already loaded in full). Returns the top matches as `MemoryEntry[]`, ranked by term frequency. No model call, no embeddings.

Progressive disclosure (see [Progressive Disclosure](#progressive-disclosure)) is the primary retrieval mechanism for `FileMemoryStore` — the agent sees the file tree in its system prompt and navigates knowledge directly using filesystem tools. The [`search_memory` tool](https://github.com/strands-agents/docs/blob/main/designs/0011-memory-manager.md) serves as a fallback: it retrieves actual content in a single tool call, preventing hallucination in cases where the agent might respond based on filenames/descriptions alone without reading the underlying files.

---
## Alternative to Progressive Disclosure: Semantic Search via Offline Indexing

Rather than relying on keyword matching for `search()`, consolidation can also build a local embedding index — the same approach managed stores like `BedrockKnowledgeBaseStore` use server-side, but run locally during the offline maintenance step. This ensures `search()` performs real semantic retrieval (handling synonyms, paraphrasing, and conceptual matches) rather than a simple keyword scan.

During `consolidate()`, an embedding model computes vectors for each knowledge file and writes them to a local index (e.g., `consolidation/embeddings.json`). At runtime, `search()` embeds the query and performs cosine similarity against the index — no model call, no tokens spent per turn. This is analogous to how Bedrock Knowledge Bases indexes documents on ingest and serves semantic search via its `RetrieveCommand`, but without the managed infrastructure.

With semantic search in place, `FileMemoryStore` works through the existing `MemoryManager` retrieval mechanisms (injection and `search_memory`) without depending on the agent's judgment to navigate files. The tradeoff: progressive disclosure costs tokens every turn (file tree in system prompt + tool calls for navigation), while semantic search costs tokens only during offline consolidation and is free at runtime.

## Integration with Existing Features

Minimal setup (defaults to `LocalFileStorage` at `~/.strands/`):

```typescript
import { Agent, MemoryManager } from "@strands-agents/sdk";
import { FileMemoryStore } from "@strands-agents/sdk/memory";

const memoryStore = new FileMemoryStore({ name: "agent-memory" });

const agent = new Agent({
    model,
    memoryManager: new MemoryManager({
        stores: [memoryStore],
        injection: false, // progressive disclosure is the recommended retrieval path
    }),
});
```

L1 and L2 can share a `Storage` instance to keep context and knowledge under one root:

```typescript
import { GithubStorage } from "./github-storage";
import { ContextManager } from "@strands-agents/sdk/context";
import { FileMemoryStore } from "@strands-agents/sdk/memory";

const storage = new GithubStorage({ owner: "myorg", repo: "agent-memory", branch: "main" });
const memoryStore = new FileMemoryStore({ name: "agent-memory", storage });

const agent = new Agent({
    model,
    contextManager: new ContextManager({ storage }),
    memoryManager: new MemoryManager({
        stores: [memoryStore],
        injection: false,
    }),
});
```

---

## Consolidation

Consolidation improves memory quality after facts accumulate. It is a developer-invoked Strands agent exposed as a method on `FileMemoryStore`. It reads stored knowledge, reasons across files, and writes changes through `Storage`.

All extracted facts land in `knowledge/facts/` by default — `FileMemoryStore.add()` writes there unless an explicit `metadata.path` override is provided. This avoids a classification model call on every extraction while still allowing programmatic writes (setup scripts, external tools) to target a specific directory. Consolidation is responsible for reorganizing files into appropriate subdirectories during offline maintenance — it may create new directories when the content warrants it (subject to programmatic guardrails), since it has full cross-file context to make informed categorization decisions.

### How It Works

```
myStore.consolidate(config)
│
├─ 1. SCOPE: process all files under knowledge/
│
├─ 2. CLUSTER: group eligible files by subdirectory
│     Clustering keeps each agent invocation focused on related files — a cluster of
│     testing facts can be deduplicated, but mixing testing facts with deploy procedures
│     would force the agent to reason across unrelated topics in a single pass.
│
│     cluster 1 (facts/): [dark-mode.md, editor-preferences.md, deploy-process.md, testing-philosophy.md]
│
├─ 3. EXECUTE: for each cluster (in parallel), invoke a Strands agent
│     Clusters operate on disjoint file sets (each file belongs to exactly one cluster),
│     so parallel agents cannot conflict.
│
│     Each agent invocation receives:
│     - model:         the LLM passed in config (does the reasoning)
│     - system prompt: built from config.operations (excluding reorganize)
│     - tools:         read_file, write_file, delete_file (thin wrappers around Storage)
│     - context:       the files in this cluster
│
│     The agent reads the cluster's files, applies the requested operations, and writes
│     changes back through Storage.
│
├─ 4. REORGANIZE: a separate final pass (after all clusters complete)
│     A single agent sees all file paths and descriptions across knowledge/ and moves
│     them to appropriate subdirectories. This runs after per-cluster operations because
│     reorganize needs cross-directory visibility that per-cluster agents don't have.
│
└─ 5. RECORD: append timestamp + summary to consolidation/changelog.md
       This serves as both an audit log and the cursor for the next "latest" run.
```

### Operations

The `operations` config controls which directives go into the agent's system prompt. They are prompt instructions — the LLM decides how to apply them using the file content and change history available in its context.

| Operation | Agent behavior | Example |
|-----------|---------------|---------|
| `deduplicate` | Merge files expressing the same fact | "User prefers dark mode" + "Theme preference: dark" → one file |
| `resolveContradictions` | Keep the more recent fact (per change history), delete the other | "Uses tabs" (April) vs "Uses spaces" (June) → keeps spaces |
| `deriveInsights` | Combine related facts into a higher-level pattern | 3 testing facts → "Testing philosophy: high-fidelity, boundary-mocked" |
| `prune` | Delete entries whose content is fully covered by a newer file | `old-deploy-process.md` superseded by `deploy-process.md` → deleted |
| `reorganize` | Move files to appropriate subdirectories based on content; may create new directories. Runs as a separate final pass with full cross-directory visibility. | Fact about debugging patterns in `facts/` → moved to `operations/debugging.md` |

### Directory Management: Hybrid Guardrails

Consolidation uses a hybrid of agent reasoning and programmatic constraints for directory management. The agent decides *which* directory a file belongs in and *what* new directories should be called — these are judgment-based decisions that benefit from cross-file context. Programmatic validation in the tool callbacks enforces structural invariants the agent cannot violate:

| Concern | Mechanism |
|---------|-----------|
| Which directory a file belongs in | Agent reasoning |
| What to name a new directory | Agent reasoning |
| Max number of directories (`maxDirectories`, default 8) | Programmatic — tool rejects write |
| Max nesting depth (one level under `knowledge/`) | Programmatic — tool rejects write |
| Directory naming format (lowercase, alphanumeric, hyphens, ≤30 chars) | Programmatic — tool rejects write |

This avoids relying solely on prompt engineering for structural constraints (which can be ignored or misinterpreted by the model) while preserving agent creativity for the organizational decisions that genuinely require judgment.

### Usage

Since Strands is a client-side SDK with no server process, consolidation needs an external trigger:

```typescript
await myStore.consolidate({
  model,
  operations: ["deduplicate", "resolveContradictions"],
});
```

Scheduling frequency is controlled by the developer — e.g., after each session for incremental cleanup, or weekly for a deep clean. See [Appendix D](#appendix-d-consolidation-examples) for the nightly vs. weekly patterns for option 2, and [Appendix E](#appendix-e-github-action-yaml) for an example GitHub Action trigger.

---

## Alternatives Considered

### 1. Branching: Separate branch per session

Each session writes to its own branch, merges back to `main` on close.

**Why rejected:** Path-based isolation (`context/{id}.md`) achieves the same separation without branch management overhead or merge conflicts.

### 2. Consolidation: Inline during agent sessions

Trigger consolidation within the agent loop (e.g., every N turns) instead of externally.

**Why rejected:** Consolidation reads many files and calls a model — running it mid-session adds latency to agent responses. Since Strands is a client-side SDK with no background process, there's no way to run it asynchronously without blocking the user. External invocation (GitHub Action, CLI) keeps the agent loop fast and gives developers cost control.

### 3. Consolidation: Deterministic rules instead of LLM

Hard-coded deduplication rules (e.g., cosine similarity > 0.95 → merge).

**Why rejected:** Rules miss semantic duplicates ("User prefers dark mode" vs. "Theme preference: dark") and can't derive insights from combining related facts. LLM judgment handles nuance. Non-determinism is mitigated by every change being versioned and reversible.

### 4. File placement: Classify at extraction time

Have `FileMemoryStore.add()` call a model to categorize each fact (e.g., preference, procedure, project fact) and write it directly to the appropriate subdirectory.

**Why rejected:** Adds a classification model call to every extraction, increasing latency and token cost during agent sessions. The classifier also only sees a single fact in isolation, leading to worse categorization than the consolidation agent, which sees all files together and can make informed cross-file decisions. Writing everything to `facts/` by default keeps `add()` fast and simple, and lets consolidation handle reorganization with full context during offline maintenance.

### 5. Retrieval: Heuristic scoring with metadata

Score files using frontmatter metadata (tags, recency, access frequency) and load top-K within a token budget. No agent involvement in retrieval.

**Why rejected:** Requires building metadata infrastructure (tag extraction, scoring weights, access counters). Vector store backends like Bedrock Knowledge Bases have embeddings and similarity scoring server-side, making programmatic scoring natural. For a filesystem store there is no equivalent infrastructure — the agent's own judgment (navigating via filenames and descriptions) is the better retrieval engine.

---

## Consequences

### What Becomes Easier

- No ongoing infrastructure costs for storage and retrieval — everything runs locally. Only LLM calls (extraction, consolidation) cost tokens, and those are controlled by the developer.
- Cross-session knowledge with zero external infrastructure (no vector DB, no managed service)
- Developer debugging — inspect the file hierarchy directly; changes are diffable
- Portability — memory directory can be copied, shared, or used to seed other agents with a knowledge base

### What Becomes Harder

- Scaling beyond ~1,000 knowledge files — file listing and search may slow down with very large trees
- Concurrent writes from multiple agent instances — simultaneous writes require coordination (file locking or single-writer constraint)
- Retrieval quality depends on model judgment — the agent must recognize when to search and what to read; if it doesn't look, relevant memories stay hidden
- Consolidation cost and non-determinism — each run calls a model, costs tokens, and may produce different results on re-runs
- Storage growth — sessions accumulate indefinitely; may need a retention policy for old session files



---

## Security Model

`FileMemoryStore` assumes single-tenant compute — one instance per user/agent. `Storage` is identity-unaware; it takes a path and performs I/O without knowledge of who is asking. It is not a multi-tenancy boundary. Deployments serving multiple users must isolate at the container or credential layer (e.g., separate containers per tenant), not within a shared `Storage` instance. Path validation is defense-in-depth against bugs, not an access control mechanism.

---

## Proposed Changes to MemoryManager (out of scope)

This feature requires two additions to the `MemoryStore` interface and `MemoryManager`, which are owned outside this project. Documented here for coordination.

### Additions to `MemoryStore` interface

```typescript
interface MemoryStore {
  // ... existing methods ...

  /** Returns text to inject into the model context. MemoryManager wires this automatically. */
  renderContent?(context: InjectionContext): Promise<string | undefined>

  /** When to run renderContent. Defaults to 'userTurn'. */
  injectionTrigger?: InjectionTrigger | ((context: InjectionContext) => boolean)
}
```

### Addition to `MemoryManager.initAgent`

After wiring search-based injection and extraction, MemoryManager loops through stores and registers injection middleware for any store that defines `renderContent`:

```typescript
private _initStoreInjection(agent: LocalAgent): void {
  for (const store of this._config.stores) {
    if (store.renderContent) {
      agent.addMiddleware(InvokeModelStage.Input, createInjectionMiddleware({
        trigger: store.injectionTrigger ?? 'userTurn',
        renderContent: (context) => store.renderContent!(context),
      }))
    }
  }
}
```

### Why

- Follows the same pattern as `getTools()` — the store declares what it needs, MemoryManager wires it
- Eliminates the need for stores to construct and return `Plugin` objects
- Store-provided injection is independent of the `injection` config (setting `injection: false` disables search-based injection only, not store-provided context like progressive disclosure)
- Any future store that needs to inject context (e.g., a store that shows available pages or a summary of recent changes) uses the same mechanism

---

## Willingness to Implement

Yes.


---
<details>
  <summary><b>Appendix A: Retrieval Worked Examples</b></summary>

```
User asks: "how should I structure these tests?"

Agent sees file tree → spots "testing-philosophy.md" (description: "Integration-first, mock at boundaries")
Agent calls: read_memory_file("knowledge/facts/testing-philosophy.md")
→ full content loaded into context, agent answers using the loaded knowledge
```

For broader queries:

```
User asks: "what do you know about our deploy process?"

Agent sees file tree → spots "deploy-process.md" (description: "Team's deployment pipeline and rollback procedures")
Agent also sees "project-architecture.md" (description: "Service boundaries and data flow")
Agent calls: read_memory_file("knowledge/facts/deploy-process.md")
→ full content loaded into context, agent answers from it
```

Fallback — when filenames and descriptions aren't enough:

```
User asks: "what was that thing about retrying failed requests?"

Agent sees file tree → no filename or description obviously matches "retrying failed requests"
Agent calls: search_memory("retry failed requests")
→ FileMemoryStore.search() keyword-matches against file content, returns relevant entries
→ agent gets content directly without guessing which file to read
```

</details>

<details>
  <summary><b>Appendix B: Extraction Configuration</b></summary>

```typescript
const myStore = new FileMemoryStore({
  name: "agent-memory",
  extraction: {
    triggers: [new InvocationTrigger()],
    extractor: new ModelExtractor({
      model,
      systemPrompt: `Extract discrete facts from the conversation. For each fact, return:
- content: the fact itself
- metadata.title: a 2-4 word slug (e.g., "testing-preferences")
- metadata.description: a one-line summary for discoverability`,
    }),
  },
});
```

</details>

<details>
  <summary><b>Appendix C: Versioning and Rollback (Nice to Have)</b></summary>

The core `FileMemoryStore` operates on `Storage` alone — no versioning required. For developers who want rollback support and richer change tracking (e.g., undoing bad consolidation), `Storage` implementations can optionally expose versioning methods. This is a nice-to-have extension, not a requirement for the initial implementation.

### Versioning Extension

Backends that support versioning can additionally implement `changesSince()` and `rollback()`:

```typescript
interface VersionedFileStorage extends Storage {
  changesSince(timestamp: number): Promise<FileChange[]>;
  rollback(path: string, timestamp: number): Promise<void>;
}

interface FileChange {
  path: string;
  timestamp: number;
  operation: "write" | "delete";
}
```

**`changesSince(timestamp)`** returns all writes and deletes after the given timestamp — would enable a future `scope: 'latest'` option for incremental consolidation (processing only files changed since the last run). **`rollback(path, timestamp)`** restores a file to its state at the given timestamp — used to undo bad consolidation.

| Implementation | How it versions |
|---------------|----------------|
| `LocalFileStorage` | Copies previous content to `.versions/{path}/{timestamp}` before overwriting; maintains a `.journal` file for `changesSince()` |
| `S3Storage` | S3 object versioning — managed by the service |
| `GithubStorage` | Git commits — `changesSince` maps to commit history, `rollback` restores from a prior commit |

### Git-Based Example

```typescript
class GithubStorage implements VersionedFileStorage {
  // Storage methods
  async put(key: string, data: Uint8Array) { /* GitHub Contents API PUT (creates commit) */ }
  async get(key: string) { /* GitHub Contents API GET */ }
  async delete(key: string) { /* GitHub Contents API DELETE */ }
  async list(prefix: string) { /* GitHub Trees API */ }

  // Versioning methods
  async changesSince(timestamp: number) { /* git log --since via Commits API */ }
  async rollback(path: string, timestamp: number) { /* restore file content from prior commit */ }
}
```

### Call Flow

When the agent extracts a fact, `FileMemoryStore.add()` writes through its `Storage` instance:

```
agent extracts "user prefers dark mode"
  → FileMemoryStore.add(content, { title: "dark-mode" })
    → Storage.put("knowledge/facts/dark-mode.md", data)
      → fs.writeFile("~/.strands/knowledge/facts/dark-mode.md", data)    // LocalFileStorage
      → s3.PutObject(...)                                                 // S3Storage
      → github.createOrUpdateFileContents(...)                            // GithubStorage
```

When consolidation runs, files are read/written through the storage backend:

```
myStore.consolidate({ model, operations: ["deduplicate"] })
  → storage.list("knowledge/")            // all files in scope
  → consolidation agent reads/writes files via Storage
```

### Usage

```typescript
const storage = new GithubStorage({ owner: "myorg", repo: "agent-memory", branch: "main" });

const memoryStore = new FileMemoryStore({
    name: "agent-memory",
    storage,
});

const agent = new Agent({
    model,
    memoryManager: new MemoryManager({ stores: [memoryStore] }),
});
`

</details>

<details>
  <summary><b>Appendix D: Consolidation Examples</b></summary>

### Full Usage Script

```typescript
// consolidate.ts — run via cron, GitHub Action, or manually
import { FileMemoryStore } from "@strands-agents/sdk/memory";

const myStore = new FileMemoryStore({ name: "agent-memory" });

// Nightly (targeted operations only)
await myStore.consolidate({
  model,
  operations: ["deduplicate", "resolveContradictions"],
});

// Weekly deep clean (all operations by default)
await myStore.consolidate({ model });
```

### Example Output

Each operation is recorded in `consolidation/changelog.md` (serves as both audit log and potential cursor for future incremental scoping):

```markdown
## 2026-06-15 02:00 (nightly)
- Consolidate(deduplicate): merged `facts/dark-mode.md` into `facts/editor-preferences.md`
- Consolidate(resolve): kept "uses spaces" over "uses tabs" (recency: June vs April)
- Consolidate(derive): synthesized `facts/testing-philosophy.md` from 3 entries
- Consolidate(prune): deleted `facts/old-deploy-process.md` (last written 2026-03-01, superseded by `facts/deploy-process.md`)
```

</details>

<details>
  <summary><b>Appendix E: GitHub Action YAML</b></summary>

```yaml
# .github/workflows/consolidate.yml
name: Memory Consolidation

on:
  schedule:
    - cron: "0 2 * * *" # nightly
  workflow_dispatch: # manual trigger

jobs:
  consolidate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - run: npx strands-memory consolidate --path ~/.strands
      - run: |
          git config user.name "strands-consolidation[bot]"
          git config user.email "consolidation@users.noreply.github.com"
      - run: git diff --quiet || (git add . && git commit -m "Consolidate: nightly maintenance" && git push)
```

</details>

<details>
  <summary><b>Appendix F: Benchmarks To Test</b></summary>

### Deep Memory Retrieval (DMR)

Have the agent accumulate knowledge across sessions stored in `FileMemoryStore`, then test whether it can recall facts from session 1 after 10+ sessions have passed. Compare recall accuracy with vs. without consolidation to measure consolidation's impact on long-horizon retrieval quality.

### File-Hierarchy Retrieval vs. Embeddings

Compare progressive disclosure (file tree + agent navigation) against embedding-based retrieval. Letta's research found that their filesystem approach scored 74.0% on the LoCoMo benchmark by storing conversational histories in files — beating specialized memory tool libraries. Evaluate `FileMemoryStore` against the same or similar benchmarks.

### Consolidation Frequency

Measure the relationship between consolidation frequency and token cost vs. retrieval quality. Research suggests diminishing returns — find the optimal cadence that preserves retrieval quality without excessive token usage.

</details>

<details>
  <summary><b>Appendix G: Success Criteria and Stretch Goals</b></summary>

### Required

| Criterion | Measure |
|-----------|---------|
| SDK integration | A working `FileMemoryStore` that plugs into `memoryManager.stores`, with both L1 and L2 sharing a `Storage` instance — passing integration tests with the existing SDK |
| `LocalFileStorage` | A `Storage` implementation backed by the local filesystem. Reads/writes files under a configurable root directory. The default backend for local development and prototyping. |
| `GithubStorage` | A `Storage` implementation backed by GitHub repos (Contents API for read/write/delete, Trees API for list). Enables shared, collaborative agent memory across teams via standard git workflows (PRs for consolidation review, branch protection for `system/`, `.github/workflows/` for scheduled consolidation). |
| Auditable history | `consolidation/changelog.md` tells a coherent story of what the agent learned and when — a developer can trace how memory evolved over time without inspecting individual file diffs |
| Consolidation quality | Benchmark showing how consolidation changes retrieval quality (e.g., DMR recall before/after consolidation runs) |
| Progressive disclosure efficiency | Benchmark measuring how progressive disclosure changes tokens loaded per turn and retrieval accuracy vs. full-context injection |
| Inspectable | A developer can browse the memory directory and diff file changes directly — the file hierarchy is human-readable and diffable |

### Stretch Goals / Nice to Have

| Criterion | Measure |
|-----------|---------|
| Versioning extension | A `VersionedStorage` interface extending `Storage` with `changesSince()` and `rollback()` for precise change tracking and undo support — see [Appendix C](#appendix-c-versioning-and-rollback-nice-to-have) |
| `scope` param | Add a `scope` parameter to `ConsolidateConfig`. `'latest'` processes only files changed since the last consolidation run via `VersionedStorage.changesSince()`. `'all'` processes everything under `knowledge/` (current default behavior). Depends on the versioning extension. |
| Semantic search (local index) | During `consolidate()`, compute embeddings for each knowledge file and write to `consolidation/embeddings.json`. At runtime, `search()` embeds the query and does cosine similarity against the index — no infrastructure, no runtime model call, scales to ~1,000 files |
| Semantic search (vector DB) | For larger-scale deployments, integrate with a vector database (SQLite + vector extension, pgvector, or a managed service) for similarity search. Appropriate when the local index approach hits scaling limits |
| Comparative benchmarks | Benchmark comparison against managed alternatives (`BedrockKnowledgeBaseStore`) and in-memory baselines showing where a local file store adds value and where it doesn't |
| End-to-end deployed example | A deployed Strands agent (code review, coding assistant, or similar) that uses `FileMemoryStore` for memory accumulation across sessions, with scheduled consolidation via GitHub Actions. Deployed for an internal team use case (e.g., a code review agent that remembers codebase patterns, or an onboarding agent that accumulates project knowledge) AND publishable as a labs/devtools sample demonstrating the full lifecycle: agent learns → memory accumulates → consolidation improves → agent gets better over time |

</details>