The Strands community has built tools and integrations for a variety of use cases. This catalog helps you discover what’s available and find packages that solve your specific needs.

Browse by category below to find tools, model providers, session managers, and platform integrations built by the community.

Community maintained

These packages are maintained by their authors, not the Strands team. Review packages before using them in production. Quality and support may vary.

## Integrations

Platform integrations help you connect Strands agents with external services and user interfaces.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [AG-UI](/docs/community/integrations/ag-ui/index.md) | AG-UI integration | ✅ | ❌ |

## Plugins

Plugins extend agent behavior by hooking into lifecycle events. Use these to add cross-cutting capabilities like policy enforcement, logging, or output control without modifying your agent code.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Agent Control](/docs/community/plugins/agent-control/index.md) | Runtime policy enforcement for agents | ✅ | ❌ |
| [Amazon AgentCore Payments](/docs/community/plugins/agentcore-payments/index.md) | Automated payment processing for agents using Amazon Bedrock AgentCore | ✅ | ❌ |
| [Amazon AgentCore Tool Search](/docs/community/plugins/agentcore-tool-search/index.md) | Semantic tool discovery for agents using Amazon Bedrock AgentCore Gateway | ✅ | ❌ |
| [Datadog AI Guard](/docs/community/plugins/datadog-ai-guard/index.md) | Real-time AI security with Datadog AI Guard | ✅ | ❌ |
| [S3 Vectors Memory](/docs/community/plugins/s3-vectors-memory/index.md) | Long-term semantic memory for Strands Agents backed by Amazon S3 Vectors | ✅ | ❌ |

## Model providers

Model providers add support for additional LLM services beyond the built-in providers. Use these to integrate with specialized or regional LLM platforms.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [CLOVA Studio](/docs/community/model-providers/clova-studio/index.md) | Naver CLOVA Studio | ✅ | ❌ |
| [Cohere](/docs/community/model-providers/cohere/index.md) | Cohere LLM | ✅ | ❌ |
| [Fireworks AI](/docs/community/model-providers/fireworksai/index.md) | Fireworks AI | ✅ | ❌ |
| [MLX](/docs/community/model-providers/mlx/index.md) | MLX | ✅ | ❌ |
| [Nebius Token Factory](/docs/community/model-providers/nebius-token-factory/index.md) | Nebius Token Factory | ✅ | ❌ |
| [NVIDIA NIM](/docs/community/model-providers/nvidia-nim/index.md) | NVIDIA NIM | ✅ | ❌ |
| [OVHcloud AI Endpoints](/docs/community/model-providers/ovhcloud-ai-endpoints/index.md) | OVHcloud AI Endpoints | ✅ | ❌ |
| [SGLang](/docs/community/model-providers/sglang/index.md) | SGLang | ✅ | ❌ |
| [vLLM](/docs/community/model-providers/vllm/index.md) | vLLM | ✅ | ❌ |
| [xAI](/docs/community/model-providers/xai/index.md) | xAI Grok | ✅ | ❌ |

## Session managers

Session managers provide alternative storage backends for conversation history. Use these when you need persistent, scalable, or distributed session storage.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Amazon AgentCore Memory](/docs/community/session-managers/agentcore-memory/index.md) | Amazon AgentCore | ✅ | ❌ |
| [Valkey/Redis](/docs/community/session-managers/strands-valkey-session-manager/index.md) | Valkey session manager | ✅ | ❌ |

## Memory stores

Memory stores are backends that give agents long-term memory across sessions. Each implements the `MemoryStore` interface and plugs into an agent through a `MemoryManager`. See the [Memory Stores overview](/docs/community/memory-stores/overview/index.md) for how they fit together.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |

## Tools

Tools extend your agents with capabilities for specific services and platforms. Each package provides one or more tools you can add to your agents.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [apify](/docs/community/tools/strands-apify/index.md) | Web scraping for social media, search engines, e-commerce, and more via Apify Actors | ✅ | ❌ |
| [deepgram](/docs/community/tools/strands-deepgram/index.md) | Deepgram speech-to-text | ✅ | ❌ |
| [google](/docs/community/tools/strands-google/index.md) | Google API integration | ✅ | ❌ |
| [hubspot](/docs/community/tools/strands-hubspot/index.md) | HubSpot CRM integration | ✅ | ❌ |
| [perplexity](/docs/community/tools/strands-perplexity/index.md) | Perplexity web search | ✅ | ❌ |
| [spraay](/docs/community/tools/strands-spraay/index.md) | Spraay batch payments on Base | ✅ | ❌ |
| [strands-sql](/docs/community/tools/strands-sql/index.md) | General-purpose SQL tool for Strands Agents — PostgreSQL, MySQL, and SQLite via SQLAlchemy. | ✅ | ❌ |
| [teams](/docs/community/tools/strands-teams/index.md) | Microsoft Teams | ✅ | ❌ |
| [telegram](/docs/community/tools/strands-telegram/index.md) | Telegram bot | ✅ | ❌ |
| [telegram-listener](/docs/community/tools/strands-telegram-listener/index.md) | Telegram listener | ✅ | ❌ |
| [UTCP](/docs/community/tools/utcp/index.md) | Universal Tool Calling Protocol | ✅ | ❌ |

## Agent Extensions

Agent extensions provide specialized agent subclasses that change how Strands agents reason and act. Use these when you need an alternative action paradigm beyond the default tool-calling approach.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Code Agent](/docs/community/agent-extensions/strands-code-agent/index.md) | A coding agent that replaces tool-calling with code generation in a persistent Python REPL | ✅ | ❌ |

## Interventions

Interventions provide composable control handlers for authorization, guardrails, and steering. Use these to add typed decision-making to your agents without modifying core logic.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Agent Governance Toolkit](/docs/community/interventions/strands-agt/index.md) | Deterministic policy enforcement via Microsoft Agent Governance Toolkit | ✅ | ❌ |

---

## Add your package

Built something useful? We’d love to feature it here.

See the [Extensions guide](/docs/contribute/contributing/extensions/index.md) for how to build and publish your package, and the [Get Featured guide](/docs/community/get-featured/index.md) for how to get listed in this catalog.