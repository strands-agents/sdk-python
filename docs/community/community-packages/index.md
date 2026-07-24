The Strands community has built tools and integrations for a variety of use cases. This catalog helps you discover what’s available and find packages that solve your specific needs.

Browse by category below to find tools, model providers, session managers, and platform integrations built by the community.

Community maintained

These packages are maintained by their authors, not the Strands team. Review packages before using them in production. Quality and support may vary.

## Integrations

Platform integrations help you connect Strands agents with external services and user interfaces.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [AG-UI](/docs/community/integrations/ag-ui/index.md) | Build streaming chat interfaces for Strands agents with the AG-UI protocol and CopilotKit front-end components. | ✅ | ❌ |

## Plugins

Plugins extend agent behavior by hooking into lifecycle events. Use these to add cross-cutting capabilities like policy enforcement, logging, or output control without modifying your agent code.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Agent Control](/docs/community/plugins/agent-control/index.md) | Enforce centrally managed runtime policies on Strands agents with Agent Control, blocking or steering violations at every step. | ✅ | ❌ |
| [Amazon AgentCore Payments](/docs/community/plugins/agentcore-payments/index.md) | Automated payment processing for agents using Amazon Bedrock AgentCore | ✅ | ❌ |
| [Amazon AgentCore Tool Search](/docs/community/plugins/agentcore-tool-search/index.md) | Semantic tool discovery for agents using Amazon Bedrock AgentCore Gateway | ✅ | ❌ |
| [Datadog AI Guard](/docs/community/plugins/datadog-ai-guard/index.md) | Protect Strands agents in real time with Datadog AI Guard, blocking prompt injection, jailbreaks, and unsafe tool calls. | ✅ | ❌ |
| [S3 Vectors Memory](/docs/community/plugins/s3-vectors-memory/index.md) | Long-term semantic memory for Strands Agents backed by Amazon S3 Vectors | ✅ | ❌ |

## Model providers

Model providers add support for additional LLM services beyond the built-in providers. Use these to integrate with specialized or regional LLM platforms.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [CLOVA Studio](/docs/community/model-providers/clova-studio/index.md) | Connect Strands agents to Naver CLOVA Studio, Korean-optimized language models on Naver Cloud Platform, via strands-clova. | ✅ | ❌ |
| [Cohere](/docs/community/model-providers/cohere/index.md) | Use Cohere language models with Strands agents through the OpenAI compatibility layer for chat, tool calling, and streaming. | ✅ | ❌ |
| [Fireworks AI](/docs/community/model-providers/fireworksai/index.md) | Run Strands agents on Fireworks AI, fast hosted inference for open-source models, using the OpenAI-compatible API. | ✅ | ❌ |
| [MLX](/docs/community/model-providers/mlx/index.md) | Run Strands agents locally on Apple Silicon with the MLX model provider, supporting inference, LoRA fine-tuning, and vision. | ✅ | ❌ |
| [Nebius Token Factory](/docs/community/model-providers/nebius-token-factory/index.md) | Use Nebius Token Factory with Strands agents for fast open-source model inference through the OpenAI-compatible API. | ✅ | ❌ |
| [NVIDIA NIM](/docs/community/model-providers/nvidia-nim/index.md) | Connect Strands agents to NVIDIA NIM inference microservices with the strands-nvidia-nim community model provider. | ✅ | ❌ |
| [OVHcloud AI Endpoints](/docs/community/model-providers/ovhcloud-ai-endpoints/index.md) | Run Strands agents on OVHcloud AI Endpoints, a European, GDPR-compliant model service with an OpenAI-compatible API. | ✅ | ❌ |
| [SGLang](/docs/community/model-providers/sglang/index.md) | Use SGLang servers as a Strands model provider with Token-In/Token-Out support for agentic reinforcement learning training. | ✅ | ❌ |
| [vLLM](/docs/community/model-providers/vllm/index.md) | Serve Strands agents from vLLM with Token-In/Token-Out support for agentic RL training via the OpenAI-compatible API. | ✅ | ❌ |
| [xAI](/docs/community/model-providers/xai/index.md) | Use xAI Grok models with Strands agents, including server-side tools for web search, X platform access, and code execution. | ✅ | ❌ |

## Session managers

Session managers provide alternative storage backends for conversation history. Use these when you need persistent, scalable, or distributed session storage.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Amazon AgentCore Memory](/docs/community/session-managers/agentcore-memory/index.md) | Persist Strands agent conversations with Amazon Bedrock AgentCore Memory, including short-term and long-term memory strategies. | ✅ | ❌ |
| [Valkey/Redis](/docs/community/session-managers/strands-valkey-session-manager/index.md) | Store Strands agent sessions in Valkey or Redis for low-latency, distributed conversation persistence across interactions. | ✅ | ❌ |

## Memory stores

Memory stores are backends that give agents long-term memory across sessions. Each implements the `MemoryStore` interface and plugs into an agent through a `MemoryManager`. See the [Memory Stores overview](/docs/community/memory-stores/overview/index.md) for how they fit together.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [AgentCore Memory Store](/docs/community/memory-stores/agentcore-memory-store/index.md) | Amazon Bedrock AgentCore Memory as a MemoryStore for long-term recall and server-side extraction | ❌ | ✅ |
| [dakera](/docs/community/memory-stores/strands-dakera/index.md) | Persistent, decay-weighted MemoryStore for agents — self-hosted, with a full-CRUD tool | ✅ | ❌ |

## Storage

Storage backends let agents and SDK subsystems persist raw bytes under string keys. Each implements the `Storage` interface and can be passed to session managers, context offloaders, or any construct that needs durable persistence. See the [Storage overview](/docs/community/storage/overview/index.md) for how they fit together.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |

## Tools

Tools extend your agents with capabilities for specific services and platforms. Each package provides one or more tools you can add to your agents.

| Package | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [apify](/docs/community/tools/strands-apify/index.md) | Web scraping for social media, search engines, e-commerce, and more via Apify Actors | ✅ | ❌ |
| [deepgram](/docs/community/tools/strands-deepgram/index.md) | Add speech-to-text, text-to-speech, and audio intelligence to Strands agents with the Deepgram voice AI platform. | ✅ | ❌ |
| [google](/docs/community/tools/strands-google/index.md) | Give Strands agents access to 200+ Google APIs, including Gmail, Drive, Calendar, Sheets, and YouTube, from a single tool. | ✅ | ❌ |
| [hubspot](/docs/community/tools/strands-hubspot/index.md) | Query HubSpot CRM data from Strands agents with read-only tools for sales intelligence, customer research, and analytics. | ✅ | ❌ |
| [perplexity](/docs/community/tools/strands-perplexity/index.md) | Real-time web search for Strands agents using the Perplexity Search API, with citations and regional filtering. | ✅ | ❌ |
| [spraay](/docs/community/tools/strands-spraay/index.md) | Send batch crypto payments from Strands agents with the Spraay Protocol: ETH or ERC-20 tokens to up to 200 recipients on Base. | ✅ | ❌ |
| [strands-sql](/docs/community/tools/strands-sql/index.md) | General-purpose SQL tool for Strands Agents — PostgreSQL, MySQL, and SQLite via SQLAlchemy. | ✅ | ❌ |
| [teams](/docs/community/tools/strands-teams/index.md) | Send Microsoft Teams notifications from Strands agents with rich Adaptive Cards and custom messaging support. | ✅ | ❌ |
| [telegram](/docs/community/tools/strands-telegram/index.md) | Control Telegram bots from Strands agents with 60+ Bot API methods for messaging, media, and chat management. | ✅ | ❌ |
| [telegram-listener](/docs/community/tools/strands-telegram-listener/index.md) | Process incoming Telegram messages in real time with Strands agents, with AI-powered auto-replies and event handling. | ✅ | ❌ |
| [UTCP](/docs/community/tools/utcp/index.md) | Let Strands agents discover and call tools directly over native protocols with the Universal Tool Calling Protocol (UTCP). | ✅ | ❌ |

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