A **memory store** is the backend that holds an agent’s long-term knowledge. The [`MemoryManager`](/docs/user-guide/concepts/memory/overview/index.md) orchestrates one or more stores to recall, inject, and extract memories across sessions. Any backend that implements the `MemoryStore` interface can plug in: see [Custom Stores](/docs/user-guide/concepts/memory/overview/index.md#custom-stores) for the contract.

The SDK ships reference stores like the [Bedrock Knowledge Base store](/docs/user-guide/concepts/memory/bedrock-knowledge-base/index.md). The packages below go further: they are **community-built** memory stores you can install and attach to an agent, backed by vector databases, managed services, and other stores the SDK does not vend itself.

Community maintained

These packages are maintained by their authors, not the Strands team. Review packages before using them in production. Quality and support may vary.

## Browse the integrations page

See the [Memory stores section of the integrations page](/integrations/?type=memory-store/index.md) for the current list, with language support and links to each package.

## Add your memory store

Built a `MemoryStore` implementation? See the [Get Featured guide](/docs/integrations/get-featured/index.md) to list it here, and the [Extensions guide](/docs/contribute/contributing/extensions/index.md) for how to build and publish a package.