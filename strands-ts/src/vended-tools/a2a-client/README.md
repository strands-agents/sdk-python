# A2A Client Tool

Invokes a remote A2A (Agent-to-Agent) agent from a Strands agent. Thin shim
over the built-in `A2AAgent` client with hardening at the tool boundary: the
underlying client has no SSRF or size guard of its own, so scheme, host, size,
redirect, and timeout checks all live here.

## Usage

```typescript
import { Agent } from '@strands-agents/sdk'
import { a2aClient } from '@strands-agents/sdk/vended-tools/a2a-client'

const agent = new Agent({ tools: [a2aClient] })
await agent.invoke('Ask the remote agent at https://agents.example.com to summarize X.')
```

A customized instance binds URL prefixes, a shorter timeout, or a
developer-supplied client factory that carries auth:

```typescript
import { makeA2AClient } from '@strands-agents/sdk/vended-tools/a2a-client'
import { ClientFactory } from '@a2a-js/sdk/client'

const boundedClient = makeA2AClient({
  allowedUrlPrefixes: ['https://agents.example.com/'],
  timeoutSeconds: 30,
  agentConfig: {
    clientFactory: new ClientFactory(),
  },
})
```

## Input and output

| Property  | Type     | Required | Description                                    |
| --------- | -------- | -------- | ---------------------------------------------- |
| `url`     | `string` | Yes      | Base URL of the remote A2A agent (http/https). |
| `message` | `string` | Yes      | Message to send. Capped at 64 KiB.             |

The tool returns the shared multi-agent result shape:

```typescript
{
  status: 'success'
  output: string // remote agent's response text, truncated at 256 KiB
  executionTimeMs: number
  remoteCard: {
    name: string // resolved from the agent card
    description: string
    url: string
  }
}
```

Cancellation and timeout surface as `DOMException` with `name === 'AbortError'`,
matching the other Strands network tools.

## Non-obvious behavior

- Only `http:` and `https:` URLs are accepted. Hostnames are DNS-resolved
  before the request goes out; any address that is loopback, private,
  CGNAT, link-local (including the AWS/GCP metadata IPs), multicast,
  reserved, or unspecified is rejected. Reserved DNS suffixes such as
  `.internal`, `.local`, `.corp`, `.home` are blocked outright.
- Fetch is wrapped so every redirect hop (card discovery included) re-runs
  the URL guard, including the developer `allowedUrlPrefixes` if set. A
  public-looking origin cannot 302 the client onto a private or off-list
  address. Redirects are capped at five hops; `Authorization`, `Cookie`,
  and `Proxy-Authorization` are stripped when the full origin (scheme, host,
  port) changes.
- The remote agent card is fetched and its own `url` is re-validated before
  the message is sent, so a malicious card that advertises a private target
  never receives the message.
- Timeout is a soft cap. The wrapper promise rejects on the deadline, but
  `A2AAgent#invoke` does not accept an `AbortSignal`, so the underlying
  request keeps running until it finishes on its own. For a hard cap, plumb
  an `AbortSignal` through `agentConfig.clientFactory`.
- DNS rebinding is not fully closed. The URL guard resolves once and checks
  that IP; the A2A SDK then opens its own socket and re-resolves. An
  attacker-controlled resolver that returns a public IP on the first lookup
  and a private one on the second can slip through. Mitigate by pinning
  the resolver, using `allowedUrlPrefixes`, or fronting the tool with an
  egress policy.
- Size caps (`maxCardBytes`, `maxResponseBytes`) are enforced after the A2A
  SDK has already read and deserialized the body. They bound what the model
  sees, not what the process reads into memory. A hostile server that streams
  gigabytes will still consume that memory before the cap fires. Front the
  tool with an egress-side response-size limit if that matters.
- If you pass your own `agentConfig.clientFactory`, the tool does not
  install the guarded fetch on your transports; you own that discipline.
