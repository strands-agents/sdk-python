import { defaultSandbox } from '$/sdk/sandbox/default.js'
import { NotASandboxLocalEnvironment } from '$/sdk/sandbox/not-a-sandbox-local-environment.js'
import { mcpServerLoader } from '$/sdk/mcp/config.js'
import { resolveServerConfigs } from '$/sdk/mcp/config.node.js'

// Integration tests don't load index.node.ts; register the node defaults.
defaultSandbox.set(new NotASandboxLocalEnvironment())
mcpServerLoader.set(resolveServerConfigs)
