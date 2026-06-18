import { defaultSandbox } from '$/sdk/sandbox/default.js'
import { NotASandboxLocalEnvironment } from '$/sdk/sandbox/not-a-sandbox-local-environment.js'

// Integration tests don't load index.node.ts; register the node default sandbox.
defaultSandbox.set(new NotASandboxLocalEnvironment())
