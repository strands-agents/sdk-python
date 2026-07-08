import { defaultSandbox } from '../sandbox/default.js'
import { NotASandboxLocalEnvironment } from '../sandbox/not-a-sandbox-local-environment.js'

// In production, index.node.ts registers this on import. Tests don't go through that entry
// point, so this setup file does it instead.
defaultSandbox.set(new NotASandboxLocalEnvironment())
