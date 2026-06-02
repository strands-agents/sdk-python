// Node entry point (selected by the "node" export condition in package.json).
// Registers Node-specific defaults, then re-exports the full public API.
// This is a load-bearing side effect -- do NOT mark this module side-effect-free
// or bundlers will tree-shake the registrations.
import { defaultSandbox } from './sandbox/default.js'
import { NotASandboxLocalEnvironment } from './sandbox/not-a-sandbox-local-environment.js'

defaultSandbox.set(new NotASandboxLocalEnvironment())

export * from './index.js'
