/**
 * Bash and sandbox-routed shell tools for Node.js environments.
 *
 * `bash` spawns a persistent bash process on the host. `makeShell` builds a
 * stateless tool that routes commands through a sandbox, which runs whichever
 * shell that sandbox provides.
 */

export { bash } from './bash.js'
export { makeShell, makeBash } from './make-shell.js'
export type { MakeShellOptions, MakeBashOptions } from './make-shell.js'
export {
  SANDBOX_SHELL_DESCRIPTION,
  SANDBOX_BASH_DESCRIPTION,
  BashTimeoutError,
  BashSessionError,
  ShellTimeoutError,
  ShellExecutionError,
} from './types.js'
export type { BashInput, BashOutput, ExecuteInput, RestartInput } from './types.js'
