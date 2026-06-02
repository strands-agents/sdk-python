/**
 * SSH sandbox — executes commands on a remote host via OpenSSH.
 */

import type { ExecuteOptions } from './base.js'
import { PosixShellSandbox, shellQuote } from './posix-shell.js'
import { streamProcess } from './stream-process.js'
import type { ExecutionResult, StreamChunk } from './types.js'

// Known-safe SSH options. Options that execute commands, tunnel traffic, or load
// external config are excluded. Reviewed and approved by AppSec.
// Full option reference: https://man.openbsd.org/ssh_config
const ALLOWED_SSH_OPTIONS = new Set([
  'addressfamily',
  'bindaddress',
  'bindinterface',
  'canonicaldomains',
  'canonicalizefallbacklocal',
  'canonicalizehostname',
  'canonicalizemaxdots',
  'canonicalizepermittedcnames',
  'checkhostip',
  'ciphers',
  'compression',
  'connectionattempts',
  'connecttimeout',
  'hostkeyalgorithms',
  'hostname',
  'identitiesonly',
  'ipqos',
  'kbdinteractiveauthentication',
  'kexalgorithms',
  'loglevel',
  'macs',
  'numberofpasswordprompts',
  'passwordauthentication',
  'port',
  'preferredauthentications',
  'pubkeyacceptedalgorithms',
  'pubkeyauthentication',
  'rekeylimit',
  'serveralivecountmax',
  'serveraliveinterval',
  'tcpkeepalive',
  'updatehostkeys',
  'user',
  'verifyhostkeydns',
])

/**
 * Options for constructing an {@link SshSandbox}.
 */
export interface SshSandboxOptions {
  /** SSH destination (e.g., `"user@host"`, `"192.168.1.10"`). */
  host: string
  /** Working directory on the remote host. */
  workingDir: string
  /** Path to SSH private key file. */
  identityFile?: string
  /** SSH port. Defaults to 22. */
  port?: number
  /** Additional SSH options passed as `-o` flags. */
  sshOptions?: string[]
  /**
   * Allow connections to hosts with unknown or changed SSH keys.
   *
   * When `false` (default), uses `StrictHostKeyChecking=accept-new` — trusts on
   * first connect but rejects if the key changes.
   * When `true`, uses `StrictHostKeyChecking=no` — disables host key verification.
   */
  allowUnknownHosts?: boolean
  /**
   * Bypass the SSH option allowlist.
   *
   * When `false` (default), unknown options throw at construction time.
   * When `true`, all options are passed through without validation.
   */
  allowUnsafeSshOptions?: boolean
}

/**
 * Execute commands on a remote host via SSH.
 *
 * Stateless — each {@link executeStreaming} call spawns a fresh `ssh` process.
 * All sessions use `BatchMode=yes` — interactive prompts are disabled and
 * authentication must be key-based.
 */
export class SshSandbox extends PosixShellSandbox {
  readonly host: string
  readonly workingDir: string
  private readonly _identityFile: string | undefined
  private readonly _port: number
  private readonly _allowUnknownHosts: boolean
  private readonly _sshOptions: string[]

  constructor(options: SshSandboxOptions) {
    super()
    this.host = options.host
    this.workingDir = options.workingDir
    this._identityFile = options.identityFile
    this._port = options.port ?? 22
    this._allowUnknownHosts = options.allowUnknownHosts ?? false
    this._sshOptions = options.sshOptions ?? []

    if (!options.allowUnsafeSshOptions) {
      for (const opt of this._sshOptions) {
        const name = opt.split(/[=\s]/, 1)[0]!
        if (!ALLOWED_SSH_OPTIONS.has(name.toLowerCase())) {
          throw new Error(`SSH option "${name}" is not allowed. Set allowUnsafeSshOptions: true to bypass.`)
        }
      }
    }
  }

  async *executeStreaming(
    command: string,
    options?: ExecuteOptions
  ): AsyncGenerator<StreamChunk | ExecutionResult, void, undefined> {
    const cwd = options?.cwd ?? this.workingDir
    const remoteCommand = `cd ${shellQuote(cwd)} && ${command}`

    const sshArgs: string[] = [
      '-o',
      `StrictHostKeyChecking=${this._allowUnknownHosts ? 'no' : 'accept-new'}`,
      '-o',
      'BatchMode=yes',
      '-p',
      String(this._port),
    ]

    if (this._identityFile) {
      sshArgs.push('-i', this._identityFile)
    }

    for (const opt of this._sshOptions) {
      sshArgs.push('-o', opt)
    }

    // ssh requires the hostname and command after all flags. Furthermore, '--'
    // terminates flag parsing so the host is always treated as a positional argument.
    // Without this, a host like '-oProxyCommand=evil' would be parsed as a flag,
    // enabling arbitrary command execution on the local machine.
    sshArgs.push('--', this.host, remoteCommand)

    yield* streamProcess('ssh', sshArgs, {
      timeout: options?.timeout,
      signal: options?.signal,
      enoentMessage: 'ssh is not installed or not on PATH',
    })
  }
}
