/**
 * Sandbox-routed code execution tool.
 *
 * Provides `makeCodeExecution` (a factory) and `codeExecution` (the default
 * instance that reads the sandbox from `context.agent.sandbox` at call time).
 * Each call runs a fresh interpreter through the sandbox; state does not
 * persist across calls.
 *
 * The tool is a thin shim over `Sandbox.executeCode`. The sandbox is the
 * security boundary; the tool refuses to execute when the agent falls back to
 * `NotASandboxLocalEnvironment` (whose name signals "no isolation").
 */

import { tool } from '../../tools/tool-factory.js'
import { z } from 'zod'
import { Sandbox } from '../../sandbox/base.js'
import { SandboxAbortError, SandboxTimeoutError } from '../../sandbox/errors.js'
import { NotASandboxLocalEnvironment } from '../../sandbox/not-a-sandbox-local-environment.js'
import {
  CODE_EXECUTION_DESCRIPTION,
  CodeSizeExceededError,
  DEFAULT_LANGUAGE,
  DEFAULT_MAX_CODE_BYTES,
  DEFAULT_MAX_OUTPUT_BYTES,
  DEFAULT_TIMEOUT_SECONDS,
  SandboxNotConfiguredError,
  TRUNCATION_MARKER,
  type CodeExecutionOutput,
} from './types.js'

/**
 * Truncate `text` to `maxBytes` UTF-8 bytes and append `TRUNCATION_MARKER` if trimmed.
 *
 * Truncation is byte-oriented (not character-oriented) because the cap
 * protects downstream consumers that may be counting bytes or tokens. If the
 * cut falls in the middle of a multi-byte code point the default
 * (non-fatal) `TextDecoder` would emit U+FFFD (`�`) in its place; we
 * strip a single trailing replacement character to match the Python side,
 * which uses `errors="ignore"` and genuinely drops the incomplete tail.
 */
function truncate(text: string, maxBytes: number): string {
  const encoded = new TextEncoder().encode(text)
  if (encoded.byteLength <= maxBytes) return text
  let trimmed = new TextDecoder('utf-8').decode(encoded.slice(0, maxBytes))
  if (trimmed.endsWith('�')) {
    trimmed = trimmed.slice(0, -1)
  }
  return trimmed + TRUNCATION_MARKER
}

/**
 * Options for {@link makeCodeExecution}.
 */
export interface MakeCodeExecutionOptions {
  /**
   * Tool name shown to the model. Defaults to `code_execution`.
   */
  name?: string
  /**
   * Description shown to the model.
   */
  description?: string
  /**
   * Interpreter to run. Fixed at factory time -- the model cannot select an
   * interpreter. Defaults to `node`. Passed through to the sandbox, which
   * validates against `LANGUAGE_PATTERN`.
   */
  language?: string
  /**
   * Upper bound on `code` size (bytes, UTF-8). Larger inputs are rejected
   * before touching the sandbox. Defaults to `DEFAULT_MAX_CODE_BYTES`.
   */
  maxCodeBytes?: number
  /**
   * Upper bound on stdout/stderr size returned to the model (bytes, UTF-8).
   * Excess is dropped and a truncation marker is appended. Defaults to
   * `DEFAULT_MAX_OUTPUT_BYTES`.
   */
  maxOutputBytes?: number
  /**
   * Timeout in seconds passed to the sandbox when the caller does not supply
   * one. Defaults to `DEFAULT_TIMEOUT_SECONDS`.
   */
  defaultTimeout?: number
}

/**
 * Create a sandbox-routed code execution tool.
 *
 * If a sandbox is passed, it is bound at creation time. Otherwise the tool
 * reads the sandbox from `context.agent.sandbox` at call time and refuses to
 * run when that sandbox is the host default (`NotASandboxLocalEnvironment`).
 */
export function makeCodeExecution(options?: MakeCodeExecutionOptions): ReturnType<typeof tool>
export function makeCodeExecution(
  sandbox: Sandbox | undefined,
  options?: MakeCodeExecutionOptions
): ReturnType<typeof tool>
export function makeCodeExecution(
  sandboxOrOptions?: Sandbox | MakeCodeExecutionOptions,
  maybeOptions?: MakeCodeExecutionOptions
): ReturnType<typeof tool> {
  const boundSandbox = sandboxOrOptions instanceof Sandbox ? sandboxOrOptions : undefined
  const options = sandboxOrOptions instanceof Sandbox || maybeOptions ? (maybeOptions ?? {}) : (sandboxOrOptions ?? {})

  const language = options.language ?? DEFAULT_LANGUAGE
  const maxCodeBytes = options.maxCodeBytes ?? DEFAULT_MAX_CODE_BYTES
  const maxOutputBytes = options.maxOutputBytes ?? DEFAULT_MAX_OUTPUT_BYTES
  const defaultTimeout = options.defaultTimeout ?? DEFAULT_TIMEOUT_SECONDS

  // Reject NaN and Infinity explicitly: `NaN <= 0` is false, so a bare `<= 0`
  // check would silently disable the cap (`codeBytes > NaN` is always false).
  if (!Number.isFinite(maxCodeBytes) || maxCodeBytes <= 0) {
    throw new Error(`maxCodeBytes must be a positive, finite number, got ${String(maxCodeBytes)}`)
  }
  if (!Number.isFinite(maxOutputBytes) || maxOutputBytes <= 0) {
    throw new Error(`maxOutputBytes must be a positive, finite number, got ${String(maxOutputBytes)}`)
  }
  if (!Number.isFinite(defaultTimeout) || defaultTimeout <= 0) {
    throw new Error(`defaultTimeout must be a positive, finite number, got ${String(defaultTimeout)}`)
  }

  const inputSchema = z.object({
    code: z.string().describe('Source code to execute in the configured language.'),
    timeout: z.number().positive().finite().optional().describe(`Timeout in seconds (default: ${defaultTimeout}).`),
  })

  return tool({
    name: options.name ?? 'code_execution',
    description: options.description ?? CODE_EXECUTION_DESCRIPTION,
    inputSchema,
    callback: async (input, context): Promise<CodeExecutionOutput> => {
      // Input validation at the tool boundary.
      const codeBytes = new TextEncoder().encode(input.code).byteLength
      if (codeBytes > maxCodeBytes) {
        throw new CodeSizeExceededError(
          `code size (${codeBytes} bytes) exceeds maximum allowed size (${maxCodeBytes} bytes)`
        )
      }

      // Prefer the factory-bound sandbox; fall back to the agent's when unbound.
      // Only require context if we need the agent's sandbox — matches the
      // sibling httpRequest, which supports direct .invoke() without context.
      const sandbox = boundSandbox ?? context?.agent.sandbox
      if (!sandbox) {
        throw new Error(
          'code_execution requires either a sandbox bound at factory time or a tool context with an agent.sandbox'
        )
      }

      // The sandbox is the security boundary. Refuse execution when the agent
      // is running against the host default -- its name says "no isolation"
      // and executing model-authored code there would be a footgun.
      if (sandbox instanceof NotASandboxLocalEnvironment) {
        throw new SandboxNotConfiguredError(
          'code_execution requires an isolating sandbox (e.g. DockerSandbox, SshSandbox) ' +
            'to be configured on the agent. Refusing to execute against ' +
            'NotASandboxLocalEnvironment, which provides no isolation.'
        )
      }

      // `globalThis.performance.now()` is monotonic; `Date.now()` can jump on NTP
      // adjustments and yield a misleading (or even negative) elapsed value.
      // This also matches the Python side's `time.monotonic()`.
      const started = globalThis.performance.now()
      let result
      try {
        result = await sandbox.executeCode(input.code, language, {
          timeout: input.timeout ?? defaultTimeout,
          ...(context ? { signal: context.agent.cancelSignal } : {}),
        })
      } catch (err) {
        // Timeouts and agent cancellation both surface as an Error whose
        // `name === 'AbortError'`, so callers can distinguish cancellation
        // from other failures with the standard web-platform check.
        if (err instanceof SandboxTimeoutError) {
          throw new DOMException(err.message, 'AbortError')
        }
        if (err instanceof SandboxAbortError || (err instanceof Error && err.name === 'AbortError')) {
          const reason = context?.agent.cancelSignal.reason
          // `AbortController.abort(reason)` accepts any value; normalize to an
          // Error subtype so callers can rely on the standard error shape.
          if (reason instanceof Error) {
            throw reason
          }
          throw new DOMException('code_execution cancelled', 'AbortError')
        }
        throw new Error((err as Error).message, { cause: err })
      }
      const elapsedMs = Math.round(globalThis.performance.now() - started)

      return {
        stdout: truncate(result.stdout, maxOutputBytes),
        stderr: truncate(result.stderr, maxOutputBytes),
        exitCode: result.exitCode,
        elapsedMs,
      }
    },
  })
}

/**
 * Default code execution tool. Reads the sandbox from the agent's context at
 * call time and refuses to run when no isolating sandbox is configured.
 */
export const codeExecution = makeCodeExecution()
