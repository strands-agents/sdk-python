import { readFile } from 'node:fs/promises'
import { join } from 'node:path'
import { WASIShim } from '@bytecodealliance/preview2-shim/instantiation'

const transpileDir = join(__dirname, '..', '..', 'dist', 'transpiled')

/** Log entry forwarded from the WASM guest to the host. */
export interface LogEntry {
  level: string
  message: string
  context?: string
}

/** Arguments passed from the WASM guest to the host tool-provider import. */
export interface CallToolArgs {
  name: string
  input: string
  toolUseId: string
}

/**
 * WIT Result type for batch tool calls (list\<result\<string, string\>\>).
 * jco does NOT unwrap list elements — the host must return the tagged variant.
 */
export type ToolResult = { tag: 'ok'; val: string } | { tag: 'err'; val: string }

/**
 * Host-side mock implementations injected into the WASM guest.
 *
 * callTool returns a plain string (success) or throws (error) — jco wraps the
 * raw return into \{tag:'ok', val\} itself for WIT result\<string, string\>.
 * callTools returns ToolResult[] directly — jco does NOT unwrap list elements.
 */
export interface HostMocks {
  log: (entry: LogEntry) => void
  callTool: (args: CallToolArgs) => string
  callTools?: (args: { calls: CallToolArgs[] }) => ToolResult[]
}

/** Compile and instantiate the WASM guest component with the given host mocks. */
export async function createGuest(mocks: HostMocks): Promise<any> {
  const getCoreModule = async (path: string): Promise<WebAssembly.Module> => {
    const bytes = await readFile(join(transpileDir, path))
    return WebAssembly.compile(bytes)
  }

  const { instantiate } = await import('../../dist/transpiled/strands-agent.js')

  return instantiate(getCoreModule, {
    'strands:agent/host-log': { log: mocks.log },
    'strands:agent/tool-provider': {
      callTool: mocks.callTool,
      callTools: mocks.callTools ?? ((args: { calls: CallToolArgs[] }) => args.calls.map(mocks.callTool)),
    },
    'strands:agent/types': {},
    ...new WASIShim().getImportObject(),
  })
}

/** Drain all batches from a guest ResponseStream into a flat event array. */
export async function drainStream(stream: any): Promise<any[]> {
  const events: any[] = []
  let batch
  while ((batch = await stream.readNext()) !== undefined) {
    events.push(...batch)
  }
  return events
}
