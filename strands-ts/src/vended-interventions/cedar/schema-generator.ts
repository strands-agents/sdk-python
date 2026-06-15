import type { ToolDefinition } from './cedar.js'

/** Adapter for `@cedar-policy/mcp-schema-generator-wasm` — generates Cedar schemas from tool definitions. */
export interface SchemaGenerator {
  generateSchema(tools: ToolDefinition[]): string
}

/** Creates a SchemaGenerator from the loaded `@cedar-policy/mcp-schema-generator-wasm` module. */
export function createSchemaGenerator(wasm: {
  generateSchema: (stub: string, toolsJson: string, configJson?: string) => string
}): SchemaGenerator {
  const defaultStub = `
namespace Agent {
  @mcp_principal
  entity User;
  @mcp_resource
  entity Resource;
  @mcp_context("session")
  type SessionContext = {
    hour_utc: Long,
    call_count: Long
  };
}
`

  return {
    generateSchema(tools: ToolDefinition[]): string {
      const config = JSON.stringify({ flattenNamespaces: true })
      const result = JSON.parse(wasm.generateSchema(defaultStub, JSON.stringify(tools), config)) as {
        schema: string | null
        error: string | null
        isOk: boolean
      }
      if (!result.isOk || !result.schema) {
        throw new Error(`Schema generation failed: ${result.error}`)
      }
      // Strip namespace wrapper and all internal namespace prefixes (e.g. Agent::TypeName → TypeName)
      const nsMatch = result.schema.match(/^namespace\s+(\w+)\s*\{/)
      const ns = nsMatch ? nsMatch[1]! : 'Agent'
      return result.schema
        .replace(/^namespace\s+\w+\s*\{/, '')
        .replace(/\}\s*$/, '')
        .replaceAll(`${ns}::`, '')
    },
  }
}
