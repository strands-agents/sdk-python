import type { CedarValueJson, TypeAndId, EntityJson } from '@cedar-policy/cedar-wasm/nodejs'
import type { ToolDefinition } from './cedar.js'

export interface SchemaGenerator {
  generateSchema(tools: ToolDefinition[]): string
  generateRequest(
    tools: ToolDefinition[],
    toolName: string,
    toolInput: Record<string, CedarValueJson>,
    principal: TypeAndId
  ): { action: TypeAndId; resource: TypeAndId; entities: EntityJson[] }
}

export function createSchemaGenerator(wasm: {
  generateSchema: (stub: string, toolsJson: string, configJson?: string) => string
  generateRequest: (
    stub: string,
    toolsJson: string,
    inputJson: string,
    principalType: string,
    principalId: string,
    resourceType: string,
    resourceId: string,
    configJson?: string
  ) => string
}): SchemaGenerator {
  const defaultStub = `
namespace Agent {
  @mcp_principal
  entity User;
  @mcp_resource
  entity Resource;
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
      return result.schema.replace(/^namespace\s+\w+\s*\{/, '').replace(/\}\s*$/, '')
    },

    generateRequest(
      tools: ToolDefinition[],
      toolName: string,
      toolInput: Record<string, CedarValueJson>,
      principal: TypeAndId
    ): { action: TypeAndId; resource: TypeAndId; entities: EntityJson[] } {
      const input = JSON.stringify({ params: { tool: toolName, args: toolInput } })
      const config = JSON.stringify({ flattenNamespaces: true })
      const result = JSON.parse(
        wasm.generateRequest(
          defaultStub,
          JSON.stringify(tools),
          input,
          principal.type,
          principal.id,
          'Resource',
          'agent',
          config
        )
      ) as {
        action: string | null
        resource: string | null
        entitiesJson: string | null
        error: string | null
        isOk: boolean
      }
      if (!result.isOk) {
        throw new Error(`Request generation failed: ${result.error}`)
      }

      return {
        action: parseEntityUid(result.action!),
        resource: parseEntityUid(result.resource!),
        entities: result.entitiesJson ? (JSON.parse(result.entitiesJson) as EntityJson[]) : [],
      }
    },
  }
}

function parseEntityUid(uid: string): TypeAndId {
  const match = uid.match(/(?:.*::)?([^:]+)::"([^"]+)"/)
  if (!match) return { type: 'Action', id: uid }
  return { type: match[1]!, id: match[2]! }
}
