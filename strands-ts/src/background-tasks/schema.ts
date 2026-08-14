import { deepCopy } from '../types/json.js'
import type { JSONSchema, JSONValue } from '../types/json.js'
import type { ToolSpec } from '../tools/types.js'

const BACKGROUND_PROPERTY = '_background'
const ALLOWED_ROOT_KEYS = new Set([
  '$id',
  '$schema',
  'title',
  'description',
  'default',
  'examples',
  'type',
  'properties',
  'required',
  'additionalProperties',
])

type BackgroundSchemaResult =
  { readonly compatible: true; readonly toolSpec: ToolSpec } | { readonly compatible: false; readonly reason: string }

export function addBackgroundSelection(toolSpec: ToolSpec): BackgroundSchemaResult {
  let copied: ToolSpec
  try {
    copied = deepCopy(toolSpec) as unknown as ToolSpec
  } catch (error) {
    return { compatible: false, reason: error instanceof Error ? error.message : String(error) }
  }

  const schema = copied.inputSchema
  if (schema === undefined) {
    copied.inputSchema = objectSchemaWithBackground({})
    return { compatible: true, toolSpec: copied }
  }
  if (!isPlainObject(schema)) {
    return { compatible: false, reason: 'input schema must be a direct object schema' }
  }

  for (const key of Object.keys(schema)) {
    if (!ALLOWED_ROOT_KEYS.has(key)) {
      return { compatible: false, reason: `unsupported root schema keyword '${key}'` }
    }
  }
  if (schema.type !== undefined && schema.type !== 'object') {
    return { compatible: false, reason: "root schema type must be 'object'" }
  }
  if (schema.properties !== undefined && !isPlainObject(schema.properties)) {
    return { compatible: false, reason: 'root schema properties must be an object' }
  }
  if (schema.additionalProperties !== undefined && typeof schema.additionalProperties !== 'boolean') {
    return { compatible: false, reason: 'schema-valued additionalProperties is not supported' }
  }
  const required = schema.required
  if (required !== undefined) {
    if (!Array.isArray(required)) {
      return { compatible: false, reason: 'root schema required must be an array' }
    }
    if (required.some((property, index) => typeof property !== 'string' || required.indexOf(property) !== index)) {
      return { compatible: false, reason: 'root schema required must contain unique property names' }
    }
  }
  if (schema.properties && BACKGROUND_PROPERTY in schema.properties) {
    return { compatible: false, reason: `schema already defines reserved property '${BACKGROUND_PROPERTY}'` }
  }
  if (required?.includes(BACKGROUND_PROPERTY)) {
    return { compatible: false, reason: `schema requires reserved property '${BACKGROUND_PROPERTY}'` }
  }

  copied.inputSchema = objectSchemaWithBackground(schema)
  return { compatible: true, toolSpec: copied }
}

export function stripBackgroundSelection(input: unknown): { readonly input: JSONValue; readonly selected?: boolean } {
  if (!isPlainObject(input)) {
    return { input: input as JSONValue }
  }
  const copied = { ...input }
  const selected = copied[BACKGROUND_PROPERTY]
  delete copied[BACKGROUND_PROPERTY]
  if (selected !== undefined && typeof selected !== 'boolean') {
    throw new TypeError(`'${BACKGROUND_PROPERTY}' must be a boolean`)
  }
  return {
    input: copied as JSONValue,
    ...(selected !== undefined && { selected }),
  }
}

function objectSchemaWithBackground(schema: JSONSchema): JSONSchema {
  const properties = schema.properties ?? {}
  return {
    ...schema,
    type: 'object',
    properties: {
      ...properties,
      [BACKGROUND_PROPERTY]: {
        type: 'boolean',
        description:
          'Run this tool call in the background. Acknowledgement is immediate; continue without waiting or polling. The final result will be delivered automatically at a later Agent boundary.',
      },
    },
  }
}

function isPlainObject(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
}
