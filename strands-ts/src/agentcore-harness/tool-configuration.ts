import type { HarnessTool } from '@aws-sdk/client-bedrock-agentcore'
import { ToolValidationError } from '../errors.js'
import type { Tool } from '../tools/tool.js'

/**
 * Merges deployed Harness tools with local host-tool declarations.
 *
 * A local host tool can bind to a same-name deployed inline function when its description and
 * input schema match. The deployed declaration remains authoritative and is emitted only once.
 *
 * @internal
 * @param deployedTools - Tools configured on the deployed Harness
 * @param hostTools - Inline-function declarations generated for local host tools
 * @returns The tool list to send with InvokeHarness
 * @throws {@link ToolValidationError} If deployed and local tool declarations conflict
 */
export function mergeHarnessTools(deployedTools: HarnessTool[], hostTools: HarnessTool[]): HarnessTool[] {
  const deployedToolsByName = new Map<string, { index: number; inferred: boolean }>()
  const namedDeployedTools = deployedTools.map((tool, index) => {
    const resolvedName = resolveDeployedToolName(tool, index)
    const existing = deployedToolsByName.get(resolvedName.name)
    if (existing !== undefined) {
      if (!existing.inferred && !resolvedName.inferred) {
        throw new ToolValidationError(
          `Deployed harness contains duplicate tool name '${resolvedName.name}'. Tool names must be unique.`
        )
      }
      throw new ToolValidationError(
        `Deployed harness tools at indexes ${existing.index} and ${index} resolve to runtime name '${resolvedName.name}'. Assign explicit unique names to the unnamed deployed tools.`
      )
    }
    deployedToolsByName.set(resolvedName.name, { index, inferred: resolvedName.inferred })
    return { tool, index, ...resolvedName }
  })

  const hostToolsByName = new Map(
    hostTools
      .filter((hostTool): hostTool is HarnessTool & { name: string } => hostTool.name !== undefined)
      .map((hostTool) => [hostTool.name, hostTool])
  )
  const boundHostToolNames = new Set<string>()

  for (const deployedTool of namedDeployedTools) {
    const hostTool = hostToolsByName.get(deployedTool.name)
    if (hostTool === undefined) continue

    if (deployedTool.inferred) {
      throw new ToolValidationError(
        `Host tool '${hostTool.name}' conflicts with the runtime-generated name of the unnamed deployed tool at index ${deployedTool.index} (type '${String(deployedTool.tool.type)}'). Assign the deployed tool an explicit unique name or rename the host tool.`
      )
    }
    assertCompatibleInlineFunction(deployedTool.tool, hostTool)
    boundHostToolNames.add(deployedTool.name)
  }

  return [...deployedTools, ...hostTools.filter((hostTool) => !boundHostToolNames.has(hostTool.name ?? ''))]
}

/** Resolves the name used as the Harness runtime tool-map key. */
function resolveDeployedToolName(
  tool: HarnessTool,
  index: number
): {
  name: string
  inferred: boolean
} {
  if (tool.name) return { name: tool.name, inferred: false }

  const prefix = ((): string => {
    switch (tool.type) {
      case 'remote_mcp':
        return 'mcp'
      case 'agentcore_gateway':
        return 'gateway'
      case 'agentcore_browser':
        return 'browser'
      case 'agentcore_code_interpreter':
        return 'code_interpreter'
      case 'inline_function':
        return 'inline_function'
      default:
        throw new ToolValidationError(
          `Unnamed deployed tool at index ${index} has unsupported type '${String(tool.type)}'; its runtime name cannot be determined safely.`
        )
    }
  })()
  return { name: `${prefix}_${index}`, inferred: true }
}

/**
 * Verifies that every local host tool is admitted by an effective Harness allowlist.
 *
 * @internal
 * @param hostTools - Local host tools
 * @param allowedTools - Effective Harness allowlist, or undefined for unrestricted access
 * @throws {@link ToolValidationError} If the allowlist excludes a local host tool
 */
export function assertHostToolsAllowed(hostTools: Tool[], allowedTools: string[] | undefined): void {
  const excludedToolNames = hostTools
    .map((hostTool) => hostTool.name)
    .filter((hostToolName) => !isHostToolAllowed(hostToolName, allowedTools))
  if (excludedToolNames.length === 0) return

  const names = excludedToolNames.map((name) => `'${name}'`).join(', ')
  const subject = excludedToolNames.length === 1 ? `Host tool ${names} is` : `Host tools ${names} are`
  const example = excludedToolNames[0]!
  throw new ToolValidationError(
    `${subject} excluded by effective allowedTools. Add '@${example}', '@${example}/${example}', a matching namespace glob, or '*' to allowedTools.`
  )
}

/**
 * Tests a host tool name against the Harness allowlist namespace rules.
 *
 * @internal
 * @param hostToolName - Local host tool name
 * @param allowedTools - Effective Harness allowlist, or undefined for unrestricted access
 * @returns Whether the host tool is admitted
 */
export function isHostToolAllowed(hostToolName: string, allowedTools: string[] | undefined): boolean {
  if (allowedTools === undefined || allowedTools.includes('*')) return true

  return allowedTools.some((pattern) => {
    const separatorIndex = pattern.indexOf('/')
    const isNamespaced = pattern.startsWith('@')
    const namespacePattern = isNamespaced
      ? pattern.slice(1, separatorIndex === -1 ? undefined : separatorIndex)
      : 'builtin'
    const toolPattern = isNamespaced ? (separatorIndex === -1 ? '*' : pattern.slice(separatorIndex + 1)) : pattern
    return matchesFnmatchcase(hostToolName, namespacePattern) && matchesFnmatchcase(hostToolName, toolPattern)
  })
}

/** Verifies that a local handler implements the contract of a deployed inline function. */
function assertCompatibleInlineFunction(deployedTool: HarnessTool, hostTool: HarnessTool & { name: string }): void {
  const name = hostTool.name
  if (deployedTool.type !== 'inline_function') {
    throw new ToolValidationError(
      `Host tool '${name}' conflicts with deployed tool type '${String(deployedTool.type)}'. A local handler can bind only to a deployed inline_function.`
    )
  }

  const deployedConfig = getInlineFunctionConfig(deployedTool)
  const hostConfig = getInlineFunctionConfig(hostTool)
  if (deployedConfig === undefined) {
    throw new ToolValidationError(`Deployed inline function '${name}' has no inline-function configuration.`)
  }
  if (hostConfig === undefined) {
    throw new ToolValidationError(`Host tool '${name}' has no inline-function configuration.`)
  }

  const mismatches: string[] = []
  if (deployedConfig.description !== hostConfig.description) mismatches.push('description')
  if (!areStructurallyEqual(deployedConfig.inputSchema, hostConfig.inputSchema)) mismatches.push('input schema')
  if (mismatches.length === 0) return

  const difference = mismatches.length === 1 ? `its ${mismatches[0]} differs` : `its ${mismatches.join(' and ')} differ`
  throw new ToolValidationError(
    `Host tool '${name}' cannot bind to the deployed inline function because ${difference}. Make the local tool definition match the deployed inline function.`
  )
}

/** Extracts the inline-function member from a Harness tool configuration union. */
function getInlineFunctionConfig(tool: HarnessTool):
  | {
      description: string | undefined
      inputSchema: unknown
    }
  | undefined {
  if (tool.config === undefined || !('inlineFunction' in tool.config)) return undefined
  return tool.config.inlineFunction
}

/** Compares JSON-like values while ignoring object-property order. */
function areStructurallyEqual(left: unknown, right: unknown): boolean {
  if (Object.is(left, right)) return true
  if (Array.isArray(left) || Array.isArray(right)) {
    return (
      Array.isArray(left) &&
      Array.isArray(right) &&
      left.length === right.length &&
      left.every((value, index) => areStructurallyEqual(value, right[index]))
    )
  }
  if (!isRecord(left) || !isRecord(right)) return false

  const leftKeys = Object.keys(left).sort()
  const rightKeys = Object.keys(right).sort()
  return (
    leftKeys.length === rightKeys.length &&
    leftKeys.every((key, index) => key === rightKeys[index] && areStructurallyEqual(left[key], right[key]))
  )
}

/** Narrows plain JSON objects used by Harness input schemas. */
function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

/** Implements Python fnmatchcase semantics used by the Harness runtime. */
function matchesFnmatchcase(value: string, pattern: string): boolean {
  const tokens = tokenizeFnmatchPattern(pattern)
  let reachable = new Set([0])

  for (const character of value) {
    reachable = starClosure(reachable, tokens)
    const next = new Set<number>()
    for (const tokenIndex of reachable) {
      const token = tokens[tokenIndex]
      if (token?.kind === 'star') {
        next.add(tokenIndex)
      } else if (token !== undefined && tokenMatches(token, character)) {
        next.add(tokenIndex + 1)
      }
    }
    reachable = next
  }

  return starClosure(reachable, tokens).has(tokens.length)
}

type FnmatchToken =
  | { kind: 'star' }
  | { kind: 'any' }
  | { kind: 'literal'; value: string }
  | { kind: 'class'; negated: boolean; literals: Set<string>; ranges: [string, string][] }
  | { kind: 'never' }

/** Converts a Python-style shell pattern into linear matching tokens. */
function tokenizeFnmatchPattern(pattern: string): FnmatchToken[] {
  const tokens: FnmatchToken[] = []
  for (let index = 0; index < pattern.length; index++) {
    const character = pattern[index]!
    if (character === '*') {
      if (tokens.at(-1)?.kind !== 'star') tokens.push({ kind: 'star' })
      continue
    }
    if (character === '?') {
      tokens.push({ kind: 'any' })
      continue
    }
    if (character !== '[') {
      tokens.push({ kind: 'literal', value: character })
      continue
    }

    const parsed = parseCharacterClass(pattern, index)
    if (parsed === undefined) {
      tokens.push({ kind: 'literal', value: '[' })
    } else {
      tokens.push(parsed.token)
      index = parsed.endIndex
    }
  }
  return tokens
}

/** Parses the bracket expressions accepted by Python fnmatchcase. */
function parseCharacterClass(
  pattern: string,
  startIndex: number
): { token: FnmatchToken; endIndex: number } | undefined {
  let endIndex = startIndex + 1
  if (pattern[endIndex] === '!') endIndex++
  if (pattern[endIndex] === ']') endIndex++
  while (endIndex < pattern.length && pattern[endIndex] !== ']') endIndex++
  if (endIndex >= pattern.length) return undefined

  let content = pattern.slice(startIndex + 1, endIndex)
  const negated = content.startsWith('!')
  if (negated) content = content.slice(1)
  if (content.length === 0) {
    return { token: negated ? { kind: 'any' } : { kind: 'never' }, endIndex }
  }

  const literals = new Set<string>()
  const ranges: [string, string][] = []
  for (let index = 0; index < content.length; index++) {
    const character = content[index]!
    if (index + 2 < content.length && content[index + 1] === '-') {
      const rangeEnd = content[index + 2]!
      if (character <= rangeEnd) ranges.push([character, rangeEnd])
      index += 2
    } else {
      literals.add(character)
    }
  }

  if (literals.size === 0 && ranges.length === 0) {
    return { token: negated ? { kind: 'any' } : { kind: 'never' }, endIndex }
  }
  return { token: { kind: 'class', negated, literals, ranges }, endIndex }
}

/** Expands pattern positions reachable through zero-length star matches. */
function starClosure(indexes: Set<number>, tokens: FnmatchToken[]): Set<number> {
  const result = new Set(indexes)
  for (const index of result) {
    if (tokens[index]?.kind === 'star') result.add(index + 1)
  }
  return result
}

/** Tests one non-star token against one input character. */
function tokenMatches(token: Exclude<FnmatchToken, { kind: 'star' }>, character: string): boolean {
  switch (token.kind) {
    case 'any':
      return true
    case 'literal':
      return token.value === character
    case 'class': {
      const included =
        token.literals.has(character) ||
        token.ranges.some(([rangeStart, rangeEnd]) => rangeStart <= character && character <= rangeEnd)
      return token.negated ? !included : included
    }
    case 'never':
      return false
  }
}
