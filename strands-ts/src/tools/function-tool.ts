import { createErrorResult, createSuccessResult, Tool } from './tool.js'
import type { InvokableTool, ToolContext } from './tool.js'
import { ToolStreamEvent } from './tool.js'
import type { ToolSpec } from './types.js'
import type { JSONSchema, JSONValue } from '../types/json.js'
import { ToolResultBlock } from '../types/messages.js'
import { InterruptError } from '../interrupt.js'

/**
 * Callback function for FunctionTool implementations.
 * The callback can return values in multiple ways, and FunctionTool handles the conversion to ToolResultBlock.
 *
 * @param input - The input parameters conforming to the tool's inputSchema
 * @param toolContext - The tool execution context with invocation state
 * @returns Can return:
 *   - AsyncGenerator: Each yielded value becomes a ToolStreamEvent, final value wrapped in ToolResultBlock
 *   - Promise: Resolved value is wrapped in ToolResultBlock
 *   - Synchronous value: Value is wrapped in ToolResultBlock
 *   - If an error is thrown, it's handled and returned as an error ToolResultBlock
 *
 * @example
 * ```typescript
 * // Async generator example
 * async function* calculator(input: unknown, context: ToolContext) {
 *   yield 'Calculating...'
 *   const result = input.a + input.b
 *   yield `Result: ${result}`
 *   return result
 * }
 *
 * // Promise example
 * async function fetchData(input: unknown, context: ToolContext) {
 *   const response = await fetch(input.url)
 *   return await response.json()
 * }
 *
 * // Synchronous example
 * function multiply(input: unknown, context: ToolContext) {
 *   return input.a * input.b
 * }
 * ```
 */
export type FunctionToolCallback = (
  input: unknown,
  toolContext: ToolContext
) => AsyncGenerator<JSONValue, JSONValue, never> | Promise<JSONValue> | JSONValue

/**
 * Configuration options for creating a FunctionTool.
 */
export interface FunctionToolConfig {
  /** The unique name of the tool */
  name: string
  /** Human-readable description of the tool's purpose */
  description: string
  /** JSON Schema defining the expected input structure. If omitted, defaults to an empty object schema. */
  inputSchema?: JSONSchema
  /** Function that implements the tool logic */
  callback: FunctionToolCallback
}

/**
 * A Tool implementation that wraps a callback function and handles all ToolResultBlock conversion.
 *
 * FunctionTool allows creating tools from existing functions without needing to manually
 * handle ToolResultBlock formatting or error handling. It supports multiple callback patterns:
 * - Async generators for streaming responses
 * - Promises for async operations
 * - Synchronous functions for immediate results
 *
 * All return values are automatically wrapped in ToolResultBlock, and errors are caught and
 * returned as error ToolResultBlocks.
 *
 * @example
 * ```typescript
 * // Create a tool with streaming
 * const streamingTool = new FunctionTool({
 *   name: 'processor',
 *   description: 'Processes data with progress updates',
 *   inputSchema: { type: 'object', properties: { data: { type: 'string' } } },
 *   callback: async function* (input: any) {
 *     yield 'Starting processing...'
 *     // Do some work
 *     yield 'Halfway done...'
 *     // More work
 *     return 'Processing complete!'
 *   }
 * })
 * ```
 */
export class FunctionTool extends Tool implements InvokableTool<unknown, JSONValue> {
  /**
   * The unique name of the tool.
   */
  readonly name: string

  /**
   * Human-readable description of what the tool does.
   */
  readonly description: string

  /**
   * OpenAPI JSON specification for the tool.
   */
  readonly toolSpec: ToolSpec

  /**
   * The callback function that implements the tool's logic.
   */
  private readonly _callback: FunctionToolCallback

  /**
   * Creates a new FunctionTool instance.
   *
   * @param config - Configuration object for the tool
   *
   * @example
   * ```typescript
   * // Tool with input schema
   * const greetTool = new FunctionTool({
   *   name: 'greeter',
   *   description: 'Greets a person by name',
   *   inputSchema: {
   *     type: 'object',
   *     properties: { name: { type: 'string' } },
   *     required: ['name']
   *   },
   *   callback: (input: any) => `Hello, ${input.name}!`
   * })
   *
   * // Tool without input (no parameters)
   * const statusTool = new FunctionTool({
   *   name: 'getStatus',
   *   description: 'Gets system status',
   *   callback: () => ({ status: 'operational' })
   * })
   * ```
   */
  constructor(config: FunctionToolConfig) {
    super()
    this.name = config.name
    this.description = config.description

    // Use provided schema or default empty object schema
    const inputSchema = config.inputSchema ?? {
      type: 'object',
      properties: {},
      additionalProperties: false,
    }

    this.toolSpec = {
      name: config.name,
      description: config.description,
      inputSchema: inputSchema,
    }
    this._callback = config.callback
  }

  /**
   * Executes the tool with streaming support.
   * Handles all callback patterns (async generator, promise, sync) and converts results to ToolResultBlock.
   *
   * @param toolContext - Context information including the tool use request and invocation state
   * @returns Async generator that yields ToolStreamEvents and returns a ToolResultBlock
   */
  async *stream(toolContext: ToolContext): AsyncGenerator<ToolStreamEvent, ToolResultBlock, unknown> {
    const { toolUse } = toolContext

    try {
      const result = this._callback(toolUse.input, toolContext)

      // Check if result is an async generator
      if (result && typeof result === 'object' && Symbol.asyncIterator in result) {
        // Handle async generator: yield each value as ToolStreamEvent, wrap final value in ToolResultBlock
        const generator = result as AsyncGenerator<unknown, unknown, unknown>

        // Iterate through all yielded values
        let iterResult = await generator.next()

        while (!iterResult.done) {
          // Each yielded value becomes a ToolStreamEvent
          yield new ToolStreamEvent({
            data: iterResult.value,
          })
          iterResult = await generator.next()
        }

        // The generator's return value (when done = true) is wrapped in ToolResultBlock
        return createSuccessResult(iterResult.value, toolUse.toolUseId)
      } else if (result instanceof Promise) {
        // Handle promise: await and wrap in ToolResultBlock
        const value = await result
        return createSuccessResult(value, toolUse.toolUseId)
      } else {
        // Handle synchronous value: wrap in ToolResultBlock
        return createSuccessResult(result, toolUse.toolUseId)
      }
    } catch (error) {
      // Re-throw InterruptError to allow interrupt handling in agent loop
      if (error instanceof InterruptError) {
        throw error
      }
      // Handle any other errors and yield as error ToolResultBlock
      return createErrorResult(error, toolUse.toolUseId)
    }
  }

  /**
   * Invokes the tool directly with raw input and returns the unwrapped result.
   *
   * Unlike stream(), this method:
   * - Returns the raw result (not wrapped in ToolResult)
   * - Consumes async generators and returns the generator's return value
   * - Lets errors throw naturally (not wrapped in error ToolResult)
   *
   * @param input - The input parameters for the tool
   * @param context - Optional tool execution context
   * @returns The unwrapped result
   */
  async invoke(input: unknown, context?: ToolContext): Promise<JSONValue> {
    const result = this._callback(input, context as ToolContext)

    if (result && typeof result === 'object' && Symbol.asyncIterator in result) {
      const generator = result as AsyncGenerator<unknown, JSONValue, undefined>
      let iterResult = await generator.next()
      while (!iterResult.done) {
        iterResult = await generator.next()
      }
      return iterResult.value
    }

    return (await result) as JSONValue
  }
}
