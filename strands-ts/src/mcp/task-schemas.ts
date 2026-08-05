import { specTypeSchemas, type CallToolResult, type StandardSchemaV1Sync } from '@modelcontextprotocol/client'
import { z } from 'zod'

import type {
  McpCancelTaskResult,
  McpCreateTaskResult,
  McpGetTaskResult,
  McpTaskStatusNotification,
  McpTaskStatusNotificationParams,
  McpUpdateTaskResult,
} from './task-types.js'

const RESULT_META_SCHEMA = z.record(z.string(), z.unknown()).optional()
const ISO_TIMESTAMP_SCHEMA = z.iso.datetime({ offset: true })
const DURATION_MS_SCHEMA = z.number().finite().int().nonnegative().max(Number.MAX_SAFE_INTEGER)
const TASK_SHAPE = {
  taskId: z.string().min(1),
  statusMessage: z.string().optional(),
  createdAt: ISO_TIMESTAMP_SCHEMA,
  lastUpdatedAt: ISO_TIMESTAMP_SCHEMA,
  ttlMs: DURATION_MS_SCHEMA.nullable(),
  pollIntervalMs: DURATION_MS_SCHEMA.optional(),
}
const TASK_PAYLOAD_FIELDS = ['inputRequests', 'result', 'error'] as const
const ACKNOWLEDGEMENT_FIELDS = [
  'taskId',
  'status',
  'statusMessage',
  'createdAt',
  'lastUpdatedAt',
  'ttlMs',
  'pollIntervalMs',
  ...TASK_PAYLOAD_FIELDS,
] as const

function fromStandardSchema<TInput, TOutput>(
  schema: StandardSchemaV1Sync<TInput, TOutput>
): z.ZodType<TOutput, unknown> {
  return z.unknown().transform((value, context) => {
    const result = schema['~standard'].validate(value)
    if (result.issues !== undefined) {
      context.addIssue({
        code: 'custom',
        message: result.issues.map((issue) => issue.message).join('; ') || 'Invalid MCP protocol value',
      })
      return z.NEVER
    }
    return result.value
  })
}

function forbidFields<TSchema extends z.ZodObject>(schema: TSchema, fields: readonly string[]): TSchema {
  return schema.superRefine((value, context) => {
    for (const field of fields) {
      if (Object.hasOwn(value, field)) {
        context.addIssue({
          code: 'custom',
          path: [field],
          message: `"${field}" is not valid for task status "${String(value.status)}"`,
        })
      }
    }
  })
}

function forbidAcknowledgementFields<TSchema extends z.ZodObject>(schema: TSchema): TSchema {
  return schema.superRefine((value, context) => {
    for (const field of ACKNOWLEDGEMENT_FIELDS) {
      if (Object.hasOwn(value, field)) {
        context.addIssue({
          code: 'custom',
          path: [field],
          message: `"${field}" is not valid in an empty task acknowledgement`,
        })
      }
    }
  })
}

function validateTaskChronology<TSchema extends z.ZodObject>(schema: TSchema): TSchema {
  return schema.superRefine((value, context) => {
    if (Date.parse(value.lastUpdatedAt as string) < Date.parse(value.createdAt as string)) {
      context.addIssue({
        code: 'custom',
        path: ['lastUpdatedAt'],
        message: '"lastUpdatedAt" must not precede "createdAt"',
      })
    }
  })
}

/** Validates the status literals defined by the MCP tasks extension. @internal */
export const McpTaskStatusSchema = z.enum(['working', 'input_required', 'completed', 'failed', 'cancelled'])

/** Validates a nested MCP `CallToolResult`. @internal */
export const McpCallToolResultSchema = fromStandardSchema(specTypeSchemas.CallToolResult)

/** Validates a server-to-client request carried by `inputRequests`. @internal */
export const McpInputRequestSchema = z.union([
  fromStandardSchema(specTypeSchemas.CreateMessageRequest),
  fromStandardSchema(specTypeSchemas.ListRootsRequest),
  fromStandardSchema(specTypeSchemas.ElicitRequest),
])

/** Validates a client response carried by `inputResponses`. @internal */
export const McpInputResponseSchema = z.union([
  fromStandardSchema(specTypeSchemas.CreateMessageResult),
  fromStandardSchema(specTypeSchemas.ListRootsResult),
  fromStandardSchema(specTypeSchemas.ElicitResult),
])

/** Validates the keyed requests carried by an input-required task. @internal */
export const McpInputRequestsSchema = z.record(z.string(), McpInputRequestSchema)

/** Validates the keyed responses sent through `tasks/update`. @internal */
export const McpInputResponsesSchema = z.record(z.string(), McpInputResponseSchema)

/** Validates a JSON-RPC error stored by a failed task. @internal */
export const McpTaskErrorSchema = z.looseObject({
  code: z.number().int(),
  message: z.string(),
  data: z.unknown().optional(),
})

const taskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: McpTaskStatusSchema,
  })
)
const workingTaskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: z.literal('working'),
  })
)
const inputRequiredTaskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: z.literal('input_required'),
    inputRequests: McpInputRequestsSchema,
  })
)
const completedTaskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: z.literal('completed'),
    result: McpCallToolResultSchema,
  })
)
const failedTaskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: z.literal('failed'),
    error: McpTaskErrorSchema,
  })
)
const cancelledTaskObjectSchema = validateTaskChronology(
  z.looseObject({
    ...TASK_SHAPE,
    status: z.literal('cancelled'),
  })
)

/** Validates metadata shared by every task state. @internal */
export const McpTaskSchema = taskObjectSchema

/** Validates a working task and rejects status-specific payloads. @internal */
export const McpWorkingTaskSchema = forbidFields(workingTaskObjectSchema, TASK_PAYLOAD_FIELDS)

/** Validates an input-required task and its outstanding requests. @internal */
export const McpInputRequiredTaskSchema = forbidFields(inputRequiredTaskObjectSchema, ['result', 'error'])

/** Validates a completed task and its nested tool result. @internal */
export const McpCompletedTaskSchema = forbidFields(completedTaskObjectSchema, ['inputRequests', 'error'])

/** Validates a failed task and its JSON-RPC error. @internal */
export const McpFailedTaskSchema = forbidFields(failedTaskObjectSchema, ['inputRequests', 'result'])

/** Validates a cancelled task and rejects status-specific payloads. @internal */
export const McpCancelledTaskSchema = forbidFields(cancelledTaskObjectSchema, TASK_PAYLOAD_FIELDS)

/** Validates every status-specific shape returned by `tasks/get`. @internal */
export const McpDetailedTaskSchema = z.union([
  McpWorkingTaskSchema,
  McpInputRequiredTaskSchema,
  McpCompletedTaskSchema,
  McpFailedTaskSchema,
  McpCancelledTaskSchema,
])

/** Validates a task handle returned instead of an immediate tool result. @internal */
export const McpCreateTaskResultSchema = taskObjectSchema.safeExtend({
  resultType: z.literal('task'),
  _meta: RESULT_META_SCHEMA,
})

const getWorkingTaskResultSchema = forbidFields(
  workingTaskObjectSchema.safeExtend({
    resultType: z.literal('complete'),
    _meta: RESULT_META_SCHEMA,
  }),
  TASK_PAYLOAD_FIELDS
)
const getInputRequiredTaskResultSchema = forbidFields(
  inputRequiredTaskObjectSchema.safeExtend({
    resultType: z.literal('complete'),
    _meta: RESULT_META_SCHEMA,
  }),
  ['result', 'error']
)
const getCompletedTaskResultSchema = forbidFields(
  completedTaskObjectSchema.safeExtend({
    resultType: z.literal('complete'),
    _meta: RESULT_META_SCHEMA,
  }),
  ['inputRequests', 'error']
)
const getFailedTaskResultSchema = forbidFields(
  failedTaskObjectSchema.safeExtend({
    resultType: z.literal('complete'),
    _meta: RESULT_META_SCHEMA,
  }),
  ['inputRequests', 'result']
)
const getCancelledTaskResultSchema = forbidFields(
  cancelledTaskObjectSchema.safeExtend({
    resultType: z.literal('complete'),
    _meta: RESULT_META_SCHEMA,
  }),
  TASK_PAYLOAD_FIELDS
)

/** Validates all five status-specific `tasks/get` result shapes. @internal */
export const McpGetTaskResultSchema = z.union([
  getWorkingTaskResultSchema,
  getInputRequiredTaskResultSchema,
  getCompletedTaskResultSchema,
  getFailedTaskResultSchema,
  getCancelledTaskResultSchema,
])

const acknowledgementResultSchema = z.looseObject({
  resultType: z.literal('complete'),
  _meta: RESULT_META_SCHEMA,
})

/** Validates the empty acknowledgement returned by `tasks/update`. @internal */
export const McpUpdateTaskResultSchema = forbidAcknowledgementFields(acknowledgementResultSchema)

/** Validates the empty acknowledgement returned by `tasks/cancel`. @internal */
export const McpCancelTaskResultSchema = forbidAcknowledgementFields(acknowledgementResultSchema)

const notificationWorkingTaskSchema = forbidFields(
  workingTaskObjectSchema.safeExtend({ _meta: RESULT_META_SCHEMA }),
  TASK_PAYLOAD_FIELDS
)
const notificationInputRequiredTaskSchema = forbidFields(
  inputRequiredTaskObjectSchema.safeExtend({ _meta: RESULT_META_SCHEMA }),
  ['result', 'error']
)
const notificationCompletedTaskSchema = forbidFields(
  completedTaskObjectSchema.safeExtend({ _meta: RESULT_META_SCHEMA }),
  ['inputRequests', 'error']
)
const notificationFailedTaskSchema = forbidFields(failedTaskObjectSchema.safeExtend({ _meta: RESULT_META_SCHEMA }), [
  'inputRequests',
  'result',
])
const notificationCancelledTaskSchema = forbidFields(
  cancelledTaskObjectSchema.safeExtend({ _meta: RESULT_META_SCHEMA }),
  TASK_PAYLOAD_FIELDS
)

/** Validates complete task state carried by a task notification. @internal */
export const McpTaskStatusNotificationParamsSchema = z.union([
  notificationWorkingTaskSchema,
  notificationInputRequiredTaskSchema,
  notificationCompletedTaskSchema,
  notificationFailedTaskSchema,
  notificationCancelledTaskSchema,
])

/** Validates a complete `notifications/tasks` JSON-RPC notification. @internal */
export const McpTaskStatusNotificationSchema = z
  .object({
    jsonrpc: z.literal('2.0'),
    method: z.literal('notifications/tasks'),
    params: McpTaskStatusNotificationParamsSchema,
  })
  .strict()

/**
 * Parses an MCP task handle returned in place of an immediate result.
 *
 * @param value - Value to parse
 * @returns The validated task handle
 * @throws ZodError if the value is not a valid task handle
 * @internal
 */
export function parseMcpCreateTaskResult(value: unknown): McpCreateTaskResult {
  return McpCreateTaskResultSchema.parse(value) as McpCreateTaskResult
}

/**
 * Parses a status-specific `tasks/get` result.
 *
 * @param value - Value to parse
 * @returns The validated task result
 * @throws ZodError if the value is not a valid task result
 * @internal
 */
export function parseMcpGetTaskResult(value: unknown): McpGetTaskResult {
  return McpGetTaskResultSchema.parse(value) as McpGetTaskResult
}

/**
 * Parses a `tasks/update` acknowledgement.
 *
 * @param value - Value to parse
 * @returns The validated update acknowledgement
 * @throws ZodError if the value is not a valid update acknowledgement
 * @internal
 */
export function parseMcpUpdateTaskResult(value: unknown): McpUpdateTaskResult {
  return McpUpdateTaskResultSchema.parse(value) as McpUpdateTaskResult
}

/**
 * Parses a `tasks/cancel` acknowledgement.
 *
 * @param value - Value to parse
 * @returns The validated cancellation acknowledgement
 * @throws ZodError if the value is not a valid cancellation acknowledgement
 * @internal
 */
export function parseMcpCancelTaskResult(value: unknown): McpCancelTaskResult {
  return McpCancelTaskResultSchema.parse(value) as McpCancelTaskResult
}

/**
 * Parses task state carried by a `notifications/tasks` notification and normalizes it for task-state handling.
 *
 * @param value - Value to parse
 * @returns The validated task state with the completed-result discriminator used by `tasks/get`
 * @throws ZodError if the value does not contain valid task notification parameters
 * @internal
 */
export function parseMcpTaskStatusNotificationParams(value: unknown): McpGetTaskResult {
  const params = McpTaskStatusNotificationParamsSchema.parse(value)
  return { ...params, resultType: 'complete' } as McpGetTaskResult
}

/**
 * Checks whether a value is a nested MCP tool result.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid tool result
 * @internal
 */
export function isMcpCallToolResult(value: unknown): value is CallToolResult {
  return McpCallToolResultSchema.safeParse(value).success
}

/**
 * Checks whether a value is an MCP task handle.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid task handle
 * @internal
 */
export function isMcpCreateTaskResult(value: unknown): value is McpCreateTaskResult {
  return McpCreateTaskResultSchema.safeParse(value).success
}

/**
 * Checks whether a value is a status-specific `tasks/get` result.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid task result
 * @internal
 */
export function isMcpGetTaskResult(value: unknown): value is McpGetTaskResult {
  return McpGetTaskResultSchema.safeParse(value).success
}

/**
 * Checks whether a value is a `tasks/update` acknowledgement.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid update acknowledgement
 * @internal
 */
export function isMcpUpdateTaskResult(value: unknown): value is McpUpdateTaskResult {
  return McpUpdateTaskResultSchema.safeParse(value).success
}

/**
 * Checks whether a value is a `tasks/cancel` acknowledgement.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid cancellation acknowledgement
 * @internal
 */
export function isMcpCancelTaskResult(value: unknown): value is McpCancelTaskResult {
  return McpCancelTaskResultSchema.safeParse(value).success
}

/**
 * Checks whether a value is valid task notification parameters.
 *
 * @param value - Value to validate
 * @returns Whether the value contains valid task notification parameters
 * @internal
 */
export function isMcpTaskStatusNotificationParams(value: unknown): value is McpTaskStatusNotificationParams {
  return McpTaskStatusNotificationParamsSchema.safeParse(value).success
}

/**
 * Checks whether a value is a complete task status notification.
 *
 * @param value - Value to validate
 * @returns Whether the value is a valid task status notification
 * @internal
 */
export function isMcpTaskStatusNotification(value: unknown): value is McpTaskStatusNotification {
  return McpTaskStatusNotificationSchema.safeParse(value).success
}
