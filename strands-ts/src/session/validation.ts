import type { Scope } from './types.js'

/**
 * Validates that an identifier contains only allowed characters.
 * Allowed characters: lowercase letters (a-z), numbers (0-9), hyphens (-), and underscores (_)
 *
 * @param id - The identifier to validate
 * @returns The validated identifier
 * @throws Error if identifier contains invalid characters
 */
export function validateIdentifier(id: string): string {
  const validPattern = /^[a-z0-9_-]+$/
  if (!validPattern.test(id)) {
    throw new Error(`Identifier '${id}' can only contain lowercase letters, numbers, hyphens, and underscores`)
  }
  return id
}

/**
 * The set of recognized snapshot scopes. Derived from the `Scope` union so adding a new
 * scope forces a compile error here until the allowlist is updated.
 * Used as an allowlist so a scope value can only ever name a known storage subtree.
 */
const VALID_SCOPES = { agent: true, multiAgent: true } satisfies Record<Scope, true>

/**
 * Validates that a scope is one of the recognized values.
 * `Scope` is a compile-time union, but this guards the storage layer against any value
 * that bypasses the type system before it is used to build a file path or object key.
 *
 * @param scope - The scope to validate
 * @returns The validated scope
 * @throws Error if scope is not a recognized value
 */
export function validateScope(scope: string): string {
  if (!(scope in VALID_SCOPES)) {
    throw new Error(`Scope '${scope}' is not a recognized session scope`)
  }
  return scope
}

/**
 * Validates that a string is a UUID v7.
 *
 * @param id - The string to validate
 * @throws Error if the string is not a valid UUID v7
 */
export function validateUuidV7(id: string): void {
  const uuidV7Pattern = /^[0-9a-f]{8}-[0-9a-f]{4}-7[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
  if (!uuidV7Pattern.test(id)) {
    throw new Error(`'${id}' is not a valid UUID v7 snapshot ID`)
  }
}
