/**
 * Regex pattern for validating language/interpreter names.
 * Allows alphanumeric characters, dots, hyphens, and underscores.
 * Rejects path separators, spaces, and shell metacharacters to prevent injection.
 */
export const LANGUAGE_PATTERN = /^[a-zA-Z0-9._-]+$/
