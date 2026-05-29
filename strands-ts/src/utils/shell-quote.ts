/**
 * Shell-escape a string for safe inclusion in a shell command.
 *
 * Wraps the value in single quotes and escapes any embedded single quotes
 * using the '\'' pattern. Single quotes disable all shell expansion
 * (variables, backticks, globbing), making this safe against injection.
 *
 * @param value - The string to escape.
 * @returns The shell-escaped string wrapped in single quotes.
 */
export function shellQuote(value: string): string {
  return "'" + value.replace(/'/g, "'\\''") + "'"
}
