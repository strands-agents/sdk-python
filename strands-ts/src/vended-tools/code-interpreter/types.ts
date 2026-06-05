/**
 * Input parameters for the code interpreter tool.
 */
export interface CodeInterpreterInput {
  /**
   * The source code to execute.
   */
  code: string

  /**
   * The interpreter to run the code with (e.g., `python3`, `node`, `ruby`).
   */
  language: string

  /**
   * Maximum execution time in seconds.
   */
  timeout?: number
}

/**
 * Result of code execution.
 */
export interface CodeInterpreterOutput {
  /**
   * Standard output from the interpreter.
   */
  stdout: string

  /**
   * Standard error from the interpreter.
   */
  stderr: string

  /**
   * Exit code returned by the interpreter.
   */
  exitCode: number

  /**
   * Binary artifacts produced by the execution (e.g., generated images,
   * charts). Each file's content is base64-encoded for JSON-serializable
   * transport. Empty for sandboxes that do not surface artifacts.
   */
  outputFiles: { name: string; content: string; mimeType: string }[]
}
