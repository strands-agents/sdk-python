// @ts-nocheck
// Imports for the Building a Custom Sandbox page snippets.

// --8<-- [start:custom_posix_imports]
import { spawn } from 'node:child_process'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type {
  ExecuteOptions,
  ExecutionResult,
  StreamChunk,
} from '@strands-agents/sdk/sandbox'
// --8<-- [end:custom_posix_imports]

// --8<-- [start:custom_tools_imports]
import type { Tool } from '@strands-agents/sdk'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
// --8<-- [end:custom_tools_imports]
