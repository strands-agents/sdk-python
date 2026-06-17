// @ts-nocheck
// Imports for the Sandbox overview page snippets.

// --8<-- [start:configuring_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:configuring_imports]

// --8<-- [start:host_default_imports]
import { Agent } from '@strands-agents/sdk'
// --8<-- [end:host_default_imports]

// --8<-- [start:override_tool_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
// --8<-- [end:override_tool_imports]

// --8<-- [start:docker_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:docker_imports]

// --8<-- [start:ssh_imports]
import { Agent } from '@strands-agents/sdk'
import { SshSandbox } from '@strands-agents/sdk/sandbox/ssh'
// --8<-- [end:ssh_imports]

// --8<-- [start:direct_use_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:direct_use_imports]

// --8<-- [start:streaming_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:streaming_imports]

// --8<-- [start:custom_posix_imports]
import { spawn } from 'node:child_process'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type { ExecuteOptions, ExecutionResult, StreamChunk } from '@strands-agents/sdk/sandbox'
// --8<-- [end:custom_posix_imports]

// --8<-- [start:custom_tools_imports]
import type { Tool } from '@strands-agents/sdk'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
import { makeFileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
// --8<-- [end:custom_tools_imports]
