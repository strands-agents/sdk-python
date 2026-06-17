// @ts-nocheck

// --8<-- [start:basic_usage_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:basic_usage_imports]

// --8<-- [start:docker_sandbox_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:docker_sandbox_imports]

// --8<-- [start:ssh_sandbox_imports]
import { Agent } from '@strands-agents/sdk'
import { SshSandbox } from '@strands-agents/sdk/sandbox/ssh'
// --8<-- [end:ssh_sandbox_imports]

// --8<-- [start:tool_context_imports]
import { tool } from '@strands-agents/sdk'
import { z } from 'zod'
// --8<-- [end:tool_context_imports]

// --8<-- [start:custom_sandbox_imports]
import { spawn } from 'node:child_process'
import { PosixShellSandbox } from '@strands-agents/sdk/sandbox'
import type { ExecuteOptions, StreamChunk, ExecutionResult } from '@strands-agents/sdk/sandbox'
// --8<-- [end:custom_sandbox_imports]

// --8<-- [start:programmatic_access_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:programmatic_access_imports]

// --8<-- [start:streaming_imports]
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
// --8<-- [end:streaming_imports]

// --8<-- [start:tool_override_imports]
import { Agent } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { makeBash } from '@strands-agents/sdk/vended-tools/bash'
// --8<-- [end:tool_override_imports]

// --8<-- [start:custom_tool_imports]
import { Agent, tool } from '@strands-agents/sdk'
import { DockerSandbox } from '@strands-agents/sdk/sandbox/docker'
import { z } from 'zod'
// --8<-- [end:custom_tool_imports]
