// @ts-nocheck
// Imports repeat across snippets so each rendered example is self-contained.

// --8<-- [start:session_imports]
import { randomUUID } from 'node:crypto'
import { AgentCoreHarnessAgent } from '@strands-agents/sdk/agentcore-harness'
// --8<-- [end:session_imports]

// --8<-- [start:stream_imports]
import { randomUUID } from 'node:crypto'
import {
  AgentCoreHarnessAgent,
  AgentCoreHarnessResultEvent,
  AgentCoreHarnessStreamUpdateEvent,
} from '@strands-agents/sdk/agentcore-harness'
// --8<-- [end:stream_imports]

// --8<-- [start:graph_imports]
import { randomUUID } from 'node:crypto'
import { Agent, Graph } from '@strands-agents/sdk'
import { AgentCoreHarnessAgent } from '@strands-agents/sdk/agentcore-harness'
// --8<-- [end:graph_imports]
