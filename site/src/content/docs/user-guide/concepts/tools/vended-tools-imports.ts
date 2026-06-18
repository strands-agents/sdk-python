// @ts-nocheck
// This file contains import snippets used in documentation examples.
// Each snippet is a standalone import block for a specific tool.
// @ts-nocheck is used because imports are intentionally repeated across snippets
// for documentation clarity — each snippet shows the complete imports needed.

// --8<-- [start:bash_import]
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
// --8<-- [end:bash_import]

// --8<-- [start:file_editor_import]
import { Agent } from '@strands-agents/sdk'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
// --8<-- [end:file_editor_import]

// --8<-- [start:http_request_import]
import { Agent } from '@strands-agents/sdk'
import { httpRequest } from '@strands-agents/sdk/vended-tools/http-request'
// --8<-- [end:http_request_import]

// --8<-- [start:notebook_import]
import { Agent } from '@strands-agents/sdk'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'
// --8<-- [end:notebook_import]

// --8<-- [start:notebook_persistence_import]
import { Agent, SessionManager, FileStorage } from '@strands-agents/sdk'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'
// --8<-- [end:notebook_persistence_import]

// --8<-- [start:combined_import]
import { Agent } from '@strands-agents/sdk'
import { bash } from '@strands-agents/sdk/vended-tools/bash'
import { fileEditor } from '@strands-agents/sdk/vended-tools/file-editor'
import { notebook } from '@strands-agents/sdk/vended-tools/notebook'
// --8<-- [end:combined_import]
