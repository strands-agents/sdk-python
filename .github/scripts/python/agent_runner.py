#!/usr/bin/env python3
"""
Strands GitHub Agent Runner
A portable agent runner for use in GitHub Actions across different repositories.
"""

import json
import os
import sys
from typing import Any

from strands import Agent
from strands.agent.conversation_manager import SlidingWindowConversationManager
from strands.session import S3SessionManager
from strands.models.bedrock import BedrockModel
from botocore.config import Config

from strands_tools import http_request, shell

# Import local GitHub tools we need
from github_tools import (
    add_issue_comment,
    create_issue,
    create_pull_request,
    get_issue,
    get_issue_comments,
    get_pull_request,
    get_pr_review_and_comments,
    list_issues,
    list_pull_requests,
    reply_to_review_comment,
    update_issue,
    update_pull_request,
)

# Import local tools we need
from handoff_to_user import handoff_to_user
from notebook import notebook
from str_replace_based_edit_tool import str_replace_based_edit_tool

# Strands configuration constants
STRANDS_MODEL_ID = "global.anthropic.claude-sonnet-4-5-20250929-v1:0"
STRANDS_MAX_TOKENS = 64000
STRANDS_BUDGET_TOKENS = 8000
STRANDS_REGION = "us-west-2"

# Default values for environment variables used only in this file
DEFAULT_SYSTEM_PROMPT = "You are an autonomous GitHub agent powered by Strands Agents SDK."

def _get_all_tools() -> list[Any]:
    return [
        # File editing
        str_replace_based_edit_tool,
        
        # System tools
        shell,
        http_request,
        
        # GitHub issue tools
        create_issue,
        get_issue,
        update_issue,
        list_issues,
        add_issue_comment,
        get_issue_comments,
        
        # GitHub PR tools
        create_pull_request,
        get_pull_request,
        update_pull_request,
        list_pull_requests,
        get_pr_review_and_comments,
        reply_to_review_comment,
        
        # Agent tools
        notebook,
        handoff_to_user,
    ]


def run_agent(query: str):
    """Run the agent with the provided query."""
    try:
        # Get tools and create model
        tools = _get_all_tools()
        
        # Create Bedrock model with inlined configuration
        additional_request_fields = {}
        additional_request_fields["anthropic_beta"] = ["interleaved-thinking-2025-05-14"]
        
        additional_request_fields["thinking"] = {
            "type": "enabled",
            "budget_tokens": STRANDS_BUDGET_TOKENS
        }
        
        model = BedrockModel(
            model_id=STRANDS_MODEL_ID,
            max_tokens=STRANDS_MAX_TOKENS,
            region_name=STRANDS_REGION,
            boto_client_config=Config(
                read_timeout=900,
                connect_timeout=900,
                retries={"max_attempts": 3, "mode": "adaptive"},
            ),
            additional_request_fields=additional_request_fields,
            cache_prompt="default",
            cache_tools="default",
        )
        system_prompt = os.getenv("INPUT_SYSTEM_PROMPT", DEFAULT_SYSTEM_PROMPT)
        session_id = os.getenv("SESSION_ID")
        s3_bucket = os.getenv("S3_SESSION_BUCKET")
        s3_prefix = os.getenv("GITHUB_REPOSITORY", "")

        if s3_bucket and session_id:
            print(f"🤖 Using session manager with session ID: {session_id}")
            session_manager = S3SessionManager(
                session_id=session_id,
                bucket=s3_bucket,
                prefix=s3_prefix,
            )
        else:
            raise ValueError("Both SESSION_ID and S3_SESSION_BUCKET must be set")

        # Create agent
        agent = Agent(
            model=model,
            system_prompt=system_prompt,
            tools=tools,
            session_manager=session_manager,
        )

        print("Processing user query...")
        result = agent(query)

        print(f"\n\nAgent Result 🤖\nStop Reason: {result.stop_reason}\nMessage: {json.dumps(result.message, indent=2)}")
    except Exception as e:
        error_msg = f"❌ Agent execution failed: {e}"
        print(error_msg)
        raise e


def main() -> None:
    """Main entry point for the agent runner."""
    try:
        # Read task from command line arguments
        if len(sys.argv) < 2:
            raise ValueError("Task argument is required")

        task = " ".join(sys.argv[1:])
        if not task.strip():
            raise ValueError("Task cannot be empty")
        print(f"🤖 Running agent with task: {task}")

        run_agent(task)

    except Exception as e:
        error_msg = f"Fatal error: {e}"
        print(error_msg)

        sys.exit(1)


if __name__ == "__main__":
    main()