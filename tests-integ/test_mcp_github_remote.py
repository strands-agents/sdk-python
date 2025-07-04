import json
import os
from typing import Any

import httpx
import pytest

from mcp.client.streamable_http import streamablehttp_client
from strands import Agent
from strands.tools.mcp import MCPClient

REPO_OWNER = "strands-agents"
REPO_NAME = "sdk-python"

GITHUB_REMOTE_MCP_URL = "https://api.githubcopilot.com/mcp/"

HEADER_AUTHORIZATION = "Authorization"
MCP_GITHUB_PAT = os.getenv("MCP_GITHUB_PAT")

TIMEOUT = 300

SYSTEM_PROMPT = "you are a helpful assistant"


# extension of official mcp.shared._httpx_utils.create_mcp_http_client with Transport
def create_mcp_http_client_with_transport(
        headers: dict[str, str] | None = None,
        timeout: httpx.Timeout | None = None,
        auth: httpx.Auth | None = None,
        transport: httpx.AsyncBaseTransport | None = None
) -> httpx.AsyncClient:
    kwargs: dict[str, Any] = {
        "follow_redirects": True,
    }

    if timeout is None:
        kwargs["timeout"] = httpx.Timeout(30.0)
    else:
        kwargs["timeout"] = timeout

    if headers is not None:
        kwargs["headers"] = headers

    if auth is not None:
        kwargs["auth"] = auth

    if transport is not None:
        kwargs["transport"] = transport

    return httpx.AsyncClient(**kwargs)


@pytest.fixture(name="github_mcp_client", autouse=True)
def fixture_streamable_http_client() -> MCPClient:
    kwargs = {
        "url": GITHUB_REMOTE_MCP_URL,
        "headers": {HEADER_AUTHORIZATION: f"Bearer {MCP_GITHUB_PAT}"},
        "timeout": TIMEOUT
    }

    def streamable_http_client():
        return streamablehttp_client(**kwargs)

    return MCPClient(streamable_http_client)


def test_list_tools_sync(github_mcp_client):
    print()
    with github_mcp_client:
        tools = github_mcp_client.list_tools_sync()
        tool_names = [tool.tool_name for tool in tools]
        print(f"tool names({len(tool_names)}): {tool_names}")
        assert len(tool_names) > 30


def test_github_list_commits(github_mcp_client):
    print()
    with github_mcp_client:
        tools = github_mcp_client.list_tools_sync()
        agent = Agent(system_prompt=SYSTEM_PROMPT, tools=tools)
        result = agent.tool.list_commits(owner=REPO_OWNER, repo=REPO_NAME)
    assert result["status"] == "success"
    commit_list = json.loads(result["content"][0]["text"])
    print (f"commit_list: {json.dumps(commit_list,indent=4)}")
    for commit in commit_list:
        assert "commit" in commit
        assert "sha" in commit
        assert "author" in commit
        assert "committer" in commit

