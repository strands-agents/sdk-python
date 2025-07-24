import base64
import os
import threading
import time
from typing import List, Literal

import pytest
from mcp import StdioServerParameters, stdio_client
from mcp.client.sse import sse_client
from mcp.client.streamable_http import streamablehttp_client
from mcp.types import ImageContent as MCPImageContent

from strands import Agent
from strands.tools.mcp.mcp_client import MCPClient
from strands.tools.mcp.mcp_types import MCPTransport
from strands.types.content import Message
from strands.types.tools import ToolUse


def start_comprehensive_mcp_server(transport: Literal["sse", "streamable-http"], port=int):
    """
    Initialize and start a comprehensive MCP server for integration testing.

    This function creates a FastMCP server instance that provides tools, prompts,
    and resources all in one server for comprehensive testing. The server uses
    Server-Sent Events (SSE) or streamable HTTP transport for communication.
    """
    from mcp.server import FastMCP

    mcp = FastMCP("Comprehensive MCP Server", port=port)

    # Tools
    @mcp.tool(description="Calculator tool which performs calculations")
    def calculator(x: int, y: int) -> int:
        return x + y

    @mcp.tool(description="Generates a custom image")
    def generate_custom_image() -> MCPImageContent:
        try:
            with open("tests_integ/yellow.png", "rb") as image_file:
                encoded_image = base64.b64encode(image_file.read())
                return MCPImageContent(type="image", data=encoded_image, mimeType="image/png")
        except Exception as e:
            print("Error while generating custom image: {}".format(e))

    # Prompts
    @mcp.prompt(description="A greeting prompt template")
    def greeting_prompt(name: str = "World") -> str:
        return f"Hello, {name}! How are you today?"

    @mcp.prompt(description="A math problem prompt template")
    def math_prompt(operation: str = "addition", difficulty: str = "easy") -> str:
        return f"Create a {difficulty} {operation} math problem and solve it step by step."

    # Resources (if supported by FastMCP - this is a placeholder for when resources are implemented)
    # @mcp.resource(description="A sample text resource")
    # def sample_resource() -> str:
    #     return "This is a sample resource content"

    mcp.run(transport=transport)


def test_mcp_client():
    """
    Comprehensive test for MCP client functionality including tools, prompts, and resources.

    Test should yield output similar to the following for tools:
    {'role': 'user', 'content': [{'text': 'add 1 and 2, then echo the result back to me'}]}
    {'role': 'assistant', 'content': [{'text': "I'll help you add 1 and 2 and then echo the result back to you.\n\nFirst, I'll calculate 1 + 2:"}, {'toolUse': {'toolUseId': 'tooluse_17ptaKUxQB20ySZxwgiI_w', 'name': 'calculator', 'input': {'x': 1, 'y': 2}}}]}
    {'role': 'user', 'content': [{'toolResult': {'status': 'success', 'toolUseId': 'tooluse_17ptaKUxQB20ySZxwgiI_w', 'content': [{'text': '3'}]}}]}
    {'role': 'assistant', 'content': [{'text': "\n\nNow I'll echo the result back to you:"}, {'toolUse': {'toolUseId': 'tooluse_GlOc5SN8TE6ti8jVZJMBOg', 'name': 'echo', 'input': {'to_echo': '3'}}}]}
    {'role': 'user', 'content': [{'toolResult': {'status': 'success', 'toolUseId': 'tooluse_GlOc5SN8TE6ti8jVZJMBOg', 'content': [{'text': '3'}]}}]}
    {'role': 'assistant', 'content': [{'text': '\n\nThe result of adding 1 and 2 is 3.'}]}
    """  # noqa: E501

    # Start comprehensive server with tools, prompts, and resources
    server_thread = threading.Thread(
        target=start_comprehensive_mcp_server, kwargs={"transport": "sse", "port": 8000}, daemon=True
    )
    server_thread.start()
    time.sleep(2)  # wait for server to startup completely

    sse_mcp_client = MCPClient(lambda: sse_client("http://127.0.0.1:8000/sse"))
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/echo_server.py"]))
    )

    with sse_mcp_client, stdio_mcp_client:
        # Test Tools functionality
        sse_tools = sse_mcp_client.list_tools_sync()
        stdio_tools = stdio_mcp_client.list_tools_sync()
        all_tools = sse_tools + stdio_tools

        agent = Agent(tools=all_tools)
        agent("add 1 and 2, then echo the result back to me")

        tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
        assert any([block["name"] == "echo" for block in tool_use_content_blocks])
        assert any([block["name"] == "calculator" for block in tool_use_content_blocks])

        # Test image generation tool
        image_prompt = """
        Generate a custom image, then tell me if the image is red, blue, yellow, pink, orange, or green. 
        RESPOND ONLY WITH THE COLOR
        """
        assert any(
            [
                "yellow".casefold() in block["text"].casefold()
                for block in agent(image_prompt).message["content"]
                if "text" in block
            ]
        )

        # Test Prompts functionality
        prompts_result = sse_mcp_client.list_prompts_sync()
        assert len(prompts_result.prompts) >= 2  # We expect at least greeting and math prompts

        prompt_names = [prompt.name for prompt in prompts_result.prompts]
        assert "greeting_prompt" in prompt_names
        assert "math_prompt" in prompt_names

        # Test get_prompt_sync with greeting prompt
        greeting_result = sse_mcp_client.get_prompt_sync("greeting_prompt", {"name": "Alice"})
        assert len(greeting_result.messages) > 0
        prompt_text = greeting_result.messages[0].content.text
        assert "Hello, Alice!" in prompt_text
        assert "How are you today?" in prompt_text

        # Test get_prompt_sync with math prompt
        math_result = sse_mcp_client.get_prompt_sync(
            "math_prompt", {"operation": "multiplication", "difficulty": "medium"}
        )
        assert len(math_result.messages) > 0
        math_text = math_result.messages[0].content.text
        assert "multiplication" in math_text
        assert "medium" in math_text
        assert "step by step" in math_text

        # Test pagination support for prompts
        prompts_with_token = sse_mcp_client.list_prompts_sync(pagination_token=None)
        assert len(prompts_with_token.prompts) >= 0

        # Test pagination support for tools (existing functionality)
        tools_with_token = sse_mcp_client.list_tools_sync(pagination_token=None)
        assert len(tools_with_token) >= 0

        # TODO: Add resources testing when resources are implemented
        # resources_result = sse_mcp_client.list_resources_sync()
        # assert len(resources_result.resources) >= 0


def test_can_reuse_mcp_client():
    stdio_mcp_client = MCPClient(
        lambda: stdio_client(StdioServerParameters(command="python", args=["tests_integ/echo_server.py"]))
    )
    with stdio_mcp_client:
        stdio_mcp_client.list_tools_sync()
        pass
    with stdio_mcp_client:
        agent = Agent(tools=stdio_mcp_client.list_tools_sync())
        agent("echo the following to me <to_echo>DOG</to_echo>")

        tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
        assert any([block["name"] == "echo" for block in tool_use_content_blocks])


@pytest.mark.skipif(
    condition=os.environ.get("GITHUB_ACTIONS") == "true",
    reason="streamable transport is failing in GitHub actions, debugging if linux compatibility issue",
)
def test_streamable_http_mcp_client():
    """Test comprehensive MCP client with streamable HTTP transport."""
    server_thread = threading.Thread(
        target=start_comprehensive_mcp_server, kwargs={"transport": "streamable-http", "port": 8001}, daemon=True
    )
    server_thread.start()
    time.sleep(2)  # wait for server to startup completely

    def transport_callback() -> MCPTransport:
        return streamablehttp_client(url="http://127.0.0.1:8001/mcp")

    sse_mcp_client = MCPClient(transport_callback)
    with sse_mcp_client:
        # Test tools
        agent = Agent(tools=sse_mcp_client.list_tools_sync())
        agent("add 1 and 2 using a calculator")

        tool_use_content_blocks = _messages_to_content_blocks(agent.messages)
        assert any([block["name"] == "calculator" for block in tool_use_content_blocks])

        # Test prompts
        prompts_result = sse_mcp_client.list_prompts_sync()
        assert len(prompts_result.prompts) >= 2

        greeting_result = sse_mcp_client.get_prompt_sync("greeting_prompt", {"name": "Charlie"})
        assert len(greeting_result.messages) > 0
        prompt_text = greeting_result.messages[0].content.text
        assert "Hello, Charlie!" in prompt_text


def _messages_to_content_blocks(messages: List[Message]) -> List[ToolUse]:
    return [block["toolUse"] for message in messages for block in message["content"] if "toolUse" in block]


@pytest.mark.asyncio
async def test_mcp_client_async():
    """Test MCP client async functionality with comprehensive server (tools, prompts, resources)."""
    server_thread = threading.Thread(
        target=start_comprehensive_mcp_server, kwargs={"transport": "sse", "port": 8005}, daemon=True
    )
    server_thread.start()
    time.sleep(2)  # wait for server to startup completely

    sse_mcp_client = MCPClient(lambda: sse_client("http://127.0.0.1:8005/sse"))

    with sse_mcp_client:
        # Test async prompts functionality
        prompts_result = await sse_mcp_client.list_prompts_async()
        assert len(prompts_result.prompts) >= 2

        prompt_names = [prompt.name for prompt in prompts_result.prompts]
        assert "greeting_prompt" in prompt_names
        assert "math_prompt" in prompt_names

        # Test get_prompt_async
        greeting_result = await sse_mcp_client.get_prompt_async("greeting_prompt", {"name": "Bob"})
        assert len(greeting_result.messages) > 0
        prompt_text = greeting_result.messages[0].content.text
        assert "Hello, Bob!" in prompt_text
        assert "How are you today?" in prompt_text

        # Test async pagination for prompts
        prompts_with_pagination = await sse_mcp_client.list_prompts_async(pagination_token="test_token")
        assert len(prompts_with_pagination.prompts) >= 0

        # Test async tools functionality (existing)
        tools_result = sse_mcp_client.list_tools_sync()
        assert len(tools_result) >= 2  # calculator and generate_custom_image

        tool_names = [tool.tool_name for tool in tools_result]
        assert "calculator" in tool_names
        assert "generate_custom_image" in tool_names
