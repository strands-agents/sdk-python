"""
UTCP Echo Server for Integration Testing

This module provides a FastAPI-based UTCP server that exposes echo tools
for testing the UTCP integration. It uses the current UTCP API with decorators.
"""

import asyncio

import uvicorn
from fastapi import FastAPI, Query
from utcp.data.utcp_manual import UtcpManual
from utcp.python_specific_tooling.tool_decorator import utcp_tool
from utcp_http import HttpCallTemplate

__version__ = "1.0.0"

app = FastAPI(title="UTCP Echo Server", version=__version__)


@app.get("/utcp", response_model=UtcpManual)
def get_utcp():
    """Return UTCP manual with available tools."""
    return UtcpManual.create_from_decorators(manual_version=__version__)


@utcp_tool(
    tool_call_template=HttpCallTemplate(name="echo_simple", url="http://127.0.0.1:8003/echo", http_method="POST")
)
@app.post("/echo")
def echo(message: str = Query(...)):
    """Echo back the input message."""
    return {"echo": message}


@utcp_tool(
    tool_call_template=HttpCallTemplate(name="echo_upper", url="http://127.0.0.1:8003/echo_upper", http_method="POST")
)
@app.post("/echo_upper")
def echo_upper(message: str = Query(...)):
    """Echo back the input message in uppercase."""
    return {"echo": message.upper()}


def start_echo_server(port: int = 8003):
    """Start the echo server."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)

    # Run in asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    start_echo_server()
