"""
UTCP Calculator Server for Integration Testing

This module provides a FastAPI-based UTCP server that exposes calculator tools
for testing the UTCP integration. It uses the current UTCP API with decorators.
"""

import asyncio

import uvicorn
from fastapi import FastAPI, Query
from utcp.data.utcp_manual import UtcpManual
from utcp.python_specific_tooling.tool_decorator import utcp_tool
from utcp_http import HttpCallTemplate

__version__ = "1.0.0"

app = FastAPI(title="UTCP Calculator Server", version=__version__)


@app.get("/utcp", response_model=UtcpManual)
def get_utcp():
    """Return UTCP manual with available tools."""
    return UtcpManual.create_from_decorators(manual_version=__version__)


@utcp_tool(
    tool_call_template=HttpCallTemplate(name="calculator_add", url="http://127.0.0.1:8002/add", http_method="POST")
)
@app.post("/add")
def add(a: float = Query(...), b: float = Query(...)):
    """Add two numbers together."""
    return {"result": a + b}


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="calculator_subtract", url="http://127.0.0.1:8002/subtract", http_method="POST"
    )
)
@app.post("/subtract")
def subtract(a: float = Query(...), b: float = Query(...)):
    """Subtract second number from first number."""
    return {"result": a - b}


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="calculator_multiply", url="http://127.0.0.1:8002/multiply", http_method="POST"
    )
)
@app.post("/multiply")
def multiply(a: float = Query(...), b: float = Query(...)):
    """Multiply two numbers together."""
    return {"result": a * b}


@utcp_tool(
    tool_call_template=HttpCallTemplate(
        name="calculator_divide", url="http://127.0.0.1:8002/divide", http_method="POST"
    )
)
@app.post("/divide")
def divide(a: float = Query(...), b: float = Query(...)):
    """Divide first number by second number."""
    if b == 0:
        raise ValueError("Division by zero")
    return {"result": a / b}


def start_calculator_server(port: int = 8002):
    """Start the calculator server."""
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="info")
    server = uvicorn.Server(config)

    # Run in asyncio event loop
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(server.serve())


if __name__ == "__main__":
    start_calculator_server()
