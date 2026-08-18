"""Integration tests for A2A executor with real file processing."""

import base64
import os
import threading
import time
from uuid import uuid4

import pytest
import requests
import uvicorn
from a2a.types import Message, Part, Role, SendMessageRequest
from google.protobuf.json_format import MessageToDict

from strands import Agent
from strands.multiagent.a2a import A2AServer


@pytest.mark.asyncio
async def test_a2a_executor_with_real_image():
    """Test A2A server processes a real image file correctly via HTTP."""
    # Read the test image file
    test_image_path = os.path.join(os.path.dirname(__file__), "resources/yellow.png")
    with open(test_image_path, "rb") as f:
        original_image_bytes = f.read()

    # Create real Strands agent
    strands_agent = Agent(name="Test Image Agent", description="Agent for testing image processing")

    # Create A2A server
    a2a_server = A2AServer(agent=strands_agent, port=9001)
    fastapi_app = a2a_server.to_fastapi_app()

    # Start server in background
    server_thread = threading.Thread(target=lambda: uvicorn.run(fastapi_app, port=9001), daemon=True)
    server_thread.start()
    time.sleep(1)  # Give server time to start

    try:
        # Build the A2A SendMessage request via the SDK's own types so the wire format
        # (camelCase field names, base64-encoded `raw` bytes) always matches the server.
        message = Message(
            message_id=str(uuid4()),
            role=Role.ROLE_USER,
            parts=[
                Part(text="What primary color is this image, respond with NONE if you are unsure"),
                Part(raw=original_image_bytes, media_type="image/png", filename="image.png"),
            ],
        )
        message_payload = {
            "jsonrpc": "2.0",
            "id": "test-image-request",
            "method": "SendMessage",
            "params": MessageToDict(SendMessageRequest(message=message)),
        }

        # Send request to A2A server
        response = requests.post(
            "http://127.0.0.1:9001", headers={"Content-Type": "application/json"}, json=message_payload, timeout=30
        )

        # Verify response
        assert response.status_code == 200
        response_data = response.json()
        assert "TASK_STATE_COMPLETED" == response_data["result"]["status"]["state"]
        all_text = " ".join(
            part["text"]
            for artifact in response_data["result"]["artifacts"]
            for part in artifact["parts"]
            if "text" in part
        ).lower()
        assert "yellow" in all_text

    except Exception as e:
        pytest.fail(f"Integration test failed: {e}")


def test_a2a_executor_image_roundtrip():
    """Test that image data survives the A2A base64 encoding/decoding roundtrip."""
    # Read the test image
    test_image_path = os.path.join(os.path.dirname(__file__), "resources/yellow.png")
    with open(test_image_path, "rb") as f:
        original_bytes = f.read()

    # Simulate A2A protocol: encode to base64 string
    base64_string = base64.b64encode(original_bytes).decode("utf-8")

    # Simulate executor decoding
    decoded_bytes = base64.b64decode(base64_string)

    # Verify perfect roundtrip
    assert decoded_bytes == original_bytes
    assert len(decoded_bytes) == len(original_bytes)

    # Verify it's actually image data (PNG signature)
    assert decoded_bytes.startswith(b"\x89PNG\r\n\x1a\n")
