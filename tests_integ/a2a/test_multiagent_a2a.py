import atexit
import os
import subprocess
import time

from strands.agent.a2a_agent import A2AAgent


def test_a2a_agent():
    # Start our A2A server
    server_path = os.path.join(os.path.dirname(__file__), "a2a_server.py")
    server = subprocess.Popen(["python", server_path])

    def cleanup():
        server.terminate()

    atexit.register(cleanup)
    time.sleep(5)  # Wait for A2A server to start

    # Connect to our A2A server
    a2a_agent = A2AAgent(endpoint="http://localhost:9000")

    # Invoke our A2A server
    result = a2a_agent("Hello there!")

    # Ensure that it was successful
    assert result.stop_reason == "end_turn"
