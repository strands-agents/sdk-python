import logging
import sys

from strands import Agent
from strands.multiagent.a2a import A2AAgent

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
    force=True,
)

# Log that we're starting
logging.info("Starting A2A server with root logger")

strands_agent = Agent(model="us.anthropic.claude-3-haiku-20240307-v1:0", callback_handler=None)
strands_a2a_agent = A2AAgent(agent=strands_agent, name="Hello World Agent", description="Just a hello world agent")
strands_a2a_agent.serve()
