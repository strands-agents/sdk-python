from a2a.types import (
    AgentCapabilities,
    AgentCard,
    AgentSkill,
)
from strands import Agent
from strands.multiagent.a2a import A2AAgent

agent = Agent(model="us.anthropic.claude-3-haiku-20240307-v1:0")


skill = AgentSkill(
    id="hello_world",
    name="Returns hello world",
    description="just returns hello world",
    tags=["hello world"],
    examples=["hi", "hello world"],
)

extended_skill = AgentSkill(
    id="super_hello_world",
    name="Returns a SUPER Hello World",
    description="A more enthusiastic greeting, only for authenticated users.",
    tags=["hello world", "super", "extended"],
    examples=["super hi", "give me a super hello"],
)

public_agent_card = AgentCard(
    name="Hello World Agent",
    description="Just a hello world agent",
    url="http://0.0.0.0:9000/",
    version="1.0.0",
    defaultInputModes=["text"],
    defaultOutputModes=["text"],
    capabilities=AgentCapabilities(streaming=True),
    skills=[skill],
)

agent = A2AAgent(agent=agent, agent_card=public_agent_card, skills=[skill])
agent.serve()
