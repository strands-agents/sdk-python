"""Integration tests for OpenAI Responses API on Bedrock Mantle with AWS credentials."""

import httpx
import pytest
from botocore.auth import SigV4Auth
from botocore.awsrequest import AWSRequest
from botocore.session import Session as BotocoreSession

from strands import Agent
from strands.models.openai_responses import OpenAIResponsesModel


class _SigV4Auth(httpx.Auth):
    """httpx Auth handler that signs requests with AWS SigV4."""

    def __init__(self, region: str):
        session = BotocoreSession()
        self.credentials = session.get_credentials().get_frozen_credentials()
        self.signer = SigV4Auth(self.credentials, "bedrock", region)

    def auth_flow(self, request: httpx.Request):
        aws_request = AWSRequest(
            method=request.method,
            url=str(request.url),
            headers=dict(request.headers),
            data=request.content,
        )
        self.signer.add_auth(aws_request)
        for key, value in aws_request.headers.items():
            request.headers[key] = value
        yield request


class _NonClosingAsyncClient(httpx.AsyncClient):
    """AsyncClient that survives the OpenAI SDK's context manager lifecycle."""

    async def aclose(self) -> None:
        pass


@pytest.fixture
def client_args():
    region = "us-east-1"
    return {
        "api_key": "unused",
        "base_url": f"https://bedrock-mantle.{region}.api.aws/v1",
        "http_client": _NonClosingAsyncClient(auth=_SigV4Auth(region)),
    }


@pytest.fixture
def model(client_args):
    return OpenAIResponsesModel(model_id="openai.gpt-oss-120b", client_args=client_args)


@pytest.fixture
def stateful_model(client_args):
    return OpenAIResponsesModel(model_id="openai.gpt-oss-120b", stateful=True, client_args=client_args)


def test_agent_invoke(model):
    agent = Agent(model=model, system_prompt="Reply in one short sentence.", callback_handler=None)
    result = agent("What is 2+2?")
    assert "4" in str(result)


def test_responses_server_side_conversation(stateful_model):
    agent = Agent(model=stateful_model, system_prompt="Reply in one short sentence.", callback_handler=None)

    agent("My name is Alice.")
    assert len(agent.messages) == 0

    result = agent("What is my name?")
    assert "alice" in str(result).lower()


def test_reasoning_content_multi_turn(client_args):
    """Test that reasoning content from gpt-oss models doesn't break multi-turn conversations."""
    model = OpenAIResponsesModel(
        model_id="openai.gpt-oss-120b",
        client_args=client_args,
        params={"reasoning": {"effort": "low"}},
    )
    agent = Agent(model=model, system_prompt="Reply in one short sentence.", callback_handler=None)

    result1 = agent("What is 2+2?")
    assert "4" in str(result1)

    # Verify reasoning content was produced
    has_reasoning = any(
        "reasoningContent" in block for msg in agent.messages if msg["role"] == "assistant" for block in msg["content"]
    )
    assert has_reasoning

    # Second turn should not raise despite reasoningContent in message history
    agent("What about 3+3?")
