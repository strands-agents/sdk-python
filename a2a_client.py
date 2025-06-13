import logging
from typing import Any
from uuid import uuid4

import httpx
from a2a.client import A2ACardResolver, A2AClient
from a2a.types import (
    AgentCard,
    MessageSendParams,
    SendMessageRequest,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
PUBLIC_AGENT_CARD_PATH = "/.well-known/agent.json"
BASE_URL = "http://localhost:9000"


async def main() -> None:
    async with httpx.AsyncClient() as httpx_client:
        # Initialize A2ACardResolver
        resolver = A2ACardResolver(
            httpx_client=httpx_client,
            base_url=BASE_URL,
        )

        # Fetch Public Agent Card and Initialize Client
        agent_card: AgentCard | None = None

        try:
            logger.info("Attempting to fetch public agent card from: {} {}", BASE_URL, PUBLIC_AGENT_CARD_PATH)
            agent_card = await resolver.get_agent_card()  # Fetches from default public path
            logger.info("Successfully fetched public agent card:")
            logger.info(agent_card.model_dump_json(indent=2, exclude_none=True))
        except Exception as e:
            logger.exception("Critical error fetching public agent card")
            raise RuntimeError("Failed to fetch the public agent card. Cannot continue.") from e

        client = A2AClient(httpx_client=httpx_client, agent_card=agent_card)
        logger.info("A2AClient initialized.")

        send_message_payload: dict[str, Any] = {
            "message": {
                "role": "user",
                "parts": [{"kind": "text", "text": "how much is 10 USD in INR?"}],
                "messageId": uuid4().hex,
            },
        }
        request = SendMessageRequest(id=str(uuid4()), params=MessageSendParams(**send_message_payload))

        response = await client.send_message(request)
        print(response.model_dump(mode="json", exclude_none=True))


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
