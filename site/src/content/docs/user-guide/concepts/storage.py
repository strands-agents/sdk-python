"""Storage documentation code examples."""

import asyncio

from strands import Agent
from strands.storage import InMemoryStorage, LocalFileStorage, S3Storage
from strands.vended_plugins.context_offloader import ContextOffloader


# --8<-- [start:basic_usage]
storage = LocalFileStorage()

agent = Agent(plugins=[
    ContextOffloader(storage=storage)
])
# --8<-- [end:basic_usage]


# --8<-- [start:in_memory]
storage = InMemoryStorage()
# --8<-- [end:in_memory]


# --8<-- [start:local_file]
storage = LocalFileStorage("./my-data/")
# --8<-- [end:local_file]


# --8<-- [start:s3]
storage = S3Storage("my-bucket", prefix="agents/prod/")
# --8<-- [end:s3]


# --8<-- [start:namespace]
from strands.storage import LocalFileStorage

storage = LocalFileStorage()

scoped = storage.namespace("project-alpha")
# Writes to "project-alpha/config.json" in the underlying store
# --8<-- [end:namespace]


# --8<-- [start:custom_backend]
from strands.storage import Storage


class RedisStorage:
    """A custom Storage backend backed by Redis."""

    def __init__(self, url: str = "redis://localhost:6379") -> None:
        import redis.asyncio as redis

        self._client = redis.from_url(url)

    async def write(self, key: str, data: bytes) -> None:
        await self._client.set(key, data)

    async def read(self, key: str) -> bytes | None:
        return await self._client.get(key)

    async def delete(self, key: str) -> None:
        await self._client.delete(key)

    async def list(self, query: str) -> list[str]:
        keys = await self._client.keys(f"{query}*")
        return sorted(k.decode() for k in keys)
# --8<-- [end:custom_backend]
