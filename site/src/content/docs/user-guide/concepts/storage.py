"""Storage documentation code examples."""

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
