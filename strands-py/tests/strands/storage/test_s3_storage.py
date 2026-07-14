"""Tests for S3Storage."""

import pytest

from strands.storage import S3Storage
from strands.types.exceptions import StorageError


@pytest.fixture
def s3_bucket():
    """Create a moto-mocked S3 bucket."""
    import boto3
    from moto import mock_aws

    with mock_aws():
        session = boto3.Session(region_name="us-east-1")
        client = session.client("s3")
        client.create_bucket(Bucket="test-bucket")
        yield session


@pytest.fixture
def storage(s3_bucket):
    return S3Storage("test-bucket", boto_session=s3_bucket)


class TestS3Storage:
    @pytest.mark.asyncio
    async def test_write_and_read(self, storage):
        await storage.write("key", b"hello")
        assert await storage.read("key") == b"hello"

    @pytest.mark.asyncio
    async def test_read_missing_returns_none(self, storage):
        assert await storage.read("nonexistent") is None

    @pytest.mark.asyncio
    async def test_write_overwrites(self, storage):
        await storage.write("key", b"first")
        await storage.write("key", b"second")
        assert await storage.read("key") == b"second"

    @pytest.mark.asyncio
    async def test_delete(self, storage):
        await storage.write("key", b"data")
        await storage.delete("key")
        assert await storage.read("key") is None

    @pytest.mark.asyncio
    async def test_delete_missing_is_noop(self, storage):
        await storage.delete("nonexistent")

    @pytest.mark.asyncio
    async def test_list_all(self, storage):
        await storage.write("b", b"")
        await storage.write("a", b"")
        await storage.write("c", b"")
        keys = await storage.list("")
        assert keys == ["a", "b", "c"]

    @pytest.mark.asyncio
    async def test_list_with_prefix(self, storage):
        await storage.write("sessions/a", b"")
        await storage.write("sessions/b", b"")
        await storage.write("offloader/x", b"")
        keys = await storage.list("sessions/")
        assert keys == ["sessions/a", "sessions/b"]

    @pytest.mark.asyncio
    async def test_key_normalization(self, storage):
        await storage.write("//foo///bar//", b"data")
        assert await storage.read("foo/bar") == b"data"

    @pytest.mark.asyncio
    async def test_rejects_path_traversal(self, storage):
        with pytest.raises(StorageError):
            await storage.write("../bad", b"data")

    @pytest.mark.asyncio
    async def test_namespace(self, storage):
        ns = storage.namespace("scope")
        await ns.write("key", b"value")
        assert await ns.read("key") == b"value"
        assert await storage.read("scope/key") == b"value"

    def test_rejects_both_region_and_session(self):
        import boto3

        with pytest.raises(StorageError, match="Cannot specify both"):
            S3Storage("bucket", region_name="us-east-1", boto_session=boto3.Session())


class TestS3StorageWithPrefix:
    @pytest.fixture
    def prefixed_storage(self, s3_bucket):
        return S3Storage("test-bucket", prefix="agents/data", boto_session=s3_bucket)

    @pytest.mark.asyncio
    async def test_prefix_scopes_keys(self, prefixed_storage, s3_bucket):
        await prefixed_storage.write("key", b"hello")
        # Verify the actual S3 key includes the prefix
        client = s3_bucket.client("s3")
        response = client.get_object(Bucket="test-bucket", Key="agents/data/key")
        assert response["Body"].read() == b"hello"

    @pytest.mark.asyncio
    async def test_list_strips_prefix(self, prefixed_storage):
        await prefixed_storage.write("a", b"")
        await prefixed_storage.write("b", b"")
        keys = await prefixed_storage.list("")
        assert keys == ["a", "b"]
