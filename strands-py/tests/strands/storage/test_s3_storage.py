"""Tests for the Amazon S3 unified storage backend."""

from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from strands.storage import S3Storage
from strands.types.exceptions import StorageError


@pytest.fixture
def mock_client():
    """A mock boto3 S3 client wired in via a patched ``boto3.Session``."""
    client = MagicMock()
    with patch("boto3.Session") as session_cls:
        session_cls.return_value.client.return_value = client
        yield client


def _client_error(code):
    return ClientError({"Error": {"Code": code, "Message": code}}, "GetObject")


class TestConstructor:
    def test_raises_when_both_boto_session_and_region_name_are_provided(self):
        with pytest.raises(StorageError):
            S3Storage("bucket", region_name="us-west-2", boto_session=object())

    def test_accepts_just_a_bucket_name(self):
        S3Storage("my-bucket")


class TestWrite:
    @pytest.mark.asyncio
    async def test_puts_object_with_the_correct_params(self, mock_client):
        mock_client.put_object.return_value = {}
        storage = S3Storage("my-bucket", prefix="agents/")

        await storage.write("sessions/abc/data.json", b"payload")

        mock_client.put_object.assert_called_once_with(
            Bucket="my-bucket", Key="agents/sessions/abc/data.json", Body=b"payload"
        )

    @pytest.mark.asyncio
    async def test_wraps_sdk_errors_in_storage_error(self, mock_client):
        mock_client.put_object.side_effect = _client_error("AccessDenied")
        storage = S3Storage("my-bucket")

        with pytest.raises(StorageError):
            await storage.write("key", bytes([1]))


class TestRead:
    @pytest.mark.asyncio
    async def test_returns_bytes_when_the_object_exists(self, mock_client):
        body = MagicMock()
        body.read.return_value = bytes([1, 2, 3])
        mock_client.get_object.return_value = {"Body": body}
        storage = S3Storage("my-bucket")

        assert await storage.read("some/key") == bytes([1, 2, 3])

    @pytest.mark.asyncio
    @pytest.mark.parametrize("code", ["NoSuchKey", "NotFound"])
    async def test_returns_none_for_missing_object(self, mock_client, code):
        mock_client.get_object.side_effect = _client_error(code)
        storage = S3Storage("my-bucket")

        assert await storage.read("missing") is None

    @pytest.mark.asyncio
    async def test_wraps_other_errors_in_storage_error(self, mock_client):
        mock_client.get_object.side_effect = _client_error("InternalError")
        storage = S3Storage("my-bucket")

        with pytest.raises(StorageError):
            await storage.read("key")


class TestDelete:
    @pytest.mark.asyncio
    async def test_deletes_object_with_the_correct_params(self, mock_client):
        mock_client.delete_object.return_value = {}
        storage = S3Storage("my-bucket", prefix="p/")

        await storage.delete("key")

        mock_client.delete_object.assert_called_once_with(Bucket="my-bucket", Key="p/key")

    @pytest.mark.asyncio
    async def test_wraps_errors_in_storage_error(self, mock_client):
        mock_client.delete_object.side_effect = _client_error("InternalError")
        storage = S3Storage("my-bucket")

        with pytest.raises(StorageError):
            await storage.delete("key")


class TestList:
    @pytest.mark.asyncio
    async def test_returns_keys_with_prefix_stripped(self, mock_client):
        mock_client.list_objects_v2.return_value = {
            "Contents": [{"Key": "prefix/a"}, {"Key": "prefix/b/c"}],
            "IsTruncated": False,
        }
        storage = S3Storage("my-bucket", prefix="prefix/")

        assert await storage.list("") == ["a", "b/c"]

    @pytest.mark.asyncio
    async def test_paginates_until_is_truncated_is_false(self, mock_client):
        mock_client.list_objects_v2.side_effect = [
            {"Contents": [{"Key": "a"}], "IsTruncated": True, "NextContinuationToken": "token1"},
            {"Contents": [{"Key": "b"}], "IsTruncated": False},
        ]
        storage = S3Storage("my-bucket")

        assert await storage.list("") == ["a", "b"]
        assert mock_client.list_objects_v2.call_count == 2

    @pytest.mark.asyncio
    async def test_terminates_when_truncated_without_a_continuation_token(self, mock_client):
        # A truncated page that omits NextContinuationToken must not loop forever.
        mock_client.list_objects_v2.return_value = {"Contents": [{"Key": "a"}], "IsTruncated": True}
        storage = S3Storage("my-bucket")

        assert await storage.list("") == ["a"]
        assert mock_client.list_objects_v2.call_count == 1

    @pytest.mark.asyncio
    async def test_wraps_errors_in_storage_error(self, mock_client):
        mock_client.list_objects_v2.side_effect = _client_error("BucketNotFound")
        storage = S3Storage("my-bucket")

        with pytest.raises(StorageError):
            await storage.list("prefix/")
