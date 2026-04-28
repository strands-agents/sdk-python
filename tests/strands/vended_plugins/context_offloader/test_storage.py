"""Tests for offload storage backends."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from strands.vended_plugins.context_offloader import (
    FileStorage,
    InMemoryStorage,
    S3Storage,
)


class TestInMemoryStorage:
    def test_round_trip(self):
        storage = InMemoryStorage()
        ref = storage.store("key_1", b"hello world")
        content, content_type = storage.retrieve(ref)
        assert content == b"hello world"
        assert content_type == "text/plain"

    def test_preserves_content_type(self):
        storage = InMemoryStorage()
        ref = storage.store("key_1", b'{"a": 1}', "application/json")
        content, content_type = storage.retrieve(ref)
        assert content == b'{"a": 1}'
        assert content_type == "application/json"

    def test_stores_binary_content(self):
        storage = InMemoryStorage()
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        ref = storage.store("key_1", img_bytes, "image/png")
        content, content_type = storage.retrieve(ref)
        assert content == img_bytes
        assert content_type == "image/png"

    def test_retrieve_missing_raises_key_error(self):
        storage = InMemoryStorage()
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent_ref")

    def test_unique_references(self):
        storage = InMemoryStorage()
        ref1 = storage.store("key_1", b"content a")
        ref2 = storage.store("key_1", b"content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1)[0] == b"content a"
        assert storage.retrieve(ref2)[0] == b"content b"

    def test_reference_format(self):
        storage = InMemoryStorage()
        ref = storage.store("tool_abc", b"content")
        assert ref.startswith("mem_")
        assert "tool_abc" in ref

    def test_thread_safety(self):
        storage = InMemoryStorage()
        refs: list[str] = []
        errors: list[Exception] = []

        def store_item(i: int):
            try:
                ref = storage.store(f"key_{i}", f"content_{i}".encode())
                refs.append(ref)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=store_item, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert len(set(refs)) == 50

    def test_stores_empty_content(self):
        storage = InMemoryStorage()
        ref = storage.store("key_1", b"")
        assert storage.retrieve(ref) == (b"", "text/plain")

    def test_clear(self):
        storage = InMemoryStorage()
        ref = storage.store("key_1", b"content")
        storage.clear()
        with pytest.raises(KeyError):
            storage.retrieve(ref)

    def test_clear_empty_storage(self):
        storage = InMemoryStorage()
        storage.clear()


class TestFileStorage:
    def test_round_trip(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path / "artifacts"))
        ref = storage.store("key_1", b"hello world")
        content, content_type = storage.retrieve(ref)
        assert content == b"hello world"
        assert content_type == "text/plain"

    def test_preserves_content_type(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        ref = storage.store("key_1", b'{"a": 1}', "application/json")
        content, content_type = storage.retrieve(ref)
        assert content == b'{"a": 1}'
        assert content_type == "application/json"

    def test_stores_binary_content(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        img_bytes = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        ref = storage.store("key_1", img_bytes, "image/png")
        content, content_type = storage.retrieve(ref)
        assert content == img_bytes
        assert content_type == "image/png"

    def test_extension_from_content_type(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        assert storage.store("k", b"text", "text/plain").endswith(".txt")
        assert storage.store("k", b"{}", "application/json").endswith(".json")
        assert storage.store("k", b"img", "image/png").endswith(".png")
        assert storage.store("k", b"pdf", "application/pdf").endswith(".pdf")

    def test_auto_creates_directory(self, tmp_path):
        artifact_dir = tmp_path / "nested" / "dir" / "artifacts"
        assert not artifact_dir.exists()
        storage = FileStorage(artifact_dir=str(artifact_dir))
        storage.store("key_1", b"content")
        assert artifact_dir.exists()

    def test_retrieve_missing_raises_key_error(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent.txt")

    def test_unique_references(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        ref1 = storage.store("key_1", b"content a")
        ref2 = storage.store("key_1", b"content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1)[0] == b"content a"
        assert storage.retrieve(ref2)[0] == b"content b"

    def test_sanitizes_path_traversal(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        ref = storage.store("../../etc/passwd", b"content")
        assert ".." not in ref
        assert "/" not in ref

    def test_metadata_survives_across_instances(self, tmp_path):
        artifact_dir = str(tmp_path / "artifacts")
        storage1 = FileStorage(artifact_dir=artifact_dir)
        ref = storage1.store("key_1", b"hello", "image/png")

        storage2 = FileStorage(artifact_dir=artifact_dir)
        content, content_type = storage2.retrieve(ref)
        assert content == b"hello"
        assert content_type == "image/png"

    def test_corrupt_metadata_fallback(self, tmp_path):
        (tmp_path / ".metadata.json").write_text("not valid json", encoding="utf-8")
        storage = FileStorage(artifact_dir=str(tmp_path))
        assert storage._content_types == {}

    def test_missing_metadata_fallback(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        ref = storage.store("key_1", b"content", "image/png")

        storage._content_types.clear()
        _, content_type = storage.retrieve(ref)
        assert content_type == "application/octet-stream"

    def test_retrieve_rejects_path_traversal(self, tmp_path):
        storage = FileStorage(artifact_dir=str(tmp_path))
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("../../etc/passwd")


class TestS3Storage:
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client that stores objects in memory."""
        client = MagicMock()
        objects: dict[str, tuple[bytes, str]] = {}

        def put_object(Bucket, Key, Body, ContentType="application/octet-stream", **kwargs):
            objects[f"{Bucket}/{Key}"] = (Body, ContentType)

        def get_object(Bucket, Key, **kwargs):
            full_key = f"{Bucket}/{Key}"
            if full_key not in objects:
                error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
                raise ClientError(error_response, "GetObject")
            body_bytes, ct = objects[full_key]
            body = MagicMock()
            body.read.return_value = body_bytes
            return {"Body": body, "ContentType": ct}

        client.put_object.side_effect = put_object
        client.get_object.side_effect = get_object
        return client

    @pytest.fixture
    def storage(self, mock_s3_client):
        with patch("boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.client.return_value = mock_s3_client
            mock_session_cls.return_value = mock_session
            return S3Storage(bucket="test-bucket", prefix="artifacts")

    def test_round_trip(self, storage):
        ref = storage.store("key_1", b"hello world")
        content, content_type = storage.retrieve(ref)
        assert content == b"hello world"
        assert content_type == "text/plain"

    def test_preserves_content_type(self, storage):
        ref = storage.store("key_1", b"img", "image/png")
        content, content_type = storage.retrieve(ref)
        assert content == b"img"
        assert content_type == "image/png"

    def test_retrieve_missing_raises_key_error(self, storage):
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent_key")

    def test_unique_references(self, storage):
        ref1 = storage.store("key_1", b"content a")
        ref2 = storage.store("key_1", b"content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1)[0] == b"content a"
        assert storage.retrieve(ref2)[0] == b"content b"

    def test_reference_includes_prefix(self, storage):
        ref = storage.store("tool_abc", b"content")
        assert ref.startswith("artifacts/")

    def test_empty_prefix(self, mock_s3_client):
        with patch("boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.client.return_value = mock_s3_client
            mock_session_cls.return_value = mock_session
            storage = S3Storage(bucket="test-bucket", prefix="")

        ref = storage.store("tool_abc", b"content")
        assert not ref.startswith("/")
        assert storage.retrieve(ref)[0] == b"content"

    def test_put_object_called_with_correct_params(self, storage, mock_s3_client):
        storage.store("key_1", b"test content", "application/json")

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"].startswith("artifacts/")
        assert call_kwargs["Body"] == b"test content"
        assert call_kwargs["ContentType"] == "application/json"

    def test_non_nosuchkey_error_propagates(self, storage, mock_s3_client):
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(ClientError, match="Forbidden"):
            storage.retrieve("some_key")
