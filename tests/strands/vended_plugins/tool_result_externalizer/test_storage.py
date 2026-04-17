"""Tests for externalization storage backends."""

import threading
from unittest.mock import MagicMock, patch

import pytest
from botocore.exceptions import ClientError

from strands.vended_plugins.tool_result_externalizer import (
    FileExternalizationStorage,
    InMemoryExternalizationStorage,
    S3ExternalizationStorage,
)


class TestInMemoryExternalizationStorage:
    def test_round_trip(self):
        storage = InMemoryExternalizationStorage()
        ref = storage.store("tool_123", "hello world")
        assert storage.retrieve(ref) == "hello world"

    def test_retrieve_missing_raises_key_error(self):
        storage = InMemoryExternalizationStorage()
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent_ref")

    def test_unique_references(self):
        storage = InMemoryExternalizationStorage()
        ref1 = storage.store("tool_123", "content a")
        ref2 = storage.store("tool_123", "content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1) == "content a"
        assert storage.retrieve(ref2) == "content b"

    def test_reference_format(self):
        storage = InMemoryExternalizationStorage()
        ref = storage.store("tool_abc", "content")
        assert ref.startswith("mem_")
        assert "tool_abc" in ref

    def test_thread_safety(self):
        storage = InMemoryExternalizationStorage()
        refs: list[str] = []
        errors: list[Exception] = []

        def store_item(i: int):
            try:
                ref = storage.store(f"tool_{i}", f"content_{i}")
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
        storage = InMemoryExternalizationStorage()
        ref = storage.store("tool_123", "")
        assert storage.retrieve(ref) == ""

    def test_stores_unicode_content(self):
        storage = InMemoryExternalizationStorage()
        content = "Hello \u2603 \U0001f600 \u4e16\u754c"
        ref = storage.store("tool_123", content)
        assert storage.retrieve(ref) == content


class TestFileExternalizationStorage:
    def test_round_trip(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path / "artifacts"))
        ref = storage.store("tool_123", "hello world")
        assert storage.retrieve(ref) == "hello world"

    def test_auto_creates_directory(self, tmp_path):
        artifact_dir = tmp_path / "nested" / "dir" / "artifacts"
        assert not artifact_dir.exists()
        storage = FileExternalizationStorage(artifact_dir=str(artifact_dir))
        storage.store("tool_123", "content")
        assert artifact_dir.exists()

    def test_retrieve_missing_raises_key_error(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent.txt")

    def test_unique_references(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        ref1 = storage.store("tool_123", "content a")
        ref2 = storage.store("tool_123", "content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1) == "content a"
        assert storage.retrieve(ref2) == "content b"

    def test_reference_is_filename(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        ref = storage.store("tool_abc", "content")
        assert ref.endswith(".txt")
        assert "tool_abc" in ref

    def test_sanitizes_path_traversal(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        ref = storage.store("../../etc/passwd", "content")
        assert ".." not in ref
        assert "/" not in ref
        assert "\\" not in ref

    def test_sanitizes_special_characters(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        ref = storage.store("tool/with spaces&special!chars", "content")
        assert " " not in ref
        assert "&" not in ref
        assert "!" not in ref

    def test_stores_unicode_content(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        content = "Hello \u2603 \U0001f600 \u4e16\u754c"
        ref = storage.store("tool_123", content)
        assert storage.retrieve(ref) == content

    def test_retrieve_rejects_path_traversal(self, tmp_path):
        storage = FileExternalizationStorage(artifact_dir=str(tmp_path))
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("../../etc/passwd")


class TestS3ExternalizationStorage:
    @pytest.fixture
    def mock_s3_client(self):
        """Create a mock S3 client that stores objects in memory."""
        client = MagicMock()
        objects: dict[str, bytes] = {}

        def put_object(Bucket, Key, Body, **kwargs):
            objects[f"{Bucket}/{Key}"] = Body

        def get_object(Bucket, Key, **kwargs):
            full_key = f"{Bucket}/{Key}"
            if full_key not in objects:
                error_response = {"Error": {"Code": "NoSuchKey", "Message": "Not found"}}
                raise ClientError(error_response, "GetObject")
            body = MagicMock()
            body.read.return_value = objects[full_key]
            return {"Body": body}

        client.put_object.side_effect = put_object
        client.get_object.side_effect = get_object
        return client

    @pytest.fixture
    def storage(self, mock_s3_client):
        with patch("boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.client.return_value = mock_s3_client
            mock_session_cls.return_value = mock_session
            return S3ExternalizationStorage(bucket="test-bucket", prefix="artifacts")

    def test_round_trip(self, storage):
        ref = storage.store("tool_123", "hello world")
        assert storage.retrieve(ref) == "hello world"

    def test_retrieve_missing_raises_key_error(self, storage):
        with pytest.raises(KeyError, match="Reference not found"):
            storage.retrieve("nonexistent_key")

    def test_unique_references(self, storage):
        ref1 = storage.store("tool_123", "content a")
        ref2 = storage.store("tool_123", "content b")
        assert ref1 != ref2
        assert storage.retrieve(ref1) == "content a"
        assert storage.retrieve(ref2) == "content b"

    def test_reference_includes_prefix(self, storage):
        ref = storage.store("tool_abc", "content")
        assert ref.startswith("artifacts/")
        assert ref.endswith(".txt")

    def test_empty_prefix(self, mock_s3_client):
        with patch("boto3.Session") as mock_session_cls:
            mock_session = MagicMock()
            mock_session.client.return_value = mock_s3_client
            mock_session_cls.return_value = mock_session
            storage = S3ExternalizationStorage(bucket="test-bucket", prefix="")

        ref = storage.store("tool_abc", "content")
        assert not ref.startswith("/")
        assert storage.retrieve(ref) == "content"

    def test_stores_unicode_content(self, storage):
        content = "Hello \u2603 \U0001f600 \u4e16\u754c"
        ref = storage.store("tool_123", content)
        assert storage.retrieve(ref) == content

    def test_sanitizes_tool_use_id(self, storage):
        ref = storage.store("../../etc/passwd", "content")
        assert ".." not in ref
        assert storage.retrieve(ref) == "content"

    def test_put_object_called_with_correct_params(self, storage, mock_s3_client):
        storage.store("tool_123", "test content")

        mock_s3_client.put_object.assert_called_once()
        call_kwargs = mock_s3_client.put_object.call_args[1]
        assert call_kwargs["Bucket"] == "test-bucket"
        assert call_kwargs["Key"].startswith("artifacts/")
        assert call_kwargs["Body"] == b"test content"
        assert call_kwargs["ContentType"] == "text/plain; charset=utf-8"

    def test_non_nosuchkey_error_propagates(self, storage, mock_s3_client):
        error_response = {"Error": {"Code": "AccessDenied", "Message": "Forbidden"}}
        mock_s3_client.get_object.side_effect = ClientError(error_response, "GetObject")

        with pytest.raises(ClientError, match="Forbidden"):
            storage.retrieve("some_key")
