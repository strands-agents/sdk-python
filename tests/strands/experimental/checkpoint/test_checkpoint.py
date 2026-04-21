"""Tests for strands.experimental.checkpoint — Checkpoint serialization."""

import pytest

from strands.experimental.checkpoint import CHECKPOINT_SCHEMA_VERSION, Checkpoint


class TestCheckpoint:
    """Checkpoint dataclass serialization tests."""

    def test_round_trip(self):
        checkpoint = Checkpoint(
            position="after_model",
            cycle_index=1,
            snapshot={"messages": []},
            app_data={"workflow_id": "wf-123"},
        )
        data = checkpoint.to_dict()
        restored = Checkpoint.from_dict(data)

        assert restored.position == checkpoint.position
        assert restored.cycle_index == checkpoint.cycle_index
        assert restored.snapshot == checkpoint.snapshot
        assert restored.app_data == checkpoint.app_data
        assert restored.schema_version == CHECKPOINT_SCHEMA_VERSION

    def test_schema_version_immutable(self):
        checkpoint = Checkpoint(position="after_tools")
        assert checkpoint.schema_version == CHECKPOINT_SCHEMA_VERSION

    def test_schema_version_mismatch_raises(self):
        data = Checkpoint(position="after_model").to_dict()
        data["schema_version"] = "0.0"
        with pytest.raises(ValueError, match="not compatible with current version"):
            Checkpoint.from_dict(data)

    def test_defaults(self):
        checkpoint = Checkpoint(position="after_model")
        assert checkpoint.cycle_index == 0
        assert checkpoint.snapshot == {}
        assert checkpoint.app_data == {}

    def test_from_dict_warns_on_unknown_fields(self, caplog):
        data = Checkpoint(position="after_tools").to_dict()
        data["unknown_future_field"] = "something"
        restored = Checkpoint.from_dict(data)
        assert restored.position == "after_tools"
        assert "unknown_future_field" in caplog.text

    def test_from_dict_missing_schema_version_raises(self):
        data = {"position": "after_model", "cycle_index": 0, "snapshot": {}, "app_data": {}}
        with pytest.raises(ValueError, match="not compatible with current version"):
            Checkpoint.from_dict(data)
