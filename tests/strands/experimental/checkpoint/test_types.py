"""Tests for CheckpointResume content-block types."""

from strands.experimental.checkpoint import CheckpointResumeContent, CheckpointResumeDict


def test_checkpoint_resume_dict_carries_serialized_checkpoint() -> None:
    block: CheckpointResumeDict = {"checkpoint": {"position": "after_model", "cycle_index": 0}}
    assert block["checkpoint"]["position"] == "after_model"
    assert block["checkpoint"]["cycle_index"] == 0


def test_checkpoint_resume_content_wraps_resume_dict() -> None:
    content: CheckpointResumeContent = {
        "checkpointResume": {"checkpoint": {"position": "after_tools", "cycle_index": 2}}
    }
    assert content["checkpointResume"]["checkpoint"]["cycle_index"] == 2
    assert content["checkpointResume"]["checkpoint"]["position"] == "after_tools"
