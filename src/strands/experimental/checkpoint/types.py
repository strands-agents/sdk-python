"""Content-block types for checkpoint resume.

Mirrors the interrupt pattern (`InterruptResponseContent` in `types/interrupt.py`).
Stays under `experimental/checkpoint/` for V0; will graduate to
`src/strands/types/checkpoint.py` when the feature exits experimental.
"""

from typing import Any, TypedDict


class CheckpointResumeDict(TypedDict):
    """Inner payload for a checkpointResume content block.

    Attributes:
        checkpoint: Serialized Checkpoint as produced by ``Checkpoint.to_dict()``.
    """

    checkpoint: dict[str, Any]


class CheckpointResumeContent(TypedDict):
    """Content block that resumes a paused durable agent.

    Pass a list containing exactly one instance of this type as the prompt to
    ``Agent.invoke_async`` / ``Agent.__call__`` to resume from a checkpoint.

    Example::

        result = await agent.invoke_async(
            [{"checkpointResume": {"checkpoint": previous_checkpoint.to_dict()}}]
        )

    Attributes:
        checkpointResume: The resume payload carrying the serialized checkpoint.
    """

    checkpointResume: CheckpointResumeDict
