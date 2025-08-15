"""Agent identifier utilities."""

import os


def validate(agent_id: str) -> str:
    """Validate agent id.

    Args:
        agent_id: Id to validate.

    Returns:
        Validated id.

    Raises:
        ValueError: If id contains path separators.
    """
    if os.path.basename(agent_id) != agent_id:
        raise ValueError(f"agent_id={agent_id} | agent id cannot contain path separators")

    return agent_id
