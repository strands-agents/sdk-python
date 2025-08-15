"""Session identifier utilities."""

import os


def validate(session_id: str) -> str:
    """Validate session id.

    Args:
        session_id: Id to validate.

    Returns:
        Validated id.

    Raises:
        ValueError: If id contains path separators.
    """
    if os.path.basename(session_id) != session_id:
        raise ValueError(f"session_id={session_id} | session id cannot contain path separators")

    return session_id
