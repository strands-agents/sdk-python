"""Strands identifier utilities."""

import enum
import os


class Identifier(enum.Enum):
    """Strands identifier types."""

    AGENT = "agent"
    SESSION = "session"


def validate(id_: str, type_: Identifier) -> str:
    """Validate strands id.

    Args:
        id_: Id to validate.
        type_: Type of the identifier (e.g., session id, agent id, etc.)

    Returns:
        Validated id.

    Raises:
        ValueError: If id contains path separators.
    """
    if os.path.basename(id_) != id_:
        raise ValueError(f"{type_.value}_id={id_} | id cannot contain path separators")

    return id_
