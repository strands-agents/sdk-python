"""Vended ``handoff_to_user`` tool.

A thin shim over the SDK's interrupt primitive. Lets an agent hand control back
to the user with a structured question (optionally multiple-choice) and thread
the user's answer through as the tool result.

Example Usage:
    ```python
    from strands import Agent
    from strands.vended_tools import handoff_to_user

    agent = Agent(tools=[handoff_to_user])
    ```
"""

from .handoff_to_user import handoff_to_user
from .types import (
    HANDOFF_TO_USER_DESCRIPTION,
    INTERRUPT_NAME,
    MAX_OPTION_LENGTH,
    MAX_OPTIONS_COUNT,
    MAX_QUESTION_LENGTH,
    HandoffAnswer,
    HandoffQuestion,
)

__all__ = [
    "HANDOFF_TO_USER_DESCRIPTION",
    "INTERRUPT_NAME",
    "MAX_OPTION_LENGTH",
    "MAX_OPTIONS_COUNT",
    "MAX_QUESTION_LENGTH",
    "HandoffAnswer",
    "HandoffQuestion",
    "handoff_to_user",
]
