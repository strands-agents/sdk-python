"""Shared types and constants for the ``handoff_to_user`` tool."""

from typing import TypedDict

from typing_extensions import NotRequired

HANDOFF_TO_USER_DESCRIPTION = (
    "Hand off to the user with a structured question when you cannot proceed without "
    "human input. Emits an interrupt carrying the question, optional multiple-choice "
    "options, and whether free-text answers are accepted. The agent pauses; on resume "
    "the user's answer is returned as the tool result. Use sparingly and only when the "
    "next step genuinely depends on information only the user can supply."
)
"""Model-facing description for the ``handoff_to_user`` tool."""


MAX_QUESTION_LENGTH = 4096
"""Maximum length of the question string (in characters)."""

MAX_OPTIONS_COUNT = 20
"""Maximum number of multiple-choice options."""

MAX_OPTION_LENGTH = 256
"""Maximum length of each option string (in characters)."""


INTERRUPT_NAME = "strands:handoff-to-user"
"""Fixed interrupt name emitted by the tool, so consumers can pattern-match on it."""


class HandoffQuestion(TypedDict):
    """The structured payload carried on the interrupt's ``reason``.

    Consumers reading the interrupt (custom UI, HITL handler) inspect
    ``interrupt.reason`` and see a dict with this shape.

    Attributes:
        question: The question the agent is asking the user.
        options: Multiple-choice options, or ``None`` for a free-text question.
        allow_free_text: Whether a free-text answer is acceptable. Ignored when
            ``options`` is provided; the consumer decides whether to also allow
            free text alongside a choice.
    """

    question: str
    options: list[str] | None
    allow_free_text: bool


class HandoffAnswer(TypedDict):
    """The shape the consumer resumes with, threaded back as the tool result.

    ``answer`` is always present; ``chose`` is optional and, when supplied,
    reports which option the consumer selected. Bare-string resume responses
    are coerced into ``{"answer": <string>}``.

    Attributes:
        answer: The human's response as free text (the primary field, always
            present).
        chose: The option string the consumer reports as selected. The tool does
            **not** validate this against the emitted ``options`` list — a HITL
            consumer may return any string here. Callers that need canonical
            matching should compare ``chose`` against their own copy of
            ``options`` themselves.
    """

    answer: str
    chose: NotRequired[str]
