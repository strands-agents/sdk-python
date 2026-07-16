"""``handoff_to_user`` tool: pause the agent to ask the user a structured question.

A thin shim over :meth:`~strands.types.tools.ToolContext.interrupt`. The tool
validates its inputs, then raises a single interrupt whose ``reason`` carries a
:class:`~strands.vended_tools.handoff_to_user.types.HandoffQuestion` payload.
The SDK halts the agent; the caller resumes with the user's response, which is
threaded back as the tool result.

No console I/O and no UI. Rendering the question and collecting the answer are
the consumer's job (a chat frontend, a Slack app, a CLI handler, etc.). If nothing
is listening the SDK's default interrupt handling applies.

Example:
    ```python
    from strands import Agent
    from strands.vended_tools import handoff_to_user

    agent = Agent(tools=[handoff_to_user])
    result = agent("Ask me which environment to deploy to.")

    if result.stop_reason == "interrupt":
        interrupt = result.interrupts[0]
        # interrupt.reason = {"question": "...", "options": [...], "allow_free_text": ...}
        responses = [
            {
                "interruptResponse": {
                    "interruptId": interrupt.id,
                    "response": {"answer": "prod", "chose": "prod"},
                }
            }
        ]
        result = agent(responses)
    ```
"""

from __future__ import annotations

from ...tools.decorator import tool
from ...types.tools import ToolContext
from .types import (
    HANDOFF_TO_USER_DESCRIPTION,
    INTERRUPT_NAME,
    MAX_OPTION_LENGTH,
    MAX_OPTIONS_COUNT,
    MAX_QUESTION_LENGTH,
    HandoffAnswer,
    HandoffQuestion,
)


def _validate_and_build_reason(
    question: str,
    options: list[str] | None,
    allow_free_text: bool,
) -> HandoffQuestion:
    """Validate the tool inputs and produce the interrupt payload.

    Args:
        question: The question the agent wants to ask the user.
        options: Optional multiple-choice options.
        allow_free_text: Whether free-text answers are accepted. Ignored when
            ``options`` is provided (consumers decide whether to also permit
            free text alongside the choice).

    Returns:
        A :class:`HandoffQuestion` dict for use as the interrupt's ``reason``.

    Raises:
        ValueError: If any input violates the size limits, or if the option list
            contains a non-string or a duplicate entry.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string")
    if len(question) > MAX_QUESTION_LENGTH:
        raise ValueError(f"question length ({len(question)}) exceeds maximum allowed length ({MAX_QUESTION_LENGTH})")

    normalized_options: list[str] | None = None
    if options is not None:
        if not isinstance(options, list):
            raise ValueError("options must be a list of strings")
        if len(options) == 0:
            raise ValueError("options must contain at least one entry when provided")
        if len(options) > MAX_OPTIONS_COUNT:
            raise ValueError(f"options count ({len(options)}) exceeds maximum allowed count ({MAX_OPTIONS_COUNT})")
        # Compare options on their trimmed value so ``["yes", "yes "]`` is
        # rejected as a duplicate — the whitespace check above already treats
        # such entries as equivalent (both are non-empty after strip).
        seen: set[str] = set()
        for i, opt in enumerate(options):
            if not isinstance(opt, str):
                raise ValueError(f"options[{i}] must be a string")
            stripped = opt.strip()
            if not stripped:
                raise ValueError(f"options[{i}] must be a non-empty string")
            if len(opt) > MAX_OPTION_LENGTH:
                raise ValueError(
                    f"options[{i}] length ({len(opt)}) exceeds maximum allowed length ({MAX_OPTION_LENGTH})"
                )
            if stripped in seen:
                raise ValueError(f"options[{i}] duplicates an earlier entry: {opt!r}")
            seen.add(stripped)
        normalized_options = list(options)

    if normalized_options is None and not allow_free_text:
        raise ValueError("handoff must accept either options or free text; got neither")

    # allow_free_text is meaningful only when there are no options; document that
    # in the payload rather than mutating the caller's intent so consumers can see
    # both fields on the wire.
    return HandoffQuestion(
        question=question,
        options=normalized_options,
        allow_free_text=bool(allow_free_text),
    )


def _coerce_response(response: object) -> HandoffAnswer:
    """Normalize the resume response into a :class:`HandoffAnswer` shape.

    - A bare string becomes ``{"answer": <string>}``.
    - A dict with an ``answer`` key (and optionally ``chose``) is passed through
      after minimal type checks.
    - Anything else raises: the tool result should be well-typed, not opaque.

    Args:
        response: Whatever the caller resumed the interrupt with.

    Returns:
        A :class:`HandoffAnswer` dict.

    Raises:
        ValueError: If the response cannot be coerced into a well-formed answer.
    """
    # The same 4096-char budget bounds the question going out and the answer
    # coming back — one direction is user prompt, the other is user answer, both
    # get plumbed into model context, so a single per-turn text budget is enough.
    if isinstance(response, str):
        if len(response) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"handoff response 'answer' length ({len(response)}) exceeds maximum allowed length "
                f"({MAX_QUESTION_LENGTH})"
            )
        return {"answer": response}
    if isinstance(response, dict):
        answer = response.get("answer")
        chose = response.get("chose")
        if not isinstance(answer, str):
            raise ValueError("handoff response 'answer' must be a string")
        if len(answer) > MAX_QUESTION_LENGTH:
            raise ValueError(
                f"handoff response 'answer' length ({len(answer)}) exceeds maximum allowed length "
                f"({MAX_QUESTION_LENGTH})"
            )
        result: HandoffAnswer = {"answer": answer}
        if chose is not None:
            if not isinstance(chose, str):
                raise ValueError("handoff response 'chose' must be a string when provided")
            if len(chose) > MAX_OPTION_LENGTH:
                raise ValueError(
                    f"handoff response 'chose' length ({len(chose)}) exceeds maximum allowed length "
                    f"({MAX_OPTION_LENGTH})"
                )
            result["chose"] = chose
        return result
    raise ValueError(
        f"handoff response must be a string or an object with an 'answer' key; got {type(response).__name__}"
    )


@tool(name="handoff_to_user", description=HANDOFF_TO_USER_DESCRIPTION, context="tool_context")
def handoff_to_user(
    question: str,
    tool_context: ToolContext,
    options: list[str] | None = None,
    allow_free_text: bool = True,
) -> HandoffAnswer:
    """Pause the agent and ask the user a structured question.

    Emits an interrupt whose ``reason`` carries the question, optional multiple-choice
    options, and whether a free-text answer is acceptable. On resume, the caller's
    response is normalized to a :class:`HandoffAnswer` and returned as the tool
    result so the model can continue the loop with the user's answer in hand.

    Args:
        question: The question to ask the user. Must be a non-empty string no longer
            than 4096 characters.
        tool_context: Injected by the framework. Not user-facing.
        options: Optional multiple-choice options (up to 20 entries, each up to 256
            characters, all distinct). When omitted, the tool asks a free-text
            question.
        allow_free_text: Whether the consumer should accept a free-text answer.
            Defaults to True. When ``options`` is provided this flag is passed
            through to the consumer as a hint but does not restrict what the
            consumer resumes with.

    Returns:
        The user's response as a :class:`HandoffAnswer` dict.
    """
    reason = _validate_and_build_reason(question, options, allow_free_text)
    response = tool_context.interrupt(INTERRUPT_NAME, reason=reason)
    return _coerce_response(response)
