"""Delivery primitives for context injection.

These fold just-in-time text into the model input *ephemerally* — into the latest user
message or the per-call system prompt, depending on the configured location — so the model
sees the augmented input for one call while the agent's durable history is never touched. Reach injection
through the ``ContextInjector`` plugin or the ``MemoryManager`` rather than these primitives
directly.
"""

from __future__ import annotations

import inspect
import logging
from collections.abc import Awaitable, Callable
from dataclasses import replace
from typing import TYPE_CHECKING, Any, Protocol

from .types import InjectionContext, InjectionLocation, InjectionTriggerPredicate

if TYPE_CHECKING:
    from .._middleware.stages import InvokeModelContext
    from ..types.content import ContentBlock, Message, Messages, SystemContentBlock, SystemPrompt

logger = logging.getLogger(__name__)


class RenderContentCallback(Protocol):
    """Renders the text to fold into the latest user message for a model call.

    Implemented by a plain function as well — the ``**kwargs`` tail lets the calling convention
    grow new keyword arguments without breaking existing callbacks.
    """

    def __call__(self, context: InjectionContext, **kwargs: Any) -> str | None | Awaitable[str | None]:
        """Return the text to inject, ``None``/``""`` to skip, or an awaitable of either."""
        ...


# The text-rendering callback. The bare ``Callable`` arm keeps the happy path
# (``lambda context: ...``) ergonomic; the ``RenderContentCallback`` arm is the forward-compatible
# Protocol for callers that opt into future keyword arguments. A callback that raises fails open
# (injection is skipped, the model call proceeds).
RenderContent = Callable[[InjectionContext], "str | None | Awaitable[str | None]"] | RenderContentCallback


def _create_injection_middleware(
    render_content: RenderContent,
    *,
    trigger: InjectionTriggerPredicate | None = None,
    location: InjectionLocation | None = None,
) -> Callable[[InvokeModelContext], Awaitable[InvokeModelContext]]:
    """Build an ``InvokeModelStage.Input`` handler that folds injected text into the conversation.

    The handler folds ``render_content``'s text into the model input, ephemerally: the model
    sees the augmented input for this one call while the agent's durable history is never
    touched. ``location`` selects where the text lands — the latest user message (default) or
    the per-call system prompt. The handler gates on the resolved trigger, asks ``render_content`` for the
    text, and returns a context with the folded messages. Anything that skips — the trigger
    not firing, ``render_content`` returning empty, or any callback raising — returns the
    context unchanged so the model call proceeds (fail open). The injected text never enters
    durable history because the input phase only rewrites the per-call context, not the
    agent's stored messages.

    Args:
        render_content: Renders the text to inject for this call. Sync or async.
        trigger: When to inject. An ``InjectionTrigger`` name selects a built-in policy
            (``"userTurn"`` — default — or ``"everyTurn"``); a predicate over the
            ``InjectionContext`` is the escape hatch. Defaults to ``"userTurn"``.
        location: Where the text lands. ``"lastUserMessage"`` (default) folds it into the
            latest user message; ``"systemPrompt"`` appends it to the per-call system prompt.

    Returns:
        An ``InvokeModelStage.Input`` handler that returns a (possibly) folded context.
    """
    resolved_trigger = _resolve_trigger(trigger)

    async def handler(context: InvokeModelContext) -> InvokeModelContext:
        agent = context.agent
        # Hand the callback its own list, so a callback that reorders/appends cannot perturb the
        # per-call context. The message dicts are shared, but the upstream InvokeModelContext is
        # already a defensive copy of agent state, so durable history is safe regardless.
        injection_context = InjectionContext(messages=list(context.messages), state=agent.state, agent=agent)

        if not resolved_trigger(injection_context):
            return context

        try:
            text = render_content(injection_context)
            if inspect.isawaitable(text):
                text = await text
        except Exception as error:  # noqa: BLE001 - fail open: a bad callback must not abort the model call.
            logger.warning("reason=<%s> | injection render_content raised | skipping injection", error)
            return context

        if text is None or not text.strip():
            return context

        if location == "systemPrompt":
            folded = replace(context, system_prompt=_fold_into_system_prompt(context.system_prompt, text))
        else:
            folded = replace(context, messages=_fold_into_last_user_message(context.messages, text))
        return await _account_for_injected_text(folded, text)

    return handler


async def _account_for_injected_text(context: InvokeModelContext, text: str) -> InvokeModelContext:
    """Fold the injected text's estimated tokens into the per-call projection.

    ``projected_input_tokens`` is computed by the event loop from the agent's durable state
    *before* input middleware runs, so it cannot see injected text. Without this correction,
    downstream consumers of the projection (the context-status middleware, window-enforcement
    middleware) would undercount the request actually sent to the provider by the size of the
    injected content. Counting failures fail open: the injection stands and the projection is
    left unchanged. When the corrected projection exceeds the model's context window limit, a
    warning is logged deterministically — the injected content is standing guidance that cannot
    be recovered by trimming durable history, so the call is allowed to proceed and surface the
    provider's error rather than silently dropping the injection.

    Args:
        context: The middleware context with the injected text already folded in.
        text: The text that was injected.

    Returns:
        The context with ``projected_input_tokens`` increased by the injected text's estimated
        token count, or the context unchanged when no projection is available or counting fails.
    """
    if context.projected_input_tokens is None:
        return context

    try:
        probe: Message = {"role": "user", "content": [{"text": text}]}
        injected_tokens = await context.agent.model.count_tokens([probe])
    except Exception as error:  # noqa: BLE001 - fail open: accounting must not abort the model call.
        logger.warning("reason=<%s> | token accounting for injected text failed | projection left unchanged", error)
        return context

    projected = context.projected_input_tokens + injected_tokens
    limit = context.agent.model.context_window_limit
    if limit is not None and projected > limit:
        logger.warning(
            "projected_input_tokens=<%d>, context_window_limit=<%d> | injected context pushes the projected"
            " input past the model's context window | the provider may reject this request",
            projected,
            limit,
        )
    return replace(context, projected_input_tokens=projected)


def _resolve_trigger(trigger: InjectionTriggerPredicate | None) -> Callable[[InjectionContext], bool]:
    """Resolve an ``InjectionTrigger`` name or predicate into a single gate predicate.

    ``"userTurn"`` maps to ``_is_user_turn`` (over ``context.messages``); ``"everyTurn"`` to an
    always-true gate; a user-supplied predicate is wrapped so that a raise fails open (logs and
    skips injection rather than aborting the model call).

    Args:
        trigger: An ``InjectionTrigger`` name, a predicate, or ``None`` (defaults to ``"userTurn"``).

    Returns:
        A predicate that, given the ``InjectionContext``, returns whether to inject this call.
    """
    if trigger is None or trigger == "userTurn":
        return lambda context: _is_user_turn(context.messages)
    if trigger == "everyTurn":
        return lambda context: True

    predicate = trigger

    def guarded(context: InjectionContext) -> bool:
        try:
            return predicate(context)
        except Exception as error:  # noqa: BLE001 - fail open: a bad predicate must not abort the model call.
            logger.warning("reason=<%s> | injection trigger raised | skipping injection", error)
            return False

    return guarded


def _is_user_turn(messages: Messages) -> bool:
    """Whether the latest message is a fresh user ask: a ``user`` message carrying no tool result.

    This is the ``"userTurn"`` policy — it distinguishes a new chat ask from an autonomous
    tool-result turn.

    Args:
        messages: The current conversation, as data.

    Returns:
        ``True`` when the latest message is a plain user ask, otherwise ``False``.
    """
    if not messages:
        return False
    last = messages[-1]
    return last["role"] == "user" and not any("toolResult" in block for block in last["content"])


def _fold_into_last_user_message(messages: Messages, text: str) -> Messages:
    """Fold ``text`` into the most recent ``user`` message as a text block, returning a NEW list.

    Folding into the existing user message (rather than inserting a standalone message) keeps
    role alternation valid in both chat and the autonomous tool loop. The block is placed to
    keep the message valid for the model:

    - A plain user ask: the text is **prepended**, leaving the user's own ask in the recency
      slot — the last thing the model reads.
    - A tool-result turn (the message carries a tool result block): the text is **appended**,
      because providers require the tool result to be the first content block in the turn that
      answers a tool use.

    The input list and its messages are never mutated. When there is no ``user`` message, the
    input list is returned unchanged.

    Args:
        messages: The conversation to fold into.
        text: The text to fold into the most recent user message.

    Returns:
        A new list with the folded message, or the input list when there is no user message.
    """
    target_index = -1
    for index in range(len(messages) - 1, -1, -1):
        if messages[index]["role"] == "user":
            target_index = index
            break
    if target_index < 0:
        return messages

    target = messages[target_index]
    injected: ContentBlock = {"text": text}
    # A tool result must stay the first block in the turn that answers a tool use, so append
    # rather than prepend when the target carries one.
    has_tool_result = any("toolResult" in block for block in target["content"])
    content = [*target["content"], injected] if has_tool_result else [injected, *target["content"]]

    folded: Message = {"role": target["role"], "content": content}
    if "metadata" in target:
        folded["metadata"] = target["metadata"]

    result = list(messages)
    result[target_index] = folded
    return result


def _fold_into_system_prompt(system_prompt: SystemPrompt, text: str) -> SystemPrompt:
    """Append ``text`` to the per-call system prompt, returning a NEW value.

    The text is **appended** so it lands after the agent's own directives — and, for a
    structured prompt, after any ``cachePoint`` marking the static prefix, keeping that prefix
    cacheable while the injected text varies call to call.

    The input is never mutated:

    - ``None``: the injected text becomes the entire prompt.
    - ``str``: the text is appended, separated by a blank line.
    - list of blocks: a new list with a trailing text block is returned; existing blocks
      (including cache points) are preserved untouched.

    Args:
        system_prompt: The per-call system prompt to fold into.
        text: The text to append.

    Returns:
        A new system prompt value carrying the injected text.
    """
    if system_prompt is None:
        return text
    if isinstance(system_prompt, str):
        return f"{system_prompt}\n\n{text}"
    injected: SystemContentBlock = {"text": text}
    return [*system_prompt, injected]
