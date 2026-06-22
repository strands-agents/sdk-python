"""Human-in-the-loop intervention handler.

Pauses agent execution before tool calls so a human can approve or deny them.
"""

import asyncio
import inspect
import json
import threading
from collections.abc import Awaitable
from typing import Any, Literal, Protocol, runtime_checkable

from ...hooks.events import BeforeToolCallEvent
from ...interventions.actions import Confirm, Deny, InterventionAction, Proceed, default_evaluate
from ...interventions.handler import InterventionHandler

_TRUST_RESPONSES = {"t", "trust"}
_TRUSTED_TOOLS_KEY = "hitl:trusted_tools"


@runtime_checkable
class AskCallback(Protocol):
    """Typed contract for an ``ask`` callback: prompt a human and return their response.

    May be sync or async (an async impl lets the agent keep serving its event loop
    while waiting). Documents the expected signature for type-checkers/IDEs; it is
    intentionally not exported -- customers just pass a function.
    """

    def __call__(self, prompt: str, **kwargs: Any) -> Any | Awaitable[Any]:
        """Prompt the human and return their response (directly or as an awaitable)."""
        ...


@runtime_checkable
class EvaluateCallback(Protocol):
    """Typed contract for an ``evaluate``/``evaluate_trust`` callback.

    Decides whether a response approves (or trusts) a tool call. Documents the
    expected signature for type-checkers/IDEs; intentionally not exported --
    customers just pass a function.
    """

    def __call__(self, response: Any, **kwargs: Any) -> bool:
        """Return whether the response approves (or trusts) the pending tool call."""
        ...


def _create_stdio_ask(include_trust: bool) -> AskCallback:
    """Build an ``ask`` callback that prompts the human on the terminal.

    Each prompt is serialized behind a ``threading.Lock`` so concurrent tool calls
    can't interleave their reads from a single stdin. The blocking ``input`` call is
    offloaded to a worker thread so the event loop keeps running while waiting. A
    plain ``threading.Lock`` (rather than ``asyncio.Lock``) is used deliberately: the
    same handler instance may be reused across agent invocations that each spin up
    their own event loop, and a threading lock is the only kind that spans them.

    A worker thread blocked on ``input`` cannot be cancelled, so if the agent run is
    cancelled or times out mid-prompt the thread stays parked until the user presses
    enter and the process won't exit cleanly. ``ask="stdio"`` is therefore meant for
    interactive CLI sessions, not for runs that may be cancelled out from under it.

    Args:
        include_trust: Show the trust option (``t``) in the prompt suffix when True.

    Returns:
        An async ``ask`` callback that reads a response from stdin.
    """
    options = "(y/n/t)" if include_trust else "(y/n)"
    lock = threading.Lock()

    def _blocking_ask(prompt: str) -> str:
        with lock:
            return input(f"{prompt} {options}: ").strip()

    async def ask(prompt: str, **kwargs: Any) -> Any:
        return await asyncio.to_thread(_blocking_ask, prompt)

    return ask


class HumanInTheLoop(InterventionHandler):
    """Human-in-the-loop intervention handler that pauses agent execution before tool calls.

    By default, ALL tools require approval and the agent pauses via interrupt/resume.
    Use ``allowed_tools`` to allow-list tools that run freely, and ``ask`` to provide
    inline prompting (CLI, custom UI).

    Example:
        ```python
        from strands import Agent
        from strands.vended_interventions.hitl import HumanInTheLoop

        # All tools require approval, agent pauses via interrupt (default)
        agent = Agent(interventions=[HumanInTheLoop()])

        # read_file runs freely, everything else pauses for approval
        agent = Agent(interventions=[HumanInTheLoop(allowed_tools=["read_file"])])

        # CLI mode - prompts in terminal inline
        agent = Agent(interventions=[HumanInTheLoop(ask="stdio")])

        # Custom UI - provide your own prompt function
        async def slack_ask(prompt: str) -> str:
            return await slack_dm(user_id, prompt)

        agent = Agent(interventions=[HumanInTheLoop(ask=slack_ask)])
        ```
    """

    name = "strands:human-in-the-loop"

    def __init__(
        self,
        *,
        allowed_tools: list[str] | None = None,
        enable_trust: bool = False,
        evaluate_trust: EvaluateCallback | None = None,
        evaluate: EvaluateCallback | None = None,
        ask: AskCallback | Literal["stdio"] | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            allowed_tools: Tools that can execute WITHOUT human approval. All other
                tools require approval. Use ``"*"`` to allow all tools. Prefix with
                ``!`` to exclude specific tools from ``"*"`` (they still require
                approval). For example, ``["read_file", "list_dir"]`` lets only those
                two run freely, while ``["*", "!delete_file"]`` lets everything run
                freely except ``delete_file``.
            enable_trust: When True, trust responses approve the tool AND remember it
                in ``agent.state`` for the rest of the session (won't ask again).
                Works in both interrupt/resume and inline ``ask`` modes. Negated
                tools (``!tool``) cannot be trusted. Defaults to False.
            evaluate_trust: Custom trust response validator. Defaults to accepting
                ``"t"``/``"trust"`` (case-insensitive). When this returns True, the
                tool is approved AND trusted for the session. Only evaluated when
                ``enable_trust`` is True.
            evaluate: Custom approval response validator. Defaults to accepting
                ``True``, ``"y"``/``"yes"`` (case-insensitive).
            ask: Controls how the human's response is collected. Omitted (default):
                uses interrupt/resume - agent pauses, caller resumes with response.
                ``"stdio"``: prompts via CLI stdin. Agent blocks inline until the
                human responds. Note that stdio mode runs a blocking ``input()`` in a
                worker thread that cannot be cancelled, so it is intended for
                interactive CLI use rather than runs that may be cancelled mid-prompt.
                Custom callable: your own (optionally async) prompt logic (Slack, web
                UI, etc.). Agent blocks inline. A custom ``ask`` should return a concrete
                response; returning ``None`` (e.g. a dismissed dialog) is treated as an
                explicit deny. If it raises, the exception propagates and aborts the run
                (fail-closed) -- catch and return a deny value inside your callback if you
                prefer to proceed on error.

        Note:
            ``name`` is a fixed class attribute, so at most one ``HumanInTheLoop`` can be
            registered per agent; layering two policies requires subclassing to rename.
        """
        # A bare string is iterable, so ``set("read_file")`` would silently become a per-char set; reject it.
        if isinstance(allowed_tools, str):
            raise ValueError("allowed_tools must be a list of tool names, not a single string")
        self._allowed_tools = set(allowed_tools or [])
        self._enable_trust = enable_trust
        self._evaluate_trust = evaluate_trust if evaluate_trust is not None else self._is_trust_response
        self._evaluate = evaluate if evaluate is not None else default_evaluate
        self._ask = _create_stdio_ask(enable_trust) if ask == "stdio" else ask

    # Implemented as async so the inline ``ask`` path can await a human's response.
    # The base lifecycle methods are typed as ``_MaybeAwaitable[...]`` (since #2800),
    # so an async override type-checks cleanly and the registry awaits it at runtime.
    async def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> InterventionAction:
        """Request human approval before executing a tool that is not allow-listed or trusted.

        Implemented as ``async`` so the inline ``ask`` path can await a human's
        response (e.g. an HTTP round-trip to a Slack/web UI) without blocking the
        agent's event loop. When no ``ask`` is configured this returns a ``Confirm``
        that pauses the agent via interrupt instead.

        Args:
            event: The tool call event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            Proceed if the tool is allow-listed, trusted, or approved inline;
            otherwise a Confirm action (pausing via interrupt when no ``ask`` is set).
        """
        tool_name = event.tool_use["name"]
        if not self._requires_approval(event):
            return Proceed()

        prompt = f'Tool "{tool_name}" requires human approval. Input: {json.dumps(event.tool_use["input"])}'

        is_negated = f"!{tool_name}" in self._allowed_tools

        # No ``ask`` configured: defer to interrupt/resume. The evaluate closure runs
        # later when the caller resumes with a response, so trust must be recorded
        # there rather than now.
        if self._ask is None:

            def evaluate(response: Any) -> bool:
                if not is_negated and self._enable_trust and self._evaluate_trust(response):
                    self._trust_tool(event, tool_name)
                    return True
                return self._evaluate(response)

            return Confirm(prompt=prompt, evaluate=evaluate)

        # Inline mode: collect the response now (awaiting async ``ask`` callbacks). A
        # trust response short-circuits to Proceed; anything else is handed to the
        # standard evaluator via Confirm so the normal approve/deny path applies.
        response = self._ask(prompt)
        if inspect.isawaitable(response):
            response = await response

        # A configured inline ``ask`` is expected to return a concrete response. ``None``
        # (e.g. a dismissed dialog) is treated as a fail-closed deny rather than passed
        # to ``Confirm``: ``Confirm`` reads ``response=None`` as "no preemptive value"
        # and falls back to interrupt/resume, which a stateless inline caller (Slack/web)
        # has no way to resume. Denying is the safe default for an approval gate.
        if response is None:
            return Deny(reason=f'Tool "{tool_name}" denied: approval callback returned no response.')

        if not is_negated and self._enable_trust and self._evaluate_trust(response):
            self._trust_tool(event, tool_name)
            return Proceed()

        return Confirm(prompt=prompt, response=response, evaluate=self._evaluate)

    def _requires_approval(self, event: BeforeToolCallEvent) -> bool:
        """Decide whether the tool call needs human approval.

        Precedence (first match wins):

        1. Negated (``!tool``) -> always requires approval (cannot be trusted)
        2. Trusted at runtime via trust response (stored in ``agent.state``) -> runs freely
        3. Wildcard (``*``) -> runs freely
        4. Explicitly listed -> runs freely
        5. Default -> requires approval

        Args:
            event: The tool call event under evaluation.

        Returns:
            True if the tool requires human approval.
        """
        tool_name = event.tool_use["name"]
        if f"!{tool_name}" in self._allowed_tools:
            return True
        trusted = event.agent.state.get(_TRUSTED_TOOLS_KEY) or []
        if tool_name in trusted:
            return False
        if "*" in self._allowed_tools:
            return False
        if tool_name in self._allowed_tools:
            return False
        return True

    def _trust_tool(self, event: BeforeToolCallEvent, tool_name: str) -> None:
        """Remember a tool as trusted for the rest of the session.

        Args:
            event: The tool call event (provides access to the agent).
            tool_name: Name of the tool to trust.
        """
        trusted = event.agent.state.get(_TRUSTED_TOOLS_KEY) or []
        if tool_name not in trusted:
            event.agent.state.set(_TRUSTED_TOOLS_KEY, [*trusted, tool_name])

    @staticmethod
    def _is_trust_response(response: Any) -> bool:
        """Check whether a response is a trust response (``"t"``/``"trust"``, case-insensitive).

        Args:
            response: The human's response value.

        Returns:
            True if the response is a trust response.
        """
        if isinstance(response, str):
            return response.lower().strip() in _TRUST_RESPONSES
        return False
