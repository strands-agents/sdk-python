"""Presidio-backed PII redaction intervention handler.

Detects and anonymizes personally identifiable information (PII) in tool I/O and
model context using `Microsoft Presidio <https://github.com/microsoft/presidio>`_,
wired to the ``Transform`` intervention action so redaction happens in-place before
content reaches the model, downstream tools, or logs.
"""

import logging
from collections.abc import Iterable
from typing import Any, Literal, cast, get_args

from ...hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from ...interventions.actions import LifecycleEvent, Proceed, Transform
from ...interventions.handler import InterventionHandler
from ...types.content import ContentBlock, Message
from ...types.tools import ToolResult, ToolResultContent

logger = logging.getLogger(__name__)

RedactionEvent = Literal[
    "before_invocation",
    "before_tool_call",
    "after_tool_call",
    "before_model_call",
    "after_model_call",
]
"""Lifecycle events that ``PresidioRedaction`` can scan for PII."""

_DEFAULT_EVENTS: tuple[RedactionEvent, ...] = (
    "before_tool_call",
    "after_tool_call",
    "before_model_call",
    "after_model_call",
)
"""Default event set: redact tool arguments, tool results, and model context.

``before_invocation`` is omitted because ``before_model_call`` already scans the
full ``agent.messages`` list (which includes the initial user input) before each
model call, so redaction is applied without scanning the same content twice.
"""

_OPERATOR_DEFAULTS: dict[str, dict[str, Any]] = {
    "replace": {},
    "redact": {},
    "hash": {"hash_type": "sha256"},
    "mask": {"masking_char": "*", "chars_to_mask": 100, "from_end": False},
}
"""Per-operator parameter defaults merged under any caller-supplied ``operator_params``.

``mask`` requires explicit parameters (Presidio has no usable default), so a
sensible full-value mask is provided; ``chars_to_mask`` is clamped to each match
length by Presidio. ``replace`` with no ``new_value`` falls back to Presidio's
``<ENTITY_TYPE>`` placeholder.
"""

_IMPORT_ERROR_MESSAGE = (
    "PresidioRedaction requires the optional 'presidio' dependencies, which are not installed. "
    "Install them with: pip install 'strands-agents[presidio]' "
    "and download a spaCy model: python -m spacy download en_core_web_lg"
)


class PresidioRedaction(InterventionHandler):
    """Redacts PII in tool I/O and model context using Microsoft Presidio.

    Returns a ``Transform`` for each configured lifecycle event whose ``apply``
    analyzes the relevant text with Presidio's ``AnalyzerEngine`` and anonymizes
    detected PII in-place with its ``AnonymizerEngine``. Events that are not
    configured (or carry no text) are a no-op (``Proceed``), so later handlers and
    the agent see unredacted content only where redaction was not requested.

    Presidio is an optional dependency. The engines are imported lazily on first
    use and a clear ``ImportError`` is raised if they are missing.

    Example:
        ```python
        from strands import Agent
        from strands.vended_interventions.presidio import PresidioRedaction

        # Redact emails and phone numbers in tool I/O and model context
        agent = Agent(
            interventions=[
                PresidioRedaction(entities=["EMAIL_ADDRESS", "PHONE_NUMBER"]),
            ],
        )

        # Mask instead of replacing, and only scan tool results
        agent = Agent(
            interventions=[
                PresidioRedaction(operator="mask", events=["after_tool_call"]),
            ],
        )
        ```

    Note:
        ``name`` is a fixed class attribute, so at most one ``PresidioRedaction``
        can be registered per agent; layering two policies requires subclassing to
        rename. Scanning ``before_model_call`` mutates ``agent.messages`` in place,
        so PII is redacted from the persisted conversation history (not just the
        model request) -- this is intentional, to avoid leaking PII through session
        storage or logs.

    Scope and limitations:
        Redaction covers the text-bearing fields Presidio can analyze: ``text``
        content blocks, ``json`` tool-result/content blocks (recursively, strings
        only), tool-call inputs (``toolUse.input`` and ``before_tool_call`` args),
        nested ``toolResult`` blocks, and ``reasoningContent`` text. Binary or opaque
        content (``image``, ``document``, ``video``, encrypted reasoning) is left
        untouched. Non-string scalars (e.g. a number that is really an SSN) are not
        redacted because Presidio operates on text. Tokens streamed to a callback
        during ``after_model_call`` generation are emitted before redaction runs, so
        configure a downstream filter if real-time stream output must also be clean.

        Detection is only as good as Presidio's recognizers and the configured
        ``score_threshold``; treat this as defense-in-depth, not a guarantee. Redaction
        is idempotent: re-scanning already-redacted text (e.g. when ``before_model_call``
        sweeps the growing history each turn) does not double-redact, since the
        placeholders contain no detectable PII. That per-call full-history sweep does
        cost analyzer time proportional to conversation length; for long conversations,
        prefer scanning at the write boundaries (``after_tool_call`` /
        ``after_model_call``) so each piece of content is redacted once as it is added.
    """

    name = "strands:presidio-redaction"

    def __init__(
        self,
        *,
        entities: Iterable[str] | None = None,
        events: Iterable[RedactionEvent] | None = None,
        operator: Literal["replace", "mask", "hash", "redact"] = "replace",
        operator_params: dict[str, Any] | None = None,
        language: str = "en",
        score_threshold: float = 0.5,
        analyzer: Any | None = None,
        anonymizer: Any | None = None,
    ) -> None:
        """Initialize the handler.

        Args:
            entities: PII entity types to detect (e.g. ``["EMAIL_ADDRESS",
                "PHONE_NUMBER", "PERSON"]``). When ``None``, every entity supported
                by the analyzer's recognizers is detected.
            events: Lifecycle events to scan. Defaults to ``before_tool_call``,
                ``after_tool_call``, ``before_model_call`` and ``after_model_call``.
                Events not listed here are a no-op.
            operator: Anonymization operator applied to detected PII. ``"replace"``
                substitutes a placeholder (Presidio's ``<ENTITY_TYPE>`` by default),
                ``"mask"`` overwrites characters with a mask character, ``"hash"``
                replaces the value with its hash, and ``"redact"`` removes it.
            operator_params: Extra parameters forwarded to Presidio's
                ``OperatorConfig`` for ``operator``, merged over sensible defaults.
                For example ``{"new_value": "<REDACTED>"}`` for ``"replace"`` or
                ``{"masking_char": "#"}`` for ``"mask"``.
            language: Language code passed to the analyzer. Defaults to ``"en"``.
            score_threshold: Minimum detection confidence (``0.0``-``1.0``) for a
                match to be redacted. Defaults to ``0.5``.
            analyzer: Pre-built Presidio ``AnalyzerEngine`` (or compatible). When
                ``None``, one is created lazily on first use. Supplying your own lets
                you customize recognizers and avoids importing Presidio for tests.
            anonymizer: Pre-built Presidio ``AnonymizerEngine`` (or compatible). When
                ``None``, one is created lazily on first use.

        Raises:
            ValueError: If ``events`` is a bare string, or contains an unknown event
                name, or if ``operator`` is not supported.
        """
        # A bare string is iterable, so ``set("before_tool_call")`` would silently
        # become a per-char set; reject it the way HumanInTheLoop rejects allowed_tools.
        if isinstance(events, str):
            raise ValueError("events must be a list of event names, not a single string")

        selected = tuple(events) if events is not None else _DEFAULT_EVENTS
        valid_events = set(get_args(RedactionEvent))
        unknown = [event for event in selected if event not in valid_events]
        if unknown:
            raise ValueError(f"Unknown redaction event(s): {unknown}. Valid events: {sorted(valid_events)}")

        if operator not in _OPERATOR_DEFAULTS:
            raise ValueError(f"Unsupported operator: '{operator}'. Valid operators: {sorted(_OPERATOR_DEFAULTS)}")

        self._entities = list(entities) if entities is not None else None
        self._events: frozenset[RedactionEvent] = frozenset(selected)
        self._operator = operator
        self._operator_params = {**_OPERATOR_DEFAULTS[operator], **(operator_params or {})}
        self._language = language
        self._score_threshold = score_threshold
        # Typed as Any: these hold Presidio engines (an optional dependency without
        # type stubs) and start as None until built lazily on first use.
        self._analyzer: Any = analyzer
        self._anonymizer: Any = anonymizer
        self._operators: dict[str, Any] | None = None
        self._reason = "Presidio PII redaction"

    def before_invocation(self, event: BeforeInvocationEvent, **kwargs: Any) -> Proceed | Transform:
        """Redact PII in the invocation's input messages before processing begins.

        Args:
            event: The before-invocation event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A ``Transform`` that redacts ``event.messages`` when this event is
            configured, otherwise ``Proceed``.
        """
        if "before_invocation" not in self._events:
            return Proceed()
        return Transform(apply=self._apply_before_invocation, reason=self._reason)

    def before_tool_call(self, event: BeforeToolCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Redact PII in a tool call's input arguments before the tool runs.

        Args:
            event: The before-tool-call event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A ``Transform`` that redacts ``event.tool_use["input"]`` when this event
            is configured, otherwise ``Proceed``.
        """
        if "before_tool_call" not in self._events:
            return Proceed()
        return Transform(apply=self._apply_before_tool_call, reason=self._reason)

    def after_tool_call(self, event: AfterToolCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Redact PII in a tool result before it reaches the model or logs.

        Args:
            event: The after-tool-call event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A ``Transform`` that redacts ``event.result`` when this event is
            configured, otherwise ``Proceed``.
        """
        if "after_tool_call" not in self._events:
            return Proceed()
        return Transform(apply=self._apply_after_tool_call, reason=self._reason)

    def before_model_call(self, event: BeforeModelCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Redact PII in the conversation context before the model is invoked.

        Args:
            event: The before-model-call event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A ``Transform`` that redacts ``event.agent.messages`` when this event is
            configured, otherwise ``Proceed``.
        """
        if "before_model_call" not in self._events:
            return Proceed()
        return Transform(apply=self._apply_before_model_call, reason=self._reason)

    def after_model_call(self, event: AfterModelCallEvent, **kwargs: Any) -> Proceed | Transform:
        """Redact PII in the model's response before it is added to history or logged.

        Args:
            event: The after-model-call event under evaluation.
            **kwargs: Additional keyword arguments for future extensibility.

        Returns:
            A ``Transform`` that redacts ``event.stop_response.message`` when this
            event is configured, otherwise ``Proceed``.
        """
        if "after_model_call" not in self._events:
            return Proceed()
        return Transform(apply=self._apply_after_model_call, reason=self._reason)

    def _apply_before_invocation(self, event: LifecycleEvent) -> None:
        """Redact PII in the invocation input messages in-place."""
        if isinstance(event, BeforeInvocationEvent) and event.messages is not None:
            self._redact_messages(event.messages)

    def _apply_before_tool_call(self, event: LifecycleEvent) -> None:
        """Redact PII in the tool input arguments in-place."""
        if isinstance(event, BeforeToolCallEvent):
            event.tool_use["input"] = self._redact_value(event.tool_use.get("input"))

    def _apply_after_tool_call(self, event: LifecycleEvent) -> None:
        """Redact PII in the tool result in-place."""
        if isinstance(event, AfterToolCallEvent):
            self._redact_tool_result(event.result)

    def _apply_before_model_call(self, event: LifecycleEvent) -> None:
        """Redact PII across the agent's conversation history in-place."""
        if isinstance(event, BeforeModelCallEvent):
            self._redact_messages(event.agent.messages)

    def _apply_after_model_call(self, event: LifecycleEvent) -> None:
        """Redact PII in the model's response message in-place."""
        if isinstance(event, AfterModelCallEvent) and event.stop_response is not None:
            self._redact_message(event.stop_response.message)

    def _redact_messages(self, messages: Iterable[Message]) -> None:
        """Redact PII across every content block of every message in-place.

        Args:
            messages: The messages whose content blocks should be scanned.
        """
        for message in messages:
            self._redact_message(message)

    def _redact_message(self, message: Message) -> None:
        """Redact PII across every content block of a single message in-place.

        Args:
            message: The message whose content blocks should be scanned.
        """
        for block in message.get("content", []):
            self._redact_content_block(block)

    def _redact_tool_result(self, result: ToolResult) -> None:
        """Redact PII across every content block of a tool result in-place.

        Args:
            result: The tool result whose content blocks should be scanned.
        """
        for block in result.get("content", []):
            self._redact_content_block(block)

    def _redact_content_block(self, block: ContentBlock | ToolResultContent) -> None:
        """Redact PII in every text-bearing field of a single content block in-place.

        Handles each content shape that can carry free text or string-valued data:
        ``text`` blocks, ``json`` blocks (tool results), nested ``toolResult`` and
        ``toolUse`` blocks, and ``reasoningContent`` (extended-thinking) text. Binary
        and opaque blocks (``image``, ``document``, ``video``, encrypted
        ``redactedContent``) are left untouched -- Presidio operates on text only.

        Args:
            block: A content block (``ContentBlock`` or ``ToolResultContent``).
        """
        # TypedDicts are plain dicts at runtime; cast to a mutable mapping so the
        # heterogeneous key access/assignment below type-checks across both shapes.
        data = cast("dict[str, Any]", block)
        if "text" in data:
            data["text"] = self._redact_text(data["text"])
        if "json" in data:
            data["json"] = self._redact_value(data["json"])
        if "reasoningContent" in data:
            reasoning_text = data["reasoningContent"].get("reasoningText")
            if reasoning_text is not None and "text" in reasoning_text:
                reasoning_text["text"] = self._redact_text(reasoning_text["text"])
        if "toolUse" in data:
            data["toolUse"]["input"] = self._redact_value(data["toolUse"].get("input"))
        if "toolResult" in data:
            self._redact_tool_result(data["toolResult"])

    def _redact_value(self, value: Any) -> Any:
        """Recursively redact PII in strings nested within a value.

        Strings are redacted directly; dicts and lists are walked so PII in nested
        argument or JSON structures is caught. Other types (numbers, bools, ``None``,
        opaque objects) are returned unchanged -- Presidio operates on text only.

        Args:
            value: A tool input or JSON value (string, mapping, sequence, or scalar).

        Returns:
            The value with any string PII redacted.
        """
        if isinstance(value, str):
            return self._redact_text(value)
        if isinstance(value, dict):
            return {key: self._redact_value(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self._redact_value(item) for item in value]
        return value

    def _redact_text(self, text: str) -> str:
        """Analyze and anonymize PII in a single string.

        Args:
            text: The text to scan.

        Returns:
            The text with detected PII anonymized, or the original text when it is
            empty or contains no detectable PII.
        """
        if not text or not text.strip():
            return text

        self._ensure_engines()
        results = self._analyzer.analyze(
            text=text,
            language=self._language,
            entities=self._entities,
            score_threshold=self._score_threshold,
        )
        if not results:
            return text

        logger.debug(
            "matches=<%d>, operator=<%s>, language=<%s> | redacting detected pii",
            len(results),
            self._operator,
            self._language,
        )
        anonymized = self._anonymizer.anonymize(
            text=text,
            analyzer_results=results,
            operators=self._get_operators(),
        )
        redacted: str = anonymized.text
        return redacted

    def _ensure_engines(self) -> None:
        """Lazily construct the Presidio analyzer and anonymizer engines.

        Raises:
            ImportError: If the optional Presidio dependencies are not installed.
        """
        if self._analyzer is not None and self._anonymizer is not None:
            return

        analyzer_engine, anonymizer_engine, _ = self._import_presidio()
        if self._analyzer is None:
            logger.debug(
                "entities=<%s>, score_threshold=<%s> | building presidio analyzer engine",
                self._entities,
                self._score_threshold,
            )
            self._analyzer = analyzer_engine()
        if self._anonymizer is None:
            self._anonymizer = anonymizer_engine()

    def _get_operators(self) -> dict[str, Any]:
        """Build (and cache) the Presidio operator configuration.

        Returns:
            A mapping with a ``"DEFAULT"`` operator config applied to every entity.

        Raises:
            ImportError: If the optional Presidio dependencies are not installed.
        """
        if self._operators is None:
            _, _, operator_config = self._import_presidio()
            self._operators = {"DEFAULT": operator_config(self._operator, self._operator_params)}
        return self._operators

    @staticmethod
    def _import_presidio() -> tuple[Any, Any, Any]:
        """Import Presidio engines lazily with a helpful error if unavailable.

        Returns:
            The ``AnalyzerEngine`` class, ``AnonymizerEngine`` class, and
            ``OperatorConfig`` class.

        Raises:
            ImportError: If the optional Presidio dependencies are not installed.
        """
        try:
            from presidio_analyzer import AnalyzerEngine  # type: ignore[import-not-found]
            from presidio_anonymizer import AnonymizerEngine  # type: ignore[import-not-found]
            from presidio_anonymizer.entities import OperatorConfig  # type: ignore[import-not-found]
        except ImportError as exc:
            raise ImportError(_IMPORT_ERROR_MESSAGE) from exc
        return AnalyzerEngine, AnonymizerEngine, OperatorConfig
