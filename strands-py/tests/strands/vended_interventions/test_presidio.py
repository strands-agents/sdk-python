"""Tests for the PresidioRedaction vended intervention handler.

The Presidio engines are mocked throughout: a ``FakeAnalyzer`` finds configured
substrings and a ``FakeAnonymizer`` replaces each match with a placeholder. This
exercises the handler's wiring (which events return a ``Transform``, what each
``apply`` mutates, and how the config knobs are honoured) without installing
Presidio or downloading a spaCy model.
"""

import unittest.mock
from dataclasses import dataclass

import pytest

from strands import Agent
from strands.hooks.events import (
    AfterModelCallEvent,
    AfterToolCallEvent,
    BeforeInvocationEvent,
    BeforeModelCallEvent,
    BeforeToolCallEvent,
)
from strands.interrupt import _InterruptState
from strands.interventions.actions import Proceed, Transform
from strands.tools import tool
from strands.vended_interventions.presidio import PresidioRedaction
from tests.fixtures.mocked_model_provider import MockedModelProvider

EMAIL = "alice@example.com"
PLACEHOLDER = "<EMAIL_ADDRESS>"


@dataclass
class _FakeResult:
    """Minimal stand-in for Presidio's ``RecognizerResult``."""

    entity_type: str
    start: int
    end: int
    score: float


class _FakeAnalyzer:
    """Finds every occurrence of configured substrings as PII matches.

    Records the kwargs of the most recent ``analyze`` call so tests can assert that
    config knobs (language, entities, score_threshold) are forwarded correctly.
    """

    def __init__(self, needles: list[str] | None = None) -> None:
        self.needles = needles if needles is not None else [EMAIL]
        self.calls: list[dict] = []

    def analyze(self, *, text, language, entities, score_threshold):
        self.calls.append(
            {
                "text": text,
                "language": language,
                "entities": entities,
                "score_threshold": score_threshold,
            }
        )
        results: list[_FakeResult] = []
        for needle in self.needles:
            start = text.find(needle)
            while start != -1:
                results.append(_FakeResult("EMAIL_ADDRESS", start, start + len(needle), 0.99))
                start = text.find(needle, start + len(needle))
        return results


@dataclass
class _FakeAnonymized:
    """Minimal stand-in for Presidio's ``EngineResult`` (only ``.text`` is used)."""

    text: str


class _FakeAnonymizer:
    """Replaces each analyzer match with a placeholder built from the operator.

    Mirrors Presidio's contract closely enough for assertions: it sorts matches in
    reverse so index-based splicing stays valid, and honours a ``new_value`` /
    ``masking_char`` operator param so config-knob tests are meaningful.
    """

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def anonymize(self, *, text, analyzer_results, operators):
        self.calls.append({"text": text, "analyzer_results": analyzer_results, "operators": operators})
        config = operators["DEFAULT"]
        replacement = self._replacement(config)
        for result in sorted(analyzer_results, key=lambda r: r.start, reverse=True):
            value = replacement if replacement is not None else f"<{result.entity_type}>"
            text = text[: result.start] + value + text[result.end :]
        return _FakeAnonymized(text=text)

    @staticmethod
    def _replacement(config) -> str | None:
        params = config.params
        if config.operator_name == "replace":
            new_value = params.get("new_value")
            return str(new_value) if new_value else None
        if config.operator_name == "mask":
            return params.get("masking_char", "*") * 4
        return None


class _FakeOperatorConfig:
    """Stand-in for Presidio's ``OperatorConfig`` (name + params holder)."""

    def __init__(self, operator_name, params=None) -> None:
        self.operator_name = operator_name
        self.params = params or {}


@pytest.fixture(autouse=True)
def patch_presidio(monkeypatch):
    """Patch the single lazy-import seam so engines build without the real library.

    Every handler in this module (whether it injects engines or builds them lazily)
    resolves Presidio classes through ``_import_presidio``; patching it here keeps the
    whole suite free of Presidio and spaCy. ``TestLazyImport`` overrides this within
    its own tests to assert the missing-dependency behaviour.
    """
    monkeypatch.setattr(
        PresidioRedaction,
        "_import_presidio",
        staticmethod(lambda: (_FakeAnalyzer, _FakeAnonymizer, _FakeOperatorConfig)),
    )


def make_handler(**kwargs) -> PresidioRedaction:
    """Build a handler whose engines are built lazily from the patched fake import."""
    return PresidioRedaction(**kwargs)


def make_agent():
    """A lightweight mock agent for direct event construction."""
    agent = unittest.mock.Mock()
    agent._interrupt_state = _InterruptState()
    agent.messages = []
    return agent


def tool_use_message(name: str, tool_use_id: str = "tool-1", tool_input: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input or {}}}],
    }


def text_message(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


class TestTransformReturnedForConfiguredEvents:
    """A Transform is returned only for configured events; others are a no-op Proceed."""

    def test_default_events_return_transform(self):
        handler = make_handler()
        agent = make_agent()

        before_tool = handler.before_tool_call(
            BeforeToolCallEvent(
                agent=agent,
                selected_tool=None,
                tool_use={"toolUseId": "t", "name": "x", "input": {}},
                invocation_state={},
            )
        )
        after_tool = handler.after_tool_call(
            AfterToolCallEvent(
                agent=agent,
                selected_tool=None,
                tool_use={"toolUseId": "t", "name": "x", "input": {}},
                invocation_state={},
                result={"toolUseId": "t", "status": "success", "content": []},
            )
        )
        before_model = handler.before_model_call(BeforeModelCallEvent(agent=agent, invocation_state={}))
        after_model = handler.after_model_call(AfterModelCallEvent(agent=agent, invocation_state={}))

        assert isinstance(before_tool, Transform)
        assert isinstance(after_tool, Transform)
        assert isinstance(before_model, Transform)
        assert isinstance(after_model, Transform)

    def test_before_invocation_is_not_scanned_by_default(self):
        handler = make_handler()
        agent = make_agent()

        action = handler.before_invocation(BeforeInvocationEvent(agent=agent, messages=[]))

        assert isinstance(action, Proceed)

    def test_unconfigured_events_return_proceed(self):
        handler = make_handler(events=["after_tool_call"])
        agent = make_agent()

        before_tool = handler.before_tool_call(
            BeforeToolCallEvent(
                agent=agent,
                selected_tool=None,
                tool_use={"toolUseId": "t", "name": "x", "input": {}},
                invocation_state={},
            )
        )
        before_model = handler.before_model_call(BeforeModelCallEvent(agent=agent, invocation_state={}))
        after_model = handler.after_model_call(AfterModelCallEvent(agent=agent, invocation_state={}))

        assert isinstance(before_tool, Proceed)
        assert isinstance(before_model, Proceed)
        assert isinstance(after_model, Proceed)

        # The one configured event still returns a Transform.
        after_tool = handler.after_tool_call(
            AfterToolCallEvent(
                agent=agent,
                selected_tool=None,
                tool_use={"toolUseId": "t", "name": "x", "input": {}},
                invocation_state={},
                result={"toolUseId": "t", "status": "success", "content": []},
            )
        )
        assert isinstance(after_tool, Transform)


class TestApplyMutatesContent:
    """The Transform's apply mutates event content in-place: PII -> placeholder."""

    def test_before_tool_call_redacts_string_input(self):
        handler = make_handler()
        agent = make_agent()
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "send", "input": {"to": EMAIL, "count": 3}},
            invocation_state={},
        )

        action = handler.before_tool_call(event)
        action.apply(event)

        assert event.tool_use["input"]["to"] == PLACEHOLDER
        # Non-string args are untouched.
        assert event.tool_use["input"]["count"] == 3

    def test_before_tool_call_redacts_nested_input(self):
        handler = make_handler()
        agent = make_agent()
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={
                "toolUseId": "t",
                "name": "send",
                "input": {"recipients": [EMAIL, "bob"], "meta": {"cc": EMAIL}},
            },
            invocation_state={},
        )

        action = handler.before_tool_call(event)
        action.apply(event)

        assert event.tool_use["input"]["recipients"] == [PLACEHOLDER, "bob"]
        assert event.tool_use["input"]["meta"]["cc"] == PLACEHOLDER

    def test_after_tool_call_redacts_result_text(self):
        handler = make_handler()
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={
                "toolUseId": "t",
                "status": "success",
                "content": [{"text": f"Reply to {EMAIL} now"}, {"image": {"format": "png"}}],
            },
        )

        action = handler.after_tool_call(event)
        action.apply(event)

        assert event.result["content"][0]["text"] == f"Reply to {PLACEHOLDER} now"
        # Non-text/non-json content blocks are left untouched.
        assert event.result["content"][1]["image"] == {"format": "png"}

    def test_before_model_call_redacts_message_history(self):
        handler = make_handler()
        agent = make_agent()
        agent.messages = [
            {"role": "user", "content": [{"text": f"Contact {EMAIL}"}]},
            {"role": "assistant", "content": [{"text": "ok"}]},
        ]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        action = handler.before_model_call(event)
        action.apply(event)

        assert agent.messages[0]["content"][0]["text"] == f"Contact {PLACEHOLDER}"
        assert agent.messages[1]["content"][0]["text"] == "ok"

    def test_after_model_call_redacts_response_message(self):
        handler = make_handler()
        agent = make_agent()
        stop_response = AfterModelCallEvent.ModelStopResponse(
            message={"role": "assistant", "content": [{"text": f"Sent to {EMAIL}"}]},
            stop_reason="end_turn",
        )
        event = AfterModelCallEvent(agent=agent, invocation_state={}, stop_response=stop_response)

        action = handler.after_model_call(event)
        action.apply(event)

        assert event.stop_response.message["content"][0]["text"] == f"Sent to {PLACEHOLDER}"

    def test_after_model_call_with_no_response_is_safe(self):
        handler = make_handler()
        agent = make_agent()
        event = AfterModelCallEvent(agent=agent, invocation_state={}, stop_response=None)

        action = handler.after_model_call(event)
        action.apply(event)  # Must not raise when the model call failed.

    def test_before_invocation_redacts_input_messages(self):
        handler = make_handler(events=["before_invocation"])
        agent = make_agent()
        messages = [{"role": "user", "content": [{"text": f"My email is {EMAIL}"}]}]
        event = BeforeInvocationEvent(agent=agent, messages=messages)

        action = handler.before_invocation(event)
        assert isinstance(action, Transform)
        action.apply(event)

        assert messages[0]["content"][0]["text"] == f"My email is {PLACEHOLDER}"

    def test_text_without_pii_is_unchanged(self):
        handler = make_handler()
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": "nothing sensitive here"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert event.result["content"][0]["text"] == "nothing sensitive here"

    def test_empty_and_whitespace_text_skips_analyzer(self):
        analyzer = _FakeAnalyzer()
        handler = make_handler(analyzer=analyzer)
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": ""}, {"text": "   "}]},
        )

        handler.after_tool_call(event).apply(event)

        assert analyzer.calls == []


class TestConfigKnobs:
    """Config knobs (entities, language, threshold, operator) are honoured."""

    def test_language_entities_and_threshold_forwarded_to_analyzer(self):
        analyzer = _FakeAnalyzer()
        handler = make_handler(
            analyzer=analyzer,
            entities=["EMAIL_ADDRESS", "PERSON"],
            language="de",
            score_threshold=0.8,
        )
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"hi {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        call = analyzer.calls[0]
        assert call["language"] == "de"
        assert call["entities"] == ["EMAIL_ADDRESS", "PERSON"]
        assert call["score_threshold"] == 0.8

    def test_default_entities_is_none_meaning_all(self):
        analyzer = _FakeAnalyzer()
        handler = make_handler(analyzer=analyzer)
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"hi {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert analyzer.calls[0]["entities"] is None

    def test_replace_operator_uses_custom_new_value(self):
        handler = PresidioRedaction(
            analyzer=_FakeAnalyzer(),
            anonymizer=_FakeAnonymizer(),
            operator="replace",
            operator_params={"new_value": "<REDACTED>"},
        )
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert event.result["content"][0]["text"] == "to <REDACTED>"

    def test_mask_operator_uses_masking_char(self):
        handler = PresidioRedaction(
            analyzer=_FakeAnalyzer(),
            anonymizer=_FakeAnonymizer(),
            operator="mask",
            operator_params={"masking_char": "#"},
        )
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert event.result["content"][0]["text"] == "to ####"

    def test_operator_config_is_built_with_selected_operator(self):
        anonymizer = _FakeAnonymizer()
        handler = PresidioRedaction(analyzer=_FakeAnalyzer(), anonymizer=anonymizer, operator="hash")
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert anonymizer.calls[0]["operators"]["DEFAULT"].operator_name == "hash"


class TestComprehensiveContentCoverage:
    """Redaction reaches every text-bearing content shape, not just plain text blocks."""

    def test_after_tool_call_redacts_json_block(self):
        handler = make_handler()
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={
                "toolUseId": "t",
                "status": "success",
                "content": [{"json": {"contact": EMAIL, "rows": [{"email": EMAIL}], "count": 2}}],
            },
        )

        handler.after_tool_call(event).apply(event)

        redacted = event.result["content"][0]["json"]
        assert redacted["contact"] == PLACEHOLDER
        assert redacted["rows"][0]["email"] == PLACEHOLDER
        # Non-string scalars in JSON are left untouched.
        assert redacted["count"] == 2

    def test_before_model_call_redacts_tool_use_input_in_history(self):
        # The assistant's tool-call args persist in history; they must be redacted too.
        handler = make_handler()
        agent = make_agent()
        agent.messages = [
            {
                "role": "assistant",
                "content": [{"toolUse": {"toolUseId": "t", "name": "send", "input": {"to": EMAIL}}}],
            }
        ]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        handler.before_model_call(event).apply(event)

        assert agent.messages[0]["content"][0]["toolUse"]["input"]["to"] == PLACEHOLDER

    def test_before_model_call_redacts_tool_result_block_in_history(self):
        # Tool results live in user-role messages as toolResult blocks; sweep must reach them.
        handler = make_handler()
        agent = make_agent()
        agent.messages = [
            {
                "role": "user",
                "content": [
                    {
                        "toolResult": {
                            "toolUseId": "t",
                            "status": "success",
                            "content": [{"text": f"found {EMAIL}"}],
                        }
                    }
                ],
            }
        ]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        handler.before_model_call(event).apply(event)

        result_text = agent.messages[0]["content"][0]["toolResult"]["content"][0]["text"]
        assert result_text == f"found {PLACEHOLDER}"

    def test_redacts_reasoning_content_text(self):
        handler = make_handler()
        agent = make_agent()
        agent.messages = [
            {
                "role": "assistant",
                "content": [{"reasoningContent": {"reasoningText": {"text": f"user is {EMAIL}"}}}],
            }
        ]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        handler.before_model_call(event).apply(event)

        assert agent.messages[0]["content"][0]["reasoningContent"]["reasoningText"]["text"] == f"user is {PLACEHOLDER}"

    def test_non_text_blocks_pass_through_untouched(self):
        handler = make_handler()
        agent = make_agent()
        image_block = {"image": {"format": "png", "source": {"bytes": b"data"}}}
        agent.messages = [{"role": "user", "content": [image_block, {"text": f"see {EMAIL}"}]}]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        handler.before_model_call(event).apply(event)

        assert agent.messages[0]["content"][0] == {"image": {"format": "png", "source": {"bytes": b"data"}}}
        assert agent.messages[0]["content"][1]["text"] == f"see {PLACEHOLDER}"


class TestToolInputEdgeCases:
    """before_tool_call handles missing / None / scalar inputs without raising."""

    def test_missing_input_key_is_safe(self):
        handler = make_handler()
        agent = make_agent()
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x"},  # no "input" key
            invocation_state={},
        )

        handler.before_tool_call(event).apply(event)  # Must not raise.

        assert event.tool_use["input"] is None

    def test_none_input_is_safe(self):
        handler = make_handler()
        agent = make_agent()
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": None},
            invocation_state={},
        )

        handler.before_tool_call(event).apply(event)

        assert event.tool_use["input"] is None

    def test_scalar_input_is_left_unchanged(self):
        handler = make_handler()
        agent = make_agent()
        event = BeforeToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": 42},
            invocation_state={},
        )

        handler.before_tool_call(event).apply(event)

        assert event.tool_use["input"] == 42


class TestHistoryPersistenceAndIdempotency:
    """after_model_call redacts the message that gets persisted; redaction is idempotent."""

    def test_after_model_call_redacts_persisted_history(self):
        @tool(name="noop")
        def noop() -> str:
            return "ok"

        model = MockedModelProvider([text_message(f"Reply to {EMAIL}")])
        agent = Agent(
            model=model,
            tools=[noop],
            interventions=[make_handler(events=["after_model_call"])],
        )

        agent("hi")

        # The final assistant message persisted to history must be redacted, not just
        # the in-flight stop_response (they share the same object reference).
        last = agent.messages[-1]
        assert last["role"] == "assistant"
        joined = " ".join(block.get("text", "") for block in last["content"])
        assert EMAIL not in joined
        assert PLACEHOLDER in joined

    def test_redaction_is_idempotent(self):
        handler = make_handler()
        agent = make_agent()
        message = {"role": "user", "content": [{"text": f"reach {EMAIL}"}]}
        agent.messages = [message]
        event = BeforeModelCallEvent(agent=agent, invocation_state={})

        handler.before_model_call(event).apply(event)
        once = message["content"][0]["text"]
        # A second sweep over already-redacted history changes nothing.
        handler.before_model_call(event).apply(event)
        twice = message["content"][0]["text"]

        assert once == twice == f"reach {PLACEHOLDER}"


class TestValidation:
    """Constructor validation mirrors HumanInTheLoop's defensive checks."""

    def test_bare_string_events_raises(self):
        with pytest.raises(ValueError, match="must be a list"):
            PresidioRedaction(events="before_tool_call")

    def test_unknown_event_raises(self):
        with pytest.raises(ValueError, match="Unknown redaction event"):
            PresidioRedaction(events=["before_tool_call", "not_an_event"])

    def test_unsupported_operator_raises(self):
        with pytest.raises(ValueError, match="Unsupported operator"):
            PresidioRedaction(operator="encrypt")


class TestLazyImport:
    """Presidio is imported lazily; a clear error is raised when it's missing."""

    def test_no_presidio_import_at_module_load(self):
        # Importing the handler module must not import Presidio (optional dep).
        import sys

        assert "presidio_analyzer" not in sys.modules
        assert "presidio_anonymizer" not in sys.modules

    def test_helpful_error_when_presidio_missing(self, monkeypatch):
        # Restore the genuine _import_presidio (the autouse fixture patched it), then
        # rely on Presidio genuinely being absent so the real ImportError path runs.
        monkeypatch.undo()
        if "presidio_analyzer" in __import__("sys").modules:
            pytest.skip("presidio is installed; cannot exercise the missing-dependency path")

        handler = PresidioRedaction()
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        with pytest.raises(ImportError, match="strands-agents\\[presidio\\]"):
            handler.after_tool_call(event).apply(event)

    def test_no_import_attempted_when_engines_injected(self, monkeypatch):
        # Injected engines short-circuit the lazy import entirely.
        def boom():
            raise AssertionError("_import_presidio must not be called when engines are injected")

        monkeypatch.setattr(PresidioRedaction, "_import_presidio", staticmethod(boom))
        # OperatorConfig is also injected so no import is needed for anonymization.
        handler = PresidioRedaction(
            analyzer=_FakeAnalyzer(),
            anonymizer=_FakeAnonymizer(),
            operator_params={"new_value": "<REDACTED>"},
        )
        # Pre-seed the cached operator config so _get_operators skips the import too.
        handler._operators = {"DEFAULT": _FakeOperatorConfig("replace", {"new_value": "<REDACTED>"})}
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)  # Must not raise.

        assert event.result["content"][0]["text"] == "to <REDACTED>"

    def test_engines_built_lazily_from_import(self):
        # No injected engines: built lazily from the (patched) import on first use.
        handler = PresidioRedaction()
        agent = make_agent()
        event = AfterToolCallEvent(
            agent=agent,
            selected_tool=None,
            tool_use={"toolUseId": "t", "name": "x", "input": {}},
            invocation_state={},
            result={"toolUseId": "t", "status": "success", "content": [{"text": f"to {EMAIL}"}]},
        )

        handler.after_tool_call(event).apply(event)

        assert event.result["content"][0]["text"] == f"to {PLACEHOLDER}"


class TestEndToEndWithAgent:
    """The handler integrates with a real Agent via the intervention registry."""

    def test_redacts_tool_input_before_execution(self):
        seen_args = []

        @tool(name="send_email")
        def send_email(to: str) -> str:
            seen_args.append(to)
            return "sent"

        model = MockedModelProvider([tool_use_message("send_email", tool_input={"to": EMAIL}), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[send_email],
            interventions=[make_handler(events=["before_tool_call"])],
        )

        result = agent("Email the report")

        assert result.stop_reason == "end_turn"
        # The tool received the redacted address, never the real PII.
        assert seen_args == [PLACEHOLDER]

    def test_redacts_tool_result_after_execution(self):
        @tool(name="lookup")
        def lookup() -> str:
            return f"The contact is {EMAIL}"

        model = MockedModelProvider([tool_use_message("lookup"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[lookup],
            interventions=[make_handler(events=["after_tool_call"])],
        )

        agent("Look it up")

        tool_results = [
            block["toolResult"]
            for message in agent.messages
            for block in message.get("content", [])
            if "toolResult" in block
        ]
        assert tool_results, "expected a tool result in history"
        redacted_text = tool_results[0]["content"][0]["text"]
        assert EMAIL not in redacted_text
        assert PLACEHOLDER in redacted_text


class TestPublicExports:
    def test_presidio_redaction_is_exported(self):
        import strands.vended_interventions as vended
        import strands.vended_interventions.presidio as presidio

        assert "PresidioRedaction" in vended.__all__
        assert presidio.__all__ == ["PresidioRedaction"]
        assert vended.PresidioRedaction is presidio.PresidioRedaction
        assert presidio.PresidioRedaction.name == "strands:presidio-redaction"
