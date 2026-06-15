"""Tests for the HumanInTheLoop vended intervention handler."""

import pytest

from strands import Agent
from strands.tools import tool
from strands.vended_interventions.hitl import HumanInTheLoop
from tests.fixtures.mocked_model_provider import MockedModelProvider


def tool_use_message(name: str, tool_use_id: str = "tool-1", tool_input: dict | None = None) -> dict:
    return {
        "role": "assistant",
        "content": [{"toolUse": {"toolUseId": tool_use_id, "name": name, "input": tool_input or {}}}],
    }


def text_message(text: str) -> dict:
    return {"role": "assistant", "content": [{"text": text}]}


class TestDefaultInterruptResume:
    def test_pauses_agent_with_interrupt_on_any_tool_call(self):
        executed = []

        @tool(name="any_tool")
        def any_tool() -> str:
            executed.append(True)
            return "result"

        model = MockedModelProvider([tool_use_message("any_tool", tool_input={"x": 1}), text_message("Done")])
        agent = Agent(model=model, tools=[any_tool], interventions=[HumanInTheLoop()])

        result = agent("Do something")

        assert result.stop_reason == "interrupt"
        assert result.interrupts is not None
        assert len(result.interrupts) == 1
        assert result.interrupts[0].name == "strands:human-in-the-loop"
        assert "any_tool" in result.interrupts[0].reason
        assert executed == []

    def test_resume_with_approval_executes_tool(self):
        executed = []

        @tool(name="any_tool")
        def any_tool() -> str:
            executed.append(True)
            return "result"

        model = MockedModelProvider([tool_use_message("any_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[any_tool], interventions=[HumanInTheLoop()])

        result = agent("Do something")
        assert result.stop_reason == "interrupt"

        interrupt_id = result.interrupts[0].id
        result = agent([{"interruptResponse": {"interruptId": interrupt_id, "response": "yes"}}])

        assert result.stop_reason == "end_turn"
        assert executed == [True]

    def test_resume_with_rejection_cancels_tool(self):
        executed = []

        @tool(name="any_tool")
        def any_tool() -> str:
            executed.append(True)
            return "result"

        model = MockedModelProvider([tool_use_message("any_tool"), text_message("Understood")])
        agent = Agent(model=model, tools=[any_tool], interventions=[HumanInTheLoop()])

        result = agent("Do something")
        assert result.stop_reason == "interrupt"

        interrupt_id = result.interrupts[0].id
        result = agent([{"interruptResponse": {"interruptId": interrupt_id, "response": "no"}}])

        assert result.stop_reason == "end_turn"
        assert executed == []


class TestInlineAskMode:
    def test_allows_tool_execution_when_approved(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask=lambda prompt: "yes")])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == [True]

    def test_denies_tool_execution_when_rejected(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Understood")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask=lambda prompt: "no")])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == []

    def test_ask_returning_none_denies_instead_of_pausing(self):
        # A configured inline ask returning None (e.g. a dismissed dialog) must
        # fail closed as a deny, not silently fall back to interrupt/resume which
        # a stateless inline caller cannot resume.
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Understood")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask=lambda prompt: None)])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == []

    def test_async_ask_returning_none_denies_instead_of_pausing(self):
        # Same fail-closed contract as the sync case, but exercising the awaited
        # branch where the async ``ask`` resolves to None.
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        async def ask(prompt: str):
            return None

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Understood")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask=ask)])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == []

    def test_supports_async_ask_function(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        async def ask(prompt: str) -> str:
            return "yes"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask=ask)])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == [True]


class TestAllowedTools:
    def test_bare_string_allowed_tools_raises(self):
        # str is iterable, so a bare string would be shredded into a per-character
        # set and silently mis-gate every tool. Reject it explicitly.
        with pytest.raises(ValueError, match="must be a list"):
            HumanInTheLoop(allowed_tools="read_file")

    def test_does_not_prompt_for_tools_in_allowed_tools(self):
        executed = []

        @tool(name="read_file")
        def read_file() -> str:
            executed.append(True)
            return "content"

        model = MockedModelProvider([tool_use_message("read_file"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[read_file],
            interventions=[HumanInTheLoop(allowed_tools=["read_file"], ask=lambda prompt: "no")],
        )

        result = agent("Read it")

        assert result.stop_reason == "end_turn"
        assert executed == [True]

    def test_prompts_for_tools_not_in_allowed_tools(self):
        executed = []

        @tool(name="delete_file")
        def delete_file() -> str:
            executed.append(True)
            return "deleted"

        model = MockedModelProvider([tool_use_message("delete_file"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[delete_file],
            interventions=[HumanInTheLoop(allowed_tools=["read_file"], ask=lambda prompt: "no")],
        )

        agent("Delete it")

        assert executed == []

    def test_allows_all_tools_except_negated_ones(self):
        exec_log = []

        @tool(name="read_file")
        def read_file() -> str:
            exec_log.append("read")
            return "content"

        @tool(name="delete_file")
        def delete_file() -> str:
            exec_log.append("delete")
            return "deleted"

        model = MockedModelProvider(
            [
                {
                    "role": "assistant",
                    "content": [
                        {"toolUse": {"toolUseId": "tool-1", "name": "read_file", "input": {}}},
                        {"toolUse": {"toolUseId": "tool-2", "name": "delete_file", "input": {}}},
                    ],
                },
                text_message("Done"),
            ]
        )
        agent = Agent(
            model=model,
            tools=[read_file, delete_file],
            interventions=[HumanInTheLoop(allowed_tools=["*", "!delete_file"], ask=lambda prompt: "no")],
        )

        agent("Do both")

        assert "read" in exec_log
        assert "delete" not in exec_log

    def test_allows_all_tools_with_wildcard(self):
        executed = []

        @tool(name="dangerous_tool")
        def dangerous_tool() -> str:
            executed.append(True)
            return "ran"

        model = MockedModelProvider([tool_use_message("dangerous_tool"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[dangerous_tool],
            interventions=[HumanInTheLoop(allowed_tools=["*"], ask=lambda prompt: "no")],
        )

        result = agent("Do it")

        assert result.stop_reason == "end_turn"
        assert executed == [True]


class TestAskCallback:
    def test_passes_tool_name_and_input_in_prompt(self):
        prompts = []

        @tool(name="send_email")
        def send_email(to: str) -> str:
            return "sent"

        def ask(prompt: str) -> str:
            prompts.append(prompt)
            return "yes"

        model = MockedModelProvider(
            [tool_use_message("send_email", tool_input={"to": "bob@example.com"}), text_message("Done")]
        )
        agent = Agent(model=model, tools=[send_email], interventions=[HumanInTheLoop(ask=ask)])

        agent("Send email")

        assert "send_email" in prompts[0]
        assert "bob@example.com" in prompts[0]

    def test_supports_custom_evaluate_function(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[my_tool],
            interventions=[
                HumanInTheLoop(
                    ask=lambda prompt: "magic-word",
                    evaluate=lambda response: response == "magic-word",
                )
            ],
        )

        result = agent("Go")

        assert result.stop_reason == "end_turn"
        assert executed == [True]


class TestTrustMode:
    def test_trusts_tool_for_session_when_response_is_t(self):
        ask_count = []

        @tool(name="my_tool")
        def my_tool() -> str:
            return "executed"

        def ask(prompt: str) -> str:
            ask_count.append(1)
            return "t"

        model = MockedModelProvider(
            [
                tool_use_message("my_tool", "tool-1"),
                tool_use_message("my_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(enable_trust=True, ask=ask)])

        agent("Run tool twice")

        assert len(ask_count) == 1

    def test_does_not_trust_when_enable_trust_is_false(self):
        ask_count = []

        @tool(name="my_tool")
        def my_tool() -> str:
            return "executed"

        def ask(prompt: str) -> str:
            ask_count.append(1)
            return "t"

        model = MockedModelProvider(
            [
                tool_use_message("my_tool", "tool-1"),
                tool_use_message("my_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(enable_trust=False, ask=ask)])

        agent("Run tool twice")

        # 't' is not recognized as approval when trust is disabled, so the tool is
        # denied both times, but ask is still called both times (no trust memory)
        assert len(ask_count) == 2

    def test_t_response_also_approves_current_tool_call(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(
            model=model,
            tools=[my_tool],
            interventions=[HumanInTheLoop(enable_trust=True, ask=lambda prompt: "t")],
        )

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == [True]

    @pytest.mark.parametrize("trust_response", ["trust", "T", "TRUST"])
    def test_trusts_with_alternate_trust_responses(self, trust_response):
        ask_count = []

        @tool(name="my_tool")
        def my_tool() -> str:
            return "executed"

        def ask(prompt: str) -> str:
            ask_count.append(1)
            return trust_response

        model = MockedModelProvider(
            [
                tool_use_message("my_tool", "tool-1"),
                tool_use_message("my_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(enable_trust=True, ask=ask)])

        agent("Run tool twice")

        assert len(ask_count) == 1

    def test_supports_custom_evaluate_trust_function(self):
        ask_count = []

        @tool(name="my_tool")
        def my_tool() -> str:
            return "executed"

        def ask(prompt: str) -> str:
            ask_count.append(1)
            return "approve-and-remember"

        model = MockedModelProvider(
            [
                tool_use_message("my_tool", "tool-1"),
                tool_use_message("my_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(
            model=model,
            tools=[my_tool],
            interventions=[
                HumanInTheLoop(
                    enable_trust=True,
                    evaluate_trust=lambda r: r == "approve-and-remember",
                    ask=ask,
                )
            ],
        )

        agent("Run tool twice")

        assert len(ask_count) == 1

    def test_negated_tools_cannot_be_trusted(self):
        ask_count = []
        executed = []

        @tool(name="danger_tool")
        def danger_tool() -> str:
            executed.append(True)
            return "ran"

        def ask(prompt: str) -> str:
            ask_count.append(1)
            return "t"

        model = MockedModelProvider(
            [
                tool_use_message("danger_tool", "tool-1"),
                tool_use_message("danger_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(
            model=model,
            tools=[danger_tool],
            interventions=[
                HumanInTheLoop(allowed_tools=["*", "!danger_tool"], enable_trust=True, ask=ask),
            ],
        )

        agent("Run danger twice")

        assert len(ask_count) == 2
        assert executed == []

    def test_trust_via_interrupt_resume_mode(self):
        """Trust responses work in interrupt/resume mode (no ask)."""
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        model = MockedModelProvider(
            [
                tool_use_message("my_tool", "tool-1"),
                tool_use_message("my_tool", "tool-2"),
                text_message("Done"),
            ]
        )
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(enable_trust=True)])

        result = agent("Run tool twice")
        assert result.stop_reason == "interrupt"

        # Resume with a trust response: approves the current call AND trusts the tool
        interrupt_id = result.interrupts[0].id
        result = agent([{"interruptResponse": {"interruptId": interrupt_id, "response": "t"}}])

        # Second call to my_tool is trusted: no new interrupt raised
        assert result.stop_reason == "end_turn"
        assert executed == [True, True]
        assert agent.state.get("hitl:trusted_tools") == ["my_tool"]


class TestStdioMode:
    def test_stdio_ask_prompts_via_input(self, monkeypatch):
        prompts = []
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return "y"

        monkeypatch.setattr("builtins.input", fake_input)

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(ask="stdio")])

        result = agent("Run tool")

        assert result.stop_reason == "end_turn"
        assert executed == [True]
        assert len(prompts) == 1
        assert "my_tool" in prompts[0]
        assert "(y/n)" in prompts[0]

    def test_stdio_ask_includes_trust_option_when_enabled(self, monkeypatch):
        prompts = []
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "executed"

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return "n"

        monkeypatch.setattr("builtins.input", fake_input)

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(enable_trust=True, ask="stdio")])

        agent("Run tool")

        assert "(y/n/t)" in prompts[0]
        # "n" must actually deny: the tool should never run in stdio mode.
        assert executed == []


class TestPublicCallbackProtocols:
    """The callback Protocols are part of the public API so customers know what to pass."""

    def test_protocols_are_exported(self):
        from strands.vended_interventions import AskCallback, EvaluateCallback, HumanInTheLoop
        from strands.vended_interventions.hitl import AskCallback as AskFromHitl
        from strands.vended_interventions.hitl import EvaluateCallback as EvaluateFromHitl

        # Re-exported from both the package and the submodule so either import works.
        assert AskCallback is AskFromHitl
        assert EvaluateCallback is EvaluateFromHitl
        assert HumanInTheLoop.name == "strands:human-in-the-loop"

    def test_callbacks_are_runtime_checkable(self):
        from strands.vended_interventions import AskCallback, EvaluateCallback

        class MyAsk:
            def __call__(self, prompt: str, **kwargs) -> str:
                return "y"

        class MyEvaluate:
            def __call__(self, response, **kwargs) -> bool:
                return response == "approve"

        assert isinstance(MyAsk(), AskCallback)
        assert isinstance(MyEvaluate(), EvaluateCallback)

    def test_custom_evaluate_accepting_kwargs_is_used(self):
        executed = []

        @tool(name="my_tool")
        def my_tool() -> str:
            executed.append(True)
            return "ok"

        # A forward-compatible callback that accepts **kwargs (the Protocol shape).
        def approve_on_emoji(response, **kwargs) -> bool:
            return response == "👍"

        model = MockedModelProvider([tool_use_message("my_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[my_tool], interventions=[HumanInTheLoop(evaluate=approve_on_emoji)])

        result = agent("Run tool")
        assert result.stop_reason == "interrupt"

        interrupt_id = result.interrupts[0].id
        resumed = agent([{"interruptResponse": {"interruptId": interrupt_id, "response": "👍"}}])
        assert resumed.stop_reason == "end_turn"
        assert executed == [True]


class TestCustomInterventionHandler:
    """A custom handler may override before_tool_call as async and be awaited at runtime.

    The registry awaits coroutine overrides, so an ``async def`` handler runs
    end-to-end. Widening the base class so async overrides also type-check
    without a per-file ``# type: ignore`` is tracked in harness-sdk#2800.
    """

    def test_async_handler_returning_proceed(self):
        from strands.interventions.actions import Proceed
        from strands.interventions.handler import InterventionHandler

        executed = []

        @tool(name="ok_tool")
        def ok_tool() -> str:
            executed.append(True)
            return "ran"

        class AsyncAllow(InterventionHandler):
            name = "test:async-allow"

            async def before_tool_call(self, event, **kwargs):
                return Proceed()

        model = MockedModelProvider([tool_use_message("ok_tool"), text_message("Done")])
        agent = Agent(model=model, tools=[ok_tool], interventions=[AsyncAllow()])

        result = agent("Run tool")
        assert result.stop_reason == "end_turn"
        assert executed == [True]
