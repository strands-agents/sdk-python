import pytest

from strands.agent.agent import Agent
from strands.tools.class_loader import load_tools_from_instance


class MyTestClass:
    def foo(self, x: int) -> int:
        """Add 1 to x."""
        return x + 1

    @staticmethod
    def bar(y: int) -> int:
        """Multiply y by 2."""
        return y * 2

    @classmethod
    def baz(cls, z: int) -> int:
        """Subtract 1 from z."""
        return z - 1

    not_a_method = 42


def test_agent_tool_invocation_for_all_method_types():
    """Test that agent.tool.{tool_name} works for instance, static, and class methods."""
    instance = MyTestClass()
    prefix = "agenttest"
    tools = load_tools_from_instance(instance, prefix=prefix)
    agent = Agent(tools=tools)
    # Instance method
    result_foo = agent.tool.agenttest_foo(x=5)
    assert result_foo["status"] == "success"
    assert result_foo["content"][0]["text"] == "6"
    # Static method
    result_bar = agent.tool.bar(y=3)
    assert result_bar["status"] == "success"
    assert result_bar["content"][0]["text"] == "6"
    # Class method
    result_baz = agent.tool.baz(z=10)
    assert result_baz["status"] == "success"
    assert result_baz["content"][0]["text"] == "9"


def test_non_callable_attributes_are_skipped():
    """Test that non-callable attributes are not loaded as tools."""

    class NonCallable:
        foo = 123

        def bar(self):
            return 1

    instance = NonCallable()
    tools = load_tools_from_instance(instance, prefix="nc")
    tool_names = {tool.tool_name for tool in tools}
    assert "nc_foo" not in tool_names
    assert "nc_bar" in tool_names


def test_error_handling_for_unconvertible_methods(monkeypatch):
    """Test that a warning is logged and method is skipped if it cannot be converted."""

    class BadClass:
        def bad(self, x):
            return x

    instance = BadClass()
    # Patch FunctionToolMetadata to raise Exception
    from strands.tools import class_loader

    orig = class_loader.FunctionToolMetadata.__init__

    def fail_init(self, func):
        raise ValueError("fail")

    monkeypatch.setattr(class_loader.FunctionToolMetadata, "__init__", fail_init)
    with pytest.raises(ValueError):
        # Direct instantiation should raise
        class_loader.GenericFunctionTool(instance.bad)
    # But loader should skip and not raise
    tools = load_tools_from_instance(instance, prefix="bad")
    assert tools == []
    # Restore
    monkeypatch.setattr(class_loader.FunctionToolMetadata, "__init__", orig)


def test_default_prefix_is_instance_id():
    """Test that the default prefix is id(instance) when no prefix is provided."""
    instance = MyTestClass()
    tools = load_tools_from_instance(instance)
    expected_prefix = str(id(instance))
    tool_names = {tool.tool_name for tool in tools}
    assert f"{expected_prefix}_foo" in tool_names
    assert "bar" in tool_names
    assert "baz" in tool_names


def test_multiple_instances_of_same_class():
    """Test loading tools from multiple instances of the same class, including a static method."""

    class Counter:
        def __init__(self, start):
            self.start = start

        def increment(self, x: int) -> int:
            return self.start + x

        @staticmethod
        def double_static(y: int) -> int:
            return y * 2

    a = Counter(10)
    b = Counter(100)
    tools_a = load_tools_from_instance(a, prefix="a")
    tools_b = load_tools_from_instance(b, prefix="b")
    agent = Agent(tools=tools_a + tools_b)
    # Call increment for each instance
    result_a = agent.tool.a_increment(x=5)
    result_b = agent.tool.b_increment(x=5)
    assert result_a["status"] == "success"
    assert result_b["status"] == "success"
    assert result_a["content"][0]["text"] == "15"
    assert result_b["content"][0]["text"] == "105"
    # Static method should be available (not disambiguated)
    print(f"agent.tool_names: {agent.tool_names}")
    result_static = agent.tool.double_static(y=7)
    assert result_static["status"] == "success"
    assert result_static["content"][0]["text"] == "14"
    # Tool names are unique for instance methods, static method is shared
    tool_names = set(agent.tool_names)
    assert "a_increment" in tool_names
    assert "b_increment" in tool_names
    assert "double_static" in tool_names
