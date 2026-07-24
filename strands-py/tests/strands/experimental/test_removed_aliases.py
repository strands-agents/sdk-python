"""Tests for experimental compatibility aliases that completed their removal lifecycle."""

import importlib

import pytest


@pytest.mark.parametrize(
    ("module_name", "attribute_name"),
    [
        ("strands.experimental.tools", "ToolProvider"),
        ("strands.experimental.hooks", "BeforeToolInvocationEvent"),
        ("strands.experimental.hooks", "AfterToolInvocationEvent"),
        ("strands.experimental.hooks", "BeforeModelInvocationEvent"),
        ("strands.experimental.hooks", "AfterModelInvocationEvent"),
        ("strands.experimental.hooks.events", "BeforeToolInvocationEvent"),
    ],
)
def test_removed_experimental_alias_is_unavailable(module_name, attribute_name):
    module = importlib.import_module(module_name)

    with pytest.raises(AttributeError):
        getattr(module, attribute_name)


@pytest.mark.parametrize(
    "module_name",
    [
        "strands.experimental.steering.core.action",
        "strands.experimental.hooks.multiagent.events",
    ],
)
def test_removed_experimental_module_is_unavailable(module_name):
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(module_name)
