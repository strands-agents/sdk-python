"""Tests for deployment packaging utilities."""

from unittest.mock import MagicMock

import pytest

from strands.experimental.deploy._exceptions import DeployPackagingException
from strands.experimental.deploy._packaging import (
    _strip_deploy_call,
    generate_agentcore_entrypoint,
)


class TestStripDeployCall:
    def test_removes_deploy_call(self):
        source = "from strands import Agent\nagent = Agent()\ndeploy(agent, name='test')\n"
        result = _strip_deploy_call(source)
        assert "deploy(" not in result
        assert "agent = Agent()" in result

    def test_removes_module_deploy_call(self):
        source = "import strands\nagent = Agent()\nstrands.deploy(agent)\n"
        result = _strip_deploy_call(source)
        assert "deploy(" not in result

    def test_removes_if_name_main(self):
        source = "agent = Agent()\nif __name__ == '__main__':\n    deploy(agent)\n"
        result = _strip_deploy_call(source)
        assert "__name__" not in result
        assert "__main__" not in result


    def test_preserves_non_deploy_code(self):
        source = "from strands import Agent\nfrom my_tools import search\nagent = Agent(tools=[search])\n"
        result = _strip_deploy_call(source)
        assert "from strands import Agent" in result
        assert "from my_tools import search" in result
        assert "Agent(tools=[search])" in result


class TestGenerateAgentcoreEntrypoint:
    def test_raises_when_caller_source_not_found(self):
        """When caller source can't be found (e.g., REPL), raises an error."""
        from unittest.mock import patch

        agent = MagicMock()
        agent.name = "my-agent"

        with patch("strands.experimental.deploy._packaging._find_caller_info", return_value=None):
            with pytest.raises(DeployPackagingException, match="Could not find the source file"):
                generate_agentcore_entrypoint(agent)
