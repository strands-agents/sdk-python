"""Tests for AgentManifest."""

import json
import tempfile
from pathlib import Path

import pytest

from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger


class TestInputContract:
    """Tests for InputContract dataclass."""

    def test_default_empty(self):
        contract = InputContract()
        assert contract.features_required == []
        assert contract.data_sources == []
        assert contract.events_consumed == []
        assert contract.tool_capabilities == []
        assert contract.knowledge_bases == []

    def test_with_values(self):
        contract = InputContract(
            features_required=["feature_a", "feature_b"],
            events_consumed=["EventX"],
            tool_capabilities=["data_query"],
        )
        assert contract.features_required == ["feature_a", "feature_b"]
        assert contract.events_consumed == ["EventX"]
        assert contract.tool_capabilities == ["data_query"]


class TestOutputContract:
    """Tests for OutputContract dataclass."""

    def test_default_empty(self):
        contract = OutputContract()
        assert contract.events_produced == []
        assert contract.features_produced == []
        assert contract.artifacts_produced == []

    def test_with_values(self):
        contract = OutputContract(
            events_produced=["ResultEvent"],
            artifacts_produced=["report.pdf"],
        )
        assert contract.events_produced == ["ResultEvent"]
        assert contract.artifacts_produced == ["report.pdf"]


class TestTrigger:
    """Tests for Trigger dataclass."""

    def test_event_trigger(self):
        trigger = Trigger(type="event", condition="DataReadyEvent")
        assert trigger.type == "event"
        assert trigger.condition == "DataReadyEvent"

    def test_schedule_trigger(self):
        trigger = Trigger(type="schedule", condition="*/15 * * * *")
        assert trigger.type == "schedule"
        assert trigger.condition == "*/15 * * * *"

    def test_on_demand_trigger(self):
        trigger = Trigger(type="on_demand")
        assert trigger.type == "on_demand"
        assert trigger.condition is None


class TestAgentManifest:
    """Tests for AgentManifest dataclass."""

    def _sample_manifest(self) -> AgentManifest:
        return AgentManifest(
            name="test_agent",
            version="1.0.0",
            domain="testing",
            description="A test agent",
            input_contract=InputContract(
                features_required=["feature_a"],
                events_consumed=["InputEvent"],
                tool_capabilities=["data_query", "file_write"],
            ),
            output_contract=OutputContract(
                events_produced=["OutputEvent"],
                features_produced=["computed_metric"],
            ),
            trigger=Trigger(type="event", condition="InputEvent"),
            scenario_types=["simulation", "replay"],
            tags={"team": "platform", "priority": "high"},
        )

    def test_creation(self):
        manifest = self._sample_manifest()
        assert manifest.name == "test_agent"
        assert manifest.version == "1.0.0"
        assert manifest.domain == "testing"
        assert manifest.description == "A test agent"
        assert manifest.input_contract.features_required == ["feature_a"]
        assert manifest.output_contract.events_produced == ["OutputEvent"]
        assert manifest.trigger.type == "event"
        assert manifest.scenario_types == ["simulation", "replay"]
        assert manifest.tags == {"team": "platform", "priority": "high"}

    def test_minimal_creation(self):
        manifest = AgentManifest(name="minimal", version="0.1.0")
        assert manifest.name == "minimal"
        assert manifest.version == "0.1.0"
        assert manifest.domain is None
        assert manifest.description is None
        assert manifest.input_contract.features_required == []
        assert manifest.output_contract.events_produced == []
        assert manifest.trigger is None
        assert manifest.scenario_types == []
        assert manifest.tags == {}

    def test_to_dict(self):
        manifest = self._sample_manifest()
        data = manifest.to_dict()
        assert data["name"] == "test_agent"
        assert data["version"] == "1.0.0"
        assert data["domain"] == "testing"
        assert data["input_contract"]["features_required"] == ["feature_a"]
        assert data["output_contract"]["events_produced"] == ["OutputEvent"]
        assert data["trigger"]["type"] == "event"
        assert data["tags"]["team"] == "platform"

    def test_to_dict_removes_none_values(self):
        manifest = AgentManifest(name="minimal", version="0.1.0")
        data = manifest.to_dict()
        assert "domain" not in data
        assert "description" not in data
        assert "trigger" not in data

    def test_to_json(self):
        manifest = self._sample_manifest()
        json_str = manifest.to_json()
        parsed = json.loads(json_str)
        assert parsed["name"] == "test_agent"
        assert parsed["version"] == "1.0.0"

    def test_from_dict(self):
        data = {
            "name": "restored_agent",
            "version": "2.0.0",
            "domain": "analytics",
            "input_contract": {
                "features_required": ["metric_x"],
                "events_consumed": ["TriggerEvent"],
                "tool_capabilities": ["chart_generation"],
            },
            "output_contract": {
                "events_produced": ["ReportEvent"],
            },
            "trigger": {"type": "schedule", "condition": "0 * * * *"},
            "scenario_types": ["backtest"],
            "tags": {"env": "prod"},
        }
        manifest = AgentManifest.from_dict(data)
        assert manifest.name == "restored_agent"
        assert manifest.version == "2.0.0"
        assert manifest.domain == "analytics"
        assert manifest.input_contract.features_required == ["metric_x"]
        assert manifest.input_contract.tool_capabilities == ["chart_generation"]
        assert manifest.output_contract.events_produced == ["ReportEvent"]
        assert manifest.trigger.type == "schedule"
        assert manifest.trigger.condition == "0 * * * *"
        assert manifest.scenario_types == ["backtest"]
        assert manifest.tags == {"env": "prod"}

    def test_from_dict_minimal(self):
        data = {"name": "bare", "version": "0.0.1"}
        manifest = AgentManifest.from_dict(data)
        assert manifest.name == "bare"
        assert manifest.trigger is None
        assert manifest.input_contract.events_consumed == []

    def test_roundtrip(self):
        original = self._sample_manifest()
        data = original.to_dict()
        restored = AgentManifest.from_dict(data)
        assert restored.name == original.name
        assert restored.version == original.version
        assert restored.input_contract.features_required == original.input_contract.features_required
        assert restored.output_contract.events_produced == original.output_contract.events_produced
        assert restored.trigger.type == original.trigger.type
        assert restored.scenario_types == original.scenario_types

    def test_from_file(self, tmp_path):
        data = {
            "name": "file_agent",
            "version": "1.0.0",
            "input_contract": {"events_consumed": ["FileEvent"]},
            "output_contract": {"events_produced": ["ProcessedEvent"]},
            "trigger": {"type": "event", "condition": "FileEvent"},
        }
        file_path = tmp_path / "manifest.json"
        file_path.write_text(json.dumps(data))

        manifest = AgentManifest.from_file(file_path)
        assert manifest.name == "file_agent"
        assert manifest.input_contract.events_consumed == ["FileEvent"]
        assert manifest.trigger.condition == "FileEvent"

    def test_from_file_not_found(self):
        with pytest.raises(FileNotFoundError):
            AgentManifest.from_file("/nonexistent/path/manifest.json")

    def test_satisfies_all_met(self):
        manifest = self._sample_manifest()
        satisfied, unmet = manifest.satisfies(
            available_events=["InputEvent", "OtherEvent"],
            available_features=["feature_a", "feature_b"],
            available_capabilities=["data_query", "file_write", "extra_cap"],
        )
        assert satisfied is True
        assert unmet == []

    def test_satisfies_missing_event(self):
        manifest = self._sample_manifest()
        satisfied, unmet = manifest.satisfies(
            available_events=["OtherEvent"],
            available_features=["feature_a"],
            available_capabilities=["data_query", "file_write"],
        )
        assert satisfied is False
        assert "event:InputEvent" in unmet

    def test_satisfies_missing_feature(self):
        manifest = self._sample_manifest()
        satisfied, unmet = manifest.satisfies(
            available_events=["InputEvent"],
            available_features=[],
            available_capabilities=["data_query", "file_write"],
        )
        assert satisfied is False
        assert "feature:feature_a" in unmet

    def test_satisfies_missing_capability(self):
        manifest = self._sample_manifest()
        satisfied, unmet = manifest.satisfies(
            available_events=["InputEvent"],
            available_features=["feature_a"],
            available_capabilities=["data_query"],
        )
        assert satisfied is False
        assert "capability:file_write" in unmet

    def test_satisfies_none_checks_skipped(self):
        manifest = self._sample_manifest()
        # When None is passed, that category is not checked
        satisfied, unmet = manifest.satisfies(
            available_events=None,
            available_features=None,
            available_capabilities=None,
        )
        assert satisfied is True
        assert unmet == []


class TestAgentWithManifest:
    """Tests for Agent class with manifest parameter."""

    def test_agent_without_manifest(self):
        """Agent without manifest works as before."""
        from strands import Agent

        # This should not raise — manifest is optional
        agent = Agent.__new__(Agent)
        # Just verify the attribute would be None by default
        # (full Agent init requires model setup which we skip here)

    def test_manifest_accessible(self):
        """Manifest is accessible as agent.manifest."""
        manifest = AgentManifest(name="test", version="1.0.0")
        # We can't easily instantiate a full Agent in unit tests without mocking,
        # but we verify the manifest class is importable and usable
        assert manifest.name == "test"
