"""Tests for strands.composition module."""

import pytest

from strands.agent.manifest import AgentManifest, InputContract, OutputContract, Trigger
from strands.composition import (
    BindingResolver,
    CompositionRegistry,
    EventRouter,
    LifecycleManager,
    Route,
)


# --- Fixtures ---


def make_manifest(name, events_consumed=None, events_produced=None,
                  features_required=None, tool_capabilities=None,
                  knowledge_bases=None, trigger_type="event", trigger_condition=None):
    """Helper to create manifests for testing."""
    return AgentManifest(
        name=name,
        version="1.0.0",
        input_contract=InputContract(
            events_consumed=events_consumed or [],
            features_required=features_required or [],
            tool_capabilities=tool_capabilities or [],
            knowledge_bases=knowledge_bases or [],
        ),
        output_contract=OutputContract(
            events_produced=events_produced or [],
        ),
        trigger=Trigger(type=trigger_type, condition=trigger_condition),
    )


class FakeAgent:
    """Minimal agent-like object for testing."""
    def __init__(self, manifest):
        self.manifest = manifest


# --- Registry Tests ---


class TestCompositionRegistry:
    def test_register_agent(self):
        registry = CompositionRegistry()
        manifest = make_manifest("agent_a", events_produced=["EventA"])
        agent = FakeAgent(manifest)
        registry.register(agent)

        assert "agent_a" in registry
        assert len(registry) == 1
        assert registry.get("agent_a") is agent

    def test_register_manifest_only(self):
        registry = CompositionRegistry()
        manifest = make_manifest("agent_b", events_consumed=["EventA"])
        registry.register_manifest(manifest)

        assert "agent_b" in registry
        assert registry.get("agent_b") is None  # no agent instance
        assert registry.get_manifest("agent_b") is manifest

    def test_register_without_manifest_raises(self):
        registry = CompositionRegistry()

        class NoManifest:
            manifest = None

        with pytest.raises(ValueError, match="without a manifest"):
            registry.register(NoManifest())

    def test_unregister(self):
        registry = CompositionRegistry()
        manifest = make_manifest("agent_a")
        registry.register_manifest(manifest)
        registry.unregister("agent_a")

        assert "agent_a" not in registry

    def test_producers_of(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["X"]))
        registry.register_manifest(make_manifest("b", events_produced=["X", "Y"]))
        registry.register_manifest(make_manifest("c", events_produced=["Y"]))

        assert set(registry.producers_of("X")) == {"a", "b"}
        assert set(registry.producers_of("Y")) == {"b", "c"}
        assert registry.producers_of("Z") == []

    def test_consumers_of(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_consumed=["X"]))
        registry.register_manifest(make_manifest("b", events_consumed=["X", "Y"]))

        assert set(registry.consumers_of("X")) == {"a", "b"}
        assert registry.consumers_of("Y") == ["b"]


# --- Resolver Tests ---


class TestBindingResolver:
    def test_all_satisfied(self):
        resolver = BindingResolver(
            available_events=["EventA"],
            available_features=["feature_x"],
            available_tools=["tool_y"],
            available_knowledge_bases=["kb_z"],
        )
        manifest = make_manifest(
            "agent",
            events_consumed=["EventA"],
            features_required=["feature_x"],
            tool_capabilities=["tool_y"],
            knowledge_bases=["kb_z"],
        )

        status = resolver.evaluate(manifest)

        assert status.state == "active"
        assert status.gaps == []
        assert "EventA" in status.resolved_events
        assert "feature_x" in status.resolved_features

    def test_missing_event(self):
        resolver = BindingResolver(available_events=[])
        manifest = make_manifest("agent", events_consumed=["MissingEvent"])

        status = resolver.evaluate(manifest)

        assert status.state == "dormant"
        assert len(status.gaps) == 1
        assert status.gaps[0].type == "event"
        assert status.gaps[0].name == "MissingEvent"

    def test_missing_feature(self):
        resolver = BindingResolver(available_features=[])
        manifest = make_manifest("agent", features_required=["missing_feature"])

        status = resolver.evaluate(manifest)

        assert status.state == "dormant"
        assert status.gaps[0].type == "feature"

    def test_missing_tool(self):
        resolver = BindingResolver(available_tools=[])
        manifest = make_manifest("agent", tool_capabilities=["missing_tool"])

        status = resolver.evaluate(manifest)

        assert status.state == "dormant"
        assert status.gaps[0].type == "tool"

    def test_missing_knowledge_base(self):
        resolver = BindingResolver(available_knowledge_bases=[])
        manifest = make_manifest("agent", knowledge_bases=["missing_kb"])

        status = resolver.evaluate(manifest)

        assert status.state == "dormant"
        assert status.gaps[0].type == "knowledge_base"

    def test_multiple_gaps(self):
        resolver = BindingResolver()
        manifest = make_manifest(
            "agent",
            events_consumed=["E"],
            features_required=["F"],
            tool_capabilities=["T"],
        )

        status = resolver.evaluate(manifest)

        assert status.state == "dormant"
        assert len(status.gaps) == 3

    def test_add_and_remove_resource(self):
        resolver = BindingResolver()
        manifest = make_manifest("agent", features_required=["feat"])

        assert resolver.evaluate(manifest).state == "dormant"

        resolver.add_feature("feat")
        assert resolver.evaluate(manifest).state == "active"

        resolver.remove_feature("feat")
        assert resolver.evaluate(manifest).state == "dormant"

    def test_evaluate_all(self):
        resolver = BindingResolver(available_events=["E1"])
        m1 = make_manifest("a", events_consumed=["E1"])
        m2 = make_manifest("b", events_consumed=["E2"])

        results = resolver.evaluate_all([m1, m2])

        assert results[0].state == "active"
        assert results[1].state == "dormant"

    def test_what_would_activate(self):
        resolver = BindingResolver(available_events=["E1"])
        m1 = make_manifest("a", events_consumed=["E1"])  # already active
        m2 = make_manifest("b", events_consumed=["E1"], features_required=["F"])  # dormant

        would = resolver.what_would_activate("feature", "F", [m1, m2])

        assert would == ["b"]

    def test_gap_report_format(self):
        resolver = BindingResolver()
        manifest = make_manifest("my_agent", features_required=["x"], tool_capabilities=["y"])

        status = resolver.evaluate(manifest)
        report = status.gap_report()

        assert "my_agent" in report
        assert "dormant" in report
        assert "feature: x" in report
        assert "tool: y" in report


# --- Router Tests ---


class TestEventRouter:
    def test_simple_route(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("producer", events_produced=["EventX"]))
        registry.register_manifest(make_manifest("consumer", events_consumed=["EventX"]))

        router = EventRouter(registry)
        routes = router.resolve_routes()

        assert len(routes) == 1
        assert routes[0].event == "EventX"
        assert routes[0].source == "producer"
        assert routes[0].targets == ["consumer"]

    def test_fan_out(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("producer", events_produced=["EventX"]))
        registry.register_manifest(make_manifest("consumer_a", events_consumed=["EventX"]))
        registry.register_manifest(make_manifest("consumer_b", events_consumed=["EventX"]))

        router = EventRouter(registry)
        routes = router.resolve_routes()

        assert len(routes) == 1
        assert set(routes[0].targets) == {"consumer_a", "consumer_b"}

    def test_chain(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["E1"]))
        registry.register_manifest(make_manifest("b", events_consumed=["E1"], events_produced=["E2"]))
        registry.register_manifest(make_manifest("c", events_consumed=["E2"]))

        router = EventRouter(registry)
        routes = router.resolve_routes()

        assert len(routes) == 2
        sources = {r.source for r in routes}
        assert sources == {"a", "b"}

    def test_no_self_routing(self):
        registry = CompositionRegistry()
        registry.register_manifest(
            make_manifest("agent", events_consumed=["E"], events_produced=["E"])
        )

        router = EventRouter(registry)
        routes = router.resolve_routes()

        assert routes == []

    def test_dependency_order(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["E1"]))
        registry.register_manifest(make_manifest("b", events_consumed=["E1"], events_produced=["E2"]))
        registry.register_manifest(make_manifest("c", events_consumed=["E2"]))

        router = EventRouter(registry)
        layers = router.dependency_order()

        assert layers[0] == ["a"]
        assert layers[1] == ["b"]
        assert layers[2] == ["c"]

    def test_routes_from(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["E1"]))
        registry.register_manifest(make_manifest("b", events_consumed=["E1"]))

        router = EventRouter(registry)

        assert len(router.routes_from("a")) == 1
        assert router.routes_from("b") == []

    def test_routes_to(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["E1"]))
        registry.register_manifest(make_manifest("b", events_consumed=["E1"]))

        router = EventRouter(registry)

        assert router.routes_to("a") == []
        assert len(router.routes_to("b")) == 1

    def test_visualize(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_produced=["E1"]))
        registry.register_manifest(make_manifest("b", events_consumed=["E1"]))

        router = EventRouter(registry)
        viz = router.visualize()

        assert "Event Flow:" in viz
        assert "a" in viz
        assert "E1" in viz
        assert "b" in viz

    def test_empty_registry(self):
        registry = CompositionRegistry()
        router = EventRouter(registry)

        assert router.resolve_routes() == []
        assert router.visualize() == "(no routes)"


# --- Lifecycle Tests ---


class TestLifecycleManager:
    def test_initial_evaluation(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("a", events_consumed=["E1"]))
        registry.register_manifest(make_manifest("b"))

        resolver = BindingResolver(available_events=[])

        manager = LifecycleManager(registry, resolver)

        assert "a" in manager.dormant_agents
        assert "b" in manager.active_agents  # no dependencies

    def test_add_resource_activates(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("agent", features_required=["feat"]))

        resolver = BindingResolver()
        manager = LifecycleManager(registry, resolver)

        assert manager.dormant_agents == ["agent"]

        transitions = manager.add_resource("feature", "feat")

        assert len(transitions) == 1
        assert transitions[0].agent_name == "agent"
        assert transitions[0].from_state == "dormant"
        assert transitions[0].to_state == "active"
        assert manager.active_agents == ["agent"]

    def test_remove_resource_deactivates(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("agent", features_required=["feat"]))

        resolver = BindingResolver(available_features=["feat"])
        manager = LifecycleManager(registry, resolver)

        assert manager.active_agents == ["agent"]

        transitions = manager.remove_resource("feature", "feat")

        assert transitions[0].to_state == "dormant"
        assert manager.dormant_agents == ["agent"]

    def test_history(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("agent", features_required=["feat"]))

        resolver = BindingResolver()
        manager = LifecycleManager(registry, resolver)

        manager.add_resource("feature", "feat")
        manager.remove_resource("feature", "feat")

        assert len(manager.history) == 2
        assert manager.history[0].to_state == "active"
        assert manager.history[1].to_state == "dormant"

    def test_get_state(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("agent", features_required=["feat"]))

        resolver = BindingResolver()
        manager = LifecycleManager(registry, resolver)

        state = manager.get_state("agent")

        assert state.name == "agent"
        assert state.state == "dormant"
        assert len(state.binding_status.gaps) == 1

    def test_summary(self):
        registry = CompositionRegistry()
        registry.register_manifest(make_manifest("active_one"))
        registry.register_manifest(make_manifest("dormant_one", features_required=["missing"]))

        resolver = BindingResolver()
        manager = LifecycleManager(registry, resolver)

        summary = manager.summary()

        assert "active_one" in summary
        assert "dormant_one" in summary
        assert "feature:missing" in summary
