"""Graph configuration loader for Strands Agents.

This module provides the GraphConfigLoader class that enables creating Graph instances
from YAML/dictionary configurations, supporting serialization and deserialization of Graph
configurations for persistence and dynamic loading scenarios.
"""

import importlib
import inspect
import logging
import re
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Set

if TYPE_CHECKING:
    from ..agent.agent_config_loader import AgentConfigLoader
    from ..swarm.swarm_config_loader import SwarmConfigLoader

from strands.agent.agent import Agent
from strands.multiagent.graph import Graph, GraphEdge, GraphNode, GraphState
from strands.multiagent.swarm import Swarm

logger = logging.getLogger(__name__)


class GraphConfigLoader:
    """Loads and serializes Strands Graph instances via YAML/dictionary configurations.

    This class provides functionality to create Graph instances from YAML/dictionary
    configurations and serialize existing Graph instances to dictionaries for
    persistence and configuration management.

    The loader supports:
    1. Loading graphs from YAML/dictionary configurations
    2. Serializing graphs to YAML-compatible dictionary configurations
    3. Agent and Swarm loading via respective ConfigLoaders
    4. All condition types through unified type discriminator
    5. Caching for performance optimization
    6. Configuration validation and error handling
    """

    def __init__(
        self, agent_loader: Optional["AgentConfigLoader"] = None, swarm_loader: Optional["SwarmConfigLoader"] = None
    ):
        """Initialize the GraphConfigLoader.

        Args:
            agent_loader: Optional AgentConfigLoader instance for loading agents.
                         If not provided, will be imported and created when needed.
            swarm_loader: Optional SwarmConfigLoader instance for loading swarms.
                         If not provided, will be imported and created when needed.
        """
        self._agent_loader = agent_loader
        self._swarm_loader = swarm_loader
        self._condition_registry = ConditionRegistry()

    def _get_agent_config_loader(self) -> "AgentConfigLoader":
        """Get or create an AgentConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            AgentConfigLoader instance.
        """
        if self._agent_loader is None:
            # Import here to avoid circular imports
            from ..agent.agent_config_loader import AgentConfigLoader

            self._agent_loader = AgentConfigLoader()
        return self._agent_loader

    def _get_swarm_config_loader(self) -> "SwarmConfigLoader":
        """Get or create a SwarmConfigLoader instance.

        This method implements lazy loading to avoid circular imports.

        Returns:
            SwarmConfigLoader instance.
        """
        if self._swarm_loader is None:
            # Import here to avoid circular imports
            from ..swarm.swarm_config_loader import SwarmConfigLoader

            self._swarm_loader = SwarmConfigLoader()
        return self._swarm_loader

    def load_graph(self, config: Dict[str, Any]) -> Graph:
        """Load a Graph from configuration dictionary.

        Args:
            config: Dictionary containing graph configuration with top-level 'graph' key.

        Returns:
            Graph instance configured according to the provided dictionary.

        Raises:
            ValueError: If required configuration is missing or invalid.
            ImportError: If specified models or tools cannot be imported.
        """
        # Validate top-level structure
        if "graph" not in config:
            raise ValueError("Configuration must contain a top-level 'graph' key")

        graph_config = config["graph"]
        if not isinstance(graph_config, dict):
            raise ValueError("The 'graph' configuration must be a dictionary")

        # Validate configuration structure
        self._validate_config(graph_config)

        # Load nodes
        nodes = self._load_nodes(graph_config.get("nodes", []))

        # Load edges with conditions
        edges = self._load_edges(graph_config.get("edges", []), nodes)

        # Load entry points
        entry_points = self._load_entry_points(graph_config.get("entry_points", []), nodes)

        # Extract graph configuration
        graph_params = self._extract_graph_parameters(graph_config)

        # Create graph
        graph = Graph(nodes=nodes, edges=edges, entry_points=entry_points, **graph_params)

        return graph

    def serialize_graph(self, graph: Graph) -> Dict[str, Any]:
        """Serialize a Graph instance to YAML-compatible dictionary configuration.

        Args:
            graph: Graph instance to serialize.

        Returns:
            Dictionary containing the graph's configuration with top-level 'graph' key.
        """
        graph_config: Dict[str, Any] = {}

        # Serialize nodes
        nodes_config = []
        for node_id, node in graph.nodes.items():
            node_config = self._serialize_node(node_id, node)
            nodes_config.append(node_config)
        graph_config["nodes"] = nodes_config

        # Serialize edges
        edges_config = []
        for edge in graph.edges:
            edge_config = self._serialize_edge(edge)
            edges_config.append(edge_config)
        graph_config["edges"] = edges_config

        # Serialize entry points
        entry_points_config: List[str] = []
        for entry_point in graph.entry_points:
            # Find the node_id for this entry point
            for node_id, node in graph.nodes.items():
                if node == entry_point:
                    entry_points_config.append(node_id)
                    break
        graph_config["entry_points"] = entry_points_config

        # Serialize graph parameters (only include non-default values)
        if graph.max_node_executions is not None:
            graph_config["max_node_executions"] = graph.max_node_executions
        if graph.execution_timeout is not None:
            graph_config["execution_timeout"] = graph.execution_timeout
        if graph.node_timeout is not None:
            graph_config["node_timeout"] = graph.node_timeout
        if graph.reset_on_revisit is not False:
            graph_config["reset_on_revisit"] = graph.reset_on_revisit

        return {"graph": graph_config}

    def _load_nodes(self, nodes_config: List[Dict[str, Any]]) -> Dict[str, GraphNode]:
        """Load graph nodes from configuration.

        Args:
            nodes_config: List of node configuration dictionaries.

        Returns:
            Dictionary mapping node_id to GraphNode instances.
        """
        nodes = {}

        for node_config in nodes_config:
            node_id = node_config["node_id"]
            node_type = node_config["type"]

            if node_type == "agent":
                if "config" in node_config:
                    # Load agent from configuration
                    # Wrap the agent config in the required top-level 'agent' key
                    agent_loader = self._get_agent_config_loader()
                    wrapped_agent_config = {"agent": node_config["config"]}
                    agent = agent_loader.load_agent(wrapped_agent_config)
                elif "reference" in node_config:
                    # Load agent from reference (string identifier)
                    agent = self._load_agent_reference(node_config["reference"])
                else:
                    raise ValueError(f"Agent node {node_id} missing config or reference")

                nodes[node_id] = GraphNode(node_id=node_id, executor=agent)

            elif node_type == "swarm":
                if "config" in node_config:
                    # Load swarm from configuration
                    # Wrap the swarm config in the required top-level 'swarm' key
                    swarm_loader = self._get_swarm_config_loader()
                    wrapped_swarm_config = {"swarm": node_config["config"]}
                    swarm = swarm_loader.load_swarm(wrapped_swarm_config)
                elif "reference" in node_config:
                    # Load swarm from reference
                    swarm = self._load_swarm_reference(node_config["reference"])
                else:
                    raise ValueError(f"Swarm node {node_id} missing config or reference")

                nodes[node_id] = GraphNode(node_id=node_id, executor=swarm)

            elif node_type == "graph":
                if "config" in node_config:
                    # Recursive graph loading
                    # Wrap the graph config in the required top-level 'graph' key
                    wrapped_graph_config = {"graph": node_config["config"]}
                    sub_graph = self.load_graph(wrapped_graph_config)
                elif "reference" in node_config:
                    # Load graph from reference
                    sub_graph = self._load_graph_reference(node_config["reference"])
                else:
                    raise ValueError(f"Graph node {node_id} missing config or reference")

                nodes[node_id] = GraphNode(node_id=node_id, executor=sub_graph)

            else:
                raise ValueError(f"Unknown node type: {node_type}")

            logger.debug("node_id=<%s>, type=<%s> | loaded graph node", node_id, node_type)

        return nodes

    def _load_edges(self, edges_config: List[Dict[str, Any]], nodes: Dict[str, GraphNode]) -> Set[GraphEdge]:
        """Load graph edges with conditions from configuration.

        Args:
            edges_config: List of edge configuration dictionaries.
            nodes: Dictionary of loaded nodes.

        Returns:
            Set of GraphEdge instances.
        """
        edges = set()

        for edge_config in edges_config:
            from_node_id = edge_config["from_node"]
            to_node_id = edge_config["to_node"]

            # Validate nodes exist
            if from_node_id not in nodes:
                raise ValueError(f"Edge references unknown from_node: {from_node_id}")
            if to_node_id not in nodes:
                raise ValueError(f"Edge references unknown to_node: {to_node_id}")

            from_node = nodes[from_node_id]
            to_node = nodes[to_node_id]

            # Load condition if present
            condition = None
            if "condition" in edge_config and edge_config["condition"] is not None:
                condition = self._condition_registry.load_condition(edge_config["condition"])

            edge = GraphEdge(from_node, to_node, condition)
            edges.add(edge)

            logger.debug("from=<%s>, to=<%s> | loaded graph edge", from_node_id, to_node_id)

        return edges

    def _load_entry_points(self, entry_points_config: List[str], nodes: Dict[str, GraphNode]) -> Set[GraphNode]:
        """Load entry points from configuration.

        Args:
            entry_points_config: List of node IDs that are entry points.
            nodes: Dictionary of loaded nodes.

        Returns:
            Set of GraphNode instances that are entry points.
        """
        entry_points = set()

        for entry_point_id in entry_points_config:
            if entry_point_id not in nodes:
                raise ValueError(f"Entry point references unknown node: {entry_point_id}")

            entry_points.add(nodes[entry_point_id])
            logger.debug("entry_point=<%s> | loaded entry point", entry_point_id)

        return entry_points

    def _extract_graph_parameters(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Extract graph-specific parameters from configuration.

        Args:
            config: Configuration dictionary.

        Returns:
            Dictionary containing graph constructor parameters.
        """
        params = {}

        # Extract parameters with validation
        if "max_node_executions" in config:
            max_executions = config["max_node_executions"]
            if max_executions is not None and (not isinstance(max_executions, int) or max_executions < 1):
                raise ValueError("max_node_executions must be a positive integer or null")
            params["max_node_executions"] = max_executions

        if "execution_timeout" in config:
            execution_timeout = config["execution_timeout"]
            if execution_timeout is not None and (
                not isinstance(execution_timeout, (int, float)) or execution_timeout <= 0
            ):
                raise ValueError("execution_timeout must be a positive number or null")
            params["execution_timeout"] = int(execution_timeout) if execution_timeout is not None else None

        if "node_timeout" in config:
            node_timeout = config["node_timeout"]
            if node_timeout is not None and (not isinstance(node_timeout, (int, float)) or node_timeout <= 0):
                raise ValueError("node_timeout must be a positive number or null")
            params["node_timeout"] = int(node_timeout) if node_timeout is not None else None

        if "reset_on_revisit" in config:
            reset_on_revisit = config["reset_on_revisit"]
            if not isinstance(reset_on_revisit, bool):
                raise ValueError("reset_on_revisit must be a boolean")
            params["reset_on_revisit"] = reset_on_revisit

        return params

    def _validate_config(self, config: Dict[str, Any]) -> None:
        """Validate graph configuration structure.

        Args:
            config: Configuration dictionary to validate.

        Raises:
            ValueError: If configuration is invalid.
        """
        if not isinstance(config, dict):
            raise ValueError(f"Graph configuration must be a dictionary, got: {type(config)}")

        # Check for required fields
        required_fields = ["nodes", "edges", "entry_points"]
        for field in required_fields:
            if field not in config:
                raise ValueError(f"Graph configuration must include '{field}' field")

        # Validate nodes
        nodes_config = config["nodes"]
        if not isinstance(nodes_config, list):
            raise ValueError("'nodes' field must be a list")

        if not nodes_config:
            raise ValueError("'nodes' list cannot be empty")

        node_ids = set()
        for i, node in enumerate(nodes_config):
            if not isinstance(node, dict):
                raise ValueError(f"Node configuration at index {i} must be a dictionary")

            if "node_id" not in node:
                raise ValueError(f"Node at index {i} missing required 'node_id' field")

            node_id = node["node_id"]
            if node_id in node_ids:
                raise ValueError(f"Duplicate node_id: {node_id}")
            node_ids.add(node_id)

            # Validate node type
            node_type = node.get("type")
            if node_type not in ["agent", "swarm", "graph"]:
                raise ValueError(f"Invalid node type: {node_type}")

        # Validate edges
        edges_config = config["edges"]
        if not isinstance(edges_config, list):
            raise ValueError("'edges' field must be a list")

        for i, edge in enumerate(edges_config):
            if not isinstance(edge, dict):
                raise ValueError(f"Edge configuration at index {i} must be a dictionary")

            required_edge_fields = ["from_node", "to_node"]
            for field in required_edge_fields:
                if field not in edge:
                    raise ValueError(f"Edge at index {i} missing required '{field}' field")

            # Validate edge references existing nodes
            if edge["from_node"] not in node_ids:
                raise ValueError(f"Edge references unknown from_node: {edge['from_node']}")
            if edge["to_node"] not in node_ids:
                raise ValueError(f"Edge references unknown to_node: {edge['to_node']}")

            # Validate condition if present
            if "condition" in edge and edge["condition"] is not None:
                self._validate_condition_config(edge["condition"])

        # Validate entry_points
        entry_points_config = config["entry_points"]
        if not isinstance(entry_points_config, list):
            raise ValueError("'entry_points' field must be a list")

        if not entry_points_config:
            raise ValueError("'entry_points' list cannot be empty")

        for entry_point in entry_points_config:
            if entry_point not in node_ids:
                raise ValueError(f"Entry point references unknown node: {entry_point}")

    def _validate_condition_config(self, condition_config: Dict[str, Any]) -> None:
        """Validate condition configuration."""
        if "type" not in condition_config:
            raise ValueError("Condition missing required 'type' field")

        condition_type = condition_config["type"]
        if condition_type not in ["function", "expression", "rule", "lambda", "template", "composite"]:
            raise ValueError(f"Invalid condition type: {condition_type}")

        # Type-specific validation
        if condition_type == "function":
            required = ["module", "function"]
            for field in required:
                if field not in condition_config:
                    raise ValueError(f"Function condition missing required field: {field}")

        elif condition_type == "expression":
            if "expression" not in condition_config:
                raise ValueError("Expression condition missing required 'expression' field")

        elif condition_type == "rule":
            if "rules" not in condition_config:
                raise ValueError("Rule condition missing required 'rules' field")

            for rule in condition_config["rules"]:
                required_rule_fields = ["field", "operator", "value"]
                for field in required_rule_fields:
                    if field not in rule:
                        raise ValueError(f"Rule missing required field: {field}")

    def _load_agent_reference(self, reference: str) -> Agent:
        """Load agent from string reference."""
        # This would implement agent lookup by reference
        # For now, raise NotImplementedError
        raise NotImplementedError("Agent reference loading not yet implemented")

    def _load_swarm_reference(self, reference: str) -> Swarm:
        """Load swarm from string reference."""
        # This would implement swarm lookup by reference
        # For now, raise NotImplementedError
        raise NotImplementedError("Swarm reference loading not yet implemented")

    def _load_graph_reference(self, reference: str) -> Graph:
        """Load graph from string reference."""
        # This would implement graph lookup by reference
        # For now, raise NotImplementedError
        raise NotImplementedError("Graph reference loading not yet implemented")

    def _serialize_node(self, node_id: str, node: GraphNode) -> Dict[str, Any]:
        """Serialize a graph node to configuration."""
        node_config = {"node_id": node_id}

        if isinstance(node.executor, Agent):
            node_config["type"] = "agent"
            agent_loader = self._get_agent_config_loader()
            node_config["config"] = agent_loader.serialize_agent(node.executor)  # type: ignore[assignment]
        elif isinstance(node.executor, Swarm):
            node_config["type"] = "swarm"
            swarm_loader = self._get_swarm_config_loader()
            node_config["config"] = swarm_loader.serialize_swarm(node.executor)  # type: ignore[assignment]
        elif isinstance(node.executor, Graph):
            node_config["type"] = "graph"
            node_config["config"] = self.serialize_graph(node.executor)  # type: ignore[assignment]
        else:
            raise ValueError(f"Unknown node executor type: {type(node.executor)}")

        return node_config

    def _serialize_edge(self, edge: GraphEdge) -> Dict[str, Any]:
        """Serialize a graph edge to configuration."""
        # This is a simplified approach - in practice you'd need to maintain
        # a mapping of nodes to IDs during serialization
        edge_config = {
            "from_node": edge.from_node.node_id,
            "to_node": edge.to_node.node_id,
            "condition": None,
        }

        # Serialize condition if present
        if edge.condition is not None:
            # This would require condition serialization logic
            # For now, we'll note that this is complex and would need
            # reverse engineering of the condition function
            edge_config["condition"] = {"type": "function", "note": "Condition serialization not implemented"}  # type: ignore[assignment]

        return edge_config


class ConditionRegistry:
    """Registry for condition functions and evaluation strategies with type-based dispatch."""

    def __init__(self) -> None:
        """Initialize the condition registry with type-based loaders."""
        self._condition_loaders = {
            "function": self._load_function_condition,
            "expression": self._load_expression_condition,
            "rule": self._load_rule_condition,
            "lambda": self._load_lambda_condition,
            "template": self._load_template_condition,
            "composite": self._load_composite_condition,
        }
        self._template_registry = self._initialize_templates()
        self.allowed_modules = ["conditions", "workflow.conditions"]
        self.max_expression_length = 500
        self.evaluation_timeout = 5.0

    def load_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition based on type discriminator.

        Args:
            config: Condition configuration with 'type' field.

        Returns:
            Callable that takes GraphState and returns bool.

        Raises:
            ValueError: If condition type is unsupported or configuration is invalid.
        """
        condition_type = config.get("type")
        if condition_type not in self._condition_loaders:
            raise ValueError(f"Unsupported condition type: {condition_type}")

        return self._condition_loaders[condition_type](config)

    def _load_function_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition from function reference.

        Config format:
        condition:
          type: "function"
          module: "my_conditions"
          function: "is_valid"
          timeout: 5.0
          default: false
        """
        module_name = config["module"]
        function_name = config["function"]
        timeout = config.get("timeout")
        default_value = config.get("default", False)

        # Validate module access
        self._validate_module_access(module_name)

        try:
            module = importlib.import_module(module_name)
            func = getattr(module, function_name)

            # Validate function signature matches expected pattern
            sig = inspect.signature(func)
            if len(sig.parameters) != 1:
                raise ValueError(f"Condition function {function_name} must accept exactly one parameter (GraphState)")

            if timeout:
                return self._wrap_with_timeout(func, timeout, default_value)

            return func  # type: ignore[no-any-return]

        except (ImportError, AttributeError) as e:
            raise ValueError(f"Cannot load condition function {module_name}.{function_name}: {e}") from e

    def _load_expression_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition from expression string.

        Config format:
        condition:
          type: "expression"
          expression: "state.results.get('validator', {}).get('status') == 'success'"
          description: "Check if validation was successful"
          default: false
        """
        expression = config["expression"]
        default_value = config.get("default", False)

        # Sanitize and validate expression
        expression = self._sanitize_expression(expression)

        # Compile expression for safety and performance
        try:
            compiled_expr = compile(expression, "<condition>", "eval")
        except SyntaxError as e:
            raise ValueError(f"Invalid expression syntax: {expression}: {e}") from e

        def condition_func(state: GraphState) -> bool:
            try:
                # Provide safe evaluation context with GraphState
                context = {
                    "__builtins__": {},
                    "state": state,
                    # Add common helper functions
                    "len": len,
                    "str": str,
                    "int": int,
                    "float": float,
                    "bool": bool,
                }

                result = eval(compiled_expr, context)
                return bool(result)

            except Exception as e:
                logger.warning("Expression condition failed: %s, returning default: %s", e, default_value)
                return bool(default_value)

        return condition_func

    def _load_rule_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition from rule configuration.

        Config format:
        condition:
          type: "rule"
          rules:
            - field: "results.validator.status"
              operator: "equals"
              value: "success"
            - field: "results.validator.confidence"
              operator: "greater_than"
              value: 0.8
          logic: "and"
        """
        rules = config["rules"]
        logic = config.get("logic", "and")

        operators = {
            "equals": lambda a, b: a == b,
            "not_equals": lambda a, b: a != b,
            "greater_than": lambda a, b: a > b,
            "less_than": lambda a, b: a < b,
            "greater_equal": lambda a, b: a >= b,
            "less_equal": lambda a, b: a <= b,
            "contains": lambda a, b: b in str(a),
            "starts_with": lambda a, b: str(a).startswith(str(b)),
            "ends_with": lambda a, b: str(a).endswith(str(b)),
            "regex_match": lambda a, b: bool(re.match(b, str(a))),
        }

        def condition_func(state: GraphState) -> bool:
            results = []

            for rule in rules:
                field_path = rule["field"]
                operator = rule["operator"]
                expected_value = rule["value"]

                try:
                    # Extract field value using dot notation from GraphState
                    field_value = self._get_nested_field(state, field_path)

                    # Apply operator
                    if operator in operators:
                        result = operators[operator](field_value, expected_value)
                        results.append(result)
                    else:
                        raise ValueError(f"Unknown operator: {operator}")
                except Exception as e:
                    logger.warning("Rule evaluation failed for field %s: %s", field_path, e)
                    results.append(False)

            # Apply logic
            if logic == "and":
                return all(results)
            elif logic == "or":
                return any(results)
            else:
                raise ValueError(f"Unknown logic operator: {logic}")

        return condition_func

    def _load_lambda_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition from lambda expression.

        Config format:
        condition:
          type: "lambda"
          expression: "lambda state: 'technical' in str(state.results.get('classifier', {}).get('result', '')).lower()"
          description: "Check for technical classification"
          timeout: 2.0
        """
        expression = config["expression"]
        timeout = config.get("timeout")
        default_value = config.get("default", False)

        # Sanitize expression
        expression = self._sanitize_expression(expression)

        try:
            # Compile and evaluate lambda
            compiled_lambda = compile(expression, "<lambda>", "eval")
            lambda_func = eval(compiled_lambda, {"__builtins__": {}})

            # Validate it's actually a lambda/function
            if not callable(lambda_func):
                raise ValueError("Lambda expression must evaluate to a callable")

            # Validate signature
            sig = inspect.signature(lambda_func)
            if len(sig.parameters) != 1:
                raise ValueError("Lambda must accept exactly one parameter (GraphState)")

            if timeout:
                return self._wrap_with_timeout(lambda_func, timeout, default_value)

            return lambda_func  # type: ignore[no-any-return]

        except Exception as e:
            raise ValueError(f"Invalid lambda expression: {expression}: {e}") from e

    def _load_template_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load condition from predefined template.

        Config format:
        condition:
          type: "template"
          template: "node_result_contains"
          parameters:
            node_id: "classifier"
            search_text: "technical"
            case_sensitive: false
        """
        template_name = config["template"]
        parameters = config.get("parameters", {})

        if template_name not in self._template_registry:
            raise ValueError(f"Unknown condition template: {template_name}")

        template_func = self._template_registry[template_name]
        return template_func(**parameters)  # type: ignore[no-any-return]

    def _load_composite_condition(self, config: Dict[str, Any]) -> Callable[[GraphState], bool]:
        """Load composite condition with multiple sub-conditions.

        Config format:
        condition:
          type: "composite"
          logic: "and"  # "and", "or", "not"
          conditions:
            - type: "function"
              module: "conditions"
              function: "is_valid"
            - type: "expression"
              expression: "state.execution_count < 10"
        """
        logic = config["logic"]
        sub_conditions = []

        for sub_config in config["conditions"]:
            sub_condition = self.load_condition(sub_config)
            sub_conditions.append(sub_condition)

        def condition_func(state: GraphState) -> bool:
            if logic == "and":
                return all(cond(state) for cond in sub_conditions)
            elif logic == "or":
                return any(cond(state) for cond in sub_conditions)
            elif logic == "not":
                if len(sub_conditions) != 1:
                    raise ValueError("NOT logic requires exactly one sub-condition")
                return not sub_conditions[0](state)
            else:
                raise ValueError(f"Unknown composite logic: {logic}")

        return condition_func

    def _initialize_templates(self) -> Dict[str, Callable]:
        """Initialize predefined condition templates."""
        return {
            "node_result_contains": self._template_node_result_contains,
            "node_execution_time_under": self._template_node_execution_time_under,
            "node_status_equals": self._template_node_status_equals,
            "execution_count_under": self._template_execution_count_under,
        }

    def _template_node_result_contains(
        self, node_id: str, search_text: str, case_sensitive: bool = True
    ) -> Callable[[GraphState], bool]:
        """Template for checking if node result contains specific text."""

        def condition_func(state: GraphState) -> bool:
            node_result = state.results.get(node_id)
            if not node_result:
                return False

            result_text = str(node_result.result)
            if not case_sensitive:
                return search_text.lower() in result_text.lower()
            return search_text in result_text

        return condition_func

    def _template_node_execution_time_under(self, node_id: str, max_time_ms: int) -> Callable[[GraphState], bool]:
        """Template for checking if node execution time is under threshold."""

        def condition_func(state: GraphState) -> bool:
            node_result = state.results.get(node_id)
            if not node_result:
                return False

            execution_time = getattr(node_result, "execution_time", 0)
            return execution_time < max_time_ms

        return condition_func

    def _template_node_status_equals(self, node_id: str, status: str) -> Callable[[GraphState], bool]:
        """Template for checking if node status equals expected value."""

        def condition_func(state: GraphState) -> bool:
            node_result = state.results.get(node_id)
            if not node_result:
                return False

            node_status = getattr(node_result, "status", None)
            return str(node_status) == status

        return condition_func

    def _template_execution_count_under(self, max_count: int) -> Callable[[GraphState], bool]:
        """Template for checking if execution count is under threshold."""

        def condition_func(state: GraphState) -> bool:
            return state.execution_count < max_count

        return condition_func

    def _validate_module_access(self, module_name: str) -> None:
        """Validate that module is in allowlist."""
        if not any(module_name.startswith(allowed) for allowed in self.allowed_modules):
            raise ValueError(f"Module {module_name} not in allowed modules: {self.allowed_modules}")

    def _sanitize_expression(self, expression: str) -> str:
        """Sanitize expression to prevent code injection."""
        if len(expression) > self.max_expression_length:
            raise ValueError(f"Expression too long: {len(expression)} > {self.max_expression_length}")

        # Check for dangerous patterns (more precise matching)
        dangerous_patterns = ["__", "import ", "exec(", "eval(", "open(", "file("]
        for pattern in dangerous_patterns:
            if pattern in expression:
                raise ValueError(f"Dangerous pattern '{pattern}' found in expression")

        return expression

    def _wrap_with_timeout(self, func: Callable, timeout: float, default_value: bool) -> Callable[[GraphState], bool]:
        """Wrap function with timeout protection."""
        import signal

        def timeout_handler(signum: int, frame: Any) -> None:
            raise TimeoutError("Condition evaluation timed out")

        def wrapped_func(state: GraphState) -> bool:
            try:
                # Set timeout
                signal.signal(signal.SIGALRM, timeout_handler)
                signal.alarm(int(timeout))

                result = func(state)

                # Clear timeout
                signal.alarm(0)
                return bool(result)

            except TimeoutError:
                logger.warning("Condition function timed out after %ss, returning default: %s", timeout, default_value)
                return default_value
            except Exception as e:
                logger.warning("Condition function failed: %s, returning default: %s", e, default_value)
                return default_value
            finally:
                signal.alarm(0)

        return wrapped_func

    def _get_nested_field(self, state: GraphState, field_path: str) -> Any:
        """Extract nested field value using dot notation."""
        parts = field_path.split(".")
        current = state

        for part in parts:
            if hasattr(current, part):
                current = getattr(current, part)
            elif isinstance(current, dict) and part in current:
                current = current[part]
            else:
                return None

        return current
