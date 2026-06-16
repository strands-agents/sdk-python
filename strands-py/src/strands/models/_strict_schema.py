"""Strict JSON schema transformation for tool definitions.

When model providers require `strict: true` on tool definitions, they also require
`"additionalProperties": false` on every `object` type in the input schema. This module
provides a utility to recursively apply that constraint.

Modeled after OpenAI's `_ensure_strict_json_schema`:
https://github.com/openai/openai-python/blob/main/src/openai/lib/_pydantic.py
"""

import copy
import logging
from typing import Any

logger = logging.getLogger(__name__)


def ensure_strict_json_schema(
    schema: dict[str, Any],
    *,
    require_all_properties: bool = False,
) -> dict[str, Any]:
    """Ensure a JSON schema conforms to strict tool use requirements.

    Creates a deep copy of the schema and recursively:
    1. Adds ``"additionalProperties": false`` to all ``object`` types that do not already define it
    2. Optionally adds all properties to the ``required`` array (needed for OpenAI)
    3. Handles ``$defs``, ``definitions``, ``anyOf``, ``allOf``, ``items``, and ``$ref``

    Args:
        schema: The JSON schema to process. A deep copy is made internally so the original is not mutated.
        require_all_properties: If True, set ``required`` to include all property keys. OpenAI strict mode
            requires this; Bedrock and Anthropic do not.

    Returns:
        A new schema dict with strict-mode constraints applied.
    """
    schema_copy = copy.deepcopy(schema)
    _apply_strict(schema_copy, root=schema_copy, require_all_properties=require_all_properties)
    return schema_copy


def _apply_strict(
    schema: dict[str, Any],
    *,
    root: dict[str, Any],
    require_all_properties: bool,
) -> None:
    """Recursively apply strict-mode constraints to a JSON schema in place.

    Args:
        schema: The schema node to process (modified in place).
        root: The root schema, used for resolving ``$ref`` pointers.
        require_all_properties: If True, add all properties to ``required``.
    """
    # Process $defs / definitions blocks
    for defs_key in ("$defs", "definitions"):
        defs = schema.get(defs_key)
        if isinstance(defs, dict):
            for def_schema in defs.values():
                if isinstance(def_schema, dict):
                    _apply_strict(def_schema, root=root, require_all_properties=require_all_properties)

    # Add additionalProperties: false to object types that lack it
    if schema.get("type") == "object" and "additionalProperties" not in schema:
        schema["additionalProperties"] = False

    # Process properties and optionally enforce required
    properties = schema.get("properties")
    if isinstance(properties, dict):
        if require_all_properties:
            schema["required"] = list(properties.keys())

        for prop_schema in properties.values():
            if isinstance(prop_schema, dict):
                _apply_strict(prop_schema, root=root, require_all_properties=require_all_properties)

    # Process array items
    items = schema.get("items")
    if isinstance(items, dict):
        _apply_strict(items, root=root, require_all_properties=require_all_properties)

    # Process anyOf variants
    any_of = schema.get("anyOf")
    if isinstance(any_of, list):
        for variant in any_of:
            if isinstance(variant, dict):
                _apply_strict(variant, root=root, require_all_properties=require_all_properties)

    # Process allOf variants
    all_of = schema.get("allOf")
    if isinstance(all_of, list):
        for entry in all_of:
            if isinstance(entry, dict):
                _apply_strict(entry, root=root, require_all_properties=require_all_properties)

    # Process oneOf variants
    one_of = schema.get("oneOf")
    if isinstance(one_of, list):
        for variant in one_of:
            if isinstance(variant, dict):
                _apply_strict(variant, root=root, require_all_properties=require_all_properties)

    # Resolve $ref combined with other keys by inlining the referenced schema
    ref = schema.get("$ref")
    if isinstance(ref, str) and len(schema) > 1:
        resolved = _resolve_ref(root, ref)
        if isinstance(resolved, dict):
            # Inline the resolved schema, giving priority to existing keys
            merged = {**copy.deepcopy(resolved), **schema}
            merged.pop("$ref", None)
            schema.clear()
            schema.update(merged)
            # Re-apply strict to the inlined schema
            _apply_strict(schema, root=root, require_all_properties=require_all_properties)


def _resolve_ref(root: dict[str, Any], ref: str) -> dict[str, Any] | None:
    """Resolve a JSON Schema ``$ref`` pointer against the root schema.

    Args:
        root: The root schema containing definitions.
        ref: A JSON pointer string (e.g., ``#/$defs/MyModel``).

    Returns:
        The resolved schema dict, or None if resolution fails.
    """
    if not ref.startswith("#/"):
        logger.warning("ref=<%s> | unexpected $ref format, skipping resolution", ref)
        return None

    path = ref[2:].split("/")
    current: Any = root
    for key in path:
        if not isinstance(current, dict) or key not in current:
            logger.warning("ref=<%s> | failed to resolve $ref path", ref)
            return None
        current = current[key]

    if not isinstance(current, dict):
        logger.warning("ref=<%s> | resolved to non-dict value", ref)
        return None

    return current


def validate_bedrock_strict_constraints(
    tool_specs: list[dict[str, Any]],
    strict_tools: bool,
) -> None:
    """Validate tool schemas against Bedrock strict-mode constraints.

    Bedrock strict mode enforces two constraints:
    1. No ``oneOf`` keyword anywhere in any tool schema
    2. Aggregate optional parameters across all tools must not exceed 24

    This function performs build-time validation to provide clear, actionable errors
    before the request reaches the Bedrock API, where failures are opaque.

    Args:
        tool_specs: List of tool specifications to validate.
        strict_tools: Whether strict tools mode is enabled.

    Raises:
        ValueError: If any constraint is violated.

    Example:
        >>> tool_specs = [{"name": "browser", "inputSchema": {"json": {...}}}]
        >>> validate_bedrock_strict_constraints(tool_specs, strict_tools=True)
        ValueError: Tool 'browser' contains unsupported 'oneOf' in input schema...
    """
    if not strict_tools:
        return

    # Check for oneOf in any tool schema
    for tool_spec in tool_specs:
        schema = tool_spec["inputSchema"]["json"]
        if _has_oneof(schema):
            raise ValueError(
                f"Tool '{tool_spec['name']}' contains unsupported 'oneOf' in input schema. "
                f"Bedrock strict mode does not support oneOf. Either:\n"
                f"  - Set strict_tools=False, or\n"
                f"  - Refactor the tool schema to avoid oneOf"
            )

    # Check aggregate optional parameter limit
    total_optional = 0
    tool_counts: list[tuple[str, int]] = []

    for tool_spec in tool_specs:
        schema = tool_spec["inputSchema"]["json"]
        optional_count = _count_optional_params(schema)
        total_optional += optional_count
        if optional_count > 0:
            tool_counts.append((tool_spec["name"], optional_count))

    if total_optional > 24:
        details = "\n".join(
            [f"  - {name}: {count} optional" for name, count in sorted(tool_counts, key=lambda x: -x[1])]
        )
        raise ValueError(
            f"Tools collectively have {total_optional} optional parameters (limit: 24). "
            f"Bedrock strict mode limits optional parameters. Either:\n"
            f"  - Set strict_tools=False, or\n"
            f"  - Reduce optional parameters by making some required or removing tools\n"
            f"Tools contributing optional params:\n{details}"
        )


def _has_oneof(schema: dict[str, Any], visited: set[int] | None = None) -> bool:
    """Recursively check if a schema contains the oneOf keyword.

    Args:
        schema: The schema dict to scan.
        visited: Set of schema object IDs already visited (prevents infinite recursion).

    Returns:
        True if oneOf is found anywhere in the schema tree, False otherwise.
    """
    if visited is None:
        visited = set()

    schema_id = id(schema)
    if schema_id in visited:
        return False
    visited.add(schema_id)

    # Direct oneOf check
    if "oneOf" in schema:
        return True

    # Recurse into nested structures
    for key in ("properties", "items", "anyOf", "allOf", "$defs", "definitions"):
        value = schema.get(key)
        if isinstance(value, dict):
            if _has_oneof(value, visited):
                return True
            # If key is properties or $defs/definitions, recurse into each sub-schema
            if key in ("properties", "$defs", "definitions"):
                for sub_schema in value.values():
                    if isinstance(sub_schema, dict) and _has_oneof(sub_schema, visited):
                        return True
        elif isinstance(value, list):
            for item in value:
                if isinstance(item, dict) and _has_oneof(item, visited):
                    return True

    return False


def _count_optional_params(schema: dict[str, Any]) -> int:
    """Count optional parameters in a schema.

    An optional parameter is a property that is not listed in the required array.

    Args:
        schema: The JSON schema to analyze.

    Returns:
        Number of optional parameters (properties not in required).
    """
    if schema.get("type") != "object":
        return 0

    properties = schema.get("properties", {})
    required = set(schema.get("required", []))
    return len(properties) - len(required)
