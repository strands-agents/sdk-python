"""Tool decorator for SDK.

This module provides the @tool decorator that transforms Python functions into SDK Agent tools with automatic metadata
extraction and validation.

The @tool decorator performs several functions:

1. Extracts function metadata (name, description, parameters) from docstrings and type hints
2. Generates a JSON schema for input validation
3. Handles two different calling patterns:
   - Standard function calls (func(arg1, arg2))
   - Tool use calls (agent.my_tool(param1="hello", param2=123))
4. Provides error handling and result formatting
5. Works with both standalone functions and class methods

Example:
    ```python
    from strands import Agent, tool

    @tool
    def my_tool(param1: str, param2: int = 42) -> dict:
        '''
        Tool description - explain what it does.

        #Args:
            param1: Description of first parameter.
            param2: Description of second parameter (default: 42).

        #Returns:
            A dictionary with the results.
        '''
        result = do_something(param1, param2)
        return {
            "status": "success",
            "content": [{"text": f"Result: {result}"}]
        }

    agent = Agent(tools=[my_tool])
    agent.my_tool(param1="hello", param2=123)
    ```
"""

import functools
import inspect
from typing import Any, Callable, Dict, Optional, Type, TypeVar, cast, get_type_hints

import docstring_parser
from pydantic import BaseModel, Field, create_model

# Type for wrapped function
T = TypeVar("T", bound=Callable[..., Any])


class FunctionToolMetadata:
    """Helper class to extract and manage function metadata for tool decoration.

    This class handles the extraction of metadata from Python functions including:

    - Function name and description from docstrings
    - Parameter names, types, and descriptions
    - Return type information
    - Creation of Pydantic models for input validation

    The extracted metadata is used to generate a tool specification that can be used by Strands Agent to understand and
    validate tool usage.
    """

    def __init__(self, func: Callable[..., Any]) -> None:
        """Initialize with the function to process.

        Args:
            func: The function to extract metadata from.
                 Can be a standalone function or a class method.
        """
        self.func = func
        self.signature = inspect.signature(func)
        self.type_hints = get_type_hints(func)

        # Parse the docstring with docstring_parser
        doc_str = inspect.getdoc(func) or ""
        self.doc = docstring_parser.parse(doc_str)

        # Get parameter descriptions from parsed docstring
        self.param_descriptions = {
            param.arg_name: param.description or f"Parameter {param.arg_name}" for param in self.doc.params
        }

        # Create a Pydantic model for validation
        self.input_model = self._create_input_model()

    def _create_input_model(self) -> Type[BaseModel]:
        """Create a Pydantic model from function signature for input validation.

        This method analyzes the function's signature, type hints, and docstring to create a Pydantic model that can
        validate input data before passing it to the function.

        Special parameters like 'self', 'cls', and 'agent' are excluded from the model.

        Returns:
            A Pydantic BaseModel class customized for the function's parameters.
        """
        field_definitions: Dict[str, Any] = {}

        for name, param in self.signature.parameters.items():
            # Skip special parameters
            if name in ("self", "cls", "agent"):
                continue

            # Get parameter type and default
            param_type = self.type_hints.get(name, Any)
            default = ... if param.default is inspect.Parameter.empty else param.default
            description = self.param_descriptions.get(name, f"Parameter {name}")

            # Create Field with description and default
            field_definitions[name] = (param_type, Field(default=default, description=description))

        # Create model name based on function name
        model_name = f"{self.func.__name__.capitalize()}Tool"

        # Create and return the model
        if field_definitions:
            return create_model(model_name, **field_definitions)
        else:
            # Handle case with no parameters
            return create_model(model_name)

    def extract_metadata(self) -> Dict[str, Any]:
        """Extract metadata from the function to create a tool specification.

        This method analyzes the function to create a standardized tool specification that Strands Agent can use to
        understand and interact with the tool.

        The specification includes:

        - name: The function name (or custom override)
        - description: The function's docstring
        - inputSchema: A JSON schema describing the expected parameters

        Returns:
            A dictionary containing the tool specification.
        """
        func_name = self.func.__name__

        # Extract function description from docstring, preserving paragraph breaks
        description = inspect.getdoc(self.func)
        if description:
            description = description.strip()
        else:
            description = func_name

        # Get schema directly from the Pydantic model
        input_schema = self.input_model.model_json_schema()

        # Clean up Pydantic-specific schema elements
        self._clean_pydantic_schema(input_schema)

        # Create tool specification
        tool_spec = {"name": func_name, "description": description, "inputSchema": {"json": input_schema}}

        return tool_spec

    def _clean_pydantic_schema(self, schema: Dict[str, Any]) -> None:
        """Clean up Pydantic schema to match Strands' expected format.

        Pydantic's JSON schema output includes several elements that aren't needed for Strands Agent tools and could
        cause validation issues. This method removes those elements and simplifies complex type structures.

        Key operations:

        1. Remove Pydantic-specific metadata (title, $defs, etc.)
        2. Process complex types like Union and Optional to simpler formats
        3. Handle nested property structures recursively

        Args:
            schema: The Pydantic-generated JSON schema to clean up (modified in place).
        """
        # Remove Pydantic metadata
        keys_to_remove = ["title", "$defs", "additionalProperties"]
        for key in keys_to_remove:
            if key in schema:
                del schema[key]

        # Process properties to clean up anyOf and similar structures
        if "properties" in schema:
            for _prop_name, prop_schema in schema["properties"].items():
                # Handle anyOf constructs (common for Optional types)
                if "anyOf" in prop_schema:
                    any_of = prop_schema["anyOf"]
                    # Handle Optional[Type] case (represented as anyOf[Type, null])
                    if len(any_of) == 2 and any(item.get("type") == "null" for item in any_of):
                        # Find the non-null type
                        for item in any_of:
                            if item.get("type") != "null":
                                # Copy the non-null properties to the main schema
                                for k, v in item.items():
                                    prop_schema[k] = v
                                # Remove the anyOf construct
                                del prop_schema["anyOf"]
                                break

                # Clean up nested properties recursively
                if "properties" in prop_schema:
                    self._clean_pydantic_schema(prop_schema)

                # Remove any remaining Pydantic metadata from properties
                for key in keys_to_remove:
                    if key in prop_schema:
                        del prop_schema[key]

    def validate_input(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate input data using the Pydantic model.

        This method ensures that the input data meets the expected schema before it's passed to the actual function. It
        converts the data to the correct types when possible and raises informative errors when not.

        Args:
            input_data: A dictionary of parameter names and values to validate.

        Returns:
            A dictionary with validated and converted parameter values.

        Raises:
            ValueError: If the input data fails validation, with details about what failed.
        """
        try:
            # Validate with Pydantic model
            validated = self.input_model(**input_data)

            # Return as dict
            return validated.model_dump()
        except Exception as e:
            # Re-raise with more detailed error message
            error_msg = str(e)
            raise ValueError(f"Validation failed for input parameters: {error_msg}") from e


def tool(func: Optional[Callable[..., Any]] = None, **tool_kwargs: Any) -> Callable[[T], T]:
    """Decorator that transforms a Python function into a Strands tool.

    This decorator seamlessly enables a function to be called both as a regular Python function and as a Strands tool.
    It extracts metadata from the function's signature, docstring, and type hints to generate an OpenAPI-compatible tool
    specification.

    When decorated, a function:

    1. Still works as a normal function when called directly with arguments
    2. Processes tool use API calls when provided with a tool use dictionary
    3. Validates inputs against the function's type hints and parameter spec
    4. Formats return values according to the expected Strands tool result format
    5. Provides automatic error handling and reporting

    Args:
        func: The function to decorate.
        **tool_kwargs: Additional tool specification options to override extracted values.
            E.g., `name="custom_name", description="Custom description"`.

    Returns:
        The decorated function with attached tool specifications.

    Example:
        ```python
        @tool
        def my_tool(name: str, count: int = 1) -> str:
            '''Does something useful with the provided parameters.

            "Args:
                name: The name to process
                count: Number of times to process (default: 1)

            "Returns:
                A message with the result
            '''
            return f"Processed {name} {count} times"

        agent = Agent(tools=[my_tool])
        agent.my_tool(name="example", count=3)
        # Returns: {
        #   "toolUseId": "123",
        #   "status": "success",
        #   "content": [{"text": "Processed example 3 times"}]
        # }
        ```
    """

    def decorator(f: T) -> T:
        # Create function tool metadata
        tool_meta = FunctionToolMetadata(f)
        tool_spec = tool_meta.extract_metadata()

        # Update with any additional kwargs
        tool_spec.update(tool_kwargs)

        # Attach TOOL_SPEC directly to the original function (critical for backward compatibility)
        f.TOOL_SPEC = tool_spec  # type: ignore

        @functools.wraps(f)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            """Tool wrapper.

            This wrapper handles two different calling patterns:

            1. Normal function calls: `func(arg1, arg2, ...)`
            2. Tool use calls: `func({"toolUseId": "id", "input": {...}}, agent=agent)`
            """
            # Initialize variables to track call type
            is_method_call = False
            instance = None

            # DETECT IF THIS IS A METHOD CALL (with 'self' as first argument)
            # If this is a method call, the first arg would be 'self' (instance)
            if len(args) > 0 and not isinstance(args[0], dict):
                try:
                    # Try to find f in the class of args[0]
                    if hasattr(args[0], "__class__"):
                        if hasattr(args[0].__class__, f.__name__):
                            # This is likely a method call with self as first argument
                            is_method_call = True
                            instance = args[0]
                            args = args[1:]  # Remove self from args
                except (AttributeError, TypeError):
                    pass

            # DETECT IF THIS IS A TOOL USE CALL
            # Check if this is a tool use call (dict with toolUseId or input)
            if (
                len(args) > 0
                and isinstance(args[0], dict)
                and (not args[0] or "toolUseId" in args[0] or "input" in args[0])
            ):
                # This is a tool use call - process accordingly
                tool_use = args[0]
                tool_use_id = tool_use.get("toolUseId", "unknown")
                tool_input = tool_use.get("input", {})

                try:
                    # Validate input against the Pydantic model
                    validated_input = tool_meta.validate_input(tool_input)

                    # Pass along the agent if provided and expected by the function
                    if "agent" in kwargs and "agent" in tool_meta.signature.parameters:
                        validated_input["agent"] = kwargs.get("agent")

                    # CALL THE ACTUAL FUNCTION based on whether it's a method or not
                    if is_method_call:
                        # For methods, pass the instance as 'self'
                        result = f(instance, **validated_input)
                    else:
                        # For standalone functions, just pass the validated inputs
                        result = f(**validated_input)

                    # FORMAT THE RESULT for Strands Agent
                    if isinstance(result, dict) and "status" in result and "content" in result:
                        # Result is already in the expected format, just add toolUseId
                        result["toolUseId"] = tool_use_id
                        return result
                    else:
                        # Wrap any other return value in the standard format
                        # Always include at least one content item for consistency
                        return {
                            "toolUseId": tool_use_id,
                            "status": "success",
                            "content": [{"text": str(result)}],
                        }

                except ValueError as e:
                    # Special handling for validation errors
                    error_msg = str(e)
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Error: {error_msg}"}],
                    }
                except Exception as e:
                    # Return error result with exception details for any other error
                    error_type = type(e).__name__
                    error_msg = str(e)
                    return {
                        "toolUseId": tool_use_id,
                        "status": "error",
                        "content": [{"text": f"Error: {error_type} - {error_msg}"}],
                    }
            else:
                # NORMAL FUNCTION CALL - pass through to the original function
                if is_method_call:
                    # Put instance back as first argument for method calls
                    return f(instance, *args, **kwargs)
                else:
                    # Standard function call
                    return f(*args, **kwargs)

        # Also attach TOOL_SPEC to wrapper for compatibility
        wrapper.TOOL_SPEC = tool_spec  # type: ignore

        # Return the wrapper
        return cast(T, wrapper)

    # Handle both @tool and @tool() syntax
    if func is None:
        return decorator
    return decorator(func)
