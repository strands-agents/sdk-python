Tool decorator for SDK.

This module provides the @tool decorator that transforms Python functions into SDK Agent tools with automatic metadata extraction and validation.

The @tool decorator performs several functions:

1.  Extracts function metadata (name, description, parameters) from docstrings and type hints
2.  Generates a JSON schema for input validation
3.  Handles two different calling patterns:

-   Standard function calls (func(arg1, arg2))
-   Tool use calls (agent.my\_tool(param1=“hello”, param2=123))

4.  Provides error handling and result formatting
5.  Works with both standalone functions and class methods

**Example**:

```python
from strands import Agent, tool

@tool
def my_tool(param1: str, param2: int = 42) -> dict:
    '''
    Tool description - explain what it does.

    `Args`:
        param1: Description of first parameter.
        param2: Description of second parameter (default: 42).

    `Returns`:
        A dictionary with the results.
    '''
    result = do_something(param1, param2)
    return \{
        "status": "success",
        "content": [\{"text": f"Result: \{result}"}]
    }

agent = Agent(tools=[my_tool])
agent.tool.my_tool(param1="hello", param2=123)
```

## FunctionToolMetadata

```python
class FunctionToolMetadata()
```

Defined in: [src/strands/tools/decorator.py:80](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L80)

Helper class to extract and manage function metadata for tool decoration.

This class handles the extraction of metadata from Python functions including:

-   Function name and description from docstrings
-   Parameter names, types, and descriptions
-   Return type information
-   Creation of Pydantic models for input validation

The extracted metadata is used to generate a tool specification that can be used by Strands Agent to understand and validate tool usage.

#### \_\_init\_\_

```python
def __init__(func: Callable[..., Any],
             context_param: str | None = None) -> None
```

Defined in: [src/strands/tools/decorator.py:94](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L94)

Initialize with the function to process.

**Arguments**:

-   `func` - The function to extract metadata from. Can be a standalone function or a class method.
-   `context_param` - Name of the context parameter to inject, if any.

#### extract\_metadata

```python
def extract_metadata() -> ToolSpec
```

Defined in: [src/strands/tools/decorator.py:285](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L285)

Extract metadata from the function to create a tool specification.

This method analyzes the function to create a standardized tool specification that Strands Agent can use to understand and interact with the tool.

The specification includes:

-   name: The function name (or custom override)
-   description: The function’s docstring description (excluding Args)
-   inputSchema: A JSON schema describing the expected parameters

**Returns**:

A dictionary containing the tool specification.

#### validate\_input

```python
def validate_input(input_data: dict[str, Any]) -> dict[str, Any]
```

Defined in: [src/strands/tools/decorator.py:371](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L371)

Validate input data using the Pydantic model.

This method ensures that the input data meets the expected schema before it’s passed to the actual function. It converts the data to the correct types when possible and raises informative errors when not.

**Arguments**:

-   `input_data` - A dictionary of parameter names and values to validate.

**Returns**:

A dictionary with validated and converted parameter values.

**Raises**:

-   `ValueError` - If the input data fails validation, with details about what failed.

#### inject\_special\_parameters

```python
def inject_special_parameters(validated_input: dict[str,
                                                    Any], tool_use: ToolUse,
                              invocation_state: dict[str, Any]) -> None
```

Defined in: [src/strands/tools/decorator.py:397](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L397)

Inject special framework-provided parameters into the validated input.

This method automatically provides framework-level context to tools that request it through their function signature.

**Arguments**:

-   `validated_input` - The validated input parameters (modified in place).
-   `tool_use` - The tool use request containing tool invocation details.
-   `invocation_state` - Caller-provided kwargs that were passed to the agent when it was invoked (agent(), agent.invoke\_async(), etc.).

#### P

Captures all parameters

#### R

Return type

## DecoratedFunctionTool

```python
class DecoratedFunctionTool(AgentTool, Generic[P, R])
```

Defined in: [src/strands/tools/decorator.py:454](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L454)

An AgentTool that wraps a function that was decorated with @tool.

This class adapts Python functions decorated with @tool to the AgentTool interface. It handles both direct function calls and tool use invocations, maintaining the function’s original behavior while adding tool capabilities.

The class is generic over the function’s parameter types (P) and return type (R) to maintain type safety.

#### \_\_init\_\_

```python
def __init__(tool_name: str, tool_spec: ToolSpec, tool_func: Callable[P, R],
             metadata: FunctionToolMetadata)
```

Defined in: [src/strands/tools/decorator.py:469](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L469)

Initialize the decorated function tool.

**Arguments**:

-   `tool_name` - The name to use for the tool (usually the function name).
-   `tool_spec` - The tool specification containing metadata for Agent integration.
-   `tool_func` - The original function being decorated.
-   `metadata` - The FunctionToolMetadata object with extracted function information.

#### \_\_get\_\_

```python
def __get__(instance: Any,
            obj_type: type | None = None) -> "DecoratedFunctionTool[P, R]"
```

Defined in: [src/strands/tools/decorator.py:493](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L493)

Descriptor protocol implementation for proper method binding.

This method enables the decorated function to work correctly when used as a class method. It binds the instance to the function call when accessed through an instance.

**Arguments**:

-   `instance` - The instance through which the descriptor is accessed, or None when accessed through the class.
-   `obj_type` - The class through which the descriptor is accessed.

**Returns**:

A new DecoratedFunctionTool with the instance bound to the function if accessed through an instance, otherwise returns self.

**Example**:

```python
class MyClass:
    @tool
    def my_tool():
        ...

instance = MyClass()
# instance of DecoratedFunctionTool that works as you'd expect
tool = instance.my_tool
```

#### \_\_call\_\_

```python
def __call__(*args: P.args, **kwargs: P.kwargs) -> R
```

Defined in: [src/strands/tools/decorator.py:526](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L526)

Call the original function with the provided arguments.

This method enables the decorated function to be called directly with its original signature, preserving the normal function call behavior.

**Arguments**:

-   `*args` - Positional arguments to pass to the function.
-   `**kwargs` - Keyword arguments to pass to the function.

**Returns**:

The result of the original function call.

#### tool\_name

```python
@property
def tool_name() -> str
```

Defined in: [src/strands/tools/decorator.py:542](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L542)

Get the name of the tool.

**Returns**:

The tool name as a string.

#### tool\_spec

```python
@property
def tool_spec() -> ToolSpec
```

Defined in: [src/strands/tools/decorator.py:551](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L551)

Get the tool specification.

**Returns**:

The tool specification dictionary containing metadata for Agent integration.

#### tool\_spec

```python
@tool_spec.setter
def tool_spec(value: ToolSpec) -> None
```

Defined in: [src/strands/tools/decorator.py:560](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L560)

Set the tool specification.

This allows runtime modification of the tool’s schema, enabling dynamic tool configurations based on feature flags or other runtime conditions.

**Arguments**:

-   `value` - The new tool specification.

**Raises**:

-   `ValueError` - If the spec fails structural validation (wrong name or missing required field).

#### tool\_type

```python
@property
def tool_type() -> str
```

Defined in: [src/strands/tools/decorator.py:585](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L585)

Get the type of the tool.

**Returns**:

The string “function” indicating this is a function-based tool.

#### stream

```python
@override
async def stream(tool_use: ToolUse, invocation_state: dict[str, Any],
                 **kwargs: Any) -> ToolGenerator
```

Defined in: [src/strands/tools/decorator.py:594](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L594)

Stream the tool with a tool use specification.

This method handles tool use streams from a Strands Agent. It validates the input, calls the function, and formats the result according to the expected tool result format.

Key operations:

1.  Extract tool use ID and input parameters
2.  Validate input against the function’s expected parameters
3.  Call the function with validated input
4.  Format the result as a standard tool result
5.  Handle and format any errors that occur

**Arguments**:

-   `tool_use` - The tool use specification from the Agent.
-   `invocation_state` - Caller-provided kwargs that were passed to the agent when it was invoked (agent(), agent.invoke\_async(), etc.).
-   `**kwargs` - Additional keyword arguments for future extensibility.

**Yields**:

Tool events with the last being the tool result.

#### supports\_hot\_reload

```python
@property
def supports_hot_reload() -> bool
```

Defined in: [src/strands/tools/decorator.py:711](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L711)

Check if this tool supports automatic reloading when modified.

**Returns**:

Always true for function-based tools.

#### get\_display\_properties

```python
@override
def get_display_properties() -> dict[str, str]
```

Defined in: [src/strands/tools/decorator.py:720](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L720)

Get properties to display in UI representations.

**Returns**:

Function properties (e.g., function name).

#### tool

```python
def tool(
    func: Callable[P, R] | None = None,
    description: str | None = None,
    inputSchema: JSONSchema | None = None,
    name: str | None = None,
    context: bool | str = False
) -> DecoratedFunctionTool[P, R] | Callable[[Callable[P, R]],
                                            DecoratedFunctionTool[P, R]]
```

Defined in: [src/strands/tools/decorator.py:744](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/tools/decorator.py#L744)

Decorator that transforms a Python function into a Strands tool.

This decorator seamlessly enables a function to be called both as a regular Python function and as a Strands tool. It extracts metadata from the function’s signature, docstring, and type hints to generate an OpenAPI-compatible tool specification.

When decorated, a function:

1.  Still works as a normal function when called directly with arguments
2.  Processes tool use API calls when provided with a tool use dictionary
3.  Validates inputs against the function’s type hints and parameter spec
4.  Formats return values according to the expected Strands tool result format
5.  Provides automatic error handling and reporting

The decorator can be used in two ways:

-   As a simple decorator: `@tool`
-   With parameters: `@tool(name="custom_name", description="Custom description")`

**Arguments**:

-   `func` - The function to decorate. When used as a simple decorator, this is the function being decorated. When used with parameters, this will be None.
-   `description` - Optional custom description to override the function’s docstring.
-   `inputSchema` - Optional custom JSON schema to override the automatically generated schema.
-   `name` - Optional custom name to override the function’s name.
-   `context` - When provided, places an object in the designated parameter. If True, the param name defaults to ‘tool\_context’, or if an override is needed, set context equal to a string to designate the param name.

**Returns**:

An AgentTool that also mimics the original function when invoked

**Example**:

```python
@tool
def my_tool(name: str, count: int = 1) -> str:
    # Does something useful with the provided parameters.
    #
    # Parameters:
    #   name: The name to process
    #   count: Number of times to process (default: 1)
    #
    # Returns:
    #   A message with the result
    return f"Processed \{name} \{count} times"

agent = Agent(tools=[my_tool])
agent.my_tool(name="example", count=3)
# Returns: \{
#   "toolUseId": "123",
#   "status": "success",
#   "content": [\{"text": "Processed example 3 times"}]
# }
```

Example with parameters:

```python
@tool(name="custom_tool", description="A tool with a custom name and description", context=True)
def my_tool(name: str, count: int = 1, tool_context: ToolContext) -> str:
    tool_id = tool_context["tool_use"]["toolUseId"]
    return f"Processed \{name} \{count} times with tool ID \{tool_id}"
```