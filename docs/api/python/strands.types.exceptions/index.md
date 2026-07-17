Exception-related type definitions for the SDK.

## EventLoopException

```python
class EventLoopException(Exception)
```

Defined in: [src/strands/types/exceptions.py:6](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L6)

Exception raised by the event loop.

#### \_\_init\_\_

```python
def __init__(original_exception: Exception, request_state: Any = None) -> None
```

Defined in: [src/strands/types/exceptions.py:9](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L9)

Initialize exception.

**Arguments**:

-   `original_exception` - The original exception that was raised.
-   `request_state` - The state of the request at the time of the exception.

## MaxTokensReachedException

```python
class MaxTokensReachedException(Exception)
```

Defined in: [src/strands/types/exceptions.py:21](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L21)

Exception raised when the model reaches its maximum token generation limit.

This exception is raised when the model stops generating tokens because it has reached the maximum number of tokens allowed for output generation. The partial message is automatically added to agent.messages and you can continue the conversation by calling the agent again.

This can occur when the model’s max\_tokens parameter is set too low for the complexity of the response, or when the model naturally reaches its configured output limit during generation.

#### \_\_init\_\_

```python
def __init__(message: str)
```

Defined in: [src/strands/types/exceptions.py:32](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L32)

Initialize the exception with an error message.

**Arguments**:

-   `message` - The error message describing the token limit issue

## ContextWindowOverflowException

```python
class ContextWindowOverflowException(Exception)
```

Defined in: [src/strands/types/exceptions.py:41](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L41)

Exception raised when the context window is exceeded.

This exception is raised when the input to a model exceeds the maximum context window size that the model can handle. This typically occurs when the combined length of the conversation history, system prompt, and current message is too large for the model to process.

## MCPClientInitializationError

```python
class MCPClientInitializationError(Exception)
```

Defined in: [src/strands/types/exceptions.py:52](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L52)

Raised when the MCP server fails to initialize properly.

## ModelThrottledException

```python
class ModelThrottledException(Exception)
```

Defined in: [src/strands/types/exceptions.py:58](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L58)

Exception raised when the model is throttled.

This exception is raised when the model is throttled by the service. This typically occurs when the service is throttling the requests from the client.

#### \_\_init\_\_

```python
def __init__(message: str) -> None
```

Defined in: [src/strands/types/exceptions.py:65](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L65)

Initialize exception.

**Arguments**:

-   `message` - The message from the service that describes the throttling.

## SessionException

```python
class SessionException(Exception)
```

Defined in: [src/strands/types/exceptions.py:77](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L77)

Exception raised when session operations fail.

## SnapshotException

```python
class SnapshotException(Exception)
```

Defined in: [src/strands/types/exceptions.py:83](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L83)

Exception raised when snapshot operations fail (e.g., unsupported schema version).

## ProviderTokenCountError

```python
class ProviderTokenCountError(Exception)
```

Defined in: [src/strands/types/exceptions.py:89](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L89)

Thrown when a model provider’s native token counting API fails.

This error is used as internal control flow within provider `count_tokens()` overrides. When caught, the provider falls back to the base class heuristic estimation.

## ToolProviderException

```python
class ToolProviderException(Exception)
```

Defined in: [src/strands/types/exceptions.py:99](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L99)

Exception raised when a tool provider fails to load or cleanup tools.

## StructuredOutputException

```python
class StructuredOutputException(Exception)
```

Defined in: [src/strands/types/exceptions.py:105](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L105)

Exception raised when structured output validation fails after maximum retry attempts.

#### \_\_init\_\_

```python
def __init__(message: str)
```

Defined in: [src/strands/types/exceptions.py:108](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L108)

Initialize the exception with details about the failure.

**Arguments**:

-   `message` - The error message describing the structured output failure

## ConcurrencyException

```python
class ConcurrencyException(Exception)
```

Defined in: [src/strands/types/exceptions.py:118](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L118)

Exception raised when concurrent invocations are attempted on an agent instance.

Agent instances maintain internal state that cannot be safely accessed concurrently. This exception is raised when an invocation is attempted while another invocation is already in progress on the same agent instance.

## IdempotencyAbortedError

```python
class IdempotencyAbortedError(Exception)
```

Defined in: [src/strands/types/exceptions.py:129](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L129)

Exception raised to duplicate invocations when the primary invocation was aborted.

When a caller provides an idempotency\_token and another invocation with the same token is already in-flight, the duplicate waits for the primary to complete. If the primary is aborted before producing a result (e.g. it lost a lock race or was cancelled), this exception is raised to all waiting duplicates.

## CheckpointException

```python
class CheckpointException(Exception)
```

Defined in: [src/strands/types/exceptions.py:139](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L139)

Exception raised when checkpoint operations fail (e.g., incompatible schema version).

## StorageError

```python
class StorageError(Exception)
```

Defined in: [src/strands/types/exceptions.py:145](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L145)

Raised when a storage operation fails.

Wraps backend-specific errors (filesystem, S3, network) with a uniform type that consumers can catch without knowing which backend is in use.

## AggregateMemoryError

```python
class AggregateMemoryError(Exception)
```

Defined in: [src/strands/types/exceptions.py:155](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L155)

Raised when one or more memory store operations fail.

**Attributes**:

-   `errors` - The underlying exceptions that caused this aggregate failure.

#### \_\_init\_\_

```python
def __init__(message: str, errors: list[BaseException]) -> None
```

Defined in: [src/strands/types/exceptions.py:162](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/types/exceptions.py#L162)

Initialize the aggregate error.

**Arguments**:

-   `message` - A human-readable description of the aggregate failure, typically naming the stores that failed.
-   `errors` - The underlying exceptions that caused this failure.