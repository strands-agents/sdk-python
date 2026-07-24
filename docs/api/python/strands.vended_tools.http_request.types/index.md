Shared types and constants for the http\_request tool.

#### HttpMethod

HTTP methods supported by the tool.

## HttpRequestOutput

```python
class HttpRequestOutput(TypedDict)
```

Defined in: [src/strands/vended\_tools/http\_request/types.py:9](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/http_request/types.py#L9)

Output of an HTTP request.

**Attributes**:

-   `status` - HTTP status code.
-   `status_text` - HTTP status reason phrase.
-   `headers` - Response headers as a plain dict (lower-cased keys).
-   `body` - Response body as text.

#### DEFAULT\_HTTP\_REQUEST\_DESCRIPTION

Description for the http\_request tool shown to the model.