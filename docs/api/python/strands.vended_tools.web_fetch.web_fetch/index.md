Web fetch tool: fetch a URL and return relevant content about it.

Provides :func:`make_web_fetch` and the default :data:`web_fetch` instance. The factory’s `mode` parameter selects the extraction strategy at construction time:

-   `agentic` (default): HTML is converted to markdown and passed to an analyst agent that answers `prompt`; the full page never enters the main agent’s context.
-   `markdown`: HTML is converted to clean markdown with scripts, styles, and noise stripped. Use when the agent needs full pages for reasoning.

The tool delegates all networking to the `httpx.AsyncClient` instance provided by the operator, giving full control over transport configuration, caching, proxies, redirects, and connection pooling.

## WebFetchError

```python
class WebFetchError(ValueError)
```

Defined in: [src/strands/vended\_tools/web\_fetch/web\_fetch.py:32](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/web_fetch/web_fetch.py#L32)

Raised when a web fetch request fails.

#### make\_web\_fetch

```python
def make_web_fetch(
        *,
        name: str = "web_fetch",
        description: str | None = None,
        max_bytes: int = _DEFAULT_MAX_BYTES,
        max_content_chars: int = _DEFAULT_MAX_CONTENT_CHARS,
        client: httpx.AsyncClient | None = None,
        model: Model | None = None,
        mode: Literal["markdown",
                      "agentic"] = "agentic") -> DecoratedFunctionTool
```

Defined in: [src/strands/vended\_tools/web\_fetch/web\_fetch.py:56](https://github.com/strands-agents/harness-sdk/blob/main/strands-py/src/strands/vended_tools/web_fetch/web_fetch.py#L56)

Create a web fetch tool.

**Arguments**:

-   `name` - Tool name. Defaults to `"web_fetch"`.
-   `description` - Tool description shown to the model. Defaults to a mode-appropriate description when `None`.
-   `max_bytes` - Maximum response body size in bytes. Responses larger than this are rejected without buffering the entire body. Defaults to 5 MiB.
-   `max_content_chars` - Maximum characters of extracted content delivered to the model or analyst. Content exceeding this is truncated with a visible marker. Defaults to 50,000.
-   `client` - Optional `httpx.AsyncClient` to use for requests. When provided, the tool uses it directly and will not close it. When `None`, a new client is created per request with `follow_redirects=True` and httpx’s default timeout (5s).
-   `model` - Optional model for the analyst. Only used when `mode='agentic'`. Resolution order: this model, then the host agent’s model, then `WebFetchError` if neither is available.
-   `mode` - Extraction mode. Defaults to `agentic`.

**Returns**:

A decorated tool that fetches a URL and extracts content according to the configured mode:

-   `agentic` (default): HTML is converted to markdown and passed to an analyst agent that answers a `prompt`; the full page never enters the main agent’s context.
-   `markdown`: HTML converted to clean markdown; other content types returned as-is.

#### web\_fetch

Default web fetch tool (agentic mode).