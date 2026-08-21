The Strands SDK provides logging infrastructure to give visibility into its operations.

(( tab "Python" ))
Strands SDK uses Python’s standard [`logging`](https://docs.python.org/3/library/logging.html) module. The SDK implements a straightforward logging approach:

1.  **Module-level Loggers**: Each module creates its own logger using `logging.getLogger(__name__)`, following Python best practices for hierarchical logging.
    
2.  **Root Logger**: All loggers are children of the “strands” root logger, making it easy to configure logging for the entire SDK.
    
3.  **Default Behavior**: By default, the SDK doesn’t configure any handlers or log levels, allowing you to integrate it with your application’s logging configuration.
(( /tab "Python" ))

(( tab "TypeScript" ))
Strands SDK provides a simple logging infrastructure with a global logger that can be configured to use your preferred logging implementation.

1.  **Logger Interface**: A simple interface (`debug`, `info`, `warn`, `error`) compatible with popular logging libraries like Pino, Winston, and the browser/Node.js console.
    
2.  **Global Logger**: A single global logger instance configured via `configureLogging()`.
    
3.  **Default Behavior**: By default, the SDK only logs warnings and errors to the console. Debug and info logs are no-ops unless you configure a custom logger.
(( /tab "TypeScript" ))

## Configuring Logging

(( tab "Python" ))
To enable logging for the Strands Agents SDK, you can configure the “strands” logger:

```python
import logging

# Configure the root strands logger
logging.getLogger("strands").setLevel(logging.DEBUG)

# Add a handler to see the logs
logging.basicConfig(
    format="%(levelname)s | %(name)s | %(message)s",
    handlers=[logging.StreamHandler()]
)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
To enable logging for the Strands Agents SDK, use the `configureLogging` function. The SDK’s logger interface is compatible with standard console and popular logging libraries.

**Using console:**

```typescript
// Use the default console for logging
configureLogging(console)
```

**Using Pino:**

```typescript
import pino from 'pino'

const pinoLogger = pino({
  level: 'debug',
  transport: {
    target: 'pino-pretty',
    options: {
      colorize: true,
    },
  },
})

configureLogging(pinoLogger)
```

**Default Behavior:**

-   By default, the SDK only logs warnings and errors using `console.warn()` and `console.error()`
-   Debug and info logs are no-ops by default (zero performance overhead)
-   Configure a custom logger with appropriate log levels to enable debug/info logging
(( /tab "TypeScript" ))

### Log Levels

The Strands Agents SDK uses standard log levels:

-   **DEBUG**: Detailed operational information for troubleshooting. Extensively used for tool registration, discovery, configuration, and execution flows.
    
-   **INFO**: General informational messages. Currently not used.
    
-   **WARNING**: Potential issues that don’t prevent operation, such as validation failures, specification errors, and compatibility warnings.
    
-   **ERROR**: Significant problems that prevent specific operations from completing successfully, such as execution failures and handler errors.
    
-   **CRITICAL**: Reserved for catastrophic failures.
    

## Key Logging Areas

(( tab "Python" ))
The Strands Agents SDK logs information in several key areas. Let’s look at what kinds of logs you might see when using the following example agent with a calculator tool:

```python
from strands import Agent
from strands_tools import calculator

# Create an agent with the calculator tool
agent = Agent(tools=[calculator])
result = agent("What is 125 * 37?")
```

When running this code with logging enabled, you’ll see logs from different components of the SDK as the agent processes the request, calls the calculator tool, and generates a response.

**Tool Registry and Execution**

Logs related to tool discovery, registration, and execution:

```plaintext
# Tool registration
DEBUG | strands.tools.registry | tool_name=<calculator> | registering tool
DEBUG | strands.tools.registry | tool_name=<calculator>, tool_type=<function>, is_dynamic=<False> | registering tool
DEBUG | strands.tools.registry | tool_name=<calculator> | loaded tool config
DEBUG | strands.tools.registry | tool_count=<1> | tools configured

# Tool discovery
DEBUG | strands.tools.registry | tools_dir=</path/to/tools> | found tools directory
DEBUG | strands.tools.registry | tools_dir=</path/to/tools> | scanning
DEBUG | strands.tools.registry | tool_modules=<['calculator', 'weather']> | discovered

# Tool validation
WARNING | strands.tools.registry | tool_name=<invalid_tool> | spec validation failed | Missing required fields in tool spec: description
DEBUG | strands.tools.registry | tool_name=<calculator> | loaded dynamic tool config

# Tool execution
DEBUG | strands.event_loop.event_loop | tool_use=<calculator_tool_use_id> | streaming

# Tool hot reloading
DEBUG | strands.tools.registry | tool_name=<calculator> | searching directories for tool
DEBUG | strands.tools.registry | tool_name=<calculator> | reloading tool
DEBUG | strands.tools.registry | tool_name=<calculator> | successfully reloaded tool
```

**Event Loop**

Logs related to the event loop processing:

```plaintext
ERROR | strands.event_loop.error_handler | an exception occurred in event_loop_cycle | ContextWindowOverflowException
DEBUG | strands.event_loop.error_handler | message_index=<5> | found message with tool results at index
```

**Model Interactions**

Logs related to interactions with foundation models:

```plaintext
DEBUG | strands.models.bedrock | config=<{'model_id': 'global.anthropic.claude-sonnet-4-6'}> | initializing
WARNING | strands.models.bedrock | bedrock threw context window overflow error
DEBUG | strands.models.bedrock | Found blocked output guardrail. Redacting output.
```
(( /tab "Python" ))

(( tab "TypeScript" ))
The TypeScript SDK currently has minimal logging, primarily focused on model interactions. Logs are generated for:

-   **Model configuration warnings**: Unsupported features (e.g., cache points in OpenAI, guard content)
-   **Model response warnings**: Invalid formats, unexpected data structures
-   **Bedrock-specific operations**: Configuration auto-detection, unsupported event types

Example logs you might see:

```plaintext
# Model configuration warnings
WARN cache points are not supported in openai system prompts, ignoring cache points
WARN guard content is not supported in openai system prompts, removing guard content block

# Model response warnings
WARN choice=<null> | invalid choice format in openai chunk
WARN tool_call=<{"type":"function","id":"xyz"}> | received tool call with invalid index

# Bedrock-specific logs
DEBUG model_id=<global.anthropic.claude-sonnet-4-6>, include_tool_result_status=<true> | auto-detected includeToolResultStatus
WARN block_key=<unknown_key> | skipping unsupported block key
WARN event_type=<unknown_type> | unsupported bedrock event type
```

Future versions will include more detailed logging for tool operations and event loop processing.
(( /tab "TypeScript" ))

## Advanced Configuration

(( tab "Python" ))
**Filtering Specific Modules**

You can configure logging for specific modules within the SDK:

```python
import logging

# Enable DEBUG logs for the tool registry only
logging.getLogger("strands.tools.registry").setLevel(logging.DEBUG)

# Set WARNING level for model interactions
logging.getLogger("strands.models").setLevel(logging.WARNING)
```

**Custom Handlers**

You can add custom handlers to process logs in different ways:

```python
import logging
import json

class JsonFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "name": record.name,
            "message": record.getMessage()
        }
        return json.dumps(log_data)

# Create a file handler with JSON formatting
file_handler = logging.FileHandler("strands_agents_sdk.log")
file_handler.setFormatter(JsonFormatter())

# Add the handler to the strands logger
logging.getLogger("strands").addHandler(file_handler)
```
(( /tab "Python" ))

(( tab "TypeScript" ))
**Custom Logger Implementation**

You can implement your own logger to integrate with your application’s logging system:

```typescript
// Declare a mock logging service type for documentation
declare const myLoggingService: {
  log(level: string, ...args: unknown[]): void
}

const customLogger: Logger = {
  debug: (...args: unknown[]) => {
    // Send to your logging service
    myLoggingService.log('DEBUG', ...args)
  },
  info: (...args: unknown[]) => {
    myLoggingService.log('INFO', ...args)
  },
  warn: (...args: unknown[]) => {
    myLoggingService.log('WARN', ...args)
  },
  error: (...args: unknown[]) => {
    myLoggingService.log('ERROR', ...args)
  },
}

configureLogging(customLogger)
```
(( /tab "TypeScript" ))

## Best Practices

(( tab "Python" ))
1.  **Configure Early**: Set up logging configuration before initializing the agent
2.  **Appropriate Levels**: Use INFO for normal operation and DEBUG for troubleshooting
3.  **Structured Log Format**: Use the structured log format shown in examples for better parsing
4.  **Performance**: Be mindful of logging overhead in production environments
5.  **Integration**: Integrate Strands Agents SDK logging with your application’s logging system
(( /tab "Python" ))

(( tab "TypeScript" ))
1.  **Configure Early**: Call `configureLogging()` before creating any Agent instances
2.  **Default Behavior**: By default, only warnings and errors are logged - configure a custom logger to see debug information
3.  **Production Performance**: Debug and info logs are no-ops by default, minimizing performance impact
4.  **Compatible Libraries**: Use established logging libraries like Pino or Winston for production deployments
5.  **Consistent Format**: Ensure your custom logger maintains consistent formatting across log levels
(( /tab "TypeScript" ))

## Related pages

- [PII Redaction](/docs/user-guide/safety-security/pii-redaction/index.md) (2 shared tags)
- [Evaluating Remote Traces](/docs/user-guide/evals-sdk/how-to/trace_providers/index.md) (1 shared tag)
- [Metrics](/docs/user-guide/observability-evaluation/metrics/index.md) (1 shared tag)
- [Observability](/docs/user-guide/observability-evaluation/observability/index.md) (1 shared tag)
- [Task Decorator](/docs/user-guide/evals-sdk/how-to/eval_task/index.md) (1 shared tag)
- [Traces](/docs/user-guide/observability-evaluation/traces/index.md) (1 shared tag)
- [Bidirectional Streaming Observability](/docs/user-guide/concepts/bidirectional-streaming/observability/index.md) (1 shared tag)
- [Operating Agents in Production](/docs/user-guide/deploy/operating-agents-in-production/index.md) (1 shared tag)
- [Root Cause Analysis](/docs/user-guide/evals-sdk/detectors/root_cause_analysis/index.md) (1 shared tag)
- [Session Diagnosis](/docs/user-guide/evals-sdk/detectors/diagnosis/index.md) (1 shared tag)
