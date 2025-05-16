# Strands Agents

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

<div align="center">
  <h2>
    A model-driven approach to building AI agents in just a few lines of code.
  </h2>

  <a href="https://strandsagents.com/">Docs</a>
  ◆ <a href="https://github.com/strands-agents/docs/tree/main/docs/examples">Samples</a>
  ◆ <a href="https://github.com/strands-agents/tools">Tools</a>
  ◆ <a href="https://github.com/strands-agents/agent-builder">Agent Builder</a>
</div>

Strands Agents is a simple yet powerful SDK that takes a model-driven approach to building and running AI agents. From simple conversational assistants to complex autonomous workflows, from local development to production deployment, Strands Agents scales with your needs.

## Feature Overview

- **Lightweight & Flexible**: Simple agent loop that just works and is fully customizable
- **Model Agnostic**: Support for Amazon Bedrock, Anthropic, Ollama, and custom providers
- **Advanced Capabilities**: Multi-agent systems, autonomous agents, and streaming support
- **Built-in MCP**: Native support for Model Context Protocol (MCP) servers, enabling access to thousands of pre-built tools

## Quick Start

```bash
# Install Strands Agents
pip install strands-agents strands-agents-tools
```

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

> **Note**: For the default Amazon Bedrock model provider, you'll need AWS credentials configured and model access enabled for Claude 3.7 Sonnet in the us-west-2 region. See the [Quickstart Guide](https://strandsagents.com/) for details on configuring other model providers.

## Installation

Ensure you have Python 3.10+ installed, then:

```bash
# Create and activate virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows use: .venv\Scripts\activate

# Install Strands and tools
pip install strands-agents strands-agents-tools
```

## Features at a Glance

### Python-Based Tools

Easily build tools using Python decorators:

```python
from strands import Agent, tool

@tool
def word_count(text: str) -> int:
    """Count words in text.

    This docstring is used by the LLM to understand the tool's purpose.
    """
    return len(text.split())

agent = Agent(tools=[word_count])
response = agent("How many words are in this sentence?")
```

### MCP Support

Seamlessly integrate Model Context Protocol (MCP) servers:

```python
from strands import Agent
from strands.tools.mcp import MCPClient
from mcp import stdio_client, StdioServerParameters

aws_docs_client = MCPClient(
    lambda: stdio_client(StdioServerParameters(command="uvx", args=["awslabs.aws-documentation-mcp-server@latest"]))
)

with aws_docs_client:
   agent = Agent(tools=aws_docs_client.list_tools_sync())
   response = agent("Tell me about Amazon Bedrock and how to use it with Python")
```

### Multiple Model Providers

Support for various model providers:

```python
from strands import Agent
from strands.models import BedrockModel
from strands.models.ollama import OllamaModel
from strands.models.llamaapi import LlamaAPIModel

# Bedrock
bedrock_model = BedrockModel(
  model_id="us.amazon.nova-pro-v1:0",
  temperature=0.3,
)
agent = Agent(model=bedrock_model)
agent("Tell me about Agentic AI")

# Ollama
ollama_modal = OllamaModel(
  host="http://localhost:11434",
  model_id="llama3"
)
agent = Agent(model=ollama_modal)
agent("Tell me about Agentic AI")

# Llama API
llama_model = LlamaAPIModel(
    model_id="Llama-4-Maverick-17B-128E-Instruct-FP8",
)
agent = Agent(model=llama_model)
response = agent("Tell me about Agentic AI")
```

Built-in providers:
 - [Amazon Bedrock](https://strandsagents.com/latest/user-guide/concepts/model-providers/amazon-bedrock/)
 - [Anthropic](https://strandsagents.com/latest/user-guide/concepts/model-providers/anthropic/)
 - [LiteLLM](https://strandsagents.com/latest/user-guide/concepts/model-providers/litellm/)
 - [LlamaAPI](https://strandsagents.com/latest/user-guide/concepts/model-providers/llamaapi/)
 - [Ollama](https://strandsagents.com/latest/user-guide/concepts/model-providers/ollama/)

Custom providers can be implemented using [Custom Providers](https://strandsagents.com/latest/user-guide/concepts/model-providers/custom_model_provider/)

### Example tools

Strands offers an optional strands-agents-tools package with pre-built tools for quick experimentation:

```python
from strands import Agent
from strands_tools import calculator
agent = Agent(tools=[calculator])
agent("What is the square root of 1764")
```

It's also available on GitHub via [strands-agents-tools](https://github.com/strands-agents/strands-agents-tools).

## Documentation

For detailed guidance & examples, explore our documentation:

- [User Guide](https://strandsagents.com/)
- [Quick Start Guide](https://strandsagents.com/latest/user-guide/quickstart/)
- [Agent Loop](https://strandsagents.com/latest/user-guide/concepts/agents/agent-loop/)
- [Examples](https://strandsagents.com/latest/examples/)
- [API Reference](https://strandsagents.com/latest/api-reference/agent/)
- [Production & Deployment Guide](https://strandsagents.com/latest/user-guide/deploy/operating-agents-in-production/)

## Contributing

We welcome contributions! See our [Contributing Guide](https://github.com/strands-agents/sdk-python/blob/main/CONTRIBUTING.md) for details on:
- Reporting bugs & features
- Development setup
- Contributing via Pull Requests
- Code of Conduct
- Reporting of security issues

## License

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE) file for details.

## ⚠️ Preview Status

Strands Agents is currently in public preview. During this period:
- APIs may change as we refine the SDK
- We welcome feedback and contributions
