A collection of sample implementations to help you get started with Strands Agents. From simple agents to complex multi-agent systems, each example illustrates key concepts and patterns you can adapt for your own projects.

## Getting Started

1.  Set up the SDK for your language:
    -   [Python quickstart](/docs/user-guide/quickstart/python/index.md) (Python 3.10+, pip)
    -   [TypeScript quickstart](/docs/user-guide/quickstart/typescript/index.md) (Node.js 22+, npm)
2.  Configure AWS credentials for Amazon Bedrock (covered in both quickstart guides above), or set up an [alternative model provider](/docs/user-guide/concepts/model-providers/index.md)
3.  Clone the examples:
    
    ```bash
    git clone https://github.com/strands-agents/harness-sdk.git
    cd harness-sdk/site/docs/examples
    ```
    
4.  Browse the examples below and follow the instructions in each one

## Agent Examples

| Example | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Structured Output](/docs/examples/structured_output/index.md) | Type-safe, validated responses | ✅ | ✅ |
| [Agents Workflows](/docs/examples/python/agents_workflows/index.md) | Sequential agent workflow pattern | ✅ |  |
| [File Operations](/docs/examples/python/file_operations/index.md) | File manipulation capabilities | ✅ |  |
| [Graph Loops](/docs/examples/python/graph_loops_example/index.md) | Graph orchestration with loops | ✅ |  |
| [Knowledge Base Agent](/docs/examples/python/knowledge_base_agent/index.md) | Knowledge base retrieval | ✅ |  |
| [MCP Calculator](/docs/examples/python/mcp_calculator/index.md) | Model Context Protocol capabilities | ✅ |  |
| [Memory Agent](/docs/examples/python/memory_agent/index.md) | Persistent memory | ✅ |  |
| [Meta Tooling](/docs/examples/python/meta_tooling/index.md) | Meta tooling capabilities | ✅ |  |
| [Multi-Agent Example](/docs/examples/python/multi_agent_example/multi_agent_example/index.md) | Multi-agent system | ✅ |  |
| [Multimodal](/docs/examples/python/multimodal/index.md) | Multimodal capabilities | ✅ |  |
| [Weather Forecaster](/docs/examples/python/weather_forecaster/index.md) | Weather forecasting agent | ✅ |  |

## Deployment Examples

Also see [Operating Agents in Production](/docs/user-guide/deploy/operating-agents-in-production/index.md) for best practices on security, monitoring, and scaling.

| Guide | Description | Python | TypeScript |
| --- | --- | --- | --- |
| [Bedrock AgentCore](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/index.md) | Serverless agent runtime | ✅ | ✅ |
| [Docker](/docs/user-guide/deploy/deploy_to_docker/index.md) | Containerized deployment | ✅ | ✅ |
| [AWS Lambda](/docs/user-guide/deploy/deploy_to_aws_lambda/index.md) | Serverless compute | ✅ |  |
| [AWS Fargate](/docs/user-guide/deploy/deploy_to_aws_fargate/index.md) | Serverless containers | ✅ |  |
| [AWS App Runner](/docs/user-guide/deploy/deploy_to_aws_apprunner/index.md) | Managed web applications | ✅ |  |
| [Amazon EC2](/docs/user-guide/deploy/deploy_to_amazon_ec2/index.md) | Virtual machines | ✅ |  |
| [Amazon EKS](/docs/user-guide/deploy/deploy_to_amazon_eks/index.md) | Managed Kubernetes | ✅ |  |
| [Kubernetes](/docs/user-guide/deploy/deploy_to_kubernetes/index.md) | Self-managed Kubernetes | ✅ |  |