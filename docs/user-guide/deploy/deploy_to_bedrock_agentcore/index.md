Amazon Bedrock AgentCore Runtime is a secure, serverless runtime purpose-built for deploying and scaling dynamic AI agents and tools using any open-source framework including Strands Agents, LangChain, LangGraph and CrewAI. It supports any protocol such as MCP and A2A, and any model from any provider including Amazon Bedrock, OpenAI, Gemini, etc. Developers can securely and reliably run any type of agent including multi-modal, real-time, or long-running agents. AgentCore Runtime helps protect sensitive data with complete session isolation, providing dedicated microVMs for each user session - critical for AI agents that maintain complex state and perform privileged operations on users’ behalf. It is highly reliable with session persistence and it can scale up to thousands of agent sessions in seconds so developers don’t have to worry about managing infrastructure and only pay for actual usage. AgentCore Runtime, using AgentCore Identity, also seamlessly integrates with the leading identity providers such as Amazon Cognito, Microsoft Entra ID, and Okta, as well as popular OAuth providers such as Google and GitHub. It supports all authentication methods, from OAuth tokens and API keys to IAM roles, so developers don’t have to build custom security infrastructure.

## Prerequisites

Before you start, you need:

-   An AWS account with appropriate [permissions](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-permissions.html)
-   Python 3.10+ or Node.js 22+
-   Optional: A container engine (Docker, Finch, or Podman) - only required for local testing and advanced deployment scenarios

---

## Choose Strands SDK Your Language

Select your preferred programming language to get started with deploying Strands agents to Amazon Bedrock AgentCore Runtime:

[Python Deployment](python/index.md)Deploy your Python Strands agent to AgentCore Runtime!

[TypeScript Deployment](typescript/index.md)Deploy your TypeScript Strands agent to AgentCore Runtime!

## Additional Resources

-   [Amazon Bedrock AgentCore Runtime Documentation](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)
-   [Strands Documentation](https://strandsagents.com/latest/)
-   [AWS IAM Documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/introduction.html)
-   [Docker Documentation](https://docs.docker.com/)
-   [Amazon Bedrock AgentCore Observability](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability.html)

## Related pages

- [Python Deployment to Amazon Bedrock AgentCore Runtime](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/python/index.md) (4 shared tags)
- [TypeScript Deployment to Amazon Bedrock AgentCore Runtime](/docs/user-guide/deploy/deploy_to_bedrock_agentcore/typescript/index.md) (4 shared tags)
- [AgentCore Evaluation Dashboard Configuration](/docs/user-guide/evals-sdk/how-to/agentcore_evaluation_dashboard/index.md) (3 shared tags)
- [Deploying Strands Agents SDK Agents to Amazon EC2](/docs/user-guide/deploy/deploy_to_amazon_ec2/index.md) (2 shared tags)
- [Deploying Strands Agents SDK Agents to Amazon EKS](/docs/user-guide/deploy/deploy_to_amazon_eks/index.md) (2 shared tags)
- [Deploying Strands Agents SDK Agents to AWS App Runner](/docs/user-guide/deploy/deploy_to_aws_apprunner/index.md) (2 shared tags)
- [Deploying Strands Agents SDK Agents to AWS Fargate](/docs/user-guide/deploy/deploy_to_aws_fargate/index.md) (2 shared tags)
- [Deploying Strands Agents SDK Agents to AWS Lambda](/docs/user-guide/deploy/deploy_to_aws_lambda/index.md) (2 shared tags)
- [Guardrails](/docs/user-guide/safety-security/guardrails/index.md) (2 shared tags)
- [Amazon Nova](/docs/user-guide/concepts/model-providers/amazon-nova/index.md) (2 shared tags)
