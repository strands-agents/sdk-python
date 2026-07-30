Strands Agents SDK provides powerful capabilities for building AI agents with access to tools and external resources. With this power comes the responsibility to ensure your AI applications are developed and deployed in an ethical, safe, and beneficial manner. This guide outlines best practices for responsible AI usage with the Strands Agents SDK. Please also reference our [Prompt Engineering](/docs/user-guide/safety-security/prompt-engineering/index.md) page for guidance on how to effectively create agents that align with responsible AI usage, and [Guardrails](/docs/user-guide/safety-security/guardrails/index.md) page for how to add mechanisms to ensure safety and security.

You can learn more about the core dimensions of responsible AI on the [AWS Responsible AI](https://aws.amazon.com/ai/responsible-ai/) site.

### Tool Design

When designing tools with Strands, follow these principles:

1.  **Least Privilege**: Tools should have the minimum permissions needed
2.  **Input Validation**: Thoroughly validate all inputs to tools
3.  **Clear Documentation**: Document tool purpose, limitations, and expected inputs
4.  **Error Handling**: Gracefully handle edge cases and invalid inputs
5.  **Audit Logging**: Log sensitive operations for review

Below is an example of a simple tool design that follows these principles:

```python
@tool
def profanity_scanner(query: str) -> str:
    """Scans text files for profanity and inappropriate content.
    Only access allowed directories."""
    # Least Privilege: Verify path is in allowed directories
    allowed_dirs = ["/tmp/safe_files_1", "/tmp/safe_files_2"]
    real_path = os.path.realpath(os.path.abspath(query.strip()))
    if not any(real_path.startswith(d) for d in allowed_dirs):
        logging.warning(f"Security violation: {query}")  # Audit Logging
        return "Error: Access denied. Path not in allowed directories."

    try:
        # Error Handling: Read file securely
        if not os.path.exists(query):
            return f"Error: File '{query}' does not exist."
        with open(query, 'r') as f:
            file_content = f.read()

        # Use Agent to scan text for profanity
        profanity_agent = Agent(
            system_prompt="""You are a content moderator. Analyze the provided text
            and identify any profanity, offensive language, or inappropriate content.
            Report the severity level (mild, moderate, severe) and suggest appropriate
            alternatives where applicable. Be thorough but avoid repeating the offensive
            content in your analysis.""",
        )

        scan_prompt = f"Scan this text for profanity and inappropriate content:\n\n{file_content}"
        return profanity_agent(scan_prompt)["message"]["content"][0]["text"]

    except Exception as e:
        logging.error(f"Error scanning file: {str(e)}")  # Audit Logging
        return f"Error scanning file: {str(e)}"
```

---

**Additional Resources:**

-   [AWS Responsible AI Policy](https://aws.amazon.com/ai/responsible-ai/policy/)
-   [Anthropic’s Responsible Scaling Policy](https://www.anthropic.com/news/anthropics-responsible-scaling-policy)
-   [Partnership on AI](https://partnershiponai.org/)
-   [AI Ethics Guidelines Global Inventory](https://inventory.algorithmwatch.org/)
-   [OECD AI Principles](https://www.oecd.org/digital/artificial-intelligence/ai-principles/)

## Related pages

- [Attack Strategies](/docs/user-guide/evals-sdk/red-teaming/strategies/index.md) (1 shared tag)
- [Harmfulness Evaluator](/docs/user-guide/evals-sdk/evaluators/harmfulness_evaluator/index.md) (1 shared tag)
- [Reading the Report](/docs/user-guide/evals-sdk/red-teaming/reading_the_report/index.md) (1 shared tag)
- [Red Teaming](/docs/user-guide/evals-sdk/red-teaming/index.md) (1 shared tag)
- [Refusal Evaluator](/docs/user-guide/evals-sdk/evaluators/refusal_evaluator/index.md) (1 shared tag)
- [Scoring Attacks](/docs/user-guide/evals-sdk/red-teaming/evaluators/index.md) (1 shared tag)
- [Stereotyping Evaluator](/docs/user-guide/evals-sdk/evaluators/stereotyping_evaluator/index.md) (1 shared tag)
- [Writing Custom Cases](/docs/user-guide/evals-sdk/red-teaming/custom_cases/index.md) (1 shared tag)
- [Trusted Message History](/docs/user-guide/safety-security/trusted-message-history/index.md) (1 shared tag)
- [Instruction Following Evaluator](/docs/user-guide/evals-sdk/evaluators/instruction_following_evaluator/index.md) (1 shared tag)
