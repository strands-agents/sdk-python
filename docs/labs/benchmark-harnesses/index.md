[Strands Benchmark Harnesses](https://github.com/strands-labs/benchmark-harnesses) is a repository of Strands-based agents and harnesses for agentic benchmarks. It is a [uv workspace](https://docs.astral.sh/uv/concepts/projects/workspaces/): the repository root coordinates one or more member packages, each with its own setup and usage.

## Simple Strands Agent (SSA)

The flagship member is **Simple Strands Agent** - a lean, hackable harness for autonomous software engineering that achieves state-of-the-art performance across software-engineering benchmarks. It pairs frontier LLMs (Claude, GPT, Gemini, and open-weight models via Bedrock / LiteLLM / vLLM) with `bash` and file-editing tools inside isolated Docker environments to analyze codebases, diagnose bugs, write patches, and verify solutions.

Despite its simplicity, SSA delivers SOTA-level results on **SWE-Bench Verified**, **SWE-Bench Pro**, and **Terminal Bench 2**. See the [technical report](https://arxiv.org/abs/2606.17454) and the [Amazon Science post](https://www.amazon.science/blog/bridging-intent-and-execution-in-agentic-systems).

### Highlights

-   **Model-agnostic** - first-class adapters for Anthropic, OpenAI, Google, xAI, Bedrock, and any OpenAI-compatible endpoint (vLLM, LiteLLM, Together, Vertex, Z.AI).
-   **Composable tools** - `bash`, `str_replace_editor`, `think`, and `submit` primitives with per-tool output clipping and timeout controls.
-   **Isolated environments** - Docker-backed sandboxes with streaming exec, automatic image resolution, and ECR support.
-   **Hydra-powered configs** - every knob is overridable from the command line; experiments are reproducible from a single YAML.
-   **Built-in benchmarking** - turnkey scripts for SWE-Bench Verified, SWE-Bench Pro, and Terminal Bench 2, including S3 result upload.

## Getting started

```bash
git clone https://github.com/strands-labs/benchmark-harnesses.git
cd benchmark-harnesses

# Recommended: sync the workspace (creates .venv with the package + deps)
uv sync
source .venv/bin/activate
```

Run a single instance:

```bash
uv run python -m ssa.run \
    --config-name=default.yaml \
    dataset.name=sbv \
    dataset.identifier=django__django-15987 \
    env.env_type=docker \
    env.docker.workdir="/testbed"
```

## Running agents safely

Agents in this repository are given shell access, which means the model can run commands in the environment where the agent starts. Run agents inside Docker or another sandbox, mount only what the agent needs, and avoid exposing credentials, SSH keys, or production data.

## Links

-   [GitHub repository](https://github.com/strands-labs/benchmark-harnesses)
-   [Technical report (arXiv)](https://arxiv.org/abs/2606.17454)
-   [Amazon Science blog post](https://www.amazon.science/blog/bridging-intent-and-execution-in-agentic-systems)