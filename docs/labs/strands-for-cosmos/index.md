[Strands for Cosmos](https://github.com/strands-labs/strands-for-cosmos) brings [NVIDIA Cosmos](https://research.nvidia.com/labs/cosmos-lab/cosmos3/) to Strands Agents - giving an agent eyes that understand physics and hands that generate video, audio, and robot actions, all on local compute. It ships four model providers (Cosmos 3 omnimodal Reasoner and Generator, plus the Cosmos-Reason2 VLM) and 45 tools spanning the full pipeline: reasoning, generation, curation, post-training, quantization, edge deployment, and evaluation.

```python
from strands import Agent
from strands_cosmos import Cosmos3ReasonerModel

agent = Agent(model=Cosmos3ReasonerModel(base_url="http://localhost:8000/v1"))
agent("Caption in detail: <video>scene.mp4</video>")
```

## Getting started

```bash
uv pip install strands-cosmos
```

The package is published to PyPI as [`strands-cosmos`](https://pypi.org/project/strands-cosmos/). Pick an extra for your task:

| Extra | Pulls in | For |
| --- | --- | --- |
| *(none)* | transformers, torch, torchvision, torchcodec, av | Cosmos-Reason2 VLM + tools |
| `cosmos3` | `openai` | Cosmos 3 reasoner client (vLLM server) |
| `cosmos3-gen` | diffusers, cosmos\_guardrail, soundfile, imageio | Cosmos 3 generator (in-process) |
| `vllm` | vllm, openai | self-hosting vLLM |
| `all` | everything (heavy) | kitchen sink |

## Two runtime surfaces

Cosmos 3 is a unified Mixture-of-Transformers that jointly **understands and generates** text, images, video, audio, and action. Strands for Cosmos exposes both:

-   **Reasoner** (`Cosmos3ReasonerModel`) - text + vision to text, served by vLLM.
-   **Generator** (`Cosmos3GeneratorModel`) - to image / video / audio / action, running in-process via Diffusers (no server).

```python
from strands_cosmos import Cosmos3GeneratorModel

gen = Cosmos3GeneratorModel(model_id="nvidia/Cosmos3-Nano")
gen.generate(mode="text2video", prompt="A robot navigates a warehouse.", out_path="vid.mp4")
gen.generate(mode="text2video-with-sound", prompt="A robot pours water.", out_path="av.mp4", enable_sound=True)
gen.generate(mode="image2video", prompt="It moves forward.", image="frame.jpg", out_path="i2v.mp4")
```

## Lightweight edge VLM

For edge and Jetson deployments, the Cosmos-Reason2 VLM runs as a Strands model provider with a tiny footprint - verified on Jetson AGX Thor with Chain-of-Thought reasoning.

```python
from strands import Agent
from strands_cosmos import CosmosVisionModel

agent = Agent(model=CosmosVisionModel(model_id="nvidia/Cosmos-Reason2-2B"))
agent("Caption in detail: <video>dashcam.mp4</video>")
agent("<image>robot_view.jpg</image> What should the robot do next?")
```

## Links

-   [GitHub repository](https://github.com/strands-labs/strands-for-cosmos)
-   [PyPI package](https://pypi.org/project/strands-cosmos/)
-   [NVIDIA Cosmos](https://research.nvidia.com/labs/cosmos-lab/cosmos3/)
-   [Strands Robots](/docs/labs/robots/index.md) - physical robot control