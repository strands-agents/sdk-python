"""Stability AI model provider.

- Docs: https://platform.stability.ai/
"""

import base64
import logging
from enum import Enum
from typing import Any, Iterable, Optional, TypedDict, cast

from typing_extensions import NotRequired, Unpack, override

from strands.types.content import Messages
from strands.types.models import Model
from strands.types.streaming import ContentBlockDelta, ContentBlockDeltaEvent, StreamEvent
from strands.types.tools import ToolSpec

from ._stabilityaiclient import StabilityAiClient, StabilityAiError

logger = logging.getLogger(__name__)


class OutputFormat(Enum):
    """Supported output formats for image generation."""

    JPEG = "jpeg"
    PNG = "png"
    WEBP = "webp"


class StylePreset(Enum):
    """Supported style presets for image generation."""

    THREE_D_MODEL = "3d-model"
    ANALOG_FILM = "analog-film"
    ANIME = "anime"
    CINEMATIC = "cinematic"
    COMIC_BOOK = "comic-book"
    DIGITAL_ART = "digital-art"
    ENHANCE = "enhance"
    FANTASY_ART = "fantasy-art"
    ISOMETRIC = "isometric"
    LINE_ART = "line-art"
    LOW_POLY = "low-poly"
    MODELING_COMPOUND = "modeling-compound"
    NEON_PUNK = "neon-punk"
    ORIGAMI = "origami"
    PHOTOGRAPHIC = "photographic"
    PIXEL_ART = "pixel-art"
    TILE_TEXTURE = "tile-texture"


class StabilityAiImageModel(Model):
    """Your custom model provider implementation."""

    class StabilityAiImageModelConfig(TypedDict):
        """Configuration your model.

        Attributes:
            model_id: ID of Custom model (required).
            params: Model parameters (e.g., max_tokens).
        """

        """
        image - the image to use as the starting point for the generation
        strength - controls how much influence the image parameter has on the output image
        aspect_ratio - the aspect ratio of the output image
        seed - the randomness seed to use for the generation
        output_format - the format of the output image
        """
        # Required parameters
        model_id: str

        # Optional parameters with defaults
        aspect_ratio: NotRequired[str]  # defaults to "1:1"
        seed: NotRequired[int]  # defaults to random
        output_format: NotRequired[OutputFormat]  # defaults to PNG
        style_preset: NotRequired[StylePreset]  # defaults to PHOTOGRAPHIC
        image: NotRequired[str]  # defaults to None
        strength: NotRequired[float]  # defaults to 0.35

    def __init__(self, api_key: str, **model_config: Unpack[StabilityAiImageModelConfig]) -> None:
        """Initialize provider instance.

        Args:
            api_key: The API key for connecting to your Custom model.
            **model_config: Configuration options for Custom model.
        """
        # Set default values for optional parameters

        defaults = {
            "output_format": OutputFormat.PNG,
        }

        # Update defaults with provided config
        config_dict = {**defaults, **dict(model_config)}

        # Convert string output_format to enum if provided as string
        if "output_format" in config_dict and isinstance(config_dict["output_format"], str):
            try:
                config_dict["output_format"] = OutputFormat(config_dict["output_format"])
            except ValueError as e:
                raise ValueError(f"output_format must be one of: {[f.value for f in OutputFormat]}") from e

        # Convert string style_preset to enum if provided as string
        if "style_preset" in config_dict and isinstance(config_dict["style_preset"], str):
            try:
                config_dict["style_preset"] = StylePreset(config_dict["style_preset"])
            except ValueError as e:
                raise ValueError(f"style_preset must be one of: {[f.value for f in StylePreset]}") from e

        self.config = cast(StabilityAiImageModel.StabilityAiImageModelConfig, config_dict)
        logger.debug("config=<%s> | initializing", self.config)

        model_id = self.config.get("model_id")
        if model_id is None:
            raise ValueError("model_id is required")
        self.client = StabilityAiClient(api_key=api_key, model_id=model_id)

    @override
    def format_request(
        self, messages: Messages, tool_specs: Optional[list[ToolSpec]] = None, system_prompt: Optional[str] = None
    ) -> dict[str, Any]:
        """Format a Stability AI model request.

        Args:
            messages: List of messages containing the conversation history
            tool_specs: Optional list of tool specifications
            system_prompt: Optional system prompt

        Returns:
            Formatted request parameters for the Stability AI API
        """
        # Extract the last user message as the prompt
        # We do not need all the previous messages as context unlike an llm
        prompt = ""

        for message in reversed(messages):
            if message["role"] == "user":
                # Find the text content in the message
                for content in message["content"]:
                    if isinstance(content, dict) and "text" in content:
                        prompt = content["text"]
                        break
                break

        if not prompt:
            raise ValueError("No user message found in the conversation")

        # Format the request
        request = {
            "prompt": prompt,
            "aspect_ratio": self.config.get("aspect_ratio", "1:1"),
            "output_format": self.config.get("output_format", OutputFormat.PNG).value,
            "style_preset": self.config.get("style_preset", StylePreset.PHOTOGRAPHIC).value,
        }

        # Add optional parameters if they exist in config
        if "seed" in self.config:
            request["seed"] = self.config["seed"]  # type: ignore[assignment]
        if self.config.get("image") is not None:
            request["image"] = self.config["image"]
            request["strength"] = self.config.get("strength", 0.35)  # type: ignore[assignment]

        return request

    @override
    def update_config(self, **model_config: Unpack[StabilityAiImageModelConfig]) -> None:  # type: ignore[override]
        """Update the model configuration with the provided arguments.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> StabilityAiImageModelConfig:
        """Get the model configuration.

        Returns:
            The model configuration.
        """
        return self.config

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format an OpenAI response event into a standardized message chunk.

        Args:
            event: A response event from the OpenAI compatible model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
                This error should never be encountered as chunk_type is controlled in the stream method.
        """
        match event["chunk_type"]:
            case "message_start":
                return {"messageStart": {"role": "assistant"}}

            case "content_start":
                return {"contentBlockStart": {"start": {}}}

            case "content_block_delta":
                # Have to do this cast as there are two different ContentBlockDelta types
                # with different structures. The cast is for mypy to explicitly understand
                # the right one, otherwise it is getting confused.
                content_block_delta = cast(
                    ContentBlockDelta,
                    {
                        "image": {
                            "format": self.config["output_format"].value,
                            "source": {"bytes": base64.b64decode(event.get("data", b""))},
                        }
                    },
                )
                content_block_delta_event = ContentBlockDeltaEvent(delta=content_block_delta)
                return {"contentBlockDelta": content_block_delta_event}

            case "content_stop":
                return {"contentBlockStop": {}}
            case "message_stop":
                return {"messageStop": {"stopReason": event["data"]}}
            case _:
                raise RuntimeError(f"chunk_type=<{event['chunk_type']} | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[Any]:
        """Send the request to the Stability AI model and get a streaming response.

        Args:
            request: The formatted request to send to the Stability AI model.

        Returns:
            An iterable of response events from the Stability AI model.

        Raises:
            StabilityAiError: If the API request fails
        """
        yield {"chunk_type": "message_start"}
        yield {"chunk_type": "content_start", "data_type": "text"}
        try:
            # Generate the image #TODO add generate_image_bytes
            response_json = self.client.generate_image_json(**request)
            # Yield the image data as a single event

            yield {"chunk_type": "content_block_delta", "data_type": "image", "data": response_json.get("image")}
            yield {"chunk_type": "content_stop", "data_type": "text"}

            yield {"chunk_type": "message_stop", "data": response_json.get("finish_reason")}
        except StabilityAiError as e:
            logger.error("Failed to generate image: %s", str(e))
            raise
