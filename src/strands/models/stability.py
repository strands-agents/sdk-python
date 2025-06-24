"""Stability AI model provider.

- Docs: https://platform.stability.ai/
"""

import base64
import logging
from typing import Any, Callable, Iterable, Optional, Type, TypedDict, TypeVar, cast

from pydantic import BaseModel
from typing_extensions import NotRequired, Unpack, override

from strands.types.content import Messages
from strands.types.exceptions import (
    ContentModerationException,
    EventLoopException,
    ModelAuthenticationException,
    ModelServiceException,
    ModelThrottledException,
    ModelValidationException,
)
from strands.types.models import Model
from strands.types.streaming import ContentBlockDelta, ContentBlockDeltaEvent, StreamEvent
from strands.types.tools import ToolSpec

from ._stabilityaiclient import (
    AuthenticationError,
    BadRequestError,
    ContentModerationError,
    InternalServerError,
    Mode,
    NetworkError,
    OutputFormat,
    PayloadTooLargeError,
    RateLimitError,
    StabilityAiClient,
    StabilityAiError,
    StylePreset,
    ValidationError,
)

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)


class Defaults:
    """Default values for Stability AI configuration."""

    ASPECT_RATIO = "1:1"
    OUTPUT_FORMAT = OutputFormat.PNG
    STYLE_PRESET = StylePreset.PHOTOGRAPHIC
    STRENGTH = 0.35
    MODE = Mode.TEXT_TO_IMAGE


class ChunkTypes:
    """Chunk type constants."""

    MESSAGE_START = "message_start"
    CONTENT_START = "content_start"
    CONTENT_BLOCK_DELTA = "content_block_delta"
    CONTENT_STOP = "content_stop"
    MESSAGE_STOP = "message_stop"


class StabilityAiImageModel(Model):
    """Stability AI image generation model provider."""

    class StabilityAiImageModelConfig(TypedDict):
        """Configuration for Stability AI image model.

        Attributes:
            model_id: ID of the Stability AI model (required).
            aspect_ratio: Aspect ratio of the output image.
            cfg_scale: CFG scale for image generation (only used for stability.sd3-5-large-v1:0).
            seed: Random seed for generation.
            output_format: Output format (jpeg, png, webp).
            style_preset: Style preset for image generation.
            image: Input image for img2img generation.
            mode: Mode of operation (text-to-image, image-to-image).
            return_json: Return JSON response with base64-encoded image data. Defaults to False.
            strength: Influence of input image on output (0.0-1.0).
        """

        # Required parameters
        model_id: str

        # Optional parameters with defaults
        aspect_ratio: NotRequired[str]  # defaults to "1:1"
        cfg_scale: NotRequired[int]  # defaults to 4. Only used for stability.sd3-5-large-v1:0
        seed: NotRequired[int]  # defaults to random
        output_format: NotRequired[OutputFormat]  # defaults to PNG
        style_preset: NotRequired[StylePreset]  # defaults to PHOTOGRAPHIC
        image: NotRequired[str]  # defaults to None
        mode: NotRequired[Mode]  # defaults to "text-to-image"
        return_json: NotRequired[bool]  # defaults to False
        strength: NotRequired[float]  # defaults to 0.35

    def __init__(self, api_key: str, **model_config: Unpack[StabilityAiImageModelConfig]) -> None:
        """Initialize the Stability AI model provider.

        Args:
            api_key: The API key for connecting to Stability AI.
            **model_config: Configuration options for the model.
        """
        # pop the return_json to avoid it being passed into the request
        self.return_json = model_config.pop("return_json", False)

        config_dict = {**{"output_format": Defaults.OUTPUT_FORMAT}, **dict(model_config)}
        self._validate_and_convert_config(config_dict)

        self.config = cast(StabilityAiImageModel.StabilityAiImageModelConfig, config_dict)
        logger.debug("config=<%s> | initializing", self.config)

        model_id = self.config.get("model_id")
        if model_id is None:
            raise ValueError("model_id is required")
        self.client = StabilityAiClient(api_key=api_key)

    def _convert_output_format(self, config_dict: dict[str, Any]) -> None:
        """Convert string output_format to enum if needed."""
        if "output_format" in config_dict and isinstance(config_dict["output_format"], str):
            try:
                config_dict["output_format"] = OutputFormat(config_dict["output_format"])
            except ValueError as e:
                valid_formats = [f.value for f in OutputFormat]
                raise ValueError(f"output_format must be one of: {valid_formats}") from e

    def _convert_style_preset(self, config_dict: dict[str, Any]) -> None:
        """Convert string style_preset to enum if needed."""
        if "style_preset" in config_dict and isinstance(config_dict["style_preset"], str):
            try:
                config_dict["style_preset"] = StylePreset(config_dict["style_preset"])
            except ValueError as e:
                valid_presets = [p.value for p in StylePreset]
                raise ValueError(f"style_preset must be one of: {valid_presets}") from e

        self.config = cast(StabilityAiImageModel.StabilityAiImageModelConfig, config_dict)
        logger.debug("config=<%s> | initializing", self.config)

    def _validate_and_convert_config(self, config_dict: dict[str, Any]) -> None:
        """Validate and convert configuration values to proper types."""
        # Validate required fields first
        if "model_id" not in config_dict:
            raise ValueError("model_id is required in configuration")

        # Validate model_id is one of the supported models
        valid_model_ids = [
            "stability.stable-image-core-v1:1",
            "stability.stable-image-ultra-v1:1",
            "stability.sd3-5-large-v1:0",
        ]
        if config_dict["model_id"] not in valid_model_ids:
            raise ValueError(f"Invalid model_id: {config_dict['model_id']}. Must be one of: {valid_model_ids}")

        if config_dict["model_id"] == "stability.sd3-5-large-v1:0":
            allowed_formats = [OutputFormat.PNG, OutputFormat.JPEG]  # Compare enum to enum
            if config_dict["output_format"] not in allowed_formats:
                raise ValueError(
                    f"'output_format' must be one of {[f.value for f in allowed_formats]} "
                    f"for {config_dict['model_id']}. Got '{config_dict['output_format']}'."
                )

        # Warn if cfg_scale is used with non-SD3.5 models
        if "cfg_scale" in config_dict and config_dict["model_id"] != "stability.sd3-5-large-v1:0":
            logger.warning(
                "cfg_scale is only supported for stability.sd3-5-large-v1:0. It will be ignored for model %s",
                config_dict["model_id"],
            )
        # Convert other fields
        self._convert_output_format(config_dict)
        self._convert_style_preset(config_dict)

    def _extract_prompt_from_messages(self, messages: Messages) -> str:
        """Extract the last user message as prompt.

        Args:
            messages: List of conversation messages.

        Returns:
            The extracted prompt text.

        Raises:
            ValueError: If no user message with text content is found.
        """
        for message in reversed(messages):
            if message["role"] == "user":
                for content in message["content"]:
                    if isinstance(content, dict) and "text" in content:
                        return content["text"]
        raise ValueError("No user message found in the conversation")

    def _build_base_request(self, prompt: str) -> dict[str, Any]:
        """Build the base request with required parameters.

        Args:
            prompt: The text prompt for image generation.

        Returns:
            Dictionary with base request parameters.
        """
        return {
            "prompt": prompt,
            "aspect_ratio": self.config.get("aspect_ratio", Defaults.ASPECT_RATIO),
            "output_format": self.config.get("output_format", Defaults.OUTPUT_FORMAT).value,
            "style_preset": self.config.get("style_preset", Defaults.STYLE_PRESET).value,
            "mode": self.config.get("mode", Defaults.MODE).value,
        }

    def _add_optional_parameters(self, request: dict[str, Any]) -> None:
        """Add optional parameters to the request if they exist in config.

        Args:
            request: The request dictionary to modify.
        """
        # Only add cfg_scale for SD3.5 model
        if "cfg_scale" in self.config and self.config["model_id"] == "stability.sd3-5-large-v1:0":
            request["cfg_scale"] = self.config["cfg_scale"]

        if "seed" in self.config:
            request["seed"] = self.config["seed"]
        if self.config.get("image") is not None:
            request["image"] = self.config["image"]
            request["strength"] = self.config.get("strength", Defaults.STRENGTH)

    @override
    def structured_output(
        self, output_model: Type[T], prompt: Messages, callback_handler: Optional[Callable] = None
    ) -> T:
        """Get structured output from the model.

        Args:
            output_model(Type[BaseModel]): The output model to use for the agent.
            prompt(Messages): The prompt to use for the agent.
            callback_handler(Optional[Callable]): Optional callback handler for processing events. Defaults to None.
        """
        return output_model()

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
            Formatted request parameters for the Stability AI API.
        """
        prompt = self._extract_prompt_from_messages(messages)
        request = self._build_base_request(prompt)
        self._add_optional_parameters(request)

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

    def _format_message_start(self) -> StreamEvent:
        """Format message start event."""
        return {"messageStart": {"role": "assistant"}}

    def _format_content_start(self) -> StreamEvent:
        """Format content start event."""
        return {"contentBlockStart": {"start": {}}}

    def _format_content_block_delta(self, event: dict[str, Any]) -> StreamEvent:
        """Format content block delta event.

        Args:
            event: The event containing image data.

        Returns:
            Formatted content block delta event.
        """
        data = event.get("data", b"")

        # Handle both bytes and base64 string
        if isinstance(data, bytes):
            image_bytes = data
        else:
            # Assume it's a base64 string
            image_bytes = base64.b64decode(data)

        content_block_delta = cast(
            ContentBlockDelta,
            {
                "image": {
                    "format": self.config["output_format"].value,
                    "source": {"bytes": image_bytes},
                }
            },
        )
        content_block_delta_event = ContentBlockDeltaEvent(delta=content_block_delta)
        return {"contentBlockDelta": content_block_delta_event}

    def _format_content_stop(self) -> StreamEvent:
        """Format content stop event."""
        return {"contentBlockStop": {}}

    def _format_message_stop(self, event: dict[str, Any]) -> StreamEvent:
        """Format message stop event.

        Args:
            event: The event containing stop reason.

        Returns:
            Formatted message stop event.
        """
        return {"messageStop": {"stopReason": event["data"]}}

    @override
    def format_chunk(self, event: dict[str, Any]) -> StreamEvent:
        """Format an event into a standardized message chunk.

        Args:
            event: A response event from the Stability AI model.

        Returns:
            The formatted chunk.

        Raises:
            RuntimeError: If chunk_type is not recognized.
        """
        chunk_type = event["chunk_type"]

        match chunk_type:
            case ChunkTypes.MESSAGE_START:
                return self._format_message_start()
            case ChunkTypes.CONTENT_START:
                return self._format_content_start()
            case ChunkTypes.CONTENT_BLOCK_DELTA:
                return self._format_content_block_delta(event)
            case ChunkTypes.CONTENT_STOP:
                return self._format_content_stop()
            case ChunkTypes.MESSAGE_STOP:
                return self._format_message_stop(event)
            case _:
                raise RuntimeError(f"chunk_type=<{chunk_type}> | unknown type")

    @override
    def stream(self, request: dict[str, Any]) -> Iterable[Any]:
        """Send the request to the Stability AI model and get a streaming response.

        Args:
            request: The formatted request to send to the Stability AI model.

        Returns:
            An iterable of response events from the Stability AI model.

        Raises:
            ModelAuthenticationException: If authentication fails
            ModelValidationException: If request validation fails
            ContentModerationException: If content is flagged
            ModelThrottledException: If rate limit is exceeded
            ModelServiceException: If server error occurs
            EventLoopException: If network error occurs
        """
        yield {"chunk_type": ChunkTypes.MESSAGE_START}
        yield {"chunk_type": ChunkTypes.CONTENT_START, "data_type": "text"}

        model_id = self.config["model_id"]

        try:
            if self.return_json:
                json_response = self.client.generate_image_json(model_id, **request)
                image_data = json_response.get("image")  # base64 string
                finish_reason = json_response.get("finish_reason", "SUCCESS")
            else:
                bytes_response = self.client.generate_image_bytes(model_id, **request)  # requests.Response object
                image_data = bytes_response.content  # raw bytes from response body
                finish_reason = bytes_response.headers.get("finish-reason", "SUCCESS")  # from HTTP header

            # Yield the image data as a single event
            yield {
                "chunk_type": ChunkTypes.CONTENT_BLOCK_DELTA,
                "data_type": "image",
                "data": image_data,  # Either bytes or base64 string
            }
            yield {"chunk_type": ChunkTypes.CONTENT_STOP, "data_type": "text"}
            yield {"chunk_type": ChunkTypes.MESSAGE_STOP, "data": finish_reason}

        except AuthenticationError as e:
            logger.error("Authentication failed: %s", str(e))
            raise ModelAuthenticationException(str(e)) from e

        except RateLimitError as e:
            logger.warning("Rate limit exceeded: %s", str(e))
            raise ModelThrottledException(str(e)) from e

        except ContentModerationError as e:
            logger.warning("Content flagged by moderation: %s", str(e))
            raise ContentModerationException(str(e)) from e

        except (ValidationError, BadRequestError, PayloadTooLargeError) as e:
            logger.error("Request validation failed: %s", str(e))
            raise ModelValidationException(str(e)) from e

        except InternalServerError as e:
            logger.error("Server error during image generation: %s", str(e))
            raise ModelServiceException(str(e), is_transient=True) from e

        except NetworkError as e:
            logger.error("Network error during image generation: %s", str(e))
            raise EventLoopException(e.original_error or e, request_state=request) from e

        except StabilityAiError as e:
            # Catch any other StabilityAiError subclasses
            logger.error("Unexpected error during image generation: %s", str(e))
            raise ModelServiceException(str(e), is_transient=False) from e
