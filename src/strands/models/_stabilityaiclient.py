import base64
from enum import Enum
from io import BytesIO
from typing import Any, BinaryIO, Dict, Optional, Union, cast

import requests
from PIL import Image

# Validation classes and functions


class Mode(Enum):
    TEXT_TO_IMAGE = "text-to-image"
    IMAGE_TO_IMAGE = "image-to-image"


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


def _validate_image_pixels_and_aspect_ratio(image: Union[str, BinaryIO]) -> None:
    """Validates the number of pixels in the 'image' field of the request.

    The image must have a total pixel count between 4,096 and 9,437,184 (inclusive).
    Not implemented yet (but required for stable image services):
    If the model is outpaint, the aspect ratio must be between 1:2.5 and 2.5:1.

    Args:
        image: Either a base64-encoded string or a BinaryIO object
    """
    # Get the raw image data
    if isinstance(image, str):
        # Decode base64 string
        try:
            image_data = base64.b64decode(image)
        except Exception as e:
            raise ValueError("Invalid base64 encoding for 'image'") from e
    else:
        # Read from BinaryIO
        image_data = image.read()
        image.seek(0)  # Reset the file pointer so it can be read again later

    # Attempt to open the image using Pillow
    try:
        with Image.open(BytesIO(image_data)) as img:
            width, height = img.size
    except Exception as e:
        raise ValueError("Unable to open or process the image data") from e

    # Check the image type based on magic bytes (JPEG, PNG, WebP)
    image_format = None
    if image_data.startswith(b"\xff\xd8\xff"):  # JPEG magic number
        image_format = "jpeg"
    elif image_data.startswith(b"\x89\x50\x4e\x47"):  # PNG magic number
        image_format = "png"
    elif image_data.startswith(b"\x52\x49\x46\x46") and image_data[8:12] == b"WEBP":  # WebP magic number
        image_format = "webp"

    if not image_format:
        raise ValueError("Unsupported image format. Only JPEG, PNG, or WebP are allowed.")

    total_pixels = width * height
    MIN_PIXELS = 4096
    MAX_PIXELS = 9437184

    if total_pixels < MIN_PIXELS or total_pixels > MAX_PIXELS:
        raise ValueError(
            f"Image total pixel count {total_pixels} is invalid. Image size (height x width) must be between "
            f"{MIN_PIXELS} and {MAX_PIXELS} pixels."
        )


class StabilityAiError(Exception):
    """Base exception for Stability AI API errors.

    Attributes:
        message: Error message
        status_code: HTTP status code (if applicable)
        response_data: Full response data from API (if available)
    """

    def __init__(self, message: str, status_code: Optional[int] = None, response_data: Optional[Dict[str, Any]] = None):
        """Initialize the exception.

        Args:
            message: Error message
            status_code: HTTP status code that caused the error
            response_data: Full response data from the API
        """
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.response_data = response_data or {}

    def __str__(self) -> str:
        """String representation of the error."""
        if self.status_code:
            return f"[HTTP {self.status_code}] {self.message}"
        return self.message


### Specific error classes for common API errors
class AuthenticationError(StabilityAiError):
    """Raised when authentication fails (401)."""

    def __init__(self, message: str = "Authentication failed", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=401, response_data=response_data)


class BadRequestError(StabilityAiError):
    """Raised when request parameters are invalid (400)."""

    def __init__(self, message: str = "Invalid request parameters", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=400, response_data=response_data)


class ContentModerationError(StabilityAiError):
    """Raised when content is flagged by moderation (403)."""

    def __init__(self, message: str = "Content flagged by moderation", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=403, response_data=response_data)


class PayloadTooLargeError(StabilityAiError):
    """Raised when request exceeds size limit (413)."""

    def __init__(self, message: str = "Request too large (max 10MiB)", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=413, response_data=response_data)


class ValidationError(StabilityAiError):
    """Raised when request validation fails (422)."""

    def __init__(self, message: str = "Request validation failed", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=422, response_data=response_data)


class RateLimitError(StabilityAiError):
    """Raised when rate limit is exceeded (429)."""

    def __init__(self, message: str = "Rate limit exceeded", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=429, response_data=response_data)


class InternalServerError(StabilityAiError):
    """Raised when server encounters an error (500)."""

    def __init__(self, message: str = "Internal server error", response_data: Optional[Dict[str, Any]] = None):
        super().__init__(message, status_code=500, response_data=response_data)


class NetworkError(StabilityAiError):
    """Raised when network request fails."""

    def __init__(self, message: str = "Network request failed", original_error: Optional[Exception] = None):
        super().__init__(message)
        self.original_error = original_error


class StabilityAiClient:
    """Client for interacting with the Stability AI API."""

    MODEL_ID_TO_BASE_URL = {
        "stability.stable-image-core-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/core",
        "stability.stable-image-ultra-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
        "stability.sd3-5-large-v1:0": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    }

    def __init__(self, api_key: str, client_id: Optional[str] = None, client_version: Optional[str] = None):
        """Initialize the Stability AI client.

        Args:
            api_key: Your Stability API key
            client_id: Optional client ID for debugging
            client_version: Optional client version for debugging
        """
        self.api_key = api_key
        self.client_id = client_id
        self.client_version = client_version

    def _get_headers(self, accept: str = "image/*") -> Dict[str, str]:
        """Get the headers for the API request.

        Args:
            accept: The accept header value (image/* or application/json)

        Returns:
            Dict of headers
        """
        headers = {"Authorization": f"Bearer {self.api_key}", "Accept": accept}

        if self.client_id:
            headers["stability-client-id"] = self.client_id
        if self.client_version:
            headers["stability-client-version"] = self.client_version

        return headers

    def generate_image_bytes(self, model_id: str, **kwargs: Any) -> requests.Response:
        """Generate an image using the Stability AI API.

        Args:
            model_id: The model ID to use for the API request.
            **kwargs: See _generate_image for available parameters

        Returns:
            requests.Response object contining image as bytes
        """
        kwargs["return_json"] = False
        return cast(requests.Response, self._generate_image(model_id, **kwargs))

    def generate_image_json(self, model_id: str, **kwargs: Any) -> Dict[str, Any]:
        """Generate an image using the Stability AI API.

        Args:
            model_id: The model ID to use for the API request.
            **kwargs: See _generate_image for available parameters

        Returns:
            JSON response with base64 image
        """
        kwargs["return_json"] = True
        return cast(Dict[str, Any], self._generate_image(model_id, **kwargs))

    def _generate_image(
        self,
        model_id: str,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "1:1",
        cfg_scale: Optional[int] = 4,
        seed: Optional[int] = None,
        output_format: Union[OutputFormat, str] = "png",
        image: Optional[BinaryIO] = None,
        mode: Union[Mode] = Mode.TEXT_TO_IMAGE,
        style_preset: Optional[str] = None,
        strength: Optional[float] = 0.35,
        return_json: bool = False,
        **extra_kwargs: Any,
    ) -> Union[requests.Response, Dict[str, Any]]:
        """Generate an image using the Stability AI API.

        Args:
            model_id: The model ID to use for the API request
            prompt: Text prompt for image generation
            negative_prompt: Optional text describing what not to include
            aspect_ratio: Aspect ratio of the output image
            cfg_scale: Optional classifier-free guidance scale, only used for stability.sd3-5-large-v1:0
            seed: Random seed for generation
            output_format: Output format (jpeg, png, webp)
            image: Optional input image for img2img
            mode: "text-to-image" or "image-to-image"
            style_preset: Optional style preset
            strength: Required when image is provided, controls influence of input image
            return_json: If True, returns JSON response with base64 image
            **extra_kwargs: Additional keyword arguments

        Returns:
            Either image bytes or JSON response with base64 image

        Raises:
            StabilityAiError: If the API request fails
        """
        # Validate and prepare the base URL
        if model_id not in self.MODEL_ID_TO_BASE_URL:
            raise ValueError(f"Invalid model_id: {model_id}. Must be one of: {list(self.MODEL_ID_TO_BASE_URL.keys())}")
        base_url = self.MODEL_ID_TO_BASE_URL[model_id]

        if isinstance(output_format, str):
            try:
                output_format = OutputFormat(output_format)
            except ValueError as e:
                raise ValueError(
                    f"Invalid output_format: {output_format}. Must be one of: {[e.value for e in OutputFormat]}"
                ) from e

        if isinstance(mode, str):
            try:
                mode = Mode(mode)
            except ValueError as e:
                raise ValueError(f"Invalid mode: {mode}. Must be one of: {[e.value for e in Mode]}") from e

        # Prepare the multipart form data
        files: Dict[str, Union[BinaryIO, str]] = {}
        data: Dict[str, Any] = {}

        # Add all parameters to data as strings
        data["prompt"] = prompt
        if negative_prompt:
            data["negative_prompt"] = negative_prompt
        if aspect_ratio:
            data["aspect_ratio"] = aspect_ratio
        if seed is not None:
            data["seed"] = seed
        if output_format:
            data["output_format"] = output_format.value
        if style_preset:
            data["style_preset"] = style_preset

        # Handle input image if provided
        if image:
            _validate_image_pixels_and_aspect_ratio(image)
            files["image"] = image

        if len(files) == 0:
            files["none"] = ""
        try:
            # Make the API request
            response = requests.post(
                base_url,
                headers=self._get_headers("application/json" if return_json else "image/*"),
                data=data,
                files=files,
            )

            # Handle successful response
            if response.status_code == 200:
                if return_json:
                    return cast(Dict[str, Any], response.json())
                # If return_json is False, return the full requests.Response object
                # This is because data like the seed and finish_reason are not included in the image bytes response
                # but need to be retreived from the headers
                else:
                    return cast(requests.Response, response)
                    # return response

            # Parse error response
            try:
                error_data = response.json()
                error_message = error_data.get("errors", error_data.get("message", "Unknown error"))
            except ValueError:
                error_data = {}
                error_message = response.text or "Unknown error"

            # Handle specific error cases
            if response.status_code == 401:
                raise AuthenticationError(
                    f"Unauthorized: check authentication credentials: {error_message}", response_data=error_data
                )
            elif response.status_code == 400:
                raise BadRequestError(f"Invalid parameters: {error_message}", response_data=error_data)
            elif response.status_code == 403:
                raise ContentModerationError("Request flagged by content moderation", response_data=error_data)
            elif response.status_code == 413:
                raise PayloadTooLargeError("Request too large (max 10MiB)", response_data=error_data)
            elif response.status_code == 422:
                raise ValidationError(f"Request rejected: {error_message}", response_data=error_data)
            elif response.status_code == 429:
                raise RateLimitError("Rate limit exceeded (max 150 requests per 10 seconds)", response_data=error_data)
            elif response.status_code == 500:
                raise InternalServerError("Internal server error", response_data=error_data)
            else:
                raise StabilityAiError(
                    f"Unexpected error: {error_message}", status_code=response.status_code, response_data=error_data
                )

        except requests.exceptions.RequestException as e:
            raise NetworkError(f"Request failed: {str(e)}", original_error=e) from e
