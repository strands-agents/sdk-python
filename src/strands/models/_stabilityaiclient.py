from typing import Any, BinaryIO, Dict, Optional, Union, cast

import requests


class StabilityAiError(Exception):
    """Base exception for Stability AI API errors."""

    pass


class StabilityAiClient:
    """Client for interacting with the Stability AI API."""

    MODEL_ID_TO_BASE_URL = {
        "stability.stable-image-core-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/core",
        "stability.stable-image-ultra-v1:1": "https://api.stability.ai/v2beta/stable-image/generate/ultra",
        "stability.sd3-5-large-v1:0": "https://api.stability.ai/v2beta/stable-image/generate/sd3",
    }

    def __init__(
        self, api_key: str, model_id: str, client_id: Optional[str] = None, client_version: Optional[str] = None
    ):
        """Initialize the Stability AI client.

        Args:
            api_key: Your Stability API key
            model_id: The model ID to use for the API request.See MODEL_ID_TO_BASE_URL for available models
            client_id: Optional client ID for debugging
            client_version: Optional client version for debugging
        """
        self.model_id = model_id
        self.base_url = self.MODEL_ID_TO_BASE_URL[model_id]
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

    def generate_image_bytes(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
        output_format: str = "png",
        image: Optional[BinaryIO] = None,
        style_preset: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> bytes:
        """Generate an image using the Stability AI API.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Optional text describing what not to include
            aspect_ratio: Aspect ratio of the output image
            seed: Random seed for generation
            output_format: Output format (jpeg, png, webp)
            image: Optional input image for img2img
            style_preset: Optional style preset
            strength: Required when image is provided, controls influence of input image

        Returns:  bytes of the image
        """
        return cast(
            bytes,
            self._generate_image(
                prompt,
                negative_prompt,
                aspect_ratio,
                seed,
                output_format,
                image,
                style_preset,
                strength,
                return_json=False,
            ),
        )

    def generate_image_json(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
        output_format: str = "png",
        image: Optional[BinaryIO] = None,
        style_preset: Optional[str] = None,
        strength: Optional[float] = None,
    ) -> Dict[str, Any]:
        """Generate an image using the Stability AI API.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Optional text describing what not to include
            aspect_ratio: Aspect ratio of the output image
            seed: Random seed for generation
            output_format: Output format (jpeg, png, webp)
            image: Optional input image for img2img
            style_preset: Optional style preset
            strength: Required when image is provided, controls influence of input image
            return_json: If True, returns JSON response with base64 image

        Returns:
            Either image bytes or JSON response with base64 image
        """
        return cast(
            Dict[str, Any],
            self._generate_image(
                prompt,
                negative_prompt,
                aspect_ratio,
                seed,
                output_format,
                image,
                style_preset,
                strength,
                return_json=True,
            ),
        )

    def _generate_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        aspect_ratio: str = "1:1",
        seed: Optional[int] = None,
        output_format: str = "png",
        image: Optional[BinaryIO] = None,
        style_preset: Optional[str] = None,
        strength: Optional[float] = None,
        return_json: bool = False,
    ) -> Union[bytes, Dict[str, Any]]:
        """Generate an image using the Stability AI API.

        Args:
            prompt: Text prompt for image generation
            negative_prompt: Optional text describing what not to include
            aspect_ratio: Aspect ratio of the output image
            seed: Random seed for generation
            output_format: Output format (jpeg, png, webp)
            image: Optional input image for img2img
            style_preset: Optional style preset
            strength: Required when image is provided, controls influence of input image
            return_json: If True, returns JSON response with base64 image

        Returns:
            Either image bytes or JSON response with base64 image

        Raises:
            StabilityAiError: If the API request fails
        """
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
            data["output_format"] = output_format
        if style_preset:
            data["style_preset"] = style_preset

        # Handle input image if provided
        if image:
            files["image"] = image

        if len(files) == 0:
            files["none"] = ""
        try:
            # Make the API request
            response = requests.post(
                self.base_url,
                headers=self._get_headers("application/json" if return_json else "image/*"),
                data=data,
                files=files,
            )

            # Handle different response status codes
            if response.status_code == 200:
                if return_json:
                    return cast(Dict[str, Any], response.json())
                return cast(bytes, response.content)
            elif response.status_code == 400:
                raise StabilityAiError(f"Invalid parameters: {response.json().get('errors', 'Unknown error')}")
            elif response.status_code == 403:
                raise StabilityAiError("Request flagged by content moderation")
            elif response.status_code == 413:
                raise StabilityAiError("Request too large (max 10MiB)")
            elif response.status_code == 422:
                raise StabilityAiError(f"Request rejected: {response.json().get('errors', 'Unknown error')}")
            elif response.status_code == 429:
                raise StabilityAiError("Rate limit exceeded (max 150 requests per 10 seconds)")
            elif response.status_code == 500:
                raise StabilityAiError("Internal server error")
            else:
                raise StabilityAiError(f"Unexpected error: {response.status_code}")

        except requests.exceptions.RequestException as e:
            raise StabilityAiError(f"Request failed: {str(e)}") from e
