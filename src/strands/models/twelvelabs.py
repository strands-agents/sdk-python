"""TwelveLabs model providers for search and video understanding.

This module provides two TwelveLabs model implementations:
- TwelveLabsSearchModel: Semantic video search using Marengo model
- TwelveLabsPegasusModel: Video understanding and Q&A using Pegasus model

- Docs: https://docs.twelvelabs.io/
"""

import hashlib
import logging
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict, Union, cast

from twelvelabs import TwelveLabs
from typing_extensions import Unpack, override

from ..types.content import Messages
from ..types.exceptions import ModelThrottledException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

logger = logging.getLogger(__name__)

DEFAULT_SEARCH_MODEL = "Marengo-retrieval-2.7"
DEFAULT_PEGASUS_MODEL = "pegasus1.2"

# Error codes for TwelveLabs API
TWELVELABS_THROTTLING_MESSAGES = ["usage_limit_exceeded", "rate_limit_exceeded"]

TWELVELABS_AUTHENTICATION_MESSAGES = ["api_key_invalid", "unauthorized"]

TWELVELABS_SEARCH_MESSAGES = [
    "search_option_not_supported",
    "search_option_combination_not_supported",
    "search_filter_invalid",
    "search_page_token_expired",
    "index_not_supported_for_search",
]

TWELVELABS_PROCESSING_MESSAGES = [
    "video_processing_failed",
    "index_not_ready",
    "video_not_found",
    "task_failed",
]


class TwelveLabsSearchModel(Model):
    """TwelveLabs model provider implementation for video search.

    This provider enables semantic video search using TwelveLabs' Marengo model
    through natural language queries, without requiring direct embedding management.
    """

    class TwelveLabsSearchConfig(TypedDict, total=False):
        """Configuration options for TwelveLabs search models.

        Attributes:
            api_key: TwelveLabs API key (falls back to TWELVELABS_API_KEY env var)
            model_id: TwelveLabs model ID (default: "Marengo-retrieval-2.7")
            index_id: Index ID for searches
            search_options: Default search modalities ["visual", "audio"]
            group_by: Default grouping for results ("video" or "clip")
            threshold: Default confidence threshold ("high", "medium", "low", "none")
            page_limit: Default maximum results per page
        """

        api_key: Optional[str]
        model_id: str
        index_id: Optional[str]
        search_options: Optional[List[Literal["visual", "audio"]]]
        group_by: Optional[Literal["video", "clip"]]
        threshold: Optional[Literal["high", "medium", "low", "none"]]
        page_limit: Optional[int]

    def __init__(
        self,
        **model_config: Unpack[TwelveLabsSearchConfig],
    ):
        """Initialize TwelveLabs search model provider.

        Args:
            **model_config: Configuration options for the TwelveLabs model.
        """
        self.config = TwelveLabsSearchModel.TwelveLabsSearchConfig(
            model_id=DEFAULT_SEARCH_MODEL,
            search_options=["visual"],
            group_by="clip",
            page_limit=10,
        )
        self.update_config(**model_config)

        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TwelveLabs API key required. Set TWELVELABS_API_KEY environment variable "
                "or pass api_key in model config"
            )

        self._api_key = api_key
        logger.debug("TwelveLabs search model provider initialized with model_id=%s", self.config["model_id"])

    @override
    def update_config(self, **model_config: Any) -> None:
        """Update the TwelveLabs search model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        # Filter to only valid TwelveLabsSearchConfig keys to avoid TypedDict type error
        valid_keys = {"api_key", "model_id", "index_id", "search_options", "group_by", "threshold", "page_limit"}
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}
        self.config.update(filtered_config)  # type: ignore[typeddict-item]

    @override
    def get_config(self) -> TwelveLabsSearchConfig:
        """Get the current TwelveLabs search model configuration.

        Returns:
            The TwelveLabs search model configuration.
        """
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a TwelveLabs search request from conversation messages.

        This method analyzes the conversation to extract search parameters
        and formats them for the TwelveLabs Search API.

        Args:
            messages: List of message objects containing search queries
            tool_specs: Tool specifications (unused for TwelveLabs)
            system_prompt: System prompt (unused for TwelveLabs)

        Returns:
            A formatted search request for TwelveLabs API
        """
        # Extract the latest user message as the search query
        search_query = ""

        for message in reversed(messages):
            if message.get("role") == "user":
                # Extract text content from the message
                content: Union[str, List[Any]] = message.get("content", [])

                # Handle both string content (legacy/test format) and list content (standard format)
                if isinstance(content, str):
                    # Direct string content
                    search_query = content
                    break
                elif isinstance(content, list):
                    # Standard List[ContentBlock] format
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            search_query = block.get("text", "")
                            break
                    if search_query:
                        break

        index_id = self.config.get("index_id")
        if not index_id:
            raise ValueError("index_id must be configured in TwelveLabsSearchConfig")

        formatted_request = {
            "query_text": search_query,
            "index_id": index_id,
            "search_options": self.config.get("search_options", ["visual", "audio"]),
            "group_by": self.config.get("group_by", "clip"),
            "threshold": self.config.get("threshold"),
            "page_limit": self.config.get("page_limit", 10),
        }

        query_preview = search_query[:50] + ("..." if len(search_query) > 50 else "")
        logger.info("Formatted TwelveLabs search request: query_text='%s', index_id='%s'", query_preview, index_id)

        return formatted_request

    @override
    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format TwelveLabs search results into streaming format.

        Args:
            event: Search result from TwelveLabs API

        Returns:
            Formatted streaming event
        """
        return cast(StreamEvent, event)

    @override
    def stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        """Perform TwelveLabs search and stream results.

        Args:
            request: Formatted search request

        Yields:
            Streaming events containing search results

        Raises:
            ModelThrottledException: If the TwelveLabs API is throttling requests
            ValueError: If required parameters are missing or invalid
        """
        query_text = request.get("query_text")
        index_id = request.get("index_id")

        query_preview = (query_text[:50] if query_text else "") + ("..." if query_text and len(query_text) > 50 else "")
        logger.info("Starting TwelveLabs search: query='%s', index='%s'", query_preview, index_id)

        if not query_text:
            logger.warning("No search query provided")
            yield from self._error_response("No search query provided in the conversation.")
            return

        if not index_id:
            logger.warning("No index ID provided")
            yield from self._error_response(
                "No index ID specified. Please provide an index ID or set a default in the model config."
            )
            return

        try:
            with TwelveLabs(self._api_key) as client:
                # Perform the search
                result = client.search.query(
                    index_id=index_id,
                    options=request.get("search_options", ["visual", "audio"]),
                    query_text=query_text,
                    group_by=request.get("group_by"),
                    threshold=request.get("threshold"),
                    page_limit=request.get("page_limit"),
                )

                # Start streaming response
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"start": {}}}

                # Format and stream results
                total_results = getattr(result.pool, "total_count", 0) if hasattr(result, "pool") else 0
                logger.info("TwelveLabs search completed: %d results", total_results)

                response_text = "Video Search Results\n\n"
                response_text += f'Query: "{query_text}"\n'
                response_text += f"Index: {index_id}\n"
                response_text += f"Found {total_results} total results\n\n"

                # Process search results
                search_results = []
                for item in result.data:
                    if request.get("group_by") == "video" and hasattr(item, "clips"):
                        # Video-grouped results
                        video_result = {"video_id": item.id, "clips": []}
                        for clip in item.clips:
                            video_result["clips"].append(
                                {
                                    "score": clip.score,
                                    "start": clip.start,
                                    "end": clip.end,
                                    "confidence": clip.confidence,
                                    "video_id": clip.video_id,
                                }
                            )
                        search_results.append(video_result)
                    else:
                        # Clip-level results
                        search_results.append(
                            {
                                "score": item.score,
                                "start": item.start,
                                "end": item.end,
                                "confidence": item.confidence,
                                "video_id": item.video_id,
                            }
                        )

                # Format detailed results
                if search_results:
                    response_text += "Top Results:\n"
                    for i, result_item in enumerate(search_results[:5], 1):
                        if request.get("group_by") == "video":
                            response_text += f"\n{i}. Video ID: {result_item['video_id']}\n"
                            response_text += f"   Found {len(result_item['clips'])} clips\n"
                            for j, clip in enumerate(result_item["clips"][:3], 1):
                                response_text += (
                                    f"   {j}. Score: {clip['score']:.3f} | "
                                    f"{clip['start']:.1f}s-{clip['end']:.1f}s | {clip['confidence']}\n"
                                )
                        else:
                            response_text += f"\n{i}. Video: {result_item['video_id']}\n"
                            response_text += f"   Score: {result_item['score']:.3f}\n"
                            response_text += f"   Time: {result_item['start']:.1f}s - {result_item['end']:.1f}s\n"
                            response_text += f"   Confidence: {result_item['confidence']}\n"
                else:
                    response_text += "No results found. Try adjusting your query or lowering the confidence threshold."

                # Stream the response text
                yield {"contentBlockDelta": {"delta": {"text": response_text}}}

                # End the response
                yield {"contentBlockStop": {}}
                yield {
                    "messageStop": {
                        "stopReason": "end_turn",
                        "additionalModelResponseFields": {
                            "search_metadata": {
                                "total_results": total_results,
                                "returned_results": len(search_results),
                                "query": query_text,
                                "index_id": index_id,
                                "model_id": self.config["model_id"],
                            }
                        },
                    }
                }

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.exception("TwelveLabs search error: %s", error_msg)

            # Check for throttling/rate limiting
            if any(throttle_msg in error_msg_lower for throttle_msg in TWELVELABS_THROTTLING_MESSAGES):
                raise ModelThrottledException(f"TwelveLabs API throttling: {error_msg}") from e

            # Check for authentication errors
            if any(auth_msg in error_msg_lower for auth_msg in TWELVELABS_AUTHENTICATION_MESSAGES):
                logger.warning("TwelveLabs authentication error")
                yield from self._error_response(f"Authentication failed: {error_msg}")
                return

            # Check for search-specific errors
            if any(search_msg in error_msg_lower for search_msg in TWELVELABS_SEARCH_MESSAGES):
                logger.warning("TwelveLabs search error")
                yield from self._error_response(f"Search error: {error_msg}")
                return

            # For other errors, stream error response
            yield from self._error_response(f"Search failed: {error_msg}")

    def search_videos(self, query_text: str, index_id: Optional[str] = None, **search_params: Any) -> Dict[str, Any]:
        """Direct video search method (non-streaming).

        Args:
            query_text: Natural language search query
            index_id: Index ID (uses default if not provided)
            **search_params: Additional search parameters

        Returns:
            Dictionary containing search results
        """
        index_id = index_id or self.config.get("index_id")

        if not index_id:
            raise ValueError("index_id required - provide directly or set index_id in config")

        try:
            with TwelveLabs(self._api_key) as client:
                result = client.search.query(
                    index_id=index_id,
                    options=search_params.get("search_options", self.config.get("search_options")),
                    query_text=query_text,
                    group_by=search_params.get("group_by", self.config.get("group_by")),
                    threshold=search_params.get("threshold", self.config.get("threshold")),
                    page_limit=search_params.get("page_limit", self.config.get("page_limit")),
                    **{
                        k: v
                        for k, v in search_params.items()
                        if k not in ["search_options", "group_by", "threshold", "page_limit"]
                    },
                )

                # Format results
                search_results = []
                total_results = getattr(result.pool, "total_count", 0) if hasattr(result, "pool") else 0

                for item in result.data:
                    if search_params.get("group_by") == "video" and hasattr(item, "clips"):
                        video_result = {
                            "video_id": item.id,
                            "clips": [
                                {
                                    "score": clip.score,
                                    "start": clip.start,
                                    "end": clip.end,
                                    "confidence": clip.confidence,
                                    "video_id": clip.video_id,
                                }
                                for clip in item.clips
                            ],
                        }
                        search_results.append(video_result)
                    else:
                        search_results.append(
                            {
                                "score": item.score,
                                "start": item.start,
                                "end": item.end,
                                "confidence": item.confidence,
                                "video_id": item.video_id,
                            }
                        )

                return {
                    "results": search_results,
                    "total_results": total_results,
                    "query": query_text,
                    "index_id": index_id,
                    "model_id": self.config["model_id"],
                }

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.exception("TwelveLabs search error: %s", error_msg)

            # Check for throttling/rate limiting
            if any(throttle_msg in error_msg_lower for throttle_msg in TWELVELABS_THROTTLING_MESSAGES):
                raise ModelThrottledException(f"TwelveLabs API throttling: {error_msg}") from e

            # Check for authentication errors
            if any(auth_msg in error_msg_lower for auth_msg in TWELVELABS_AUTHENTICATION_MESSAGES):
                logger.warning("TwelveLabs authentication error")
                raise ValueError(f"TwelveLabs authentication error: {error_msg}") from e

            # Check for search-specific errors
            if any(search_msg in error_msg_lower for search_msg in TWELVELABS_SEARCH_MESSAGES):
                logger.warning("TwelveLabs search error")
                raise ValueError(f"TwelveLabs search error: {error_msg}") from e

            # For other errors, re-raise the original exception
            raise

    def _error_response(self, message: str) -> Iterable[StreamEvent]:
        """Generate standardized error response events.

        Args:
            message: Error message to display

        Yields:
            Streaming events for error response
        """
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"Error: {message}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    @override
    def structured_output(self, output_model: Any, prompt: Any, callback_handler: Any = None) -> Any:
        """TwelveLabs search does not support structured output.

        Raises:
            NotImplementedError: TwelveLabs search models do not support structured output.
        """
        raise NotImplementedError("TwelveLabs search models do not support structured output.")


class TwelveLabsPegasusModel(Model):
    """TwelveLabs model provider implementation for video understanding.

    This provider enables video analysis using TwelveLabs' Pegasus model,
    supporting video upload, indexing, and natural language Q&A about video content.
    """

    class TwelveLabsPegasusConfig(TypedDict, total=False):
        """Configuration options for TwelveLabs Pegasus models.

        Attributes:
            api_key: TwelveLabs API key (falls back to TWELVELABS_API_KEY env var)
            model_id: TwelveLabs model ID (default: "pegasus1.2")
            index_id: Index ID for video storage
            temperature: Generation temperature (0.0-1.0)
            video_id: Default video ID for analysis
            engine_options: Pegasus engine options (visual, audio)
        """

        api_key: Optional[str]
        model_id: str
        index_id: Optional[str]
        temperature: Optional[float]
        video_id: Optional[str]
        engine_options: Optional[List[Literal["visual", "audio"]]]

    def __init__(
        self,
        **model_config: Unpack[TwelveLabsPegasusConfig],
    ):
        """Initialize TwelveLabs Pegasus model provider.

        Args:
            **model_config: Configuration options for the TwelveLabs model.
        """
        self.config = TwelveLabsPegasusModel.TwelveLabsPegasusConfig(
            model_id=DEFAULT_PEGASUS_MODEL,
            temperature=0.7,
            engine_options=["visual", "audio"],
        )
        self.update_config(**model_config)

        # Get API key from config or environment
        api_key = self.config.get("api_key") or os.getenv("TWELVELABS_API_KEY")
        if not api_key:
            raise ValueError(
                "TwelveLabs API key required. Set TWELVELABS_API_KEY environment variable "
                "or pass api_key in model config"
            )

        self._api_key = api_key
        self.video_cache: Dict[str, str] = {}  # Hash -> video_id mapping
        logger.debug("TwelveLabs Pegasus model provider initialized with model_id=%s", self.config["model_id"])

    @override
    def update_config(self, **model_config: Any) -> None:
        """Update the TwelveLabs Pegasus model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        # Filter to only valid TwelveLabsPegasusConfig keys to avoid TypedDict type error
        valid_keys = {"api_key", "model_id", "index_id", "temperature", "video_id", "engine_options"}
        filtered_config = {k: v for k, v in model_config.items() if k in valid_keys}
        self.config.update(filtered_config)  # type: ignore[typeddict-item]

    @override
    def get_config(self) -> TwelveLabsPegasusConfig:
        """Get the current TwelveLabs Pegasus model configuration.

        Returns:
            The TwelveLabs Pegasus model configuration.
        """
        return self.config

    @override
    def format_request(
        self,
        messages: Messages,
        tool_specs: Optional[List[ToolSpec]] = None,
        system_prompt: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Format a TwelveLabs Pegasus request from conversation messages.

        This method extracts the prompt and video information from messages
        and formats them for the TwelveLabs Pegasus API.

        Args:
            messages: List of message objects containing prompts and video content
            tool_specs: Tool specifications (unused for TwelveLabs)
            system_prompt: System prompt (unused for TwelveLabs)

        Returns:
            A formatted request for TwelveLabs Pegasus API
        """
        if not messages:
            raise ValueError("No messages provided")

        # Extract prompt from latest user message
        prompt = ""
        for message in reversed(messages):
            if message.get("role") == "user":
                content: Union[str, List[Any]] = message.get("content", [])

                if isinstance(content, str):
                    prompt = content
                    break
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict) and "text" in block:
                            prompt = block.get("text", "")
                            break
                    if prompt:
                        break

        if not prompt.strip():
            raise ValueError("Pegasus requires a text prompt")

        # Use configured video_id
        video_id = self.config.get("video_id")

        if not video_id:
            raise ValueError("No video_id provided in configuration")

        formatted_request = {
            "video_id": video_id,
            "prompt": prompt,
            "temperature": self.config.get("temperature", 0.7),
            "engine_options": self.config.get("engine_options", ["visual", "audio"]),
        }

        prompt_preview = prompt[:50] + ("..." if len(prompt) > 50 else "")
        logger.info("Formatted Pegasus request: prompt='%s', video_id='%s'", prompt_preview, video_id)

        return formatted_request

    def _upload_and_index_video(self, video_bytes: bytes) -> str:
        """Upload video bytes and return video_id, with caching.

        Args:
            video_bytes: Raw video content to upload

        Returns:
            video_id of the uploaded video
        """
        # Use hash for deduplication
        video_hash = hashlib.sha256(video_bytes).hexdigest()

        if video_hash in self.video_cache:
            logger.info("Using cached video_id for hash %s...", video_hash[:8])
            return self.video_cache[video_hash]

        logger.info("Uploading new video (hash: %s)...", video_hash[:8])

        # Use configured index_id
        index_id = self.config.get("index_id")
        if not index_id:
            raise ValueError("index_id must be configured for video uploads")

        with TwelveLabs(self._api_key) as client:
            # Upload video
            task = client.task.create(index_id=index_id, file=video_bytes)
            logger.info("Upload task created: %s", task.id)

            # Wait for completion
            task.wait_for_done(sleep_interval=5)

            if task.status != "ready":
                raise RuntimeError(f"Video indexing failed: {task.status}")

            video_id = str(task.video_id)
            self.video_cache[video_hash] = video_id

            # Automatically set as default video for analysis
            self.config["video_id"] = video_id
            logger.info("Video indexed successfully! Video ID: %s (set as default)", video_id)

            return video_id

    @override
    def format_chunk(self, event: Dict[str, Any]) -> StreamEvent:
        """Format TwelveLabs Pegasus response into streaming format.

        Args:
            event: Response from TwelveLabs API

        Returns:
            Formatted streaming event
        """
        return cast(StreamEvent, event)

    @override
    def stream(self, request: Dict[str, Any]) -> Iterable[StreamEvent]:
        """Execute Pegasus call and stream response.

        Args:
            request: Formatted Pegasus request

        Yields:
            Streaming events containing Pegasus response

        Raises:
            ModelThrottledException: If the TwelveLabs API is throttling requests
            ValueError: If required parameters are missing or invalid
        """
        video_id = request.get("video_id")
        prompt = request.get("prompt")
        temperature = request.get("temperature", 0.7)
        engine_options = request.get("engine_options", ["visual", "audio"])

        prompt_preview = (prompt[:50] if prompt else "") + ("..." if prompt and len(prompt) > 50 else "")
        logger.info("Starting Pegasus generation: prompt='%s', video='%s'", prompt_preview, video_id)

        if not prompt:
            logger.warning("No prompt provided")
            yield from self._error_response("No prompt provided in the conversation.")
            return

        if not video_id:
            logger.warning("No video ID provided")
            yield from self._error_response("No video ID specified. Please provide a video or set video_id in config.")
            return

        try:
            with TwelveLabs(self._api_key) as client:
                # Execute Pegasus generation
                response = client.generate.text(
                    video_id=video_id,
                    prompt=prompt,
                    temperature=temperature,
                )

                # Start streaming response
                yield {"messageStart": {"role": "assistant"}}
                yield {"contentBlockStart": {"start": {}}}

                # Stream the response text
                if hasattr(response, "data"):
                    yield {"contentBlockDelta": {"delta": {"text": response.data}}}
                else:
                    # Handle response format variations
                    response_text = str(response)
                    yield {"contentBlockDelta": {"delta": {"text": response_text}}}

                # End the response
                yield {"contentBlockStop": {}}
                yield {
                    "messageStop": {
                        "stopReason": "end_turn",
                        "additionalModelResponseFields": {
                            "pegasus_metadata": {
                                "video_id": video_id,
                                "model_id": self.config["model_id"],
                                "temperature": temperature,
                                "engine_options": engine_options,
                            }
                        },
                    }
                }

                logger.info("Pegasus generation completed successfully")

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.exception("TwelveLabs Pegasus error: %s", error_msg)

            # Check for throttling/rate limiting
            if any(throttle_msg in error_msg_lower for throttle_msg in TWELVELABS_THROTTLING_MESSAGES):
                raise ModelThrottledException(f"TwelveLabs API throttling: {error_msg}") from e

            # Check for authentication errors
            if any(auth_msg in error_msg_lower for auth_msg in TWELVELABS_AUTHENTICATION_MESSAGES):
                logger.warning("TwelveLabs authentication error")
                yield from self._error_response(f"Authentication failed: {error_msg}")
                return

            # Check for processing errors
            if any(proc_msg in error_msg_lower for proc_msg in TWELVELABS_PROCESSING_MESSAGES):
                logger.warning("TwelveLabs processing error")
                yield from self._error_response(f"Video processing error: {error_msg}")
                return

            # For other errors, stream error response
            yield from self._error_response(f"Pegasus generation failed: {error_msg}")

    def upload_video(self, video_path: str) -> str:
        """Upload a video file and return its video_id.

        Args:
            video_path: Path to the video file

        Returns:
            video_id of the uploaded video
        """
        with open(video_path, "rb") as video_file:
            video_bytes = video_file.read()

        return self._upload_and_index_video(video_bytes)

    def analyze_video(self, prompt: str, video_id: Optional[str] = None, **kwargs: Any) -> str:
        """Direct video analysis method (non-streaming).

        Args:
            prompt: Natural language question about the video
            video_id: Video ID (uses default if not provided)
            **kwargs: Additional parameters (temperature, etc)

        Returns:
            Pegasus response text
        """
        video_id = video_id or self.config.get("video_id")

        if not video_id:
            raise ValueError("video_id required - provide directly or set video_id in config")

        temperature = kwargs.get("temperature", self.config.get("temperature", 0.7))

        try:
            with TwelveLabs(self._api_key) as client:
                response = client.generate.text(
                    video_id=video_id,
                    prompt=prompt,
                    temperature=temperature,
                )

                if hasattr(response, "data"):
                    return str(response.data)
                else:
                    return str(response)

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.exception("TwelveLabs Pegasus error: %s", error_msg)

            # Check for throttling/rate limiting
            if any(throttle_msg in error_msg_lower for throttle_msg in TWELVELABS_THROTTLING_MESSAGES):
                raise ModelThrottledException(f"TwelveLabs API throttling: {error_msg}") from e

            # Check for authentication errors
            if any(auth_msg in error_msg_lower for auth_msg in TWELVELABS_AUTHENTICATION_MESSAGES):
                raise ValueError(f"TwelveLabs authentication error: {error_msg}") from e

            # Check for processing errors
            if any(proc_msg in error_msg_lower for proc_msg in TWELVELABS_PROCESSING_MESSAGES):
                raise ValueError(f"TwelveLabs processing error: {error_msg}") from e

            # For other errors, re-raise the original exception
            raise

    def _error_response(self, message: str) -> Iterable[StreamEvent]:
        """Generate standardized error response events.

        Args:
            message: Error message to display

        Yields:
            Streaming events for error response
        """
        yield {"messageStart": {"role": "assistant"}}
        yield {"contentBlockStart": {"start": {}}}
        yield {"contentBlockDelta": {"delta": {"text": f"Error: {message}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}

    @override
    def structured_output(self, output_model: Any, prompt: Any, callback_handler: Any = None) -> Any:
        """TwelveLabs Pegasus does not support structured output.

        Raises:
            NotImplementedError: TwelveLabs Pegasus models do not support structured output.
        """
        raise NotImplementedError("TwelveLabs Pegasus models do not support structured output.")
