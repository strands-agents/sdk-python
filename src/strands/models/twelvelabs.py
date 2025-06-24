"""TwelveLabs Search model provider with Marengo integration.

This model provider integrates TwelveLabs Search API with the Marengo model,
allowing for semantic video search through Strands Agents SDK.

- Docs: https://docs.twelvelabs.io/
"""

import logging
import os
from typing import Any, Dict, Iterable, List, Literal, Optional, TypedDict, Union, cast

from typing_extensions import Required, Unpack, override

from ..types.content import ContentBlock, Messages  
from ..types.exceptions import ContextWindowOverflowException, ModelThrottledException
from ..types.models import Model
from ..types.streaming import StreamEvent
from ..types.tools import ToolSpec

try:
    from twelvelabs import TwelveLabs
    TWELVELABS_AVAILABLE = True
except ImportError:
    TWELVELABS_AVAILABLE = False

logger = logging.getLogger(__name__)

DEFAULT_TWELVELABS_MODEL = "Marengo-retrieval-2.7"

# Error codes for TwelveLabs Search API
TWELVELABS_THROTTLING_MESSAGES = [
    "usage_limit_exceeded"
]

TWELVELABS_AUTHENTICATION_MESSAGES = [
    "api_key_invalid"
]

TWELVELABS_SEARCH_MESSAGES = [
    "search_option_not_supported",
    "search_option_combination_not_supported", 
    "search_filter_invalid",
    "search_page_token_expired",
    "index_not_supported_for_search"
]


class TwelveLabsModel(Model):
    """TwelveLabs model provider implementation for video search.
    
    This provider enables semantic video search using TwelveLabs' Marengo model
    through natural language queries, without requiring direct embedding management.
    """

    class TwelveLabsConfig(TypedDict, total=False):
        """Configuration options for TwelveLabs models.

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
        **model_config: Unpack[TwelveLabsConfig],
    ):
        """Initialize TwelveLabs model provider.

        Args:
            **model_config: Configuration options for the TwelveLabs model.
        """
        if not TWELVELABS_AVAILABLE:
            raise ImportError(
                "TwelveLabs SDK not available. Install with: pip install twelvelabs-python"
            )

        self.config = TwelveLabsModel.TwelveLabsConfig(
            model_id=DEFAULT_TWELVELABS_MODEL,
            search_options=["visual"],  # Changed from ["visual", "audio"] to just ["visual"]
            group_by="clip",
            page_limit=10
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
        logger.debug("TwelveLabs model provider initialized with model_id=%s", self.config["model_id"])

    @override
    def update_config(self, **model_config: Unpack[TwelveLabsConfig]) -> None:
        """Update the TwelveLabs model configuration.

        Args:
            **model_config: Configuration overrides.
        """
        self.config.update(model_config)

    @override
    def get_config(self) -> TwelveLabsConfig:
        """Get the current TwelveLabs model configuration.

        Returns:
            The TwelveLabs model configuration.
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
                content = message.get("content", [])
                if isinstance(content, str):
                    search_query = content
                    break
                elif isinstance(content, list):
                    for block in content:
                        if isinstance(block, dict):
                            if block.get("type") == "text" or "text" in block:
                                search_query = block.get("text", "")
                            break
                    if search_query:
                        break
        
        index_id = self.config.get("index_id")
        if not index_id:
            raise ValueError("index_id must be configured in TwelveLabsConfig")

        formatted_request = {
            "query_text": search_query,
            "index_id": index_id,
            "search_options": self.config.get("search_options", ["visual", "audio"]),
            "group_by": self.config.get("group_by", "clip"),
            "threshold": self.config.get("threshold"),
            "page_limit": self.config.get("page_limit", 10),
        }
        
        logger.info(f"Formatted TwelveLabs search request: query_text='{search_query[:50]}{'...' if len(search_query) > 50 else ''}', index_id='{index_id}'")
        
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
        
        logger.info(f"Starting TwelveLabs search: query='{query_text[:50] if query_text else ''}{'...' if query_text and len(query_text) > 50 else ''}', index='{index_id}'")
        
        if not query_text:
            logger.warning("No search query provided")
            yield from self._error_response("No search query provided in the conversation.")
            return
            
        if not index_id:
            logger.warning("No index ID provided")
            yield from self._error_response("No index ID specified. Please provide an index ID or set a default in the model config.")
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
                yield {"contentBlockStart": {"start": {"text": ""}}}

                # Format and stream results
                total_results = getattr(result.pool, 'total_count', 0) if hasattr(result, 'pool') else 0
                logger.info(f"TwelveLabs search completed: {total_results} results")
                
                response_text = "Video Search Results\n\n"
                response_text += f"Query: \"{query_text}\"\n"
                response_text += f"Index: {index_id}\n"
                response_text += f"Found {total_results} total results\n\n"

                # Process search results
                search_results = []
                for item in result.data:
                    if request.get("group_by") == "video" and hasattr(item, 'clips'):
                        # Video-grouped results
                        video_result = {
                            "video_id": item.id,
                            "clips": []
                        }
                        for clip in item.clips:
                            video_result["clips"].append({
                                "score": clip.score,
                                "start": clip.start,
                                "end": clip.end,
                                "confidence": clip.confidence,
                                "video_id": clip.video_id,
                            })
                        search_results.append(video_result)
                    else:
                        # Clip-level results
                        search_results.append({
                            "score": item.score,
                            "start": item.start,
                            "end": item.end,
                            "confidence": item.confidence,
                            "video_id": item.video_id,
                        })

                # Format detailed results
                if search_results:
                    response_text += "Top Results:\n"
                    for i, result_item in enumerate(search_results[:5], 1):
                        if request.get("group_by") == "video":
                            response_text += f"\n{i}. Video ID: {result_item['video_id']}\n"
                            response_text += f"   Found {len(result_item['clips'])} clips\n"
                            for j, clip in enumerate(result_item['clips'][:3], 1):
                                response_text += f"   {j}. Score: {clip['score']:.3f} | {clip['start']:.1f}s-{clip['end']:.1f}s | {clip['confidence']}\n"
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
                                "model_id": self.config["model_id"]
                            }
                        }
                    }
                }

        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.error(f"TwelveLabs search error: {error_msg}", exc_info=True)
            
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

    def search_videos(
        self,
        query_text: str,
        index_id: Optional[str] = None,
        **search_params: Any
    ) -> Dict[str, Any]:
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
                    **{k: v for k, v in search_params.items() 
                       if k not in ["search_options", "group_by", "threshold", "page_limit"]}
                )
                
                # Format results
                search_results = []
                total_results = getattr(result.pool, 'total_count', 0) if hasattr(result, 'pool') else 0
                
                for item in result.data:
                    if search_params.get("group_by") == "video" and hasattr(item, 'clips'):
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
                            ]
                        }
                        search_results.append(video_result)
                    else:
                        search_results.append({
                            "score": item.score,
                            "start": item.start,
                            "end": item.end,
                            "confidence": item.confidence,
                            "video_id": item.video_id,
                        })
                
                return {
                    "results": search_results,
                    "total_results": total_results,
                    "query": query_text,
                    "index_id": index_id,
                    "model_id": self.config["model_id"]
                }
                
        except Exception as e:
            error_msg = str(e)
            error_msg_lower = error_msg.lower()
            logger.error(f"TwelveLabs search error: {error_msg}", exc_info=True)
            
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
        yield {"contentBlockStart": {"start": {"text": ""}}}
        yield {"contentBlockDelta": {"delta": {"text": f"Error: {message}"}}}
        yield {"contentBlockStop": {}}
        yield {"messageStop": {"stopReason": "end_turn"}}