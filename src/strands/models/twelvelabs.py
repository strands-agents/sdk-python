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
from ..types.exceptions import ModelThrottledException
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
            default_index_id: Default index ID for searches (can be overridden per query)
            search_options: Default search modalities ["visual", "audio"]
            group_by: Default grouping for results ("video" or "clip")
            threshold: Default confidence threshold ("high", "medium", "low", "none")
            page_limit: Default maximum results per page
        """
        
        api_key: Optional[str]
        model_id: str
        default_index_id: Optional[str]
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
        logger.debug(f"format_request called with {len(messages)} messages")
        
        # Extract the latest user message as the search query
        search_query = ""
        index_id = self.config.get("default_index_id")
        
        for i, message in enumerate(reversed(messages)):
            logger.debug(f"Checking message {i} (reversed): role={message.get('role')}")
            if message.get("role") == "user":
                # Extract text content from the message
                content = message.get("content", [])
                logger.debug(f"User message content type: {type(content)}")
                if isinstance(content, str):
                    search_query = content
                    logger.debug(f"Extracted query from string content: '{search_query}'")
                    break
                elif isinstance(content, list):
                    logger.debug(f"Content is list with {len(content)} items")
                    for j, block in enumerate(content):
                        logger.debug(f"Block {j}: type={type(block)}")
                        if isinstance(block, dict):
                            logger.debug(f"Block {j} dict keys: {list(block.keys())}")
                            logger.debug(f"Block {j} content: {block}")
                            if block.get("type") == "text" or "text" in block:
                                search_query = block.get("text", "")
                                logger.debug(f"Extracted query from text block: '{search_query}'")
                            break
                    if search_query:
                        break
        
        # Parse search parameters from the query (basic extraction)
        # In a production implementation, you might use NLP to better extract parameters
        query_lower = search_query.lower()
        
        # Extract index_id if mentioned in query
        if "index" in query_lower:
            # Simple pattern matching - could be more sophisticated
            words = search_query.split()
            for i, word in enumerate(words):
                if "index" in word.lower() and i + 1 < len(words):
                    potential_id = words[i + 1].strip(".,!?")
                    if potential_id and not potential_id.lower() in ["id", "is", "are"]:
                        index_id = potential_id
                        break

        formatted_request = {
            "query_text": search_query,
            "index_id": index_id,
            "search_options": self.config.get("search_options", ["visual", "audio"]),
            "group_by": self.config.get("group_by", "clip"),
            "threshold": self.config.get("threshold"),
            "page_limit": self.config.get("page_limit", 10),
        }
        
        logger.info(f"Formatted request: query_text='{search_query}', index_id='{index_id}'")
        logger.debug(f"Full formatted request: {formatted_request}")
        
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
        
        logger.debug(f"TwelveLabs stream() called with request: {request}")
        logger.debug(f"Extracted query_text: '{query_text}'")
        logger.debug(f"Extracted index_id: '{index_id}'")
        
        if not query_text:
            logger.warning("No search query provided in request")
            yield {
                "messageStart": {"role": "assistant"},
                "contentBlockStart": {"start": {"text": ""}},
                "contentBlockDelta": {
                    "delta": {"text": "Error: No search query provided in the conversation."}
                },
                "contentBlockStop": {},
                "messageStop": {"stopReason": "end_turn"}
            }
            return
            
        if not index_id:
            logger.warning("No index ID provided in request")
            yield {
                "messageStart": {"role": "assistant"},
                "contentBlockStart": {"start": {"text": ""}},
                "contentBlockDelta": {
                    "delta": {"text": "Error: No index ID specified. Please provide an index ID or set a default in the model config."}
                },
                "contentBlockStop": {},
                "messageStop": {"stopReason": "end_turn"}
            }
            return

        try:
            logger.info(f"Making TwelveLabs API call with query: '{query_text}', index: '{index_id}'")
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
                
                logger.debug(f"TwelveLabs API call successful, result type: {type(result)}")

                # Start streaming response
                logger.debug("Yielding messageStart event")
                yield {"messageStart": {"role": "assistant"}}
                logger.debug("Yielding contentBlockStart event")
                yield {"contentBlockStart": {"start": {"text": ""}}}

                # Format and stream results
                total_results = getattr(result.pool, 'total_count', 0) if hasattr(result, 'pool') else 0
                logger.info(f"Search returned {total_results} total results")
                
                response_text = f"🎬 Video Search Results\n\n"
                response_text += f"Query: \"{query_text}\"\n"
                response_text += f"Index: {index_id}\n"
                response_text += f"Found {total_results} total results\n\n"

                # Process search results
                search_results = []
                logger.debug(f"Processing {len(result.data) if hasattr(result, 'data') else 0} search result items")
                for i, item in enumerate(result.data):
                    logger.debug(f"Processing result item {i}: type={type(item)}")
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
                logger.debug(f"Processed {len(search_results)} search results")

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
                logger.debug(f"Yielding response text of length: {len(response_text)}")
                yield {"contentBlockDelta": {"delta": {"text": response_text}}}

                # End the response
                logger.debug("Yielding contentBlockStop event")
                yield {"contentBlockStop": {}}
                logger.debug("Yielding messageStop event with metadata")
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
                logger.info("TwelveLabs stream completed successfully")

        except Exception as e:
            error_msg = str(e)
            logger.error(f"TwelveLabs search error: {error_msg}", exc_info=True)
            
            # Check for throttling
            if "throttl" in error_msg.lower() or "rate limit" in error_msg.lower():
                raise ModelThrottledException(f"TwelveLabs API throttling: {error_msg}")
            
            # Stream error response
            logger.debug("Yielding error response events")
            yield {"messageStart": {"role": "assistant"}}
            yield {"contentBlockStart": {"start": {"text": ""}}}
            yield {
                "contentBlockDelta": {
                    "delta": {"text": f"❌ Search failed: {error_msg}"}
                }
            }
            yield {"contentBlockStop": {}}
            yield {"messageStop": {"stopReason": "end_turn"}}
            logger.info("Error response stream completed")

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
        index_id = index_id or self.config.get("default_index_id")

        if not index_id:
            raise ValueError("index_id required - provide directly or set default_index_id in config")
            
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
            logger.error(f"TwelveLabs search error: {str(e)}")
            raise