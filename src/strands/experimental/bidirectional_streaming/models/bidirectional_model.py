"""Bidirectional model interface for real-time streaming conversations.

INTERFACE PURPOSE:
-----------------
Declares bidirectional capabilities separate from existing Model hierarchy to maintain 
clean separation of concerns. Models choose to implement this interface explicitly 
for bidirectional streaming support.

PROVIDER ABSTRACTION:
--------------------
Abstracts incompatible initialization patterns: Nova Sonic's event-driven sequences, 
Google's WebSocket setup, OpenAI's dual protocol support. Normalizes different tool 
calling approaches and handles provider-specific session management with varying 
time limits and connection patterns.

SESSION-BASED APPROACH:
----------------------
Unlike existing Model interface's stateless request-response pattern where each 
stream() call processes complete messages independently, BidirectionalModel introduces 
session-based approach where create_bidirectional_connection() establishes persistent 
connections supporting real-time bidirectional communication during active generation.
"""

import abc
import logging
from typing import Any, AsyncIterable, Dict, List, Optional

from ....types.content import Messages
from ....types.tools import ToolSpec
from ..types.bidirectional_streaming import AudioInputEvent

logger = logging.getLogger(__name__)

class BidirectionalModelSession(abc.ABC):
    """Model-specific session interface for bidirectional communication."""
    
    @abc.abstractmethod
    async def receive_events(self) -> AsyncIterable[Dict[str, Any]]:
        """Receive events from model in provider-agnostic format.
        
        Normalizes different provider event formats so the event loop
        can process all providers uniformly.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_audio_content(self, audio_input: AudioInputEvent) -> None:
        """Send audio content to model during session.
        
        Manages complex audio encoding and provider-specific event sequences
        while presenting simple AudioInputEvent interface to Agent.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_text_content(self, text: str, **kwargs) -> None:
        """Send text content processed concurrently with ongoing generation.
        
        Enables natural interruption and follow-up questions without session restart.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_interrupt(self) -> None:
        """Send interruption signal to immediately stop generation.
        
        Critical for responsive conversational experiences where users
        can naturally interrupt mid-response.
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_tool_result(self, tool_use_id: str, result: Dict[str, Any]) -> None:
        """Send tool execution result to model in provider-specific format.
        
        Each provider handles result formatting according to their protocol:
        - Nova Sonic: toolResult events with JSON content
        - Google Live API: toolResponse with specific structure  
        - OpenAI Realtime: function call responses with call_id correlation
        """
        raise NotImplementedError
    
    @abc.abstractmethod
    async def send_tool_error(self, tool_use_id: str, error: str) -> None:
        """Send tool execution error to model in provider-specific format."""
        raise NotImplementedError
    
    @abc.abstractmethod
    async def close(self) -> None:
        """Close session and cleanup resources with graceful termination."""
        raise NotImplementedError


class BidirectionalModel(abc.ABC):
    """Interface for models that support bidirectional streaming.
    
    Separate from Model to maintain clean separation of concerns.
    Models choose to implement this interface explicitly.
    """
    
    @abc.abstractmethod
    async def create_bidirectional_connection(
        self,
        system_prompt: Optional[str] = None,
        tools: Optional[List[ToolSpec]] = None,
        messages: Optional[Messages] = None,
        **kwargs
    ) -> BidirectionalModelSession:
        """Create bidirectional session with model-specific implementation.
        
        Abstracts complex provider-specific initialization while presenting
        uniform interface to Agent.
        """
        raise NotImplementedError

