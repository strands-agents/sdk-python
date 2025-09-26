"""Bidirectional Agent for real-time streaming conversations.

AGENT PURPOSE:
-------------
Provides type-safe constructor and session management for real-time audio/text 
interaction. Serves as the bidirectional equivalent to invoke_async() → stream_async() 
but establishes sessions that continue indefinitely with concurrent task management.

ARCHITECTURAL APPROACH:
----------------------
While invoke_async() creates single request-response cycles that terminate after 
stop_reason: "end_turn" with sequential tool processing, start_conversation() 
establishes persistent sessions with concurrent processing of model events, tool 
execution, and user input without session termination.

DESIGN CHOICE:
-------------
Uses dedicated BidirectionalAgent class (Option 1 from design document) for:
- Type safety with no conditional behavior based on model type
- Separation of concerns - solely focused on bidirectional streaming  
- Future proofing - allows changes without implications to existing Agent class
"""

import asyncio
import logging
from typing import AsyncIterable, List, Optional

from strands.tools.registry import ToolRegistry
from strands.types.content import Messages

from ..event_loop.bidirectional_event_loop import start_bidirectional_connection, stop_bidirectional_connection
from ..models.bidirectional_model import BidirectionalModel
from ..types.bidirectional_streaming import AudioInputEvent, BidirectionalStreamEvent
from ..utils.debug import log_event, log_flow

logger = logging.getLogger(__name__)


class BidirectionalAgent:
    """Agent for bidirectional streaming conversations.
    
    Provides type-safe constructor and session management for real-time 
    audio/text interaction with concurrent processing capabilities.
    """
    
    def __init__(
        self,
        model: BidirectionalModel,
        tools: Optional[List] = None,
        system_prompt: Optional[str] = None,
        messages: Optional[Messages] = None
    ):
        """Initialize bidirectional agent with required model and optional configuration.
        
        Args:
            model: BidirectionalModel instance supporting streaming sessions.
            tools: Optional list of tools available to the model.
            system_prompt: Optional system prompt for conversations.
            messages: Optional conversation history to initialize with.
        """
        self.model = model
        self.system_prompt = system_prompt
        self.messages = messages or []
        
        # Initialize tool registry using existing Strands infrastructure
        self.tool_registry = ToolRegistry()
        if tools:
            self.tool_registry.process_tools(tools)
        self.tool_registry.initialize_tools()
        
        # Initialize tool executor for concurrent execution
        from strands.tools.executors import ConcurrentToolExecutor
        self.tool_executor = ConcurrentToolExecutor()
        
        # Session management
        self._session = None
        self._output_queue = asyncio.Queue()
    
    async def start_conversation(self) -> None:
        """Initialize persistent bidirectional session for real-time interaction.
        
        Creates provider-specific session and starts concurrent background tasks
        for model events, tool execution, and session lifecycle management.
        
        Raises:
            ValueError: If conversation already active.
            ConnectionError: If session creation fails.
        """
        if self._session and self._session.active:
            raise ValueError("Conversation already active. Call end_conversation() first.")
        
        log_flow("conversation_start", "initializing session")
        self._session = await start_bidirectional_connection(self)
        log_event("conversation_ready")
    
    async def send_text(self, text: str) -> None:
        """Send text input during active session without interrupting model generation.
        
        Args:
            text: Text message to send to the model.
            
        Raises:
            ValueError: If no active session.
        """
        self._validate_active_session()
        log_event("text_sent", length=len(text))
        await self._session.model_session.send_text_content(text)
    
    async def send_audio(self, audio_input: AudioInputEvent) -> None:
        """Send audio input during active session for real-time speech interaction.
        
        Args:
            audio_input: AudioInputEvent containing audio data and configuration.
            
        Raises:
            ValueError: If no active session.
        """
        self._validate_active_session()
        await self._session.model_session.send_audio_content(audio_input)
        
    async def receive(self) -> AsyncIterable[BidirectionalStreamEvent]:
        """Receive output events from the model including audio, text.
        
        Provides access to model output events processed by background tasks.
        Events include audio output, text responses, tool calls, and session updates.
        
        Yields:
            BidirectionalStreamEvent: Events from the model session.
        """
        while self._session and self._session.active:
            try:
                event = await asyncio.wait_for(self._output_queue.get(), timeout=0.1)
                yield event
            except asyncio.TimeoutError:
                continue
    
    async def interrupt(self) -> None:
        """Interrupt current model generation and switch to listening mode.
        
        Sends interruption signal to immediately stop generation and clear 
        pending audio output for responsive conversational experience.
        
        Raises:
            ValueError: If no active session.
        """
        self._validate_active_session()
        await self._session.model_session.send_interrupt()
    
    async def end_conversation(self) -> None:
        """End session and cleanup resources including background tasks.
        
        Performs graceful session termination with proper resource cleanup
        including background task cancellation and connection closure.
        """
        if self._session:
            await stop_bidirectional_connection(self._session)
            self._session = None
    
    def _validate_active_session(self) -> None:
        """Validate that an active session exists.
        
        Raises:
            ValueError: If no active session.
        """
        if not self._session or not self._session.active:
            raise ValueError("No active conversation. Call start_conversation() first.")

