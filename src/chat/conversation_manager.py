"""
Conversation management for Mail Mind project.
Handles chat history, context tracking, and conversation state.
"""

import json
import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime, timedelta
from dataclasses import dataclass, asdict
from collections import deque


@dataclass
class ConversationExchange:
    """Single exchange in a conversation."""
    timestamp: datetime
    user_message: str
    bot_response: str
    intent: str
    entities: Dict[str, Any]
    session_id: str
    exchange_id: str


@dataclass
class ConversationSummary:
    """Summary of conversation statistics."""
    session_id: str
    start_time: datetime
    last_activity: datetime
    message_count: int
    primary_intents: List[str]
    key_entities: Dict[str, List[Any]]


class ConversationManager:
    """
    Manages conversation history and context for chat sessions.
    Provides memory and context awareness for better user experience.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the conversation manager.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Configuration parameters
        self.max_conversation_length = self.config.get("max_conversation_length", 100)
        self.context_window_size = self.config.get("context_window_size", 10)
        self.session_timeout = timedelta(hours=self.config.get("session_timeout_hours", 24))
        
        # Storage for conversations
        self.conversations: Dict[str, deque] = {}
        self.session_metadata: Dict[str, Dict[str, Any]] = {}
        
        # Context tracking
        self.active_contexts: Dict[str, Dict[str, Any]] = {}
        
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize the conversation manager."""
        try:
            # Load persisted conversations if available
            await self._load_conversations()
            self.is_initialized = True
            self.logger.info("ConversationManager initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize ConversationManager: {e}")
            raise
    
    async def add_exchange(self, 
                         session_id: str, 
                         user_message: str, 
                         bot_response: str, 
                         intent: str,
                         entities: Optional[Dict[str, Any]] = None) -> str:
        """
        Add a conversation exchange to the history.
        
        Args:
            session_id: Unique session identifier
            user_message: The user's message
            bot_response: The bot's response
            intent: Detected intent for the exchange
            entities: Extracted entities
            
        Returns:
            Exchange ID for the added exchange
        """
        if not self.is_initialized:
            raise RuntimeError("ConversationManager not initialized")
        
        exchange_id = f"{session_id}_{datetime.now().isoformat()}"
        
        exchange = ConversationExchange(
            timestamp=datetime.now(),
            user_message=user_message,
            bot_response=bot_response,
            intent=intent,
            entities=entities or {},
            session_id=session_id,
            exchange_id=exchange_id
        )
        
        # Initialize session if needed
        if session_id not in self.conversations:
            await self._initialize_session(session_id)
        
        # Add exchange to conversation history
        self.conversations[session_id].append(exchange)
        
        # Maintain conversation length limit
        if len(self.conversations[session_id]) > self.max_conversation_length:
            self.conversations[session_id].popleft()
        
        # Update session metadata
        await self._update_session_metadata(session_id, exchange)
        
        # Update active context
        await self._update_context(session_id, exchange)
        
        self.logger.debug(f"Added exchange {exchange_id} to session {session_id}")
        return exchange_id
    
    async def get_history(self, 
                         session_id: str, 
                         limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get conversation history for a session.
        
        Args:
            session_id: Session identifier
            limit: Maximum number of exchanges to return
            
        Returns:
            List of conversation exchanges as dictionaries
        """
        if session_id not in self.conversations:
            return []
        
        history = list(self.conversations[session_id])
        
        if limit:
            history = history[-limit:]
        
        return [self._exchange_to_dict(exchange) for exchange in history]
    
    async def get_context(self, session_id: str) -> Dict[str, Any]:
        """
        Get current conversation context for a session.
        
        Args:
            session_id: Session identifier
            
        Returns:
            Context dictionary with relevant information
        """
        if session_id not in self.active_contexts:
            return {}
        
        context = self.active_contexts[session_id].copy()
        
        # Add recent history for context
        recent_history = await self.get_history(session_id, self.context_window_size)
        context["recent_history"] = recent_history
        
        # Add conversation summary
        if session_id in self.conversations:
            context["conversation_length"] = len(self.conversations[session_id])
            context["session_start"] = self.session_metadata[session_id]