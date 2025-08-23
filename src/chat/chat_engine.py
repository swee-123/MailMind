"""
Chat processing engine for Mail Mind project.
Handles the main chat logic and response generation.
"""

import logging
import asyncio
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass

from .query_processor import QueryProcessor
from .conversation_manager import ConversationManager


@dataclass
class ChatResponse:
    """Structure for chat responses."""
    content: str
    confidence: float
    response_type: str
    metadata: Dict[str, Any]
    timestamp: datetime


class ChatEngine:
    """
    Main chat processing engine that orchestrates query processing,
    conversation management, and response generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the chat engine.
        
        Args:
            config: Configuration dictionary for the chat engine
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.query_processor = QueryProcessor(config)
        self.conversation_manager = ConversationManager(config)
        
        # Engine state
        self.is_initialized = False
        self.active_sessions = {}
        
        self.logger.info("ChatEngine initialized")
    
    async def initialize(self):
        """Initialize the chat engine and its components."""
        try:
            await self.query_processor.initialize()
            await self.conversation_manager.initialize()
            self.is_initialized = True
            self.logger.info("ChatEngine initialization completed")
        except Exception as e:
            self.logger.error(f"Failed to initialize ChatEngine: {e}")
            raise
    
    async def process_message(self, 
                            message: str, 
                            session_id: str,
                            context: Optional[Dict[str, Any]] = None) -> ChatResponse:
        """
        Process a chat message and generate a response.
        
        Args:
            message: The user's message
            session_id: Unique session identifier
            context: Additional context for processing
            
        Returns:
            ChatResponse object containing the response and metadata
        """
        if not self.is_initialized:
            raise RuntimeError("ChatEngine not initialized. Call initialize() first.")
        
        start_time = datetime.now()
        
        try:
            # Get conversation context
            conversation_context = await self.conversation_manager.get_context(session_id)
            
            # Process the query
            query_result = await self.query_processor.process_query(
                message, 
                conversation_context, 
                context
            )
            
            # Generate response based on query result
            response_content = await self._generate_response(query_result, conversation_context)
            
            # Create response object
            response = ChatResponse(
                content=response_content,
                confidence=query_result.confidence,
                response_type=query_result.intent,
                metadata={
                    "processing_time": (datetime.now() - start_time).total_seconds(),
                    "query_entities": query_result.entities,
                    "session_id": session_id
                },
                timestamp=datetime.now()
            )
            
            # Update conversation history
            await self.conversation_manager.add_exchange(
                session_id, message, response_content, query_result.intent
            )
            
            self.logger.info(f"Processed message for session {session_id}")
            return response
            
        except Exception as e:
            self.logger.error(f"Error processing message: {e}")
            return ChatResponse(
                content="I apologize, but I encountered an error processing your request. Please try again.",
                confidence=0.0,
                response_type="error",
                metadata={"error": str(e)},
                timestamp=datetime.now()
            )
    
    async def _generate_response(self, query_result, conversation_context) -> str:
        """
        Generate a response based on query processing results.
        
        Args:
            query_result: Result from query processing
            conversation_context: Current conversation context
            
        Returns:
            Generated response string
        """
        try:
            intent = query_result.intent
            entities = query_result.entities
            confidence = query_result.confidence
            
            if confidence < 0.3:
                return await self._handle_low_confidence_query(query_result)
            
            # Route to specific handlers based on intent
            if intent == "email_search":
                return await self._handle_email_search(entities, conversation_context)
            elif intent == "email_summary":
                return await self._handle_email_summary(entities, conversation_context)
            elif intent == "contact_lookup":
                return await self._handle_contact_lookup(entities, conversation_context)
            elif intent == "schedule_query":
                return await self._handle_schedule_query(entities, conversation_context)
            elif intent == "general_question":
                return await self._handle_general_question(query_result, conversation_context)
            else:
                return await self._handle_unknown_intent(query_result)
                
        except Exception as e:
            self.logger.error(f"Error in response generation: {e}")
            return "I'm having trouble understanding your request. Could you please rephrase it?"
    
    async def _handle_email_search(self, entities, context) -> str:
        """Handle email search queries."""
        # This would integrate with your email backend
        sender = entities.get("sender", "")
        subject = entities.get("subject", "")
        date_range = entities.get("date_range", "")
        
        # Placeholder for email search logic
        if sender:
            return f"I found several emails from {sender}. Here are the most recent ones..."
        elif subject:
            return f"I found emails with subject containing '{subject}'..."
        else:
            return "I can help you search for emails. Please specify a sender, subject, or date range."
    
    async def _handle_email_summary(self, entities, context) -> str:
        """Handle email summary requests."""
        timeframe = entities.get("timeframe", "today")
        return f"Here's a summary of your emails from {timeframe}..."
    
    async def _handle_contact_lookup(self, entities, context) -> str:
        """Handle contact lookup queries."""
        contact_name = entities.get("contact_name", "")
        return f"Here's what I found about {contact_name}..."
    
    async def _handle_schedule_query(self, entities, context) -> str:
        """Handle schedule-related queries."""
        date = entities.get("date", "today")
        return f"Here's your schedule for {date}..."
    
    async def _handle_general_question(self, query_result, context) -> str:
        """Handle general questions about the system."""
        return "I can help you with emails, contacts, and scheduling. What would you like to know?"
    
    async def _handle_unknown_intent(self, query_result) -> str:
        """Handle queries with unknown intent."""
        return "I'm not sure how to help with that. I can assist with emails, contacts, and scheduling. What would you like to do?"
    
    async def _handle_low_confidence_query(self, query_result) -> str:
        """Handle queries with low confidence scores."""
        return "I'm not entirely sure what you're looking for. Could you please be more specific?"
    
    async def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status information for a chat session."""
        conversation_history = await self.conversation_manager.get_history(session_id)
        return {
            "session_id": session_id,
            "message_count": len(conversation_history),
            "last_activity": conversation_history[-1]["timestamp"] if conversation_history else None,
            "active": session_id in self.active_sessions
        }
    
    async def end_session(self, session_id: str):
        """End a chat session and clean up resources."""
        await self.conversation_manager.end_session(session_id)
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
        self.logger.info(f"Ended session {session_id}")
    
    async def shutdown(self):
        """Shutdown the chat engine and cleanup resources."""
        self.logger.info("Shutting down ChatEngine")
        await self.query_processor.shutdown()
        await self.conversation_manager.shutdown()
        self.is_initialized = False