"""
Chat interface module for Mail Mind project.

This module provides a comprehensive chat interface for natural language
interaction with email data, including query processing, conversation
management, and response generation.
"""

from .chat_engine import ChatEngine
from .query_processor import QueryProcessor
from .conversation_manager import ConversationManager

__version__ = "1.0.0"
__author__ = "Mail Mind Team"

__all__ = [
    "ChatEngine",
    "QueryProcessor", 
    "ConversationManager"
]

# Module-level configuration
DEFAULT_CONFIG = {
    "max_conversation_length": 100,
    "query_timeout": 30,
    "enable_context_memory": True,
    "default_response_format": "conversational"
}