"""
Natural language query processing for Mail Mind project.
Handles intent recognition, entity extraction, and query understanding.
"""

import re
import logging
import asyncio
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum


class QueryIntent(Enum):
    """Enumeration of possible query intents."""
    EMAIL_SEARCH = "email_search"
    EMAIL_SUMMARY = "email_summary"
    CONTACT_LOOKUP = "contact_lookup"
    SCHEDULE_QUERY = "schedule_query"
    GENERAL_QUESTION = "general_question"
    UNKNOWN = "unknown"


@dataclass
class QueryResult:
    """Result structure for processed queries."""
    intent: str
    confidence: float
    entities: Dict[str, Any]
    original_query: str
    processed_query: str
    suggestions: List[str]


class QueryProcessor:
    """
    Natural language query processor that extracts intent and entities
    from user messages to enable intelligent email interaction.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the query processor.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Intent patterns for rule-based classification
        self.intent_patterns = {
            QueryIntent.EMAIL_SEARCH: [
                r"(?i)\b(find|search|look for|show me).*(email|message|mail)",
                r"(?i)\b(emails? from|messages? from)",
                r"(?i)\b(emails? about|messages? about)",
                r"(?i)\b(emails? containing|messages? containing)",
            ],
            QueryIntent.EMAIL_SUMMARY: [
                r"(?i)\b(summary|summarize|overview).*(email|inbox|messages?)",
                r"(?i)\b(what.*(new|latest|recent).*(email|message|mail))",
                r"(?i)\b(inbox summary|daily summary)",
            ],
            QueryIntent.CONTACT_LOOKUP: [
                r"(?i)\b(contact|person|people).*(named|called)",
                r"(?i)\b(who is|tell me about|find contact)",
                r"(?i)\b(phone number|email address|contact info)",
            ],
            QueryIntent.SCHEDULE_QUERY: [
                r"(?i)\b(schedule|calendar|meeting|appointment)",
                r"(?i)\b(what.*(today|tomorrow|this week))",
                r"(?i)\b(when.*(meeting|call|appointment))",
            ],
            QueryIntent.GENERAL_QUESTION: [
                r"(?i)\b(help|how|what can you|what do you)",
                r"(?i)\b(thanks|thank you|goodbye|bye)",
            ]
        }
        
        # Entity extraction patterns
        self.entity_patterns = {
            "email_address": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "person_name": r'\b(?:from|to|with|by)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b',
            "date": r'\b(?:today|tomorrow|yesterday|\d{1,2}[-/]\d{1,2}[-/]\d{2,4}|\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{2,4})\b',
            "subject": r'(?i)(?:subject|about|regarding|re:)\s*["\']?([^"\']+)["\']?',
            "timeframe": r'\b(?:today|yesterday|this week|last week|this month|last month|\d+\s+days?\s+ago)\b'
        }
        
        self.is_initialized = False
        
    async def initialize(self):
        """Initialize the query processor."""
        try:
            # Load any required models or resources here
            # For now, using rule-based approach
            self.is_initialized = True
            self.logger.info("QueryProcessor initialized")
        except Exception as e:
            self.logger.error(f"Failed to initialize QueryProcessor: {e}")
            raise
    
    async def process_query(self, 
                          query: str, 
                          conversation_context: Optional[Dict[str, Any]] = None,
                          additional_context: Optional[Dict[str, Any]] = None) -> QueryResult:
        """
        Process a natural language query and extract intent and entities.
        
        Args:
            query: The user's natural language query
            conversation_context: Previous conversation context
            additional_context: Additional context information
            
        Returns:
            QueryResult object with extracted information
        """
        if not self.is_initialized:
            raise RuntimeError("QueryProcessor not initialized")
        
        try:
            # Preprocess the query
            processed_query = self._preprocess_query(query)
            
            # Extract intent
            intent, confidence = await self._extract_intent(processed_query, conversation_context)
            
            # Extract entities
            entities = await self._extract_entities(processed_query, intent)
            
            # Apply context if available
            if conversation_context:
                entities = self._apply_conversation_context(entities, conversation_context)
            
            # Generate suggestions
            suggestions = await self._generate_suggestions(intent, entities)
            
            return QueryResult(
                intent=intent.value,
                confidence=confidence,
                entities=entities,
                original_query=query,
                processed_query=processed_query,
                suggestions=suggestions
            )
            
        except Exception as e:
            self.logger.error(f"Error processing query: {e}")
            return QueryResult(
                intent=QueryIntent.UNKNOWN.value,
                confidence=0.0,
                entities={},
                original_query=query,
                processed_query=query,
                suggestions=["Could you please rephrase your question?"]
            )
    
    def _preprocess_query(self, query: str) -> str:
        """
        Preprocess the query by cleaning and normalizing text.
        
        Args:
            query: Raw query string
            
        Returns:
            Preprocessed query string
        """
        # Basic text cleaning
        query = query.strip()
        query = re.sub(r'\s+', ' ', query)  # Normalize whitespace
        
        # Handle common contractions
        contractions = {
            "can't": "cannot",
            "won't": "will not",
            "don't": "do not",
            "didn't": "did not",
            "haven't": "have not",
            "hasn't": "has not",
            "isn't": "is not",
            "aren't": "are not"
        }
        
        for contraction, expansion in contractions.items():
            query = re.sub(f"\\b{contraction}\\b", expansion, query, flags=re.IGNORECASE)
        
        return query
    
    async def _extract_intent(self, 
                            query: str, 
                            context: Optional[Dict[str, Any]] = None) -> Tuple[QueryIntent, float]:
        """
        Extract the intent from the query using pattern matching.
        
        Args:
            query: Preprocessed query string
            context: Conversation context
            
        Returns:
            Tuple of (intent, confidence_score)
        """
        intent_scores = {}
        
        # Calculate scores for each intent based on pattern matching
        for intent, patterns in self.intent_patterns.items():
            score = 0.0
            matches = 0
            
            for pattern in patterns:
                if re.search(pattern, query):
                    matches += 1
                    score += 1.0
            
            if matches > 0:
                intent_scores[intent] = score / len(patterns)
        
        # Apply context boost if available
        if context and context.get("last_intent"):
            last_intent = context["last_intent"]
            if last_intent in intent_scores:
                intent_scores[QueryIntent(last_intent)] *= 1.2
        
        # Return the highest scoring intent
        if intent_scores:
            best_intent = max(intent_scores, key=intent_scores.get)
            confidence = min(intent_scores[best_intent], 1.0)
            return best_intent, confidence
        
        return QueryIntent.UNKNOWN, 0.0
    
    async def _extract_entities(self, query: str, intent: QueryIntent) -> Dict[str, Any]:
        """
        Extract relevant entities from the query based on the detected intent.
        
        Args:
            query: Preprocessed query string
            intent: Detected intent
            
        Returns:
            Dictionary of extracted entities
        """
        entities = {}
        
        # Extract basic entities using regex patterns
        for entity_type, pattern in self.entity_patterns.items():
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if entity_type == "person_name":
                    entities[entity_type] = [match.strip() for match in matches]
                else:
                    entities[entity_type] = matches
        
        # Intent-specific entity extraction
        if intent == QueryIntent.EMAIL_SEARCH:
            entities.update(self._extract_email_search_entities(query))
        elif intent == QueryIntent.EMAIL_SUMMARY:
            entities.update(self._extract_summary_entities(query))
        elif intent == QueryIntent.CONTACT_LOOKUP:
            entities.update(self._extract_contact_entities(query))
        elif intent == QueryIntent.SCHEDULE_QUERY:
            entities.update(self._extract_schedule_entities(query))
        
        return entities
    
    def _extract_email_search_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities specific to email search queries."""
        entities = {}
        
        # Look for sender information
        sender_match = re.search(r'(?i)(?:from|by|sent by)\s+([^\s,]+(?:\s+[^\s,]+)*)', query)
        if sender_match:
            entities["sender"] = sender_match.group(1).strip()
        
        # Look for subject information
        subject_match = re.search(r'(?i)(?:subject|about|regarding|with subject)\s*[:\'""]?\s*([^"\']+)', query)
        if subject_match:
            entities["subject"] = subject_match.group(1).strip()
        
        # Look for keywords
        keyword_match = re.search(r'(?i)(?:containing|with|including)\s+([^\s,]+(?:\s+[^\s,]+)*)', query)
        if keyword_match:
            entities["keywords"] = keyword_match.group(1).strip()
        
        return entities
    
    def _extract_summary_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities specific to email summary queries."""
        entities = {}
        
        # Look for timeframe
        timeframe_patterns = [
            r'(?i)\b(today|yesterday|this week|last week|this month|last month)\b',
            r'(?i)\b(\d+)\s+(days?|weeks?|months?)\s+ago\b'
        ]
        
        for pattern in timeframe_patterns:
            match = re.search(pattern, query)
            if match:
                entities["timeframe"] = match.group(0)
                break
        
        return entities
    
    def _extract_contact_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities specific to contact lookup queries."""
        entities = {}
        
        # Look for contact names
        name_patterns = [
            r'(?i)(?:contact|person|find)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)',
            r'(?i)(?:named|called)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)'
        ]
        
        for pattern in name_patterns:
            match = re.search(pattern, query)
            if match:
                entities["contact_name"] = match.group(1).strip()
                break
        
        return entities
    
    def _extract_schedule_entities(self, query: str) -> Dict[str, Any]:
        """Extract entities specific to schedule queries."""
        entities = {}
        
        # Look for dates
        date_patterns = [
            r'(?i)\b(today|tomorrow|yesterday)\b',
            r'(?i)\b(this|next|last)\s+(week|month|day)\b',
            r'\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b'
        ]
        
        for pattern in date_patterns:
            match = re.search(pattern, query)
            if match:
                entities["date"] = match.group(0)
                break
        
        return entities
    
    def _apply_conversation_context(self, 
                                 entities: Dict[str, Any], 
                                 context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Apply conversation context to enhance entity extraction.
        
        Args:
            entities: Initially extracted entities
            context: Conversation context
            
        Returns:
            Enhanced entities dictionary
        """
        # If no explicit entities found, try to use context
        if not entities and context.get("last_entities"):
            # Inherit some entities from previous query
            last_entities = context["last_entities"]
            for key in ["sender", "subject", "contact_name"]:
                if key in last_entities:
                    entities[key] = last_entities[key]
        
        return entities
    
    async def _generate_suggestions(self, 
                                  intent: QueryIntent, 
                                  entities: Dict[str, Any]) -> List[str]:
        """
        Generate helpful suggestions based on the detected intent and entities.
        
        Args:
            intent: Detected query intent
            entities: Extracted entities
            
        Returns:
            List of suggestion strings
        """
        suggestions = []
        
        if intent == QueryIntent.EMAIL_SEARCH:
            if not entities.get("sender") and not entities.get("subject"):
                suggestions.append("Try specifying a sender: 'emails from john@example.com'")
                suggestions.append("Or search by subject: 'emails about project update'")
        
        elif intent == QueryIntent.EMAIL_SUMMARY:
            suggestions.append("You can ask for summaries of specific time periods")
            suggestions.append("Try: 'summarize emails from last week'")
        
        elif intent == QueryIntent.CONTACT_LOOKUP:
            suggestions.append("I can help find contact information")
            suggestions.append("Try: 'find contact John Smith'")
        
        elif intent == QueryIntent.SCHEDULE_QUERY:
            suggestions.append("Ask about your schedule for specific days")
            suggestions.append("Try: 'what's my schedule for tomorrow'")
        
        elif intent == QueryIntent.UNKNOWN:
            suggestions.append("I can help with emails, contacts, and scheduling")
            suggestions.append("Try asking: 'show me recent emails' or 'what's my schedule today'")
        
        return suggestions[:3]  # Limit to 3 suggestions
    
    async def shutdown(self):
        """Shutdown the query processor and cleanup resources."""
        self.logger.info("Shutting down QueryProcessor")
        self.is_initialized = False