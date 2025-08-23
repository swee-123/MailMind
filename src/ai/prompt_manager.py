# src/ai/prompt_manager.py
import yaml
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from langchain.prompts import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.chains import LLMChain, SimpleSequentialChain, SequentialChain
from langchain.schema import BaseOutputParser
from langchain.output_parsers import PydanticOutputParser, OutputFixingParser
from langchain.schema.runnable import RunnablePassthrough, RunnableParallel
from pydantic import BaseModel, Field
import json
import re

from src.ai.model_manager import ai_manager
from src.utils.cache_manager import cache_manager

logger = logging.getLogger(__name__)

# Pydantic models for structured outputs
class EmailPriorityOutput(BaseModel):
    priority_score: float = Field(description="Priority score from 0-10", ge=0, le=10)
    urgency_level: str = Field(description="Urgency level: critical, high, medium, low")
    importance_factors: List[str] = Field(description="List of importance factors")
    confidence: float = Field(description="Confidence in analysis", ge=0, le=1)

class ReplyRequirementOutput(BaseModel):
    requires_reply: bool = Field(description="Whether email requires a reply")
    urgency: str = Field(description="Reply urgency: immediate, today, this_week, when_convenient, no_reply")
    suggested_tone: str = Field(description="Suggested reply tone: formal, casual, urgent, friendly")
    key_points: List[str] = Field(description="Key points to address in reply")

class EmailSummaryOutput(BaseModel):
    summary: str = Field(description="Concise email summary")
    key_topics: List[str] = Field(description="Main topics discussed")
    action_items: List[str] = Field(description="Action items for recipient")
    sentiment: str = Field(description="Email sentiment: positive, neutral, negative, urgent")
    
class DraftReplyOutput(BaseModel):
    subject: str = Field(description="Reply subject line")
    body: str = Field(description="Reply body content")
    tone: str = Field(description="Tone used in reply")
    confidence: float = Field(description="Confidence in generated reply", ge=0, le=1)

class ChatResponseOutput(BaseModel):
    response: str = Field(description="Chat response content")
    suggestions: List[str] = Field(description="Follow-up suggestions", max_items=3)
    context_used: List[str] = Field(description="Context sources used")

class PromptManager:
    """Advanced LangChain prompt management with structured outputs"""
    
    def _init_(self, config_path: str = "config/prompts"):
        self.config_path = Path(config_path)
        self.prompts: Dict[str, Dict] = {}
        self.chains: Dict[str, Any] = {}
        self._load_all_prompts()
        self._initialize_chains()
    
    def _load_all_prompts(self):
        """Load all prompt configurations"""
        prompt_files = [
            "prioritization.yaml",
            "summarization.yaml", 
            "reply_generation.yaml",
            "chat_responses.yaml"
        ]
        
        for filename in prompt_files:
            filepath = self.config_path / filename
            if filepath.exists():
                with open(filepath, 'r') as f:
                    prompt_data = yaml.safe_load(f)
                    category = filename.replace('.yaml', '')
                    self.prompts[category] = prompt_data
            else:
                logger.warning(f"Prompt file not found: {filepath}")
                self._create_default_prompts(filename.replace('.yaml', ''))
    
    def _create_default_prompts(self, category: str):
        """Create default prompts if config files don't exist"""
        defaults = {
            "prioritization": {
                "system_prompt": "You are an AI email assistant specialized in analyzing email priority and importance.",
                "priority_analysis": """
                Analyze this email for priority and importance:
                
                Subject: {subject}
                Sender: {sender}
                Content: {content}
                
                Consider:
                - Sender importance and relationship
                - Urgency indicators in content
                - Business impact
                - Time sensitivity
                - Action requirements
                
                Provide structured analysis with priority score (0-10), urgency level, and importance factors.
                """,
                "reply_detection": """
                Determine if this email requires a reply:
                
                Subject: {subject}
                Content: {content}
                
                Look for:
                - Direct questions
                - Requests for information or action
                - Meeting invitations
                - Decisions needed
                - Deadlines mentioned
                
                Specify if reply is needed and urgency level.
                """
            },
            "summarization": {
                "system_prompt": "You are an AI assistant that creates concise, accurate email summaries.",
                "email_summary": """
                Summarize this email concisely:
                
                Subject: {subject}
                Sender: {sender}
                Content: {content}
                
                Provide:
                - 2-3 sentence summary
                - Key topics discussed
                - Action items for recipient
                - Overall sentiment
                """,
                "thread_summary": """
                Summarize this email thread:
                
                {thread_content}
                
                Provide chronological summary focusing on:
                - Main decisions made
                - Outstanding action items
                - Key participants
                - Current status
                """
            },
            "reply_generation": {
                "system_prompt": "You are an AI assistant that generates professional email replies.",
                "draft_reply": """
                Generate a professional reply to this email:
                
                Original Subject: {subject}
                Sender: {sender}
                Content: {content}
                
                Context: {context}
                Tone: {tone}
                Key points to address: {key_points}
                
                Generate appropriate subject line and body content.
                """,
                "follow_up": """
                Generate a follow-up email for:
                
                Previous Subject: {subject}
                Previous Content: {previous_content}
                Days since last email: {days_elapsed}
                
                Create polite follow-up maintaining professional tone.
                """
            },
            "chat_responses": {
                "system_prompt": "You are MailMind, an intelligent email assistant that helps users manage and understand their emails.",
                "general_query": """
                User query: {query}
                Email context: {email_context}
                
                Provide helpful response based on the user's email data and query.
                Include actionable suggestions when appropriate.
                """,
                "email_search": """
                Find emails matching: {search_criteria}
                
                Available emails: {email_summaries}
                
                Return relevant emails and explain why they match.
                """
            }
        }
        
        self.prompts[category] = defaults.get(category, {})
    
    def _initialize_chains(self):
        """Initialize LangChain chains for different tasks"""
        
        # Email Priority Analysis Chain
        self.chains["priority_analysis"] = self._create_priority_chain()
        
        # Email Summarization Chain
        self.chains["email_summary"] = self._create_summary_chain()
        
        # Reply Generation Chain
        self.chains["reply_generation"] = self._create_reply_chain()
        
        # Chat Response Chain
        self.chains["chat_response"] = self._create_chat_chain()
        
        # Multi-step Email Processing Chain
        self.chains["full_analysis"] = self._create_full_analysis_chain()
    
    def _create_priority_chain(self):
        """Create email priority analysis chain"""
        system_template = self.prompts.get("prioritization", {}).get("system_prompt", "")
        priority_template = self.prompts.get("prioritization", {}).get("priority_analysis", "")
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(priority_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        # Use output parser for structured output
        output_parser = PydanticOutputParser(pydantic_object=EmailPriorityOutput)
        
        return chat_prompt | output_parser
    
    def _create_summary_chain(self):
        """Create email summarization chain"""
        system_template = self.prompts.get("summarization", {}).get("system_prompt", "")
        summary_template = self.prompts.get("summarization", {}).get("email_summary", "")
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(summary_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        output_parser = PydanticOutputParser(pydantic_object=EmailSummaryOutput)
        
        return chat_prompt | output_parser
    
    def _create_reply_chain(self):
        """Create reply generation chain"""
        system_template = self.prompts.get("reply_generation", {}).get("system_prompt", "")
        reply_template = self.prompts.get("reply_generation", {}).get("draft_reply", "")
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(reply_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        output_parser = PydanticOutputParser(pydantic_object=DraftReplyOutput)
        
        return chat_prompt | output_parser
    
    def _create_chat_chain(self):
        """Create chat response chain"""
        system_template = self.prompts.get("chat_responses", {}).get("system_prompt", "")
        chat_template = self.prompts.get("chat_responses", {}).get("general_query", "")
        
        system_message = SystemMessagePromptTemplate.from_template(system_template)
        human_message = HumanMessagePromptTemplate.from_template(chat_template)
        
        chat_prompt = ChatPromptTemplate.from_messages([system_message, human_message])
        
        output_parser = PydanticOutputParser(pydantic_object=ChatResponseOutput)
        
        return chat_prompt | output_parser
    
    def _create_full_analysis_chain(self):
        """Create comprehensive email analysis chain"""
        # This chains together multiple analysis steps
        priority_chain = self.chains["priority_analysis"]
        summary_chain = self.chains["email_summary"]
        
        # Use RunnableParallel to run multiple chains in parallel
        full_chain = RunnableParallel({
            "priority": priority_chain,
            "summary": summary_chain
        })
        
        return full_chain
    
    async def analyze_email_priority(self, subject: str, sender: str, content: str, **kwargs) -> EmailPriorityOutput:
        """Analyze email priority using LangChain"""
        try:
            # Check cache first
            cache_key = f"priority_{hash(f'{subject}{sender}{content[:100]}')}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return EmailPriorityOutput(**cached_result)
            
            # Get current AI provider
            provider = ai_manager.get_provider()
            
            # Prepare input
            input_data = {
                "subject": subject,
                "sender": sender,
                "content": content[:2000],  # Limit content length
                **kwargs
            }
            
            # Run chain
            result = await self._run_chain_async("priority_analysis", input_data, provider)
            
            # Cache result
            await cache_manager.set(cache_key, result.dict(), ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Priority analysis failed: {str(e)}")
            # Return fallback result
            return EmailPriorityOutput(
                priority_score=5.0,
                urgency_level="medium",
                importance_factors=["Analysis failed"],
                confidence=0.1
            )
    
    async def summarize_email(self, subject: str, sender: str, content: str, **kwargs) -> EmailSummaryOutput:
        """Summarize email using LangChain"""
        try:
            cache_key = f"summary_{hash(f'{subject}{content[:100]}')}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return EmailSummaryOutput(**cached_result)
            
            provider = ai_manager.get_provider()
            
            input_data = {
                "subject": subject,
                "sender": sender,
                "content": content,
                **kwargs
            }
            
            result = await self._run_chain_async("email_summary", input_data, provider)
            
            await cache_manager.set(cache_key, result.dict(), ttl=3600)
            
            return result
            
        except Exception as e:
            logger.error(f"Email summarization failed: {str(e)}")
            return EmailSummaryOutput(
                summary="Summarization failed",
                key_topics=[],
                action_items=[],
                sentiment="neutral"
            )
    
    async def generate_reply(self, subject: str, sender: str, content: str, 
                           context: str = "", tone: str = "professional", 
                           key_points: List[str] = None, **kwargs) -> DraftReplyOutput:
        """Generate reply using LangChain"""
        try:
            cache_key = f"reply_{hash(f'{subject}{content[:100]}{tone}')}"
            cached_result = await cache_manager.get(cache_key)
            if cached_result:
                return DraftReplyOutput(**cached_result)
            
            provider = ai_manager.get_provider()
            
            input_data = {
                "subject": subject,
                "sender": sender,
                "content": content,
                "context": context,
                "tone": tone,
                "key_points": ", ".join(key_points or []),
                **kwargs
            }
            
            result = await self._run_chain_async("reply_generation", input_data, provider)
            
            await cache_manager.set(cache_key, result.dict(), ttl=1800)  # Shorter cache for replies
            
            return result
            
        except Exception as e:
            logger.error(f"Reply generation failed: {str(e)}")
            return DraftReplyOutput(
                subject=f"Re: {subject}",
                body="I apologize, but I'm unable to generate a reply at this time.",
                tone=tone,
                confidence=0.1
            )
    
    async def process_chat_query(self, query: str, email_context: str = "", **kwargs) -> ChatResponseOutput:
        """Process chat query using LangChain"""
        try:
            provider = ai_manager.get_provider()
            
            input_data = {
                "query": query,
                "email_context": email_context,
                **kwargs
            }
            
            result = await self._run_chain_async("chat_response", input_data, provider)
            
            return result
            
        except Exception as e:
            logger.error(f"Chat query processing failed: {str(e)}")
            return ChatResponseOutput(
                response="I'm sorry, I couldn't process your query at this time.",
                suggestions=[],
                context_used=[]
            )
    
    async def full_email_analysis(self, subject: str, sender: str, content: str, **kwargs):
        """Run comprehensive email analysis"""
        try:
            provider = ai_manager.get_provider()
            
            input_data = {
                "subject": subject,
                "sender": sender,
                "content": content,
                **kwargs
            }
            
            # Run parallel analysis
            result = await self._run_chain_async("full_analysis", input_data, provider)
            
            return result
            
        except Exception as e:
            logger.error(f"Full email analysis failed: {str(e)}")
            return {
                "priority": EmailPriorityOutput(priority_score=5.0, urgency_level="medium", importance_factors=[], confidence=0.1),
                "summary": EmailSummaryOutput(summary="Analysis failed", key_topics=[], action_items=[], sentiment="neutral")
            }
    
    async def _run_chain_async(self, chain_name: str, input_data: Dict, provider) -> Any:
        """Run LangChain chain asynchronously"""
        import asyncio
        
        chain = self.chains.get(chain_name)
        if not chain:
            raise ValueError(f"Chain {chain_name} not found")
        
        # Convert to async execution
        def run_sync():
            # For now, using synchronous execution
            # In production, you'd use async-compatible chain execution
            try:
                if hasattr(chain, 'invoke'):
                    return chain.invoke(input_data)
                else:
                    return chain.run(**input_data)
            except Exception as e:
                logger.error(f"Chain execution error: {str(e)}")
                raise
        
        # Run in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, run_sync)
    
    def get_prompt_template(self, category: str, template_name: str) -> str:
        """Get specific prompt template"""
        return self.prompts.get(category, {}).get(template_name, "")
    
    def update_prompt_template(self, category: str, template_name: str, template: str):
        """Update prompt template"""
        if category not in self.prompts:
            self.prompts[category] = {}
        self.prompts[category][template_name] = template
        
        # Reinitialize affected chains
        self._initialize_chains()
    
    def save_prompts_to_file(self, category: str):
        """Save prompts to configuration file"""
        filepath = self.config_path / f"{category}.yaml"
        filepath.parent.mkdir(parents=True, exist_ok=True)
        
        with open(filepath, 'w') as f:
            yaml.dump(self.prompts.get(category, {}), f, default_flow_style=False)

# Global instance
prompt_manager = PromptManager()