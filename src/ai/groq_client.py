# ai/groq_client.py
import os
import json
import asyncio
from typing import Dict, List, Optional, Any, AsyncGenerator
from groq import Groq, AsyncGroq
import logging
from datetime import datetime

class GroqClient:
    """Groq API client for MailMind AI operations"""
    
    # Available models
    MODELS = {
        'llama-3.1-70b': 'llama-3.1-70b-versatile',
        'llama-3.1-8b': 'llama-3.1-8b-instant',
        'mixtral-8x7b': 'mixtral-8x7b-32768',
        'gemma-7b': 'gemma-7b-it'
    }
    
    def __init__(self, api_key: Optional[str] = None, model: str = 'llama-3.1-8b'):
        self.api_key = api_key or os.getenv('GROQ_API_KEY')
        if not self.api_key:
            raise ValueError("Groq API key is required. Set GROQ_API_KEY environment variable.")
        
        self.model = self.MODELS.get(model, model)
        self.client = Groq(api_key=self.api_key)
        self.async_client = AsyncGroq(api_key=self.api_key)
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.rate_limit_requests = 100  # per minute
        self.rate_limit_tokens = 100000  # per minute
        self._request_timestamps = []
        self._token_usage = []
    
    def chat_completion(self, messages: List[Dict[str, str]], 
                       temperature: float = 0.7,
                       max_tokens: int = 1024,
                       stream: bool = False) -> Dict[str, Any]:
        """Generate chat completion using Groq"""
        try:
            self._check_rate_limit()
            
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=stream
            )
            
            if stream:
                return response
            
            self._update_usage_tracking(response)
            
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage.dict() if hasattr(response, 'usage') else {},
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            self.logger.error(f"Groq API error: {e}")
            raise
    
    async def async_chat_completion(self, messages: List[Dict[str, str]],
                                  temperature: float = 0.7,
                                  max_tokens: int = 1024) -> Dict[str, Any]:
        """Async chat completion"""
        try:
            self._check_rate_limit()
            
            response = await self.async_client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens
            )
            
            self._update_usage_tracking(response)
            
            return {
                'content': response.choices[0].message.content,
                'usage': response.usage.dict() if hasattr(response, 'usage') else {},
                'model': response.model,
                'finish_reason': response.choices[0].finish_reason
            }
            
        except Exception as e:
            self.logger.error(f"Async Groq API error: {e}")
            raise
    
    def stream_completion(self, messages: List[Dict[str, str]],
                         temperature: float = 0.7,
                         max_tokens: int = 1024) -> AsyncGenerator[str, None]:
        """Stream completion tokens"""
        try:
            stream = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                stream=True
            )
            
            for chunk in stream:
                if chunk.choices[0].delta.content is not None:
                    yield chunk.choices[0].delta.content
                    
        except Exception as e:
            self.logger.error(f"Streaming error: {e}")
            raise
    
    def _check_rate_limit(self):
        """Check and enforce rate limiting"""
        current_time = datetime.now().timestamp()
        
        # Clean old timestamps (older than 1 minute)
        self._request_timestamps = [
            ts for ts in self._request_timestamps 
            if current_time - ts < 60
        ]
        
        if len(self._request_timestamps) >= self.rate_limit_requests:
            raise Exception("Rate limit exceeded: too many requests per minute")
        
        self._request_timestamps.append(current_time)
    
    def _update_usage_tracking(self, response):
        """Update token usage tracking"""
        if hasattr(response, 'usage'):
            current_time = datetime.now().timestamp()
            self._token_usage.append({
                'timestamp': current_time,
                'tokens': response.usage.total_tokens
            })
            
            # Clean old usage data
            self._token_usage = [
                usage for usage in self._token_usage 
                if current_time - usage['timestamp'] < 60
            ]
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get current usage statistics"""
        current_time = datetime.now().timestamp()
        recent_requests = len([
            ts for ts in self._request_timestamps 
            if current_time - ts < 60
        ])
        recent_tokens = sum([
            usage['tokens'] for usage in self._token_usage 
            if current_time - usage['timestamp'] < 60
        ])
        
        return {
            'requests_last_minute': recent_requests,
            'tokens_last_minute': recent_tokens,
            'rate_limit_requests': self.rate_limit_requests,
            'rate_limit_tokens': self.rate_limit_tokens
        }

