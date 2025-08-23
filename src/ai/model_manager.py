# src/ai/model_manager.py
import os
import asyncio
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Any
from dataclasses import dataclass
import openai
import google.generativeai as genai
from groq import Groq
import anthropic
from langchain.schema import BaseMessage
from langchain.chat_models import ChatOpenAI, ChatAnthropic
from langchain.llms import OpenAI

@dataclass
class ModelConfig:
    """Configuration for AI models"""
    name: str
    provider: str
    model_id: str
    api_key: str
    max_tokens: int = 4000
    temperature: float = 0.7
    timeout: int = 30

class AIProvider(ABC):
    """Abstract base class for AI providers"""
    
    @abstractmethod
    async def generate_response(self, prompt: str, **kwargs) -> str:
        pass
    
    @abstractmethod
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        pass

class OpenAIProvider(AIProvider):
    """OpenAI API provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = openai.AsyncOpenAI(api_key=config.api_key)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = await self.client.chat.completions.create(
                model=self.config.model_id,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"OpenAI API error: {str(e)}")

class GroqProvider(AIProvider):
    """Groq API provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = Groq(api_key=config.api_key)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_id,
                messages=[{"role": "user", "content": prompt}],
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                self.client.chat.completions.create,
                model=self.config.model_id,
                messages=messages,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature)
            )
            return response.choices[0].message.content
        except Exception as e:
            raise Exception(f"Groq API error: {str(e)}")

class AnthropicProvider(AIProvider):
    """Anthropic Claude API provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.client = anthropic.AsyncAnthropic(api_key=config.api_key)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = await self.client.messages.create(
                model=self.config.model_id,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=[{"role": "user", "content": prompt}]
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # Convert messages to Anthropic format
            anthropic_messages = []
            for msg in messages:
                if msg["role"] in ["user", "assistant"]:
                    anthropic_messages.append({
                        "role": msg["role"],
                        "content": msg["content"]
                    })
            
            response = await self.client.messages.create(
                model=self.config.model_id,
                max_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                temperature=kwargs.get('temperature', self.config.temperature),
                messages=anthropic_messages
            )
            return response.content[0].text
        except Exception as e:
            raise Exception(f"Anthropic API error: {str(e)}")

class GeminiProvider(AIProvider):
    """Google Gemini API provider"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        genai.configure(api_key=config.api_key)
        self.model = genai.GenerativeModel(config.model_id)
    
    async def generate_response(self, prompt: str, **kwargs) -> str:
        try:
            response = await asyncio.to_thread(
                self.model.generate_content,
                prompt,
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], **kwargs) -> str:
        try:
            # Convert messages to Gemini chat format
            chat_history = []
            for msg in messages[:-1]:  # All but last message
                chat_history.append({
                    "role": "user" if msg["role"] == "user" else "model",
                    "parts": [msg["content"]]
                })
            
            chat = self.model.start_chat(history=chat_history)
            response = await asyncio.to_thread(
                chat.send_message,
                messages[-1]["content"],
                generation_config=genai.types.GenerationConfig(
                    max_output_tokens=kwargs.get('max_tokens', self.config.max_tokens),
                    temperature=kwargs.get('temperature', self.config.temperature)
                )
            )
            return response.text
        except Exception as e:
            raise Exception(f"Gemini API error: {str(e)}")

class AIModelManager:
    """Central manager for all AI model providers"""
    
    def __init__(self):
        self.providers: Dict[str, AIProvider] = {}
        self.configs: Dict[str, ModelConfig] = {}
        self.current_provider: Optional[str] = None
        self._load_configurations()
    
    def _load_configurations(self):
        """Load model configurations from environment variables"""
        # OpenAI
        if os.getenv("OPENAI_API_KEY"):
            self.configs["openai"] = ModelConfig(
                name="OpenAI GPT",
                provider="openai",
                model_id=os.getenv("OPENAI_MODEL", "gpt-4-turbo-preview"),
                api_key=os.getenv("OPENAI_API_KEY")
            )
        
        # Groq
        if os.getenv("GROQ_API_KEY"):
            self.configs["groq"] = ModelConfig(
                name="Groq",
                provider="groq",
                model_id=os.getenv("GROQ_MODEL", "mixtral-8x7b-32768"),
                api_key=os.getenv("GROQ_API_KEY")
            )
        
        # Anthropic
        if os.getenv("ANTHROPIC_API_KEY"):
            self.configs["anthropic"] = ModelConfig(
                name="Claude",
                provider="anthropic",
                model_id=os.getenv("ANTHROPIC_MODEL", "claude-3-sonnet-20240229"),
                api_key=os.getenv("ANTHROPIC_API_KEY")
            )
        
        # Google Gemini
        if os.getenv("GOOGLE_API_KEY"):
            self.configs["gemini"] = ModelConfig(
                name="Gemini",
                provider="gemini",
                model_id=os.getenv("GEMINI_MODEL", "gemini-1.5-pro"),
                api_key=os.getenv("GOOGLE_API_KEY")
            )
        
        # Set default provider
        if self.configs:
            self.current_provider = list(self.configs.keys())[0]
    
    def initialize_provider(self, provider_name: str) -> AIProvider:
        """Initialize a specific AI provider"""
        if provider_name not in self.configs:
            raise ValueError(f"Provider {provider_name} not configured")
        
        config = self.configs[provider_name]
        
        if provider_name == "openai":
            return OpenAIProvider(config)
        elif provider_name == "groq":
            return GroqProvider(config)
        elif provider_name == "anthropic":
            return AnthropicProvider(config)
        elif provider_name == "gemini":
            return GeminiProvider(config)
        else:
            raise ValueError(f"Unknown provider: {provider_name}")
    
    def get_provider(self, provider_name: Optional[str] = None) -> AIProvider:
        """Get AI provider instance"""
        provider_name = provider_name or self.current_provider
        
        if not provider_name:
            raise ValueError("No AI provider configured")
        
        if provider_name not in self.providers:
            self.providers[provider_name] = self.initialize_provider(provider_name)
        
        return self.providers[provider_name]
    
    def set_current_provider(self, provider_name: str):
        """Set the current active provider"""
        if provider_name not in self.configs:
            raise ValueError(f"Provider {provider_name} not configured")
        self.current_provider = provider_name
    
    def get_available_providers(self) -> List[str]:
        """Get list of available providers"""
        return list(self.configs.keys())
    
    def get_provider_info(self, provider_name: str) -> Dict[str, Any]:
        """Get information about a specific provider"""
        if provider_name not in self.configs:
            return {}
        
        config = self.configs[provider_name]
        return {
            "name": config.name,
            "provider": config.provider,
            "model": config.model_id,
            "max_tokens": config.max_tokens,
            "temperature": config.temperature
        }
    
    async def generate_response(self, prompt: str, provider_name: Optional[str] = None, **kwargs) -> str:
        """Generate response using specified or current provider"""
        provider = self.get_provider(provider_name)
        return await provider.generate_response(prompt, **kwargs)
    
    async def generate_chat_response(self, messages: List[Dict[str, str]], provider_name: Optional[str] = None, **kwargs) -> str:
        """Generate chat response using specified or current provider"""
        provider = self.get_provider(provider_name)
        return await provider.generate_chat_response(messages, **kwargs)

# Global instance
ai_manager = AIModelManager()