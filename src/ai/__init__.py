# ai/__init__.py
"""MailMind AI/ML Module for intelligent email processing"""

from .groq_client import GroqClient
from .prompt_manager import PromptManager
from .response_generator import ResponseGenerator
from .summarizer import EmailSummarizer

__all__ = ['GroqClient', 'PromptManager', 'ResponseGenerator', 'EmailSummarizer']
