"""
Chat Interface Component
======================

Provides an interactive chat interface for email assistance and AI interactions.
"""

import streamlit as st
from datetime import datetime
from typing import List, Dict, Any, Optional, Callable
import json

class ChatInterface:
    """Component for interactive chat functionality with AI assistance."""
    
    def _init_(self):
        self.message_types = {
            'user': {'icon': '👤', 'color': '#1f77b4'},
            'assistant': {'icon': '🤖', 'color': '#2ca02c'},
            'system': {'icon': '⚙', 'color': '#ff7f0e'},
            'error': {'icon': '⚠', 'color': '#d62728'}
        }
        
        self.quick_actions = [
            "📝 Draft a reply",
            "📋 Summarize emails", 
            "🔍 Find important emails",
            "📊 Show email analytics"
        ]