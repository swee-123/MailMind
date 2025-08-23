"""
Mail Project UI Package
======================

This package contains all user interface components for the mail management system.

Components:
- components/: Reusable UI components
- pages/: Streamlit application pages
- utils/: UI utility functions and session management
"""

_version_ = "1.0.0"
_author_ = "Mail Project Team"

# Import main components for easy access
from .components.email_card import EmailCard
from .components.priority_dashboard import PriorityDashboard
from .components.chat_interface import ChatInterface
from .components.time_filter import TimeFilter

from .utils.formatting import format_email_content, format_timestamp
from .utils.session_state import SessionStateManager

_all_ = [
    'EmailCard',
    'PriorityDashboard', 
    'ChatInterface',
    'TimeFilter',
    'format_email_content',
    'format_timestamp',
    'SessionStateManager'
]