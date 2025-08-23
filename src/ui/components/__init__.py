"""
Mail Project UI Package
======================

This package contains all user interface components for the mail management system.

Components:
- components/: Reusable UI components
- pages/: Streamlit application pages
- utils/: UI utility functions and session management
"""

__version__ = "1.0.0"
__author__ = "Mail Project Team"

# Import main components for easy access
from src.ui.components.email_card import EmailCard
from src.ui.components.priority_dashboard import PriorityDashboard
from src.ui.components.chat_interface import ChatInterface
from src.ui.components.time_filter import TimeFilter

from src.ui.utils.formatting import format_email_content, format_timestamp
from src.ui.utils.session_state import SessionStateManager

__all__ = [
    'EmailCard',
    'PriorityDashboard', 
    'ChatInterface',
    'TimeFilter',
    'format_email_content',
    'format_timestamp',
    'SessionStateManager'
]
