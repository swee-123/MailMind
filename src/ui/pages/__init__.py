"""
Pages Package
=============

Streamlit application pages for the mail management system.
"""

from .dashboard import Dashboard
from .priority_view import PriorityView
from .reply_manager import ReplyManager
from .chat_page import ChatPage

_all_ = [
    'Dashboard',
    'PriorityView',
    'ReplyManager',
    'ChatPage'
]