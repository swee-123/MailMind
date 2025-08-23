"""
UI Utilities Package

This package provides utilities for UI operations including:
- Markdown and display formatting
- Session state management
- Common UI helper functions
"""

from .formatting import (
    format_markdown,
    create_code_block,
    format_json,
    create_collapsible_section,
    format_table,
    sanitize_html,
    truncate_text,
    format_timestamp,
    create_badge,
    format_error_message
)

from .session_state import (
    SessionStateManager,
    get_session_value,
    set_session_value,
    clear_session,
    session_exists,
    init_session_defaults,
    update_session_value,
    get_or_create_session_list,
    session_counter
)

__version__ = "1.0.0"
__author__ = "Your Name"

__all__ = [
    # Formatting utilities
    "format_markdown",
    "create_code_block", 
    "format_json",
    "create_collapsible_section",
    "format_table",
    "sanitize_html",
    "truncate_text",
    "format_timestamp",
    "create_badge",
    "format_error_message",
    
    # Session state utilities
    "SessionStateManager",
    "get_session_value",
    "set_session_value", 
    "clear_session",
    "session_exists",
    "init_session_defaults",
    "update_session_value",
    "get_or_create_session_list",
    "session_counter"
]