"""
Email Card Component
==================

Displays individual email information in a card format.
"""

import streamlit as st
from datetime import datetime
from typing import Dict, Any, Optional
import re

class EmailCard:
    """Component for displaying email information in a card format."""
    
    def _init_(self):
        self.priority_colors = {
            'high': '#ff4444',
            'medium': '#ff8800', 
            'low': '#44aa44',
            'urgent': '#cc0000'
        }
        
        self.status_colors = {
            'unread': '#2196F3',
            'read': '#9E9E9E',
            'replied': '#4CAF50',
            'archived': '#607D8B'
        }

    def render(self, email_data: Dict[str, Any], show_full: bool = False) -> None:
        """
        Render an email card.
        
        Args:
            email_data: Dictionary containing email information
            show_full: Whether to show full email content or preview
        """
        with st.container():
            # Main card container
            with st.expander(
                f"📧 {email_data.get('subject', 'No Subject')}", 
                expanded=show_full
            ):
                # Header section
                self._render_header(email_data)
                
                # Priority and status badges
                self._render_badges(email_data)
                
                # Email content
                self._render_content(email_data, show_full)
                
                # Action buttons
                self._render_actions(email_data)

    def _render_header(self, email_data: Dict[str, Any]) -> None:
        """Render email header with sender, recipient, and timestamp."""
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            sender = email_data.get('sender', 'Unknown')
            st.write(f"*From:* {sender}")
            
        with col2:
            recipient = email_data.get('recipient', 'Unknown')
            st.write(f"*To:* {recipient}")
            
        with col3:
            timestamp = email_data.get('timestamp')
            if timestamp:
                formatted_time = self._format_timestamp(timestamp)
                st.write(f"*Date:* {formatted_time}")

    def _render_badges(self, email_data: Dict[str, Any]) -> None:
        """Render priority and status badges."""
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            priority = email_data.get('priority', 'medium').lower()
            priority_color = self.priority_colors.get(priority, '#888888')
            st.markdown(
                f'<span style="background-color: {priority_color}; '
                f'color: white; padding: 2px 8px; border-radius: 12px; '
                f'font-size: 12px; font-weight: bold;">'
                f'{priority.upper()} PRIORITY</span>', 
                unsafe_allow_html=True
            )
            
        with col2:
            status = email_data.get('status', 'unread').lower()
            status_color = self.status_colors.get(status, '#888888')
            st.markdown(
                f'<span style="background-color: {status_color}; '
                f'color: white; padding: 2px 8px; border-radius: 12px; '
                f'font-size: 12px; font-weight: bold;">'
                f'{status.upper()}</span>', 
                unsafe_allow_html=True
            )

    def _render_content(self, email_data: Dict[str, Any], show_full: bool) -> None:
        """Render email content with preview or full view."""
        content = email_data.get('content', '')
        
        if not show_full and len(content) > 200:
            # Show preview
            preview = content[:200] + "..."
            st.write("*Preview:*")
            st.write(preview)
            
            if st.button(f"Read Full Email", key=f"expand_{email_data.get('id')}"):
                st.session_state[f"show_full_{email_data.get('id')}"] = True
                st.rerun()
        else:
            # Show full content
            st.write("*Content:*")
            
            # Check if content contains HTML
            if self._is_html(content):
                # Render as HTML (sanitized)
                safe_html = self._sanitize_html(content)
                st.markdown(safe_html, unsafe_allow_html=True)
            else:
                # Render as plain text
                st.write(content)

    def _render_actions(self, email_data: Dict[str, Any]) -> None:
        """Render action buttons for the email."""
        col1, col2, col3, col4, col5 = st.columns(5)
        
        email_id = email_data.get('id')
        
        with col1:
            if st.button("✉ Reply", key=f"reply_{email_id}"):
                st.session_state['reply_to'] = email_id
                st.session_state['active_page'] = 'reply_manager'
                
        with col2:
            if st.button("↪ Forward", key=f"forward_{email_id}"):
                st.session_state['forward_email'] = email_id
                
        with col3:
            current_status = email_data.get('status', 'unread')
            new_status = 'read' if current_status == 'unread' else 'unread'
            if st.button(f"📖 Mark as {new_status.title()}", key=f"status_{email_id}"):
                self._update_email_status(email_id, new_status)
                
        with col4:
            if st.button("📁 Archive", key=f"archive_{email_id}"):
                self._update_email_status(email_id, 'archived')
                
        with col5:
            if st.button("🗑 Delete", key=f"delete_{email_id}"):
                if st.session_state.get(f'confirm_delete_{email_id}'):
                    self._delete_email(email_id)
                    del st.session_state[f'confirm_delete_{email_id}']
                else:
                    st.session_state[f'confirm_delete_{email_id}'] = True
                    st.warning("Click again to confirm deletion")

    def render_compact(self, email_data: Dict[str, Any]) -> None:
        """Render a compact version of the email card for list views."""
        with st.container():
            col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
            
            with col1:
                subject = email_data.get('subject', 'No Subject')
                sender = email_data.get('sender', 'Unknown')
                st.write(f"{subject}")
                st.caption(f"From: {sender}")
                
            with col2:
                content = email_data.get('content', '')
                preview = content[:50] + "..." if len(content) > 50 else content
                st.caption(preview)
                
            with col3:
                priority = email_data.get('priority', 'medium')
                color = self.priority_colors.get(priority.lower(), '#888888')
                st.markdown(
                    f'<div style="background-color: {color}; width: 20px; '
                    f'height: 20px; border-radius: 50%; margin: auto;"></div>',
                    unsafe_allow_html=True
                )
                
            with col4:
                timestamp = email_data.get('timestamp')
                if timestamp:
                    formatted_time = self._format_timestamp(timestamp, compact=True)
                    st.caption(formatted_time)

    def _format_timestamp(self, timestamp: Any, compact: bool = False) -> str:
        """Format timestamp for display."""
        if isinstance(timestamp, str):
            try:
                dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return timestamp
        elif isinstance(timestamp, datetime):
            dt = timestamp
        else:
            return str(timestamp)
            
        now = datetime.now(dt.tzinfo) if dt.tzinfo else datetime.now()
        diff = now - dt
        
        if compact:
            if diff.days > 0:
                return f"{diff.days}d"
            elif diff.seconds > 3600:
                return f"{diff.seconds // 3600}h"
            else:
                return f"{diff.seconds // 60}m"
        else:
            if diff.days > 0:
                return dt.strftime("%Y-%m-%d")
            elif diff.seconds > 3600:
                return dt.strftime("%I:%M %p")
            else:
                return f"{diff.seconds // 60} minutes ago"

    def _is_html(self, content: str) -> bool:
        """Check if content contains HTML tags."""
        return bool(re.search(r'<[^>]+>', content))

    def _sanitize_html(self, html_content: str) -> str:
        """Sanitize HTML content for safe display."""
        # Basic HTML sanitization - remove script tags and dangerous attributes
        import re
        
        # Remove script tags
        html_content = re.sub(r'<script[^>]>.?</script>', '', html_content, flags=re.DOTALL | re.IGNORECASE)
        
        # Remove dangerous attributes
        html_content = re.sub(r'on\w+\s*=\s*["\'][^"\']*["\']', '', html_content, flags=re.IGNORECASE)
        
        return html_content

    def _update_email_status(self, email_id: str, status: str) -> None:
        """Update email status in session state."""
        if 'emails' not in st.session_state:
            st.session_state['emails'] = []
            
        for email in st.session_state['emails']:
            if email.get('id') == email_id:
                email['status'] = status
                break
                
        st.success(f"Email marked as {status}")
        st.rerun()

    def _delete_email(self, email_id: str) -> None:
        """Delete email from session state."""
        if 'emails' not in st.session_state:
            return
            
        st.session_state['emails'] = [
            email for email in st.session_state['emails'] 
            if email.get('id') != email_id
        ]
        
        st.success("Email deleted successfully")
        st.rerun()