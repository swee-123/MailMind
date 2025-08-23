"""
Main Dashboard Page
==================

The primary dashboard view for email management.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..components.email_card import EmailCard
from ..components.priority_dashboard import PriorityDashboard
from ..components.time_filter import TimeFilter
from ..utils.session_state import SessionStateManager
from ..utils.formatting import format_email_content


class DashboardPage:
    """Main dashboard page class."""
    
    def _init_(self):
        self.email_card = EmailCard()
        self.priority_dashboard = PriorityDashboard()
        self.time_filter = TimeFilter()
        self.session_manager = SessionStateManager()
        
        # Initialize demo data if needed
        self._initialize_demo_data()

    def render(self) -> None:
        """Render the complete dashboard page."""
        st.set_page_config(
            page_title="Email Dashboard",
            page_icon="📧",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Page header
        self._render_header()
        
        # Sidebar
        self._render_sidebar()
        
        # Main content
        view_mode = st.session_state.get('dashboard_view', 'overview')
        
        if view_mode == 'overview':
            self._render_overview()
        elif view_mode == 'list':
            self._render_email_list()
        elif view_mode == 'analytics':
            self._render_analytics()
        elif view_mode == 'timeline':
            self._render_timeline()

    def _render_header(self) -> None:
        """Render page header with navigation."""
        st.title("📧 Email Management Dashboard")
        
        # Quick stats in header
        emails = self.session_manager.get_emails()
        if emails:
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total = len(emails)
                st.metric("Total Emails", total)
                
            with col2:
                unread = len([e for e in emails if e.get('status') == 'unread'])
                st.metric("Unread", unread, delta=f"-{unread}" if unread > 0 else "0")
                
            with col3:
                urgent = len([e for e in emails if e.get('priority') == 'urgent'])
                st.metric("Urgent", urgent, delta="⚠" if urgent > 0 else "✅")
                
            with col4:
                replied = len([e for e in emails if e.get('status') == 'replied'])
                response_rate = (replied / total * 100) if total > 0 else 0
                st.metric("Response Rate", f"{response_rate:.1f}%")
        
        st.divider()

    def _render_sidebar(self) -> None:
        """Render sidebar with navigation and controls."""
        with st.sidebar:
            st.header("🎛 Dashboard Controls")
            
            # View mode selection
            view_mode = st.selectbox(
                "Dashboard View:",
                ["overview", "list", "analytics", "timeline"],
                format_func=lambda x: {
                    "overview": "📊 Overview",
                    "list": "📋 Email List", 
                    "analytics": "📈 Analytics",
                    "timeline": "📅 Timeline"
                }[x],
                key="dashboard_view"
            )
            
            st.divider()
            
            # Quick filters
            self._render_quick_filters()
            
            st.divider()
            
            # Quick actions
            self._render_quick_actions()
            
            st.divider()
            
            # Email management
            self._render_email_management()

    def _render_quick_filters(self) -> None:
        """Render quick filter controls."""
        st.subheader("🔍 Quick Filters")
        
        emails = self.session_manager.get_emails()
        
        # Status filter
        status_options = ['all', 'unread', 'read', 'replied', 'archived']
        selected_status = st.selectbox(
            "Status:",
            status_options,
            key="status_filter"
        )
        
        # Priority filter
        priority_options = ['all', 'urgent', 'high', 'medium', 'low']
        selected_priority = st.selectbox(
            "Priority:",
            priority_options,
            key="priority_filter"
        )
        
        # Time filter (compact)
        filtered_emails = self.time_filter.render_compact(emails, "sidebar_time")
        
        # Apply additional filters
        if selected_status != 'all':
            filtered_emails = [e for e in filtered_emails if e.get('status') == selected_status]
            
        if selected_priority != 'all':
            filtered_emails = [e for e in filtered_emails if e.get('priority') == selected_priority]
        
        # Update session state with filtered emails
        st.session_state['filtered_emails'] = filtered_emails
        
        # Show filter results
        st.caption(f"Showing {len(filtered_emails)} of {len(emails)} emails")

    def _render_quick_actions(self) -> None:
        """Render quick action buttons."""
        st.subheader("⚡ Quick Actions")
        
        if st.button("✉ Compose Email", use_container_width=True):
            st.session_state['active_page'] = 'reply_manager'
            st.session_state['compose_new'] = True
            st.rerun()
        
        if st.button("🔄 Refresh Emails", use_container_width=True):
            self._refresh_emails()
            
        if st.button("📊 Priority Analysis", use_container_width=True):
            st.session_state['dashboard_view'] = 'analytics'
            st.rerun()
            
        if st.button("💬 Open Chat", use_container_width=True):
            st.session_state['active_page'] = 'chat_page'
            st.rerun()

    def _render_email_management(self) -> None:
        """Render email management controls."""
        st.subheader("📁 Email Management")
        
        # Bulk actions
        if st.button("📖 Mark All as Read", use_container_width=True):
            self._mark_all_read()
            
        if st.button("📦 Archive Old Emails", use_container_width=True):
            self._archive_old_emails()
            
        # Export options
        if st.button("📤 Export Emails", use_container_width=True):
            self._export_emails()
            
        # Import emails
        uploaded_file = st.file_uploader(
            "Import Emails:",
            type=['json', 'csv', 'eml'],
            key="email_import"
        )
        
        if uploaded_file:
            if st.button("📥 Import"):
                self._import_emails(uploaded_file)

    def _render_overview(self) -> None:
        """Render overview dashboard."""
        emails = st.session_state.get('filtered_emails', self.session_manager.get_emails())
        
        # Priority dashboard
        self.priority_dashboard.render(emails)
        
        st.divider()
        
        # Recent emails
        st.subheader("📬 Recent Emails")
        recent_emails = sorted(
            emails, 
            key=lambda x: self._get_timestamp(x.get('timestamp')), 
            reverse=True
        )[:5]
        
        for email in recent_emails:
            self.email_card.render_compact(email)

    def _render_email_list(self) -> None:
        """Render email list view."""
        emails = st.session_state.get('filtered_emails', self.session_manager.get_emails())
        
        if not emails:
            st.info("No emails to display. Try adjusting your filters or importing some emails.")
            return
        
        # List controls
        col1, col2, col3 = st.columns([2, 1, 1])
        
        with col1:
            st.subheader(f"📋 Email List ({len(emails)} emails)")
            
        with col2:
            sort_by = st.selectbox(
                "Sort by:",
                ["timestamp", "priority", "sender", "subject"],
                key="email_sort"
            )
            
        with col3:
            sort_order = st.selectbox(
                "Order:",
                ["newest", "oldest"] if sort_by == "timestamp" else ["asc", "desc"],
                key="sort_order"
            )
        
        # Sort emails
        reverse = (sort_order in ["newest", "desc"])
        if sort_by == "timestamp":
            sorted_emails = sorted(
                emails,
                key=lambda x: self._get_timestamp(x.get('timestamp')),
                reverse=reverse
            )
        else:
            sorted_emails = sorted(
                emails,
                key=lambda x: str(x.get(sort_by, '')).lower(),
                reverse=reverse
            )
        
        st.divider()
        
        # Display mode
        display_mode = st.radio(
            "Display Mode:",
            ["Compact", "Detailed"],
            horizontal=True,
            key="list_display_mode"
        )
        
        # Pagination
        emails_per_page = 10
        total_pages = (len(sorted_emails) + emails_per_page - 1) // emails_per_page
        
        if total_pages > 1:
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                current_page = st.selectbox(
                    "Page:",
                    range(1, total_pages + 1),
                    key="email_list_page"
                )
        else:
            current_page = 1
        
        # Calculate page range
        start_idx = (current_page - 1) * emails_per_page
        end_idx = start_idx + emails_per_page
        page_emails = sorted_emails[start_idx:end_idx]
        
        # Display emails
        for i, email in enumerate(page_emails):
            if display_mode == "Compact":
                self.email_card.render_compact(email)
            else:
                with st.expander(f"📧 {email.get('subject', 'No Subject')}", expanded=False):
                    self.email_card.render(email, show_full=False)

    def _render_analytics(self) -> None:
        """Render analytics dashboard."""
        emails = st.session_state.get('filtered_emails', self.session_manager.get_emails())
        
        # Full priority dashboard with analytics
        self.priority_dashboard.render(emails)
        
        st.divider()
        
        # Time analytics
        time_stats = self.time_filter.get_time_statistics(emails)
        
        if time_stats:
            st.subheader("📊 Time Analysis")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.write("*Date Range:*")
                if 'date_range' in time_stats:
                    date_range = time_stats['date_range']
                    st.write(f"• Earliest: {date_range['earliest']}")
                    st.write(f"• Latest: {date_range['latest']}")
                    st.write(f"• Span: {date_range['span_days']} days")
            
            with col2:
                st.write("*Email Volume:*")
                st.write(f"• Total emails: {time_stats['total_emails']}")
                st.write(f"• With timestamps: {time_stats['emails_with_timestamp']}")
                
                if time_stats['date_range']['span_days'] > 0:
                    avg_per_day = time_stats['emails_with_timestamp'] / time_stats['date_range']['span_days']
                    st.write(f"• Average per day: {avg_per_day:.1f}")

    def _render_timeline(self) -> None:
        """Render timeline view."""
        emails = st.session_state.get('filtered_emails', self.session_manager.get_emails())
        
        # Timeline visualization
        self.time_filter.render_timeline_view(emails)

    def _get_timestamp(self, timestamp: Any) -> datetime:
        """Get datetime from timestamp for sorting."""
        if isinstance(timestamp, str):
            try:
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
            except:
                return datetime.min
        elif isinstance(timestamp, datetime):
            return timestamp
        else:
            return datetime.min

    def _initialize_demo_data(self) -> None:
        """Initialize demo email data if none exists."""
        if not self.session_manager.get_emails():
            demo_emails = [
                {
                    'id': '1',
                    'subject': 'Quarterly Report Review',
                    'sender': 'manager@company.com',
                    'recipient': 'user@company.com',
                    'content': 'Please review the attached quarterly report and provide your feedback by Friday.',
                    'timestamp': datetime.now().isoformat(),
                    'priority': 'high',
                    'status': 'unread'
                },
                {
                    'id': '2', 
                    'subject': 'Team Meeting Tomorrow',
                    'sender': 'colleague@company.com',
                    'recipient': 'user@company.com',
                    'content': 'Reminder about our team meeting tomorrow at 10 AM in conference room B.',
                    'timestamp': (datetime.now() - timedelta(hours=2)).isoformat(),
                    'priority': 'medium',
                    'status': 'read'
                },
                {
                    'id': '3',
                    'subject': 'URGENT: Server Maintenance Tonight',
                    'sender': 'it@company.com',
                    'recipient': 'user@company.com', 
                    'content': 'Emergency server maintenance scheduled for tonight from 11 PM to 2 AM. Please save your work.',
                    'timestamp': (datetime.now() - timedelta(hours=1)).isoformat(),
                    'priority': 'urgent',
                    'status': 'unread'
                }
            ]
            
            self.session_manager.set_emails(demo_emails)

    def _refresh_emails(self) -> None:
        """Refresh email data."""
        # In a real application, this would fetch from email server
        st.success("✅ Emails refreshed!")
        st.rerun()

    def _mark_all_read(self) -> None:
        """Mark all filtered emails as read."""
        emails = st.session_state.get('filtered_emails', [])
        updated_count = 0
        
        for email in emails:
            if email.get('status') == 'unread':
                email['status'] = 'read'
                updated_count += 1
        
        self.session_manager.update_emails()
        st.success(f"✅ Marked {updated_count} emails as read!")
        st.rerun()

    def _archive_old_emails(self) -> None:
        """Archive emails older than 30 days."""
        emails = self.session_manager.get_emails()
        cutoff_date = datetime.now() - timedelta(days=30)
        archived_count = 0
        
        for email in emails:
            timestamp = self._get_timestamp(email.get('timestamp'))
            if timestamp < cutoff_date and email.get('status') != 'archived':
                email['status'] = 'archived'
                archived_count += 1
        
        self.session_manager.update_emails()
        st.success(f"✅ Archived {archived_count} old emails!")
        st.rerun()

    def _export_emails(self) -> None:
        """Export emails to file."""
        emails = st.session_state.get('filtered_emails', self.session_manager.get_emails())
        
        # Export format selection in modal/expander
        with st.expander("📤 Export Options", expanded=True):
            export_format = st.selectbox(
                "Export Format:",
                ["JSON", "CSV", "Text"],
                key="export_format"
            )
            
            if st.button("Generate Export"):
                exported_data = self.time_filter.export_filtered_emails(emails, export_format.lower())
                
                st.download_button(
                    label=f"📥 Download {export_format} File",
                    data=exported_data,
                    file_name=f"emails_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.{export_format.lower()}",
                    mime=f"application/{export_format.lower()}" if export_format.lower() == 'json' else "text/plain"
                )

    def _import_emails(self, uploaded_file) -> None:
        """Import emails from uploaded file."""
        try:
            if uploaded_file.type == "application/json":
                import json
                data = json.loads(uploaded_file.read())
                
                if isinstance(data, list):
                    current_emails = self.session_manager.get_emails()
                    # Add unique IDs if missing
                    for i, email in enumerate(data):
                        if 'id' not in email:
                            email['id'] = f"imported_{len(current_emails) + i}"
                    
                    current_emails.extend(data)
                    self.session_manager.set_emails(current_emails)
                    st.success(f"✅ Imported {len(data)} emails!")
                else:
                    st.error("Invalid JSON format. Expected a list of email objects.")
            
            else:
                st.error("Unsupported file format. Please use JSON files.")
                
        except Exception as e:
            st.error(f"Error importing emails: {str(e)}")
        
        st.rerun()