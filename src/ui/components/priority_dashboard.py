"""
Priority Dashboard Component
==========================

Visualizes email priorities and provides analytics dashboard.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

class PriorityDashboard:
    """Component for displaying email priority analytics and visualization."""
    
    def _init_(self):
        self.priority_order = ['urgent', 'high', 'medium', 'low']
        self.priority_colors = {
            'urgent': '#cc0000',
            'high': '#ff4444',
            'medium': '#ff8800',
            'low': '#44aa44'
        }
        
        self.status_colors = {
            'unread': '#2196F3',
            'read': '#9E9E9E', 
            'replied': '#4CAF50',
            'archived': '#607D8B'
        }

    def render(self, emails: List[Dict[str, Any]]) -> None:
        """
        Render the complete priority dashboard.
        
        Args:
            emails: List of email dictionaries
        """
        st.header("📊 Email Priority Dashboard")
        
        if not emails:
            st.info("No emails to display in dashboard.")
            return
            
        # Summary metrics
        self._render_summary_metrics(emails)
        
        st.divider()
        
        # Priority distribution charts
        col1, col2 = st.columns(2)
        with col1:
            self._render_priority_pie_chart(emails)
        with col2:
            self._render_status_distribution(emails)
            
        st.divider()
        
        # Time-based analytics
        self._render_time_analytics(emails)
        
        st.divider()
        
        # Priority trends
        self._render_priority_trends(emails)

    def _render_summary_metrics(self, emails: List[Dict[str, Any]]) -> None:
        """Render summary metrics cards."""
        # Calculate metrics
        total_emails = len(emails)
        urgent_count = len([e for e in emails if e.get('priority', '').lower() == 'urgent'])
        high_priority_count = len([e for e in emails if e.get('priority', '').lower() in ['urgent', 'high']])
        unread_count = len([e for e in emails if e.get('status', '').lower() == 'unread'])
        
        # Response rate calculation
        replied_count = len([e for e in emails if e.get('status', '').lower() == 'replied'])
        response_rate = (replied_count / total_emails * 100) if total_emails > 0 else 0
        
        # Display metrics in columns
        col1, col2, col3, col4, col5 = st.columns(5)
        
        with col1:
            st.metric(
                label="📧 Total Emails",
                value=total_emails
            )
            
        with col2:
            st.metric(
                label="🚨 Urgent",
                value=urgent_count,
                delta=f"{urgent_count/total_emails*100:.1f}%" if total_emails > 0 else "0%"
            )
            
        with col3:
            st.metric(
                label="⚡ High Priority",
                value=high_priority_count,
                delta=f"{high_priority_count/total_emails*100:.1f}%" if total_emails > 0 else "0%"
            )
            
        with col4:
            st.metric(
                label="👁 Unread",
                value=unread_count,
                delta=f"-{unread_count}" if unread_count > 0 else "0"
            )
            
        with col5:
            st.metric(
                label="✅ Response Rate",
                value=f"{response_rate:.1f}%",
                delta=f"{replied_count} replied"
            )

    def _render_priority_pie_chart(self, emails: List[Dict[str, Any]]) -> None:
        """Render priority distribution pie chart."""
        st.subheader("Priority Distribution")
        
        # Count emails by priority
        priority_counts = {}
        for email in emails:
            priority = email.get('priority', 'medium').lower()
            priority_counts[priority] = priority_counts.get(priority, 0) + 1
        
        if priority_counts:
            # Create pie chart
            fig = px.pie(
                values=list(priority_counts.values()),
                names=list(priority_counts.keys()),
                color=list(priority_counts.keys()),
                color_discrete_map=self.priority_colors,
                title="Email Priority Breakdown"
            )
            
            fig.update_traces(
                textposition='inside', 
                textinfo='percent+label',
                hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
            )
            
            fig.update_layout(
                showlegend=True,
                height=400,
                font=dict(size=12)
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No priority data available.")

    def _render_status_distribution(self, emails: List[Dict[str, Any]]) -> None:
        """Render email status distribution."""
        st.subheader("Status Distribution")
        
        # Count emails by status
        status_counts = {}
        for email in emails:
            status = email.get('status', 'unread').lower()
            status_counts[status] = status_counts.get(status, 0) + 1
        
        if status_counts:
            # Create horizontal bar chart
            fig = go.Figure(data=[
                go.Bar(
                    y=list(status_counts.keys()),
                    x=list(status_counts.values()),
                    orientation='h',
                    marker_color=[self.status_colors.get(status, '#888888') 
                                for status in status_counts.keys()],
                    text=list(status_counts.values()),
                    textposition='auto'
                )
            ])
            
            fig.update_layout(
                title="Email Status Breakdown",
                xaxis_title="Count",
                yaxis_title="Status",
                height=400,
                showlegend=False
            )
            
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No status data available.")

    def _render_time_analytics(self, emails: List[Dict[str, Any]]) -> None:
        """Render time-based email analytics."""
        st.subheader("⏰ Time Analytics")
        
        # Parse timestamps and create DataFrame
        email_data = []
        for email in emails:
            timestamp = email.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, datetime):
                        dt = timestamp
                    else:
                        continue
                        
                    email_data.append({
                        'datetime': dt,
                        'date': dt.date(),
                        'hour': dt.hour,
                        'day_of_week': dt.strftime('%A'),
                        'priority': email.get('priority', 'medium').lower(),
                        'status': email.get('status', 'unread').lower()
                    })
                except:
                    continue
        
        if not email_data:
            st.info("No timestamp data available for analysis.")
            return
            
        df = pd.DataFrame(email_data)
        
        # Time-based visualizations
        col1, col2 = st.columns(2)
        
        with col1:
            # Emails by hour of day
            hourly_counts = df.groupby('hour').size().reset_index(name='count')
            
            fig = px.line(
                hourly_counts,
                x='hour',
                y='count',
                title='Emails by Hour of Day',
                markers=True
            )
            
            fig.update_layout(
                xaxis_title="Hour (24h format)",
                yaxis_title="Number of Emails",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Emails by day of week
            day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
            daily_counts = df.groupby('day_of_week').size().reindex(day_order, fill_value=0).reset_index(name='count')
            
            fig = px.bar(
                daily_counts,
                x='day_of_week',
                y='count',
                title='Emails by Day of Week'
            )
            
            fig.update_layout(
                xaxis_title="Day of Week",
                yaxis_title="Number of Emails",
                height=300
            )
            
            st.plotly_chart(fig, use_container_width=True)

    def _render_priority_trends(self, emails: List[Dict[str, Any]]) -> None:
        """Render priority trends over time."""
        st.subheader("📈 Priority Trends")
        
        # Parse email data for trends
        trend_data = []
        for email in emails:
            timestamp = email.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, datetime):
                        dt = timestamp
                    else:
                        continue
                        
                    trend_data.append({
                        'date': dt.date(),
                        'priority': email.get('priority', 'medium').lower()
                    })
                except:
                    continue
        
        if not trend_data:
            st.info("No data available for trend analysis.")
            return
            
        df = pd.DataFrame(trend_data)
        
        # Group by date and priority
        trend_df = df.groupby(['date', 'priority']).size().unstack(fill_value=0)
        trend_df = trend_df.reset_index()
        
        # Create stacked area chart
        fig = go.Figure()
        
        for priority in self.priority_order:
            if priority in trend_df.columns:
                fig.add_trace(go.Scatter(
                    x=trend_df['date'],
                    y=trend_df[priority],
                    mode='lines',
                    fill='tonexty' if priority != self.priority_order[0] else 'tozeroy',
                    name=priority.title(),
                    line=dict(color=self.priority_colors[priority]),
                    stackgroup='one'
                ))
        
        fig.update_layout(
            title='Email Priority Trends Over Time',
            xaxis_title='Date',
            yaxis_title='Number of Emails',
            height=400,
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)

    def render_priority_filter(self, emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Render priority filter controls and return filtered emails.
        
        Args:
            emails: List of email dictionaries
            
        Returns:
            Filtered list of emails
        """
        st.subheader("🎯 Priority Filter")
        
        # Priority multiselect
        available_priorities = list(set(
            email.get('priority', 'medium').lower() 
            for email in emails
        ))
        
        selected_priorities = st.multiselect(
            "Filter by Priority:",
            options=available_priorities,
            default=available_priorities,
            key="priority_filter"
        )
        
        # Status multiselect
        available_statuses = list(set(
            email.get('status', 'unread').lower()
            for email in emails
        ))
        
        selected_statuses = st.multiselect(
            "Filter by Status:",
            options=available_statuses,
            default=available_statuses,
            key="status_filter"
        )
        
        # Date range filter
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "From Date:",
                value=datetime.now().date() - timedelta(days=30),
                key="start_date_filter"
            )
        with col2:
            end_date = st.date_input(
                "To Date:",
                value=datetime.now().date(),
                key="end_date_filter"
            )
        
        # Apply filters
        filtered_emails = []
        for email in emails:
            # Check priority filter
            if email.get('priority', 'medium').lower() not in selected_priorities:
                continue
                
            # Check status filter
            if email.get('status', 'unread').lower() not in selected_statuses:
                continue
                
            # Check date filter
            timestamp = email.get('timestamp')
            if timestamp:
                try:
                    if isinstance(timestamp, str):
                        dt = datetime.fromisoformat(timestamp.replace('Z', '+00:00'))
                    elif isinstance(timestamp, datetime):
                        dt = timestamp
                    else:
                        continue
                        
                    email_date = dt.date()
                    if not (start_date <= email_date <= end_date):
                        continue
                except:
                    continue
            
            filtered_emails.append(email)
        
        # Show filter results
        st.info(f"Showing {len(filtered_emails)} of {len(emails)} emails")
        
        return filtered_emails

    def render_quick_stats(self, emails: List[Dict[str, Any]]) -> None:
        """Render quick statistics sidebar."""
        with st.sidebar:
            st.subheader("📊 Quick Stats")
            
            if not emails:
                st.info("No emails to analyze")
                return
            
            # Priority breakdown
            st.write("*Priority Breakdown:*")
            priority_counts = {}
            for email in emails:
                priority = email.get('priority', 'medium').lower()
                priority_counts[priority] = priority_counts.get(priority, 0) + 1
            
            for priority in self.priority_order:
                if priority in priority_counts:
                    count = priority_counts[priority]
                    percentage = (count / len(emails)) * 100
                    st.write(f"• {priority.title()}: {count} ({percentage:.1f}%)")
            
            st.divider()
            
            # Action items
            st.write("*Action Items:*")
            unread_count = len([e for e in emails if e.get('status', '').lower() == 'unread'])
            urgent_count = len([e for e in emails if e.get('priority', '').lower() == 'urgent'])
            
            if urgent_count > 0:
                st.error(f"🚨 {urgent_count} urgent emails need attention!")
            if unread_count > 0:
                st.warning(f"👁 {unread_count} unread emails")
            else:
                st.success("✅ All emails have been read!")
                
            # Weekly goal
            today = datetime.now().date()
            week_start = today - timedelta(days=today.weekday())
            week_emails = [
                e for e in emails 
                if e.get('timestamp') and self._get_email_date(e['timestamp']) >= week_start
            ]
            
            if week_emails:
                st.write(f"*This Week:* {len(week_emails)} emails")

    def _get_email_date(self, timestamp: Any) -> Optional[datetime]:
        """Helper to extract date from timestamp."""
        try:
            if isinstance(timestamp, str):
                return datetime.fromisoformat(timestamp.replace('Z', '+00:00')).date()
            elif isinstance(timestamp, datetime):
                return timestamp.date()
        except:
            pass
        return None