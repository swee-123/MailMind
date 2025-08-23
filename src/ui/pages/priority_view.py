import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go

def show_priority_view():
    """Main function to display the priority emails page"""
    
    st.title("📧 Priority Email Dashboard")
    st.markdown("---")
    
    # Initialize session state for sample data if not exists
    if 'priority_emails' not in st.session_state:
        st.session_state.priority_emails = generate_sample_emails()
    
    # Sidebar filters
    with st.sidebar:
        st.header("🔍 Filters")
        
        # Priority level filter
        priority_levels = ["High", "Medium", "Low", "Critical"]
        selected_priorities = st.multiselect(
            "Priority Levels",
            priority_levels,
            default=["High", "Critical"]
        )
        
        # Status filter
        status_options = ["Unread", "Read", "Replied", "Flagged"]
        selected_status = st.multiselect(
            "Email Status",
            status_options,
            default=["Unread", "Flagged"]
        )
        
        # Date range filter
        date_range = st.date_input(
            "Date Range",
            value=[datetime.now() - timedelta(days=7), datetime.now()],
            format="YYYY-MM-DD"
        )
        
        # Sender filter
        st.text_input("Filter by Sender", key="sender_filter")
    
    # Filter the emails based on selections
    filtered_emails = filter_emails(
        st.session_state.priority_emails,
        selected_priorities,
        selected_status,
        date_range,
        st.session_state.get('sender_filter', '')
    )
    
    # Main content area with tabs
    tab1, tab2, tab3 = st.tabs(["📊 Overview", "📋 Email List", "📈 Analytics"])
    
    with tab1:
        show_overview(filtered_emails)
    
    with tab2:
        show_email_list(filtered_emails)
    
    with tab3:
        show_analytics(filtered_emails)


def generate_sample_emails():
    """Generate sample email data for demonstration"""
    import random
    
    senders = [
        "john.doe@company.com", "sarah.smith@client.com", "mike.jones@partner.org",
        "anna.wilson@supplier.net", "david.brown@customer.com", "lisa.garcia@team.com"
    ]
    
    subjects = [
        "Urgent: Project deadline approaching",
        "Meeting reschedule request",
        "Budget approval needed",
        "Client feedback on proposal",
        "System maintenance notification",
        "Invoice payment reminder",
        "New feature request",
        "Performance review discussion"
    ]
    
    priorities = ["Critical", "High", "Medium", "Low"]
    statuses = ["Unread", "Read", "Replied", "Flagged"]
    
    emails = []
    for i in range(50):
        email = {
            'id': f'email_{i+1}',
            'sender': random.choice(senders),
            'subject': random.choice(subjects),
            'priority': random.choice(priorities),
            'status': random.choice(statuses),
            'received_date': datetime.now() - timedelta(days=random.randint(0, 30)),
            'content_preview': f"This is a preview of email content {i+1}..."
        }
        emails.append(email)
    
    return pd.DataFrame(emails)


def filter_emails(emails_df, priorities, statuses, date_range, sender_filter):
    """Filter emails based on user selections"""
    filtered = emails_df.copy()
    
    if priorities:
        filtered = filtered[filtered['priority'].isin(priorities)]
    
    if statuses:
        filtered = filtered[filtered['status'].isin(statuses)]
    
    if len(date_range) == 2:
        start_date, end_date = date_range
        filtered = filtered[
            (filtered['received_date'].dt.date >= start_date) &
            (filtered['received_date'].dt.date <= end_date)
        ]
    
    if sender_filter:
        filtered = filtered[
            filtered['sender'].str.contains(sender_filter, case=False, na=False)
        ]
    
    return filtered


def show_overview(emails_df):
    """Display overview metrics and charts"""
    if emails_df.empty:
        st.warning("No emails match the current filters.")
        return
    
    # Key metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", len(emails_df))
    
    with col2:
        unread_count = len(emails_df[emails_df['status'] == 'Unread'])
        st.metric("Unread", unread_count)
    
    with col3:
        high_priority = len(emails_df[emails_df['priority'].isin(['Critical', 'High'])])
        st.metric("High Priority", high_priority)
    
    with col4:
        flagged_count = len(emails_df[emails_df['status'] == 'Flagged'])
        st.metric("Flagged", flagged_count)
    
    st.markdown("---")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        # Priority distribution pie chart
        priority_counts = emails_df['priority'].value_counts()
        fig_pie = px.pie(
            values=priority_counts.values,
            names=priority_counts.index,
            title="Email Distribution by Priority",
            color_discrete_map={
                'Critical': '#ff4444',
                'High': '#ff8800',
                'Medium': '#ffcc00',
                'Low': '#44ff44'
            }
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Status distribution bar chart
        status_counts = emails_df['status'].value_counts()
        fig_bar = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            title="Email Distribution by Status",
            labels={'x': 'Status', 'y': 'Count'}
        )
        st.plotly_chart(fig_bar, use_container_width=True)
    
    # Email timeline
    st.subheader("📅 Email Timeline")
    emails_by_date = emails_df.groupby(emails_df['received_date'].dt.date).size().reset_index()
    emails_by_date.columns = ['date', 'count']
    
    fig_timeline = px.line(
        emails_by_date,
        x='date',
        y='count',
        title="Emails Received Over Time",
        markers=True
    )
    st.plotly_chart(fig_timeline, use_container_width=True)


def show_email_list(emails_df):
    """Display detailed email list with actions"""
    if emails_df.empty:
        st.warning("No emails match the current filters.")
        return
    
    st.subheader(f"📋 Email List ({len(emails_df)} emails)")
    
    # Bulk actions
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("Mark All as Read"):
            st.success("All emails marked as read!")
    with col2:
        if st.button("Archive Selected"):
            st.success("Selected emails archived!")
    
    # Email list
    for idx, email in emails_df.iterrows():
        with st.expander(
            f"{'🔴' if email['priority'] in ['Critical', 'High'] else '🟡' if email['priority'] == 'Medium' else '🟢'} "
            f"{email['subject']} - {email['sender']}"
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**From:** {email['sender']}")
                st.write(f"**Subject:** {email['subject']}")
                st.write(f"**Received:** {email['received_date'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Priority:** {email['priority']}")
                st.write(f"**Status:** {email['status']}")
                st.write("**Preview:**")
                st.text(email['content_preview'])
            
            with col2:
                st.write("**Actions:**")
                if st.button(f"Reply", key=f"reply_{email['id']}"):
                    st.info("Redirecting to reply manager...")
                if st.button(f"Flag", key=f"flag_{email['id']}"):
                    st.success("Email flagged!")
                if st.button(f"Archive", key=f"archive_{email['id']}"):
                    st.success("Email archived!")


def show_analytics(emails_df):
    """Display advanced analytics and insights"""
    if emails_df.empty:
        st.warning("No emails match the current filters.")
        return
    
    st.subheader("📈 Email Analytics")
    
    # Top senders
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Top Senders**")
        top_senders = emails_df['sender'].value_counts().head(10)
        fig_senders = px.bar(
            x=top_senders.values,
            y=top_senders.index,
            orientation='h',
            title="Most Active Senders"
        )
        st.plotly_chart(fig_senders, use_container_width=True)
    
    with col2:
        st.write("**Response Time Analysis**")
        # Simulate response times for demonstration
        import numpy as np
        response_times = np.random.exponential(2, len(emails_df))
        
        fig_hist = px.histogram(
            x=response_times,
            nbins=20,
            title="Response Time Distribution (hours)",
            labels={'x': 'Hours', 'y': 'Count'}
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Heatmap of email activity
    st.write("**Email Activity Heatmap**")
    emails_df['hour'] = emails_df['received_date'].dt.hour
    emails_df['day_of_week'] = emails_df['received_date'].dt.day_name()
    
    heatmap_data = emails_df.pivot_table(
        values='id',
        index='day_of_week',
        columns='hour',
        aggfunc='count',
        fill_value=0
    )
    
    fig_heatmap = px.imshow(
        heatmap_data,
        title="Email Activity by Day and Hour",
        labels={'x': 'Hour of Day', 'y': 'Day of Week', 'color': 'Email Count'}
    )
    st.plotly_chart(fig_heatmap, use_container_width=True)


if __name__ == "__main__":
    show_priority_view()