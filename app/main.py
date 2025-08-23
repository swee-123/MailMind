"""
MAILMIND2.0 - Streamlit Web Interface
Advanced email management and automation system - Web UI
"""

import streamlit as st
import asyncio
import sys
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Optional
import logging

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from app import get_config, setup_logging
from app.logging_config import get_logger

# Set up logging for Streamlit
setup_logging()
logger = get_logger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="MAILMIND2.0",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

class StreamlitMailMindApp:
    """Streamlit-compatible version of MailMind application"""
    
    def __init__(self):
        self.config = get_config()
        self.initialized = False
    
    def init_session_state(self):
        """Initialize Streamlit session state"""
        if 'app_initialized' not in st.session_state:
            st.session_state.app_initialized = False
        if 'email_data' not in st.session_state:
            st.session_state.email_data = []
        if 'connection_status' not in st.session_state:
            st.session_state.connection_status = "Disconnected"
    
    async def init_components(self):
        """Initialize application components (async version for Streamlit)"""
        try:
            # Initialize without signal handlers
            await self._init_database()
            await self._init_email_connections()
            await self._init_cache()
            
            self.initialized = True
            st.session_state.app_initialized = True
            st.session_state.connection_status = "Connected"
            logger.info("MAILMIND2.0 components initialized for Streamlit")
            
        except Exception as e:
            logger.error(f"Failed to initialize components: {e}")
            st.session_state.connection_status = "Error"
            raise
    
    async def _init_database(self):
        """Initialize database connections"""
        # Placeholder - implement your database initialization
        logger.info("Database initialization (placeholder)")
    
    async def _init_email_connections(self):
        """Initialize email connections"""
        # Placeholder - implement your email connections
        logger.info("Email connections initialization (placeholder)")
    
    async def _init_cache(self):
        """Initialize cache system"""
        # Placeholder - implement your cache initialization
        logger.info("Cache initialization (placeholder)")


def main():
    """Main Streamlit application"""
    
    # Initialize app
    if 'mailmind_app' not in st.session_state:
        st.session_state.mailmind_app = StreamlitMailMindApp()
    
    app = st.session_state.mailmind_app
    app.init_session_state()
    
    # Header
    st.title("📧 MAILMIND2.0")
    st.markdown("*Advanced Email Management & Automation System*")
    
    # Sidebar
    with st.sidebar:
        st.header("System Status")
        
        # Connection status
        status_color = {
            "Connected": "🟢",
            "Disconnected": "🔴",
            "Error": "🟡"
        }
        st.markdown(f"{status_color.get(st.session_state.connection_status, '🔴')} **Status:** {st.session_state.connection_status}")
        
        # Initialize button
        if st.button("Initialize System"):
            with st.spinner("Initializing MAILMIND2.0..."):
                try:
                    # Run async initialization
                    asyncio.run(app.init_components())
                    st.success("System initialized successfully!")
                except Exception as e:
                    st.error(f"Initialization failed: {e}")
        
        st.markdown("---")
        
        # Navigation
        st.header("Navigation")
        page = st.selectbox(
            "Select Page",
            ["Dashboard", "Email Management", "Automation", "Configuration", "Logs"]
        )
    
    # Main content area
    if page == "Dashboard":
        show_dashboard()
    elif page == "Email Management":
        show_email_management()
    elif page == "Automation":
        show_automation()
    elif page == "Configuration":
        show_configuration()
    elif page == "Logs":
        show_logs()


def show_dashboard():
    """Dashboard page"""
    st.header("📊 Dashboard")
    
    # Metrics row
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Emails", "1,234", "↑ 5.6%")
    
    with col2:
        st.metric("Processed Today", "89", "↑ 12")
    
    with col3:
        st.metric("Automation Rules", "15", "→ 0")
    
    with col4:
        st.metric("Response Rate", "94.2%", "↑ 2.1%")
    
    # Charts row
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Email Volume")
        # Create chart data
        chart_data = pd.DataFrame({
            'Date': pd.date_range('2025-08-01', periods=20, freq='D'),
            'Emails': np.random.randint(50, 150, 20)
        })
        st.line_chart(chart_data.set_index('Date'))
    
    with col2:
        st.subheader("🎯 Processing Status")
        # Create status data
        status_data = pd.DataFrame({
            'Status': ['Processed', 'Pending', 'Error'],
            'Count': [850, 45, 15]
        })
        st.bar_chart(status_data.set_index('Status'))
    
    # Recent activity
    st.subheader("📋 Recent Activity")
    
    activity_data = {
        'Time': ['14:30:45', '14:25:12', '14:20:33', '14:15:07', '14:10:21'],
        'Action': ['Email Processed', 'Rule Triggered', 'Auto-Reply Sent', 'Email Classified', 'Connection Check'],
        'Status': ['✅ Success', '✅ Success', '✅ Success', '✅ Success', '✅ Success'],
        'Details': ['invoice@company.com processed', 'Priority rule applied', 'Response to customer query', 'Spam email filtered', 'All connections healthy']
    }
    
    st.dataframe(pd.DataFrame(activity_data), use_container_width=True)


def show_email_management():
    """Email management page"""
    st.header("📧 Email Management")
    
    # Tabs for different email functions
    tab1, tab2, tab3 = st.tabs(["📥 Inbox", "📤 Sent", "🗂️ Filters"])
    
    with tab1:
        st.subheader("Inbox Management")
        
        # Email list - create the data using pandas
        email_data = {
            'From': ['user1@example.com', 'user2@example.com', 'admin@system.com'],
            'Subject': ['Meeting Request', 'Project Update', 'System Notification'],
            'Date': ['2025-08-21 14:30', '2025-08-21 13:45', '2025-08-21 12:15'],
            'Status': ['Unread', 'Read', 'Processed'],
            'Priority': ['High', 'Medium', 'Low']
        }
        
        # Create DataFrame
        df = pd.DataFrame(email_data)
        st.dataframe(df, use_container_width=True)
        
        # Bulk actions
        st.subheader("Bulk Actions")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("Mark All Read"):
                st.success("All emails marked as read")
        
        with col2:
            if st.button("Process All"):
                st.success("All emails queued for processing")
        
        with col3:
            if st.button("Archive Old"):
                st.success("Old emails archived")
    
    with tab2:
        st.subheader("Sent Emails")
        st.info("Sent email tracking will be displayed here")
    
    with tab3:
        st.subheader("Email Filters")
        
        # Add new filter
        with st.expander("➕ Add New Filter"):
            filter_name = st.text_input("Filter Name")
            condition = st.selectbox("Condition", ["Subject contains", "From equals", "Body contains"])
            value = st.text_input("Value")
            action = st.selectbox("Action", ["Move to folder", "Mark as priority", "Auto-reply"])
            
            if st.button("Create Filter"):
                st.success(f"Filter '{filter_name}' created successfully!")
        
        # Existing filters
        st.subheader("Active Filters")
        filter_data = {
            'Name': ['Spam Filter', 'Priority Clients', 'Auto Responses'],
            'Condition': ['Subject contains "SPAM"', 'From in priority list', 'Subject contains "Auto:"'],
            'Action': ['Delete', 'Mark High Priority', 'Send Auto Reply'],
            'Status': ['Active', 'Active', 'Paused']
        }
        st.dataframe(pd.DataFrame(filter_data), use_container_width=True)


def show_automation():
    """Automation page"""
    st.header("🤖 Automation Rules")
    
    # Create new automation rule
    with st.expander("➕ Create New Automation Rule"):
        rule_name = st.text_input("Rule Name")
        
        col1, col2 = st.columns(2)
        with col1:
            trigger = st.selectbox("Trigger", [
                "New Email Received",
                "Keyword Detected", 
                "Time-based",
                "Priority Email"
            ])
        
        with col2:
            action = st.selectbox("Action", [
                "Send Auto-Reply",
                "Forward Email",
                "Create Task",
                "Send Notification"
            ])
        
        conditions = st.text_area("Conditions (JSON format)")
        
        if st.button("Create Automation Rule"):
            st.success(f"Automation rule '{rule_name}' created!")
    
    # Existing rules
    st.subheader("Active Automation Rules")
    
    rules_data = {
        'Rule Name': ['Welcome New Users', 'Priority Support', 'Weekly Reports'],
        'Trigger': ['New Registration Email', 'VIP Customer Email', 'Time: Every Monday'],
        'Action': ['Send Welcome Template', 'Immediate Notification', 'Generate Report'],
        'Status': ['✅ Active', '✅ Active', '⏸️ Paused'],
        'Last Triggered': ['2 hours ago', '30 minutes ago', '3 days ago']
    }
    
    st.dataframe(pd.DataFrame(rules_data), use_container_width=True)
    
    # Rule performance
    st.subheader("📊 Automation Performance")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Rules Executed Today", "47", "↑ 8")
    with col2:
        st.metric("Success Rate", "98.3%", "↑ 0.5%")
    with col3:
        st.metric("Time Saved", "3.2 hours", "↑ 0.8h")


def show_configuration():
    """Configuration page"""
    st.header("⚙️ Configuration")
    
    # Tabs for different config sections
    tab1, tab2, tab3, tab4 = st.tabs(["📧 Email Settings", "🗄️ Database", "🔧 System", "🔐 Security"])
    
    with tab1:
        st.subheader("Email Server Configuration")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.text_input("IMAP Server", value="imap.gmail.com")
            st.number_input("IMAP Port", value=993)
            st.text_input("Username", value="user@example.com")
            st.text_input("Password", type="password")
        
        with col2:
            st.text_input("SMTP Server", value="smtp.gmail.com")
            st.number_input("SMTP Port", value=587)
            st.checkbox("Use TLS", value=True)
            st.checkbox("Use SSL", value=False)
        
        if st.button("Test Email Connection"):
            with st.spinner("Testing connection..."):
                # Simulate connection test
                import time
                time.sleep(2)
                st.success("✅ Email connection successful!")
    
    with tab2:
        st.subheader("Database Configuration")
        st.text_input("Database URL", value="postgresql://localhost:5432/mailmind")
        st.number_input("Connection Pool Size", value=10)
        st.checkbox("Enable Connection Pooling", value=True)
    
    with tab3:
        st.subheader("System Settings")
        st.number_input("Max Concurrent Processes", value=5)
        st.number_input("Email Check Interval (seconds)", value=300)
        st.selectbox("Log Level", ["DEBUG", "INFO", "WARNING", "ERROR"])
    
    with tab4:
        st.subheader("Security Settings")
        st.checkbox("Enable 2FA", value=False)
        st.checkbox("Encrypt Stored Passwords", value=True)
        st.number_input("Session Timeout (minutes)", value=60)


def show_logs():
    """Logs page"""
    st.header("📋 System Logs")
    
    # Log filters
    col1, col2, col3 = st.columns(3)
    
    with col1:
        log_level = st.selectbox("Log Level", ["ALL", "DEBUG", "INFO", "WARNING", "ERROR"])
    
    with col2:
        date_filter = st.date_input("Date")
    
    with col3:
        if st.button("🔄 Refresh Logs"):
            st.rerun()
    
    # Sample log entries
    log_entries = [
        {"Time": "14:35:22", "Level": "INFO", "Module": "email.processor", "Message": "Email processed successfully: invoice@company.com"},
        {"Time": "14:34:15", "Level": "DEBUG", "Module": "database.connection", "Message": "Database query executed in 0.05s"},
        {"Time": "14:33:08", "Level": "WARNING", "Module": "email.connection", "Message": "IMAP connection timeout, retrying..."},
        {"Time": "14:32:45", "Level": "INFO", "Module": "automation.rule", "Message": "Priority rule triggered for VIP customer"},
        {"Time": "14:31:30", "Level": "ERROR", "Module": "email.processor", "Message": "Failed to process email: invalid format"},
    ]
    
    # Display logs
    for entry in log_entries:
        level_color = {
            "DEBUG": "🔍",
            "INFO": "ℹ️",
            "WARNING": "⚠️",
            "ERROR": "❌"
        }
        
        st.text(f"{level_color.get(entry['Level'], '•')} {entry['Time']} [{entry['Level']}] {entry['Module']}: {entry['Message']}")
    
    # Log download
    st.download_button(
        "📥 Download Full Logs",
        data="Sample log file content...",
        file_name=f"mailmind_logs_{date_filter}.txt",
        mime="text/plain"
    )


if __name__ == "__main__":
    main()