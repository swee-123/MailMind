"""
Time Filter Component
===================

Provides time-based filtering controls for emails.
"""

import streamlit as st
from datetime import datetime, timedelta, date
from typing import List, Dict, Any, Tuple, Optional
import calendar

class TimeFilter:
    """Component for time-based email filtering."""
    
    def _init_(self):
        self.preset_ranges = {
            "Today": lambda: (datetime.now().date(), datetime.now().date()),
            "Yesterday": lambda: (
                datetime.now().date() - timedelta(days=1),
                datetime.now().date() - timedelta(days=1)
            ),
            "Last 7 days": lambda: (
                datetime.now().date() - timedelta(days=7),
                datetime.now().date()
            ),
            "Last 30 days": lambda: (
                datetime.now().date() - timedelta(days=30),
                datetime.now().date()
            ),
            "This week": lambda: self._get_week_range(),
            "This month": lambda: self._get_month_range(),
            "Last month": lambda: self._get_last_month_range(),
            "This year": lambda: self._get_year_range(),
        }

    def render(self, emails: List[Dict[str, Any]], key_prefix: str = "time_filter") -> List[Dict[str, Any]]:
        """
        Render time filter controls and return filtered emails.
        
        Args:
            emails: List of email dictionaries
            key_prefix: Prefix for streamlit component keys
            
        Returns:
            Filtered list of emails
        """
        st.subheader("🕐 Time Filter")
        
        # Filter type selection
        filter_type = st.radio(
            "Filter Type:",
            ["Quick Presets", "Custom Range", "Advanced"],
            horizontal=True,
            key=f"{key_prefix}_filter_type"
        )
        
        if filter_type == "Quick Presets":
            return self._render_preset_filter(emails, key_prefix)
        elif filter_type == "Custom Range":
            return self._render_custom_range_filter(emails, key_prefix)
        else:
            return self._render_advanced_filter(emails, key_prefix)

    def _render_preset_filter(self, emails: List[Dict[str, Any]], key_prefix: str) -> List[Dict[str, Any]]:
        """Render preset time range filter."""
        selected_preset = st.selectbox(
            "Select Time Range:",
            list(self.preset_ranges.keys()),
            key=f"{key_prefix}_preset"
        )
        
        # Get date range for selected preset
        start_date, end_date = self.preset_ranges[selected_preset]()
        
        # Show selected range
        st.info(f"📅 Showing emails from {start_date} to {end_date}")
        
        # Apply filter
        filtered_emails = self._filter_emails_by_date_range(emails, start_date, end_date)
        
        # Show results summary
        self._show_filter_results(emails, filtered_emails, f"{selected_preset}")
        
        return filtered_emails

    def _render_custom_range_filter(self, emails: List[Dict[str, Any]], key_prefix: str) -> List[Dict[str, Any]]:
        """Render custom date range filter."""
        col1, col2 = st.columns(2)
        
        # Default to last 30 days
        default_start = datetime.now().date() - timedelta(days=30)
        default_end = datetime.now().date()
        
        with col1:
            start_date = st.date_input(
                "From Date:",
                value=default_start,
                key=f"{key_prefix}_start_date"
            )
            
        with col2:
            end_date = st.date_input(
                "To Date:",
                value=default_end,
                key=f"{key_prefix}_end_date"
            )
        
        # Validate date range
        if start_date > end_date:
            st.error("Start date must be before end date!")
            return emails
        
        # Time of day filter
        include_time = st.checkbox(
            "Include specific time filtering",
            key=f"{key_prefix}_include_time"
        )
        
        if include_time:
            col1, col2 = st.columns(2)
            with col1:
                start_time = st.time_input(
                    "From Time:",
                    value=datetime.min.time(),
                    key=f"{key_prefix}_start_time"
                )
            with col2:
                end_time = st.time_input(
                    "To Time:",
                    value=datetime.max.time().replace(microsecond=0),
                    key=f"{key_prefix}_end_time"
                )
        else:
            start_time = end_time = None
        
        # Apply filter
        filtered_emails = self._filter_emails_by_date_range(
            emails, start_date, end_date, start_time, end_time
        )
        
        # Show results
        range_text = f"{start_date} to {end_date}"
        if include_time:
            range_text += f" ({start_time} - {end_time})"
        
        self._show_filter_results(emails, filtered_emails, range_text)
        
        return filtered_emails

    def _render_advanced_filter(self, emails: List[Dict[str, Any]], key_prefix: str) -> List[Dict[str, Any]]:
        """Render advanced time filtering options."""
        st.write("*Advanced Time Filtering*")
        
        # Multiple filter criteria
        filter_criteria = []
        
        # Day of week filter
        if st.checkbox("Filter by day of week", key=f"{key_prefix}_dow_enable"):
            days_of_week = [
                'Monday', 'Tuesday', 'Wednesday', 'Thursday', 
                'Friday', 'Saturday', 'Sunday'
            ]
            selected_days = st.multiselect(
                "Select days:",
                days_of_week,
                default=days_of_week[:5],  # Weekdays by default
                key=f"{key_prefix}_days"
            )
            filter_criteria.append(('day_of_week', selected_days))
        
        # Hour range filter
        if st.checkbox("Filter by hour of day", key=f"{key_prefix}_hour_enable"):
            hour_range = st.slider(
                "Hour range (24-hour format):",
                0, 23, (9, 17),
                key=f"{key_prefix}_hours"
            )
            filter_criteria.append(('hour_range', hour_range))
        
        # Relative time filter
        if st.checkbox("Relative time filter", key=f"{key_prefix}_relative_enable"):
            col1, col2 = st.columns(2)
            with col1:
                time_value = st.number_input(
                    "Time value:",
                    min_value=1,
                    value=7,
                    key=f"{key_prefix}_relative_value"
                )
            with col2:
                time_unit = st.selectbox(
                    "Time unit:",
                    ["hours", "days", "weeks", "months"],
                    key=f"{key_prefix}_relative_unit"
                )
            
            direction = st.radio(
                "Direction:",
                ["Last", "Next"],
                key=f"{key_prefix}_relative_direction"
            )
            
            filter_criteria.append(('relative', (time_value, time_unit, direction)))
        
        # Apply advanced filters
        filtered_emails = self._apply_advanced_filters(emails, filter_criteria)
        
        # Show results
        criteria_text = ", ".join([f"{criteria[0]}: {criteria[1]}" for criteria in filter_criteria])
        self._show_filter_results(emails, filtered_emails, f"Advanced ({criteria_text})")
        
        return filtered_emails

    def _filter_emails_by_date_range(self, 
                                   emails: List[Dict[str, Any]], 
                                   start_date: date, 
                                   end_date: date,
                                   start_time: Optional[datetime] = None,
                                   end_time: Optional[datetime] = None) -> List[Dict[str, Any]]:
        """Filter emails by date range."""
        filtered_emails = []
        
        for email in emails:
            email_datetime = self._extract_datetime(email.get('timestamp'))
            if not email_datetime:
                continue
            
            email_date = email_datetime.date()
            
            # Check date range
            if not (start_date <= email_date <= end_date):
                continue
                
            # Check time range if specified
            if start_time and end_time:
                email_time = email_datetime.time()
                if not (start_time <= email_time <= end_time):
                    continue
            
            filtered_emails.append(email)
        
        return filtered_emails

    def _apply_advanced_filters(self, 
                              emails: List[Dict[str, Any]], 
                              filter_criteria: List[Tuple]) -> List[Dict[str, Any]]:
        """Apply advanced filter criteria."""
        filtered_emails = []
        
        for email in emails:
            email_datetime = self._extract_datetime(email.get('timestamp'))
            if not email_datetime:
                continue
            
            # Check all criteria
            passes_all_filters = True
            
            for criteria_type, criteria_value in filter_criteria:
                if criteria_type == 'day_of_week':
                    day_name = email_datetime.strftime('%A')
                    if day_name not in criteria_value:
                        passes_all_filters = False
                        break
                        
                elif criteria_type == 'hour_range':
                    email_hour = email_datetime.hour
                    start_hour, end_hour = criteria_value
                    if not (start_hour <= email_hour <= end_hour):
                        passes_all_filters = False
                        break
                        
                elif criteria_type == 'relative':
                    time_value, time_unit, direction = criteria_value
                    now = datetime.now()
                    
                    if time_unit == 'hours':
                        delta = timedelta(hours=time_value)
                    elif time_unit == 'days':
                        delta = timedelta(days=time_value)
                    elif time_unit == 'weeks':
                        delta = timedelta(weeks=time_value)
                    elif time_unit == 'months':
                        delta = timedelta(days=time_value * 30)  # Approximate
                    
                    if direction == 'Last':
                        cutoff_time = now - delta
                        if email_datetime < cutoff_time:
                            passes_all_filters = False
                            break
                    else:  # Next
                        cutoff_time = now + delta