"""
Date utilities - Common date and time manipulation functions
"""

import re
from datetime import datetime, date, timezone, timedelta
from typing import Optional, Union, List
from dateutil import parser
from dateutil.relativedelta import relativedelta


def get_current_timestamp() -> str:
    """Get current timestamp in ISO format."""
    return datetime.now(timezone.utc).isoformat()

def get_current_date() -> str:
    """Get current date in YYYY-MM-DD format."""
    return datetime.now().strftime('%Y-%m-%d')

def format_date(
    date_obj: Union[datetime, date, str], 
    format_str: str = '%Y-%m-%d'
) -> str:
    """
    Format a date object or string to specified format.
    
    Args:
        date_obj: Date object, datetime object, or date string
        format_str: Format string (default: '%Y-%m-%d')
    
    Returns:
        Formatted date string
    """
    if isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    
    if isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    return date_obj.strftime(format_str)

def parse_date(date_string: str, default_date: Optional[datetime] = None) -> datetime:
    """
    Parse a date string into a datetime object.
    
    Args:
        date_string: Date string to parse
        default_date: Default date to use for missing components
    
    Returns:
        Parsed datetime object
    
    Raises:
        ValueError: If date string cannot be parsed
    """
    if not date_string or not isinstance(date_string, str):
        raise ValueError("Date string cannot be empty or None")
    
    try:
        # Use dateutil parser for flexible parsing
        return parser.parse(date_string, default=default_date)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Unable to parse date string '{date_string}': {e}")

def is_valid_date(date_string: str) -> bool:
    """
    Check if a string represents a valid date.
    
    Args:
        date_string: Date string to validate
    
    Returns:
        True if valid date, False otherwise
    """
    try:
        parse_date(date_string)
        return True
    except ValueError:
        return False

def days_between(date1: Union[str, datetime, date], date2: Union[str, datetime, date]) -> int:
    """
    Calculate days between two dates.
    
    Args:
        date1: First date
        date2: Second date
    
    Returns:
        Number of days between dates (positive if date2 > date1)
    """
    if isinstance(date1, str):
        date1 = parse_date(date1).date()
    elif isinstance(date1, datetime):
        date1 = date1.date()
    
    if isinstance(date2, str):
        date2 = parse_date(date2).date()
    elif isinstance(date2, datetime):
        date2 = date2.date()
    
    return (date2 - date1).days

def add_days(date_obj: Union[str, datetime, date], days: int) -> datetime:
    """Add days to a date."""
    if isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    return date_obj + timedelta(days=days)

def add_months(date_obj: Union[str, datetime, date], months: int) -> datetime:
    """Add months to a date."""
    if isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    return date_obj + relativedelta(months=months)

def get_week_start(date_obj: Union[str, datetime, date] = None) -> datetime:
    """Get the start of the week (Monday) for a given date."""
    if date_obj is None:
        date_obj = datetime.now()
    elif isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    days_since_monday = date_obj.weekday()
    return date_obj - timedelta(days=days_since_monday)

def get_month_start(date_obj: Union[str, datetime, date] = None) -> datetime:
    """Get the first day of the month for a given date."""
    if date_obj is None:
        date_obj = datetime.now()
    elif isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    return date_obj.replace(day=1, hour=0, minute=0, second=0, microsecond=0)

def get_year_start(date_obj: Union[str, datetime, date] = None) -> datetime:
    """Get the first day of the year for a given date."""
    if date_obj is None:
        date_obj = datetime.now()
    elif isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    return date_obj.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)

def format_time_ago(date_obj: Union[str, datetime, date]) -> str:
    """
    Format a date as 'time ago' string (e.g., '2 hours ago', '3 days ago').
    
    Args:
        date_obj: Date to format
    
    Returns:
        Human-readable time ago string
    """
    if isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, date) and not isinstance(date_obj, datetime):
        date_obj = datetime.combine(date_obj, datetime.min.time())
    
    now = datetime.now(timezone.utc)
    if date_obj.tzinfo is None:
        date_obj = date_obj.replace(tzinfo=timezone.utc)
    
    diff = now - date_obj
    
    if diff.days > 365:
        years = diff.days // 365
        return f"{years} year{'s' if years != 1 else ''} ago"
    elif diff.days > 30:
        months = diff.days // 30
        return f"{months} month{'s' if months != 1 else ''} ago"
    elif diff.days > 0:
        return f"{diff.days} day{'s' if diff.days != 1 else ''} ago"
    elif diff.seconds > 3600:
        hours = diff.seconds // 3600
        return f"{hours} hour{'s' if hours != 1 else ''} ago"
    elif diff.seconds > 60:
        minutes = diff.seconds // 60
        return f"{minutes} minute{'s' if minutes != 1 else ''} ago"
    else:
        return "just now"

def is_weekend(date_obj: Union[str, datetime, date]) -> bool:
    """Check if a date falls on a weekend."""
    if isinstance(date_obj, str):
        date_obj = parse_date(date_obj)
    elif isinstance(date_obj, datetime):
        date_obj = date_obj.date()
    
    return date_obj.weekday() >= 5  # Saturday = 5, Sunday = 6

def get_business_days(start_date: Union[str, datetime, date], 
                     end_date: Union[str, datetime, date]) -> int:
    """Get number of business days between two dates (excluding weekends)."""
    if isinstance(start_date, str):
        start_date = parse_date(start_date).date()
    elif isinstance(start_date, datetime):
        start_date = start_date.date()
    
    if isinstance(end_date, str):
        end_date = parse_date(end_date).date()
    elif isinstance(end_date, datetime):
        end_date = end_date.date()
    
    business_days = 0
    current_date = start_date
    
    while current_date <= end_date:
        if not is_weekend(current_date):
            business_days += 1
        current_date += timedelta(days=1)
    
    return business_days