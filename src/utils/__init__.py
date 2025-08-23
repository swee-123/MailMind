"""
Utils package - Common utility functions and classes
"""

from .cache_manager import CacheManager
from .date_utils import (
    format_date,
    parse_date,
    get_current_timestamp,
    days_between,
    is_valid_date
)
from .error_handler import (
    ErrorHandler,
    handle_error,
    log_error,
    CustomError
)
from .text_utils import (
    clean_text,
    truncate_text,
    sanitize_filename,
    extract_keywords,
    count_words
)

__version__ = "1.0.0"
__all__ = [
    'CacheManager',
    'format_date',
    'parse_date', 
    'get_current_timestamp',
    'days_between',
    'is_valid_date',
    'ErrorHandler',
    'handle_error',
    'log_error',
    'CustomError',
    'clean_text',
    'truncate_text',
    'sanitize_filename',
    'extract_keywords',
    'count_words'
]