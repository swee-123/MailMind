"""
Error Handler - Centralized error handling and logging utilities
"""

import logging
import traceback
import sys
from datetime import datetime
from typing import Optional, Dict, Any, Callable, Type
from functools import wraps
from pathlib import Path


class CustomError(Exception):
    """Base custom exception class with additional context."""
    
    def __init__(self, message: str, error_code: Optional[str] = None, 
                 context: Optional[Dict[str, Any]] = None):
        super().__init__(message)
        self.message = message
        self.error_code = error_code or 'GENERIC_ERROR'
        self.context = context or {}
        self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert error to dictionary format."""
        return {
            'error_type': self.__class__.__name__,
            'message': self.message,
            'error_code': self.error_code,
            'context': self.context,
            'timestamp': self.timestamp
        }


class ValidationError(CustomError):
    """Raised when input validation fails."""
    pass


class ConfigurationError(CustomError):
    """Raised when configuration is invalid."""
    pass


class ProcessingError(CustomError):
    """Raised when data processing fails."""
    pass


class ErrorHandler:
    """Centralized error handling and logging."""
    
    def __init__(self, log_file: Optional[str] = None, log_level: int = logging.INFO):
        """
        Initialize error handler.
        
        Args:
            log_file: Path to log file (optional)
            log_level: Logging level
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        # Clear existing handlers
        self.logger.handlers.clear()
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        console_handler.setFormatter(console_formatter)
        self.logger.addHandler(console_handler)
        
        # File handler (if specified)
        if log_file:
            log_path = Path(log_file)
            log_path.parent.mkdir(parents=True, exist_ok=True)
            
            file_handler = logging.FileHandler(log_file)
            file_formatter = logging.Formatter(
                '%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
            )
            file_handler.setFormatter(file_formatter)
            self.logger.addHandler(file_handler)
    
    def log_error(self, error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
        """
        Log an error with context and traceback.
        
        Args:
            error: The exception that occurred
            context: Additional context information
        """
        context = context or {}
        
        error_info = {
            'error_type': type(error).__name__,
            'error_message': str(error),
            'context': context,
            'traceback': traceback.format_exc()
        }
        
        self.logger.error(f"Error occurred: {error_info}")
    
    def log_warning(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log a warning message with context."""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.warning(message)
    
    def log_info(self, message: str, context: Optional[Dict[str, Any]] = None) -> None:
        """Log an info message with context."""
        if context:
            message = f"{message} | Context: {context}"
        self.logger.info(message)
    
    def handle_error(self, error: Exception, 
                    default_return: Any = None,
                    re_raise: bool = False,
                    context: Optional[Dict[str, Any]] = None) -> Any:
        """
        Handle an error by logging it and optionally re-raising.
        
        Args:
            error: The exception that occurred
            default_return: Value to return if not re-raising
            re_raise: Whether to re-raise the exception
            context: Additional context information
        
        Returns:
            default_return value if not re-raising
        
        Raises:
            The original exception if re_raise is True
        """
        self.log_error(error, context)
        
        if re_raise:
            raise error
        
        return default_return


# Global error handler instance
_global_error_handler = ErrorHandler()

def get_error_handler() -> ErrorHandler:
    """Get the global error handler instance."""
    return _global_error_handler

def log_error(error: Exception, context: Optional[Dict[str, Any]] = None) -> None:
    """Log an error using the global error handler."""
    _global_error_handler.log_error(error, context)

def handle_error(error: Exception, 
                default_return: Any = None,
                re_raise: bool = False,
                context: Optional[Dict[str, Any]] = None) -> Any:
    """Handle an error using the global error handler."""
    return _global_error_handler.handle_error(error, default_return, re_raise, context)

def error_handler(default_return: Any = None, 
                 log_errors: bool = True,
                 re_raise: bool = False,
                 context: Optional[Dict[str, Any]] = None):
    """
    Decorator for automatic error handling.
    
    Args:
        default_return: Value to return if error occurs
        log_errors: Whether to log errors
        re_raise: Whether to re-raise exceptions
        context: Additional context for logging
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_errors:
                    func_context = {
                        'function': func.__name__,
                        'args': str(args)[:200],  # Limit string length
                        'kwargs': str(kwargs)[:200]
                    }
                    if context:
                        func_context.update(context)
                    
                    log_error(e, func_context)
                
                if re_raise:
                    raise e
                
                return default_return
        
        return wrapper
    return decorator

def safe_execute(func: Callable, *args, 
                default_return: Any = None,
                log_errors: bool = True,
                context: Optional[Dict[str, Any]] = None, **kwargs) -> Any:
    """
    Safely execute a function with error handling.
    
    Args:
        func: Function to execute
        *args: Positional arguments for the function
        default_return: Value to return if error occurs
        log_errors: Whether to log errors
        context: Additional context for logging
        **kwargs: Keyword arguments for the function
    
    Returns:
        Function result or default_return if error occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        if log_errors:
            func_context = {
                'function': func.__name__,
                'args': str(args)[:200],
                'kwargs': str(kwargs)[:200]
            }
            if context:
                func_context.update(context)
            
            log_error(e, func_context)
        
        return default_return

def try_convert(value: Any, target_type: Type, 
               default: Any = None,
               log_errors: bool = False) -> Any:
    """
    Safely convert a value to target type.
    
    Args:
        value: Value to convert
        target_type: Target type to convert to
        default: Default value if conversion fails
        log_errors: Whether to log conversion errors
    
    Returns:
        Converted value or default if conversion fails
    """
    try:
        return target_type(value)
    except (ValueError, TypeError, AttributeError) as e:
        if log_errors:
            log_error(e, {
                'value': str(value)[:100],
                'target_type': target_type.__name__,
                'operation': 'type_conversion'
            })
        return default