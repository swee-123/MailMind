"""
Logging configuration for MAILMIND2.0
Centralized logging setup with file rotation, structured logging, and performance monitoring
"""

import os
import sys
import logging
import logging.config
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional
import json
from logging.handlers import RotatingFileHandler, TimedRotatingFileHandler


class JSONFormatter(logging.Formatter):
    """Custom JSON formatter for structured logging"""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON"""
        log_entry = {
            'timestamp': datetime.fromtimestamp(record.created).isoformat(),
            'level': record.levelname,
            'logger': record.name,
            'message': record.getMessage(),
            'module': record.module,
            'function': record.funcName,
            'line': record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_entry['exception'] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ['name', 'msg', 'args', 'levelname', 'levelno', 'pathname', 
                          'filename', 'module', 'lineno', 'funcName', 'created', 
                          'msecs', 'relativeCreated', 'thread', 'threadName', 
                          'processName', 'process', 'stack_info', 'exc_info', 'exc_text']:
                log_entry[key] = value
        
        return json.dumps(log_entry)


class ColoredFormatter(logging.Formatter):
    """Colored console formatter for better readability"""
    
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }
    
    def format(self, record: logging.LogRecord) -> str:
        """Format with colors for console output"""
        if not sys.stdout.isatty():
            # No colors if not a terminal
            return super().format(record)
        
        color = self.COLORS.get(record.levelname, self.COLORS['RESET'])
        reset = self.COLORS['RESET']
        
        # Color the level name
        record.levelname = f"{color}{record.levelname}{reset}"
        
        return super().format(record)


class PerformanceFilter(logging.Filter):
    """Filter to add performance metrics to log records"""
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Add performance information to log records"""
        # Add process and thread info
        record.process_name = record.processName
        record.thread_name = record.threadName
        
        return True


class SensitiveDataFilter(logging.Filter):
    """Filter to remove sensitive data from logs"""
    
    SENSITIVE_PATTERNS = [
        'password', 'token', 'key', 'secret', 'credential',
        'authorization', 'auth', 'session'
    ]
    
    def filter(self, record: logging.LogRecord) -> bool:
        """Remove sensitive data from log messages"""
        message = record.getMessage().lower()
        
        # Check if message contains sensitive data
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message:
                # Replace sensitive values with asterisks
                record.msg = self._sanitize_message(record.msg)
                if record.args:
                    record.args = tuple(
                        self._sanitize_value(arg) for arg in record.args
                    )
                break
        
        return True
    
    def _sanitize_message(self, message: str) -> str:
        """Sanitize message string"""
        for pattern in self.SENSITIVE_PATTERNS:
            if pattern in message.lower():
                # Simple replacement - could be more sophisticated
                return message.replace(str(message), "[REDACTED]")
        return message
    
    def _sanitize_value(self, value: Any) -> Any:
        """Sanitize individual values"""
        if isinstance(value, str) and any(p in value.lower() for p in self.SENSITIVE_PATTERNS):
            return "[REDACTED]"
        return value


def setup_logging(config: Optional[Dict[str, Any]] = None) -> None:
    """
    Setup logging configuration for the application
    
    Args:
        config: Optional logging configuration dictionary
    """
    if config is None:
        config = get_default_logging_config()
    
    # Create log directory if it doesn't exist
    log_file = config.get('handlers', {}).get('file', {}).get('filename')
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
    
    # Apply logging configuration
    logging.config.dictConfig(config)
    
    # Log startup message
    logger = logging.getLogger(__name__)
    logger.info("Logging system initialized successfully")


def get_default_logging_config() -> Dict[str, Any]:
    """Get default logging configuration"""
    
    # Get configuration from environment or use defaults
    log_level = os.getenv('LOG_LEVEL', 'INFO').upper()
    log_file = os.getenv('LOG_FILE', 'logs/mailmind.log')
    log_format = os.getenv('LOG_FORMAT', 'detailed')
    enable_json = os.getenv('LOG_JSON', 'false').lower() == 'true'
    enable_colors = os.getenv('LOG_COLORS', 'true').lower() == 'true'
    
    # Base formatters
    formatters = {
        'standard': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'detailed': {
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        },
        'json': {
            '()': JSONFormatter,
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    }
    
    # Add colored formatter if enabled
    if enable_colors:
        formatters['colored'] = {
            '()': ColoredFormatter,
            'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            'datefmt': '%Y-%m-%d %H:%M:%S'
        }
    
    # Filters
    filters = {
        'performance': {
            '()': PerformanceFilter
        },
        'sensitive': {
            '()': SensitiveDataFilter
        }
    }
    
    # Handlers
    handlers = {
        'console': {
            'class': 'logging.StreamHandler',
            'level': log_level,
            'formatter': 'colored' if enable_colors else 'standard',
            'filters': ['sensitive'],
            'stream': 'ext://sys.stdout'
        },
        'file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': log_level,
            'formatter': 'json' if enable_json else log_format,
            'filters': ['performance', 'sensitive'],
            'filename': log_file,
            'maxBytes': 10485760,  # 10MB
            'backupCount': 5,
            'encoding': 'utf8'
        },
        'error_file': {
            'class': 'logging.handlers.RotatingFileHandler',
            'level': 'ERROR',
            'formatter': 'json' if enable_json else 'detailed',
            'filters': ['performance', 'sensitive'],
            'filename': log_file.replace('.log', '_error.log'),
            'maxBytes': 5242880,  # 5MB
            'backupCount': 3,
            'encoding': 'utf8'
        }
    }
    
    # Loggers configuration
    loggers = {
        '': {  # Root logger
            'level': log_level,
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'mailmind': {
            'level': log_level,
            'handlers': ['console', 'file', 'error_file'],
            'propagate': False
        },
        'app': {
            'level': log_level,
            'handlers': ['console', 'file'],
            'propagate': False
        },
        'sqlalchemy.engine': {
            'level': 'WARNING',
            'handlers': ['file'],
            'propagate': False
        },
        'urllib3': {
            'level': 'WARNING',
            'handlers': ['file'],
            'propagate': False
        }
    }
    
    return {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': formatters,
        'filters': filters,
        'handlers': handlers,
        'loggers': loggers
    }


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the specified name
    
    Args:
        name: Logger name (usually __name__)
    
    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)


def set_log_level(level: str, logger_name: str = '') -> None:
    """
    Set logging level for a specific logger
    
    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        logger_name: Name of logger (empty for root logger)
    """
    logger = logging.getLogger(logger_name)
    logger.setLevel(getattr(logging, level.upper()))


def add_file_handler(logger_name: str, filename: str, 
                    level: str = 'INFO', 
                    format_type: str = 'detailed') -> None:
    """
    Add a file handler to a specific logger
    
    Args:
        logger_name: Name of the logger
        filename: Path to log file
        level: Log level for this handler
        format_type: Format type ('standard', 'detailed', 'json')
    """
    logger = logging.getLogger(logger_name)
    
    # Create directory if needed
    Path(filename).parent.mkdir(parents=True, exist_ok=True)
    
    # Create handler
    handler = RotatingFileHandler(
        filename=filename,
        maxBytes=10485760,  # 10MB
        backupCount=5,
        encoding='utf8'
    )
    
    # Set level
    handler.setLevel(getattr(logging, level.upper()))
    
    # Set formatter
    if format_type == 'json':
        formatter = JSONFormatter()
    elif format_type == 'detailed':
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(module)s:%(funcName)s:%(lineno)d - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    else:
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
    
    handler.setFormatter(formatter)
    
    # Add filters
    handler.addFilter(SensitiveDataFilter())
    
    # Add handler to logger
    logger.addHandler(handler)


def configure_third_party_loggers() -> None:
    """Configure logging for third-party libraries"""
    
    # Reduce verbosity of common third-party loggers
    noisy_loggers = [
        'urllib3.connectionpool',
        'requests.packages.urllib3',
        'sqlalchemy.pool',
        'sqlalchemy.dialects',
        'asyncio'
    ]
    
    for logger_name in noisy_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


class LogContext:
    """Context manager for adding context to log messages"""
    
    def __init__(self, logger: logging.Logger, **context):
        self.logger = logger
        self.context = context
        self.old_factory = None
    
    def __enter__(self):
        old_factory = logging.getLogRecordFactory()
        
        def record_factory(*args, **kwargs):
            record = old_factory(*args, **kwargs)
            for key, value in self.context.items():
                setattr(record, key, value)
            return record
        
        logging.setLogRecordFactory(record_factory)
        self.old_factory = old_factory
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        logging.setLogRecordFactory(self.old_factory)


# Initialize logging on module import
if not logging.getLogger().handlers:
    setup_logging()