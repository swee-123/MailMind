"""
MAILMIND2.0 Application Package
Main application package initialization
"""

import logging
from .config import get_config, Config
from .logging_config import setup_logging

# Setup logging as early as possible
setup_logging()

# Get logger for this module
logger = logging.getLogger(__name__)

# Application metadata
__version__ = "2.0.0"
__app_name__ = "MAILMIND2.0"
__description__ = "Advanced email management and automation system"

# Initialize configuration
try:
    config = get_config()
    logger.info(f"Starting {__app_name__} v{__version__}")
    logger.info(f"Configuration loaded successfully")
except Exception as e:
    logger.error(f"Failed to load configuration: {e}")
    raise

# Export main components
__all__ = [
    '__version__',
    '__app_name__',
    '__description__',
    'config',
    'Config',
    'get_config',
    'setup_logging'
]