# src/auth/auth_utils.py
"""
Authentication utilities for credential management and validation.
Provides helper functions for working with OAuth2 credentials.
"""

import os
import json
import logging
from datetime import datetime, timezone
from typing import Optional, Dict, Any, Union
from google.oauth2.credentials import Credentials
from google.auth.transport.requests import Request

logger = logging.getLogger(__name__)


def save_credentials(credentials: Credentials, 
                    file_path: str = 'token.json') -> bool:
    """
    Save OAuth2 credentials to file.
    
    Args:
        credentials: Google OAuth2 credentials object
        file_path: Path to save credentials file
        
    Returns:
        bool: True if saved successfully, False otherwise
    """
    try:
        with open(file_path, 'w') as f:
            f.write(credentials.to_json())
        logger.info(f"Credentials saved to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save credentials: {str(e)}")
        return False


def load_credentials(file_path: str = 'token.json',
                    scopes: Optional[list] = None) -> Optional[Credentials]:
    """
    Load OAuth2 credentials from file.
    
    Args:
        file_path: Path to credentials file
        scopes: Required OAuth2 scopes
        
    Returns:
        Credentials object or None if failed to load
    """
    try:
        if not os.path.exists(file_path):
            logger.warning(f"Credentials file not found: {file_path}")
            return None
        
        creds = Credentials.from_authorized_user_file(file_path, scopes)
        logger.info(f"Credentials loaded from {file_path}")
        return creds
        
    except Exception as e:
        logger.error(f"Failed to load credentials: {str(e)}")
        return None


def is_token_valid(credentials: Optional[Credentials]) -> bool:
    """
    Check if OAuth2 token is valid and not expired.
    
    Args:
        credentials: Google OAuth2 credentials object
        
    Returns:
        bool: True if token is valid, False otherwise
    """
    if not credentials:
        return False
    
    # Check if credentials are valid
    if not credentials.valid:
        # If expired but has refresh token, it can be refreshed
        if credentials.expired and credentials.refresh_token:
            logger.info("Token expired but can be refreshed")
            return True
        else:
            logger.warning("Token invalid and cannot be refreshed")
            return False
    
    logger.debug("Token is valid")
    return True


def refresh_token_if_needed(credentials: Credentials) -> bool:
    """
    Refresh OAuth2 token if expired.
    
    Args:
        credentials: Google OAuth2 credentials object
        
    Returns:
        bool: True if token is valid (refreshed if needed), False otherwise
    """
    try:
        if not credentials:
            logger.error("No credentials provided")
            return False
        
        if credentials.expired and credentials.refresh_token:
            logger.info("Refreshing expired token")
            credentials.refresh(Request())
            logger.info("Token refreshed successfully")
            return True
        elif credentials.valid:
            logger.debug("Token is already valid")
            return True
        else:
            logger.error("Token cannot be refreshed")
            return False
            
    except Exception as e:
        logger.error(f"Failed to refresh token: {str(e)}")
        return False


def revoke_credentials(credentials: Optional[Credentials],
                      token_file: str = 'token.json') -> bool:
    """
    Revoke OAuth2 credentials and remove token file.
    
    Args:
        credentials: Google OAuth2 credentials object
        token_file: Path to token file to remove
        
    Returns:
        bool: True if revoked successfully, False otherwise
    """
    try:
        success = True
        
        # Revoke the token if credentials exist and are valid
        if credentials and credentials.valid:
            try:
                credentials.revoke(Request())
                logger.info("OAuth2 token revoked successfully")
            except Exception as e:
                logger.error(f"Failed to revoke token: {str(e)}")
                success = False
        
        # Remove token file
        if os.path.exists(token_file):
            try:
                os.remove(token_file)
                logger.info(f"Token file removed: {token_file}")
            except Exception as e:
                logger.error(f"Failed to remove token file: {str(e)}")
                success = False
        
        return success
        
    except Exception as e:
        logger.error(f"Failed to revoke credentials: {str(e)}")
        return False


def get_token_info(credentials: Optional[Credentials]) -> Dict[str, Any]:
    """
    Get detailed information about OAuth2 token.
    
    Args:
        credentials: Google OAuth2 credentials object
        
    Returns:
        Dict containing token information
    """
    if not credentials:
        return {
            'valid': False,
            'expired': None,
            'expiry': None,
            'has_refresh_token': False,
            'scopes': None
        }
    
    try:
        expiry_str = None
        if credentials.expiry:
            expiry_str = credentials.expiry.isoformat()
        
        return {
            'valid': credentials.valid,
            'expired': credentials.expired,
            'expiry': expiry_str,
            'has_refresh_token': bool(credentials.refresh_token),
            'scopes': getattr(credentials, '_scopes', None),
            'client_id': getattr(credentials, '_client_id', None)
        }
        
    except Exception as e:
        logger.error(f"Failed to get token info: {str(e)}")
        return {'error': str(e)}


def validate_credentials_file(file_path: str) -> bool:
    """
    Validate OAuth2 credentials file format.
    
    Args:
        file_path: Path to credentials file
        
    Returns:
        bool: True if file is valid, False otherwise
    """
    try:
        if not os.path.exists(file_path):
            logger.error(f"Credentials file not found: {file_path}")
            return False
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        # Check for required fields in OAuth2 credentials
        required_fields = ['installed', 'web']
        if not any(field in data for field in required_fields):
            logger.error("Invalid credentials file format")
            return False
        
        # Check for client configuration
        config = data.get('installed') or data.get('web')
        if not config:
            logger.error("No client configuration found")
            return False
        
        required_config_fields = ['client_id', 'client_secret', 'auth_uri', 'token_uri']
        missing_fields = [field for field in required_config_fields if field not in config]
        
        if missing_fields:
            logger.error(f"Missing required fields: {missing_fields}")
            return False
        
        logger.info("Credentials file validation successful")
        return True
        
    except json.JSONDecodeError:
        logger.error("Invalid JSON in credentials file")
        return False
    except Exception as e:
        logger.error(f"Failed to validate credentials file: {str(e)}")
        return False
