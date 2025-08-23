# src/auth/__init__.py
"""
Authentication module for Gmail integration.
Provides OAuth2 authentication and utilities for Gmail API access.
"""

from .gmail_auth import GmailAuthenticator
from .auth_utils import (
    save_credentials,
    load_credentials,
    is_token_valid,
    refresh_token_if_needed,
    revoke_credentials
)

__all__ = [
    'GmailAuthenticator',
    'save_credentials',
    'load_credentials',
    'is_token_valid',
    'refresh_token_if_needed',
    'revoke_credentials'
]
