# src/auth/gmail_auth.py
"""
Gmail OAuth2 authentication implementation.
Handles the complete OAuth2 flow for Gmail API access.
"""

import os
import json
import logging
from typing import Optional, Dict, Any, List
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError

logger = logging.getLogger(__name__)


class GmailAuthenticator:
    """Handles Gmail OAuth2 authentication and API service creation."""
    
    # Gmail API scopes
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify',
        'https://www.googleapis.com/auth/gmail.compose'
    ]
    
    def __init__(self, credentials_file: str = 'credentials.json', 
                 token_file: str = 'token.json'):
        """
        Initialize the Gmail authenticator.
        
        Args:
            credentials_file: Path to OAuth2 credentials file from Google Console
            token_file: Path to store/load user token
        """
        self.credentials_file = credentials_file
        self.token_file = token_file
        self.creds: Optional[Credentials] = None
        self.service = None
        
    def authenticate(self) -> bool:
        """
        Perform OAuth2 authentication flow.
        
        Returns:
            bool: True if authentication successful, False otherwise
        """
        try:
            # Load existing token if available
            if os.path.exists(self.token_file):
                self.creds = Credentials.from_authorized_user_file(
                    self.token_file, self.SCOPES
                )
                logger.info("Loaded existing credentials from token file")
            
            # Refresh token if expired but valid refresh token exists
            if self.creds and not self.creds.valid:
                if self.creds.expired and self.creds.refresh_token:
                    logger.info("Refreshing expired token")
                    self.creds.refresh(Request())
                else:
                    logger.warning("Token invalid and no refresh token available")
                    self.creds = None
            
            # If no valid credentials, initiate OAuth flow
            if not self.creds or not self.creds.valid:
                if not os.path.exists(self.credentials_file):
                    logger.error(f"Credentials file not found: {self.credentials_file}")
                    return False
                
                logger.info("Starting OAuth2 authentication flow")
                flow = InstalledAppFlow.from_client_secrets_file(
                    self.credentials_file, self.SCOPES
                )
                self.creds = flow.run_local_server(port=0)
                logger.info("OAuth2 authentication completed")
            
            # Save credentials for future use
            self._save_token()
            
            # Build Gmail service
            self.service = build('gmail', 'v1', credentials=self.creds)
            logger.info("Gmail service built successfully")
            
            return True
            
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            return False
    
    def get_service(self):
        """
        Get authenticated Gmail service object.
        
        Returns:
            Gmail service object or None if not authenticated
        """
        if not self.service:
            if not self.authenticate():
                return None
        return self.service
    
    def is_authenticated(self) -> bool:
        """
        Check if user is currently authenticated.
        
        Returns:
            bool: True if authenticated, False otherwise
        """
        return (self.creds is not None and 
                self.creds.valid and 
                self.service is not None)
    
    def get_user_info(self) -> Optional[Dict[str, Any]]:
        """
        Get authenticated user's Gmail profile information.
        
        Returns:
            Dict containing user profile info or None if failed
        """
        try:
            if not self.is_authenticated():
                logger.warning("Not authenticated - cannot get user info")
                return None
            
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email': profile.get('emailAddress'),
                'messages_total': profile.get('messagesTotal'),
                'threads_total': profile.get('threadsTotal'),
                'history_id': profile.get('historyId')
            }
            
        except HttpError as e:
            logger.error(f"Failed to get user info: {str(e)}")
            return None
    
    def revoke_access(self) -> bool:
        """
        Revoke access and clear stored credentials.
        
        Returns:
            bool: True if successfully revoked, False otherwise
        """
        try:
            if self.creds and self.creds.valid:
                # Revoke the token
                self.creds.revoke(Request())
                logger.info("Access token revoked")
            
            # Clear stored token
            if os.path.exists(self.token_file):
                os.remove(self.token_file)
                logger.info("Token file removed")
            
            # Clear instance variables
            self.creds = None
            self.service = None
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to revoke access: {str(e)}")
            return False
    
    def _save_token(self) -> None:
        """Save credentials to token file."""
        try:
            with open(self.token_file, 'w') as token:
                token.write(self.creds.to_json())
            logger.debug(f"Token saved to {self.token_file}")
        except Exception as e:
            logger.error(f"Failed to save token: {str(e)}")