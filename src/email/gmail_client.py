# src/email/gmail_client.py
import asyncio
import base64
import email
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import re

from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import html2text

from src.auth.gmail_auth import GmailAuthenticator
from src.email.prioritizer import EmailData, email_prioritizer
from src.utils.cache_manager import cache_manager
from src.utils.text_utils import clean_email_content
from src.utils.date_utils import parse_email_date

logger = logging.getLogger(__name__)

class GmailClient:
    """Enhanced Gmail API client with real-time synchronization"""
    
    def __init__(self):
        self.service = None
        self.authenticator = GmailAuthenticator()
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = True
        self.html_converter.ignore_images = True
        self._setup_html_converter()
        
    def _setup_html_converter(self):
        """Configure HTML to text converter"""
        self.html_converter.ignore_emphasis = True
        self.html_converter.body_width = 0  # Don't wrap lines
        self.html_converter.unicode_snob = True
        self.html_converter.bypass_tables = False
    
    async def initialize(self) -> bool:
        """Initialize Gmail client with authentication"""
        try:
            credentials = await self.authenticator.get_credentials()
            if not credentials:
                logger.error("Failed to get Gmail credentials")
                return False
                
            self.service = build('gmail', 'v1', credentials=credentials)
            logger.info("Gmail client initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize Gmail client: {str(e)}")
            return False
    
    async def get_profile(self) -> Optional[Dict]:
        """Get Gmail profile information"""
        try:
            if not self.service:
                await self.initialize()
            
            profile = self.service.users().getProfile(userId='me').execute()
            return {
                'email': profile.get('emailAddress'),
                'messages_total': profile.get('messagesTotal', 0),
                'threads_total': profile.get('threadsTotal', 0),
                'history_id': profile.get('historyId')
            }
        except HttpError as e:
            logger.error(f"Failed to get Gmail profile: {str(e)}")
            return None
    
    async def fetch_emails(self, query: str = '', max_results: int = 100, 
                          page_token: str = None) -> Tuple[List[EmailData], Optional[str]]:
        """Fetch emails with advanced query support"""
        try:
            if not self.service:
                await self.initialize()
            
            # Build query parameters
            params = {
                'userId': 'me',
                'q': query,
                'maxResults': min(max_results, 500)  # Gmail API limit
            }
            
            if page_token:
                params['pageToken'] = page_token
            
            # Execute query
            result = self.service.users().messages().list(**params).execute()
            messages = result.get('messages', [])
            next_page_token = result.get('nextPageToken')
            
            # Fetch detailed email data
            emails = await self._fetch_message_details(messages)
            
            return emails, next_page_token
            
        except HttpError as e:
            logger.error(f"Failed to fetch emails: {str(e)}")
            return [], None
    
    async def _fetch_message_details(self, message_refs: List[Dict]) -> List[EmailData]:
        """Fetch detailed information for multiple messages"""
        if not message_refs:
            return []
        
        emails = []
        
        # Batch process messages for efficiency
        batch_size = 10
        for i in range(0, len(message_refs), batch_size):
            batch = message_refs[i:i + batch_size]
            batch_emails = await self._process_message_batch(batch)
            emails.extend(batch_emails)
        
        return emails
    
    async def _process_message_batch(self, message_refs: List[Dict]) -> List[EmailData]:
        """Process a batch of messages efficiently"""
        tasks = [self._fetch_single_message(msg_ref['id']) for msg_ref in message_refs]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Filter out exceptions and None results
        emails = [result for result in results if isinstance(result, EmailData)]
        return emails
    
    async def _fetch_single_message(self, message_id: str) -> Optional[EmailData]:
        """Fetch detailed information for a single message"""
        try:
            # Check cache first
            cache_key = f"gmail_message_{message_id}"
            cached_email = await cache_manager.get(cache_key)
            if cached_email:
                return EmailData(**cached_email)
            
            # Fetch from Gmail API
            message = self.service.users().messages().get(
                userId='me', 
                id=message_id,
                format='full'
            ).execute()
            
            # Parse message
            email_data = await self._parse_gmail_message(message)
            
            # Cache the result
            if email_data:
                await cache_manager.set(cache_key, email_data.__dict__, ttl=3600)
            
            return email_data
            
        except HttpError as e:
            logger.error(f"Failed to fetch message {message_id}: {str(e)}")
            return None
    
    async def _parse_gmail_message(self, message: Dict) -> Optional[EmailData]:
        """Parse Gmail API message into EmailData"""
        try:
            payload = message.get('payload', {})
            headers = payload.get('headers', [])
            
            # Extract headers
            header_dict = {h['name'].lower(): h['value'] for h in headers}
            
            subject = header_dict.get('subject', 'No Subject')
            sender_raw = header_dict.get('from', 'Unknown Sender')
            recipient = header_dict.get('to', 'Unknown Recipient')
            date_str = header_dict.get('date', '')
            
            # Parse sender
            sender, sender_email = self._parse_email_address(sender_raw)
            
            # Parse date
            timestamp = parse_email_date(date_str)
            
            # Extract content
            content = await self._extract_message_content(payload)
            
            # Extract attachments
            attachments = self._extract_attachments(payload)
            
            # Get labels
            label_ids = message.get('labelIds', [])
            labels = await self._resolve_label_names(label_ids)
            
            return EmailData(
                message_id=message['id'],
                subject=subject,
                sender=sender,
                sender_email=sender_email,
                recipient=recipient,
                content=content,
                timestamp=timestamp,
                thread_id=message.get('threadId'),
                attachments=attachments,
                labels=labels
            )
            
        except Exception as e:
            logger.error(f"Failed to parse Gmail message: {str(e)}")
            return None
    
    def _parse_email_address(self, address_str: str) -> Tuple[str, str]:
        """Parse email address string into name and email"""
        # Pattern for "Name <email@domain.com>" format
        match = re.match(r'^(.+?)\s*<(.+?)>$', address_str.strip())
        if match:
            name = match.group(1).strip().strip('"')
            email = match.group(2).strip()
            return name, email
        else:
            # Assume it's just an email address
            email = address_str.strip()
            return email.split('@')[0], email
    
    async def _extract_message_content(self, payload: Dict) -> str:
        """Extract text content from Gmail message payload"""
        try:
            # Handle different payload structures
            if 'parts' in payload:
                # Multipart message
                text_content = ""
                for part in payload['parts']:
                    part_content = await self._extract_part_content(part)
                    if part_content:
                        text_content += part_content + "\n\n"
                return clean_email_content(text_content)
            else:
                # Single part message
                return await self._extract_part_content(payload)
        
        except Exception as e:
            logger.error(f"Failed to extract message content: {str(e)}")
            return "Content extraction failed"
    
    async def _extract_part_content(self, part: Dict) -> str:
        """Extract content from a message part"""
        mime_type = part.get('mimeType', '')
        
        # Skip attachments and other non-text parts
        if 'filename' in part and part['filename']:
            return ""
        
        # Handle different content types
        if mime_type == 'text/plain':
            return self._decode_message_data(part.get('body', {}).get('data', ''))
        elif mime_type == 'text/html':
            html_content = self._decode_message_data(part.get('body', {}).get('data', ''))
            return self.html_converter.handle(html_content)
        elif 'parts' in part:
            # Nested multipart
            content = ""
            for subpart in part['parts']:
                subcontent = await self._extract_part_content(subpart)
                if subcontent:
                    content += subcontent + "\n"
            return content
        
        return ""
    
    def _decode_message_data(self, data: str) -> str:
        """Decode base64url encoded message data"""
        if not data:
            return ""
        
        try:
            # Gmail uses base64url encoding
            decoded_bytes = base64.urlsafe_b64decode(data + '==')  # Add padding
            return decoded_bytes.decode('utf-8', errors='ignore')
        except Exception as e:
            logger.error(f"Failed to decode message data: {str(e)}")
            return ""
    
    def _extract_attachments(self, payload: Dict) -> List[str]:
        """Extract attachment filenames from message payload"""
        attachments = []
        
        def extract_from_parts(parts):
            for part in parts:
                if part.get('filename'):
                    attachments.append(part['filename'])
                elif 'parts' in part:
                    extract_from_parts(part['parts'])
        
        if 'parts' in payload:
            extract_from_parts(payload['parts'])
        elif payload.get('filename'):
            attachments.append(payload['filename'])
        
        return attachments
    
    async def _resolve_label_names(self, label_ids: List[str]) -> List[str]:
        """Resolve label IDs to label names"""
        if not label_ids:
            return []
        
        try:
            # Cache label mappings
            cache_key = "gmail_labels"
            label_mapping = await cache_manager.get(cache_key)
            
            if not label_mapping:
                # Fetch all labels
                labels_result = self.service.users().labels().list(userId='me').execute()
                labels = labels_result.get('labels', [])
                
                label_mapping = {label['id']: label['name'] for label in labels}
                await cache_manager.set(cache_key, label_mapping, ttl=3600)
            
            return [label_mapping.get(label_id, label_id) for label_id in label_ids]
            
        except Exception as e:
            logger.error(f"Failed to resolve label names: {str(e)}")
            return label_ids  # Return IDs as fallback
    
    async def fetch_recent_emails(self, hours: int = 24, include_sent: bool = False) -> List[EmailData]:
        """Fetch recent emails from specified time period"""
        try:
            # Build time-based query
            after_timestamp = int((datetime.now() - timedelta(hours=hours)).timestamp())
            query = f"after:{after_timestamp}"
            
            if not include_sent:
                query += " -in:sent"
            
            emails, _ = await self.fetch_emails(query=query, max_results=200)
            return emails
            
        except Exception as e:
            logger.error(f"Failed to fetch recent emails: {str(e)}")
            return []
    
    async def fetch_high_priority_emails(self, max_results: int = 50) -> List[Tuple[EmailData, Any]]:
        """Fetch and prioritize emails"""
        try:
            # Fetch recent emails
            emails = await self.fetch_recent_emails(hours=48)
            
            if not emails:
                return []
            
            # Prioritize emails
            prioritized_results = await email_prioritizer.batch_prioritize(emails)
            
            # Return top priority emails
            return prioritized_results[:max_results]
            
        except Exception as e:
            logger.error(f"Failed to fetch high priority emails: {str(e)}")
            return []
    
    async def send_email(self, to: str, subject: str, body: str, 
                        reply_to_message_id: str = None, thread_id: str = None) -> bool:
        """Send email through Gmail API"""
        try:
            if not self.service:
                await self.initialize()
            
            # Build email message
            message = self._build_email_message(to, subject, body, reply_to_message_id, thread_id)
            
            # Send email
            result = self.service.users().messages().send(
                userId='me',
                body={'raw': message}
            ).execute()
            
            logger.info(f"Email sent successfully: {result['id']}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to send email: {str(e)}")
            return False
    
    def _build_email_message(self, to: str, subject: str, body: str, 
                           reply_to_message_id: str = None, thread_id: str = None) -> str:
        """Build email message in RFC2822 format"""
        import email.mime.text
        import email.mime.multipart
        
        # Create message
        msg = email.mime.text.MIMEText(body)
        msg['to'] = to
        msg['subject'] = subject
        
        if reply_to_message_id:
            msg['In-Reply-To'] = reply_to_message_id
            msg['References'] = reply_to_message_id
        
        if thread_id:
            msg['Thread-Topic'] = thread_id
        
        # Encode message
        raw_message = base64.urlsafe_b64encode(msg.as_bytes()).decode('utf-8')
        return raw_message
    
    async def search_emails(self, query: str, max_results: int = 100) -> List[EmailData]:
        """Search emails with natural language query"""
        try:
            # Convert natural language to Gmail search syntax
            gmail_query = await self._convert_to_gmail_query(query)
            
            # Fetch emails
            emails, _ = await self.fetch_emails(query=gmail_query, max_results=max_results)
            
            return emails
            
        except Exception as e:
            logger.error(f"Failed to search emails: {str(e)}")
            return []
    
    async def _convert_to_gmail_query(self, natural_query: str) -> str:
        """Convert natural language query to Gmail search syntax"""
        # Simple conversion logic - in production, use more sophisticated NLP
        query_lower = natural_query.lower()
        gmail_query = natural_query
        
        # Handle common patterns
        if 'today' in query_lower:
            today = datetime.now().strftime('%Y/%m/%d')
            gmail_query = gmail_query.replace('today', f'after:{today}')
        
        if 'yesterday' in query_lower:
            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y/%m/%d')
            gmail_query = gmail_query.replace('yesterday', f'after:{yesterday}')
        
        if 'this week' in query_lower:
            week_ago = (datetime.now() - timedelta(days=7)).strftime('%Y/%m/%d')
            gmail_query = gmail_query.replace('this week', f'after:{week_ago}')
        
        if 'unread' in query_lower:
            gmail_query = gmail_query.replace('unread', 'is:unread')
        
        if 'important' in query_lower:
            gmail_query = gmail_query.replace('important', 'is:important')
        
        return gmail_query
    
    async def get_thread_emails(self, thread_id: str) -> List[EmailData]:
        """Get all emails in a thread"""
        try:
            if not self.service:
                await self.initialize()
            
            # Get thread
            thread = self.service.users().threads().get(
                userId='me',
                id=thread_id,
                format='full'
            ).execute()
            
            # Parse messages in thread
            messages = thread.get('messages', [])
            emails = []
            
            for message in messages:
                email_data = await self._parse_gmail_message(message)
                if email_data:
                    emails.append(email_data)
            
            # Sort by timestamp
            emails.sort(key=lambda x: x.timestamp)
            
            return emails
            
        except HttpError as e:
            logger.error(f"Failed to get thread emails: {str(e)}")
            return []
    
    async def mark_as_read(self, message_ids: List[str]) -> bool:
        """Mark emails as read"""
        try:
            if not self.service:
                await self.initialize()
            
            # Batch modify messages
            self.service.users().messages().batchModify(
                userId='me',
                body={
                    'ids': message_ids,
                    'removeLabelIds': ['UNREAD']
                }
            ).execute()
            
            return True
            
        except HttpError as e:
            logger.error(f"Failed to mark messages as read: {str(e)}")
            return False
    
    async def add_labels(self, message_ids: List[str], label_ids: List[str]) -> bool:
        """Add labels to emails"""
        try:
            if not self.service:
                await self.initialize()
            
            self.service.users().messages().batchModify(
                userId='me',
                body={
                    'ids': message_ids,
                    'addLabelIds': label_ids
                }
            ).execute()
            
            return True
            
        except HttpError as e:
            logger.error(f"Failed to add labels: {str(e)}")
            return False
    
    async def setup_push_notifications(self, webhook_url: str) -> bool:
        """Setup Gmail push notifications"""
        try:
            if not self.service:
                await self.initialize()
            
            # Watch for changes
            watch_request = {
                'labelIds': ['INBOX'],
                'topicName': webhook_url
            }
            
            result = self.service.users().watch(
                userId='me',
                body=watch_request
            ).execute()
            
            logger.info(f"Push notifications setup: {result}")
            return True
            
        except HttpError as e:
            logger.error(f"Failed to setup push notifications: {str(e)}")
            return False

# src/auth/gmail_auth.py
import os
import logging
from pathlib import Path
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google_auth_oauthlib.flow import InstalledAppFlow
from typing import Optional

logger = logging.getLogger(__name__)

class GmailAuthenticator:
    """Handle Gmail OAuth2 authentication"""
    
    SCOPES = [
        'https://www.googleapis.com/auth/gmail.readonly',
        'https://www.googleapis.com/auth/gmail.send',
        'https://www.googleapis.com/auth/gmail.modify'
    ]
    
    def __init__(self, credentials_dir: str = "credentials"):
        self.credentials_dir = Path(credentials_dir)
        self.credentials_file = self.credentials_dir / "credentials.json"
        self.token_file = self.credentials_dir / "token.json"
        
        # Create credentials directory if it doesn't exist
        self.credentials_dir.mkdir(exist_ok=True)
    
    async def get_credentials(self) -> Optional[Credentials]:
        """Get valid Gmail API credentials"""
        creds = None
        
        # Load existing token
        if self.token_file.exists():
            creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
        
        # Refresh or create new credentials
        if not creds or not creds.valid:
            if creds and creds.expired and creds.refresh_token:
                try:
                    creds.refresh(Request())
                    logger.info("Gmail credentials refreshed")
                except Exception as e:
                    logger.error(f"Failed to refresh credentials: {str(e)}")
                    creds = None
            
            if not creds:
                creds = await self._create_new_credentials()
        
        # Save credentials
        if creds:
            with open(self.token_file, 'w') as token:
                token.write(creds.to_json())
            logger.info("Gmail credentials saved")
        
        return creds
    
    async def _create_new_credentials(self) -> Optional[Credentials]:
        """Create new OAuth2 credentials"""
        try:
            if not self.credentials_file.exists():
                logger.error(f"Credentials file not found: {self.credentials_file}")
                logger.error("Please download credentials.json from Google Cloud Console")
                return None
            
            # Run OAuth flow
            flow = InstalledAppFlow.from_client_secrets_file(
                str(self.credentials_file), self.SCOPES
            )
            
            # Use local server for OAuth callback
            creds = flow.run_local_server(port=0)
            
            logger.info("New Gmail credentials created")
            return creds
            
        except Exception as e:
            logger.error(f"Failed to create new credentials: {str(e)}")
            return None
    
    def revoke_credentials(self) -> bool:
        """Revoke stored credentials"""
        try:
            if self.token_file.exists():
                self.token_file.unlink()
                logger.info("Gmail credentials revoked")
                return True
        except Exception as e:
            logger.error(f"Failed to revoke credentials: {str(e)}")
        
        return False
    
    def is_authenticated(self) -> bool:
        """Check if user is authenticated"""
        if not self.token_file.exists():
            return False
        
        try:
            creds = Credentials.from_authorized_user_file(str(self.token_file), self.SCOPES)
            return creds and creds.valid
        except Exception:
            return False

# Global instance
gmail_client = GmailClient()
