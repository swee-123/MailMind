# email/email_processor.py
import re
import html2text
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass
import base64
import logging

@dataclass
class ProcessedEmail:
    id: str
    thread_id: str
    sender: str
    sender_name: str
    recipients: List[str]
    subject: str
    body_text: str
    body_html: str
    timestamp: datetime
    attachments: List[Dict]
    labels: List[str]
    priority_score: float = 0.0
    sentiment: str = 'neutral'
    category: str = 'general'

class EmailProcessor:
    """Process and parse email messages"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.html_converter = html2text.HTML2Text()
        self.html_converter.ignore_links = False
        
    def process_message(self, raw_message: Dict) -> ProcessedEmail:
        """Process raw Gmail message into structured format"""
        try:
            headers = self._extract_headers(raw_message)
            body_text, body_html = self._extract_body(raw_message)
            attachments = self._extract_attachments(raw_message)
            
            # Parse timestamp
            timestamp = datetime.fromtimestamp(
                int(raw_message.get('internalDate', 0)) / 1000)
            
            email = ProcessedEmail(
                id=raw_message['id'],
                thread_id=raw_message['threadId'],
                sender=headers.get('from', ''),
                sender_name=self._extract_sender_name(headers.get('from', '')),
                recipients=self._parse_recipients(headers),
                subject=headers.get('subject', ''),
                body_text=body_text,
                body_html=body_html,
                timestamp=timestamp,
                attachments=attachments,
                labels=raw_message.get('labelIds', [])
            )
            
            return email
            
        except Exception as e:
            self.logger.error(f"Error processing message {raw_message.get('id')}: {e}")
            raise
    
    def _extract_headers(self, message: Dict) -> Dict[str, str]:
        """Extract email headers"""
        headers = {}
        payload = message.get('payload', {})
        
        for header in payload.get('headers', []):
            headers[header['name'].lower()] = header['value']
            
        return headers
    
    def _extract_body(self, message: Dict) -> Tuple[str, str]:
        """Extract email body (text and HTML)"""
        payload = message.get('payload', {})
        body_text = ''
        body_html = ''
        
        def decode_body(data: str) -> str:
            if data:
                return base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
            return ''
        
        # Handle single part message
        if payload.get('body', {}).get('data'):
            body_text = decode_body(payload['body']['data'])
            
        # Handle multipart message
        elif payload.get('parts'):
            for part in payload['parts']:
                mime_type = part.get('mimeType', '')
                
                if mime_type == 'text/plain' and part.get('body', {}).get('data'):
                    body_text = decode_body(part['body']['data'])
                elif mime_type == 'text/html' and part.get('body', {}).get('data'):
                    body_html = decode_body(part['body']['data'])
                    
                # Handle nested multipart
                elif part.get('parts'):
                    for nested_part in part['parts']:
                        nested_mime = nested_part.get('mimeType', '')
                        if nested_mime == 'text/plain' and nested_part.get('body', {}).get('data'):
                            body_text = decode_body(nested_part['body']['data'])
                        elif nested_mime == 'text/html' and nested_part.get('body', {}).get('data'):
                            body_html = decode_body(nested_part['body']['data'])
        
        # Convert HTML to text if no plain text available
        if not body_text and body_html:
            body_text = self.html_converter.handle(body_html)
            
        return body_text.strip(), body_html.strip()
    
    def _extract_attachments(self, message: Dict) -> List[Dict]:
        """Extract attachment information"""
        attachments = []
        payload = message.get('payload', {})
        
        def process_parts(parts: List[Dict]):
            for part in parts:
                if part.get('filename'):
                    attachment = {
                        'filename': part['filename'],
                        'mime_type': part.get('mimeType', ''),
                        'size': part.get('body', {}).get('size', 0),
                        'attachment_id': part.get('body', {}).get('attachmentId', '')
                    }
                    attachments.append(attachment)
                
                if part.get('parts'):
                    process_parts(part['parts'])
        
        if payload.get('parts'):
            process_parts(payload['parts'])
            
        return attachments
    
    def _extract_sender_name(self, from_header: str) -> str:
        """Extract sender name from FROM header"""
        if '<' in from_header and '>' in from_header:
            name_part = from_header.split('<')[0].strip()
            return name_part.strip('"')
        return from_header
    
    def _parse_recipients(self, headers: Dict[str, str]) -> List[str]:
        """Parse all recipients (TO, CC, BCC)"""
        recipients = []
        
        for field in ['to', 'cc', 'bcc']:
            if field in headers:
                # Simple email extraction (can be improved with proper parsing)
                emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', 
                                  headers[field])
                recipients.extend(emails)
                
        return list(set(recipients))  # Remove duplicates
    
    def extract_keywords(self, email: ProcessedEmail) -> List[str]:
        """Extract keywords from email content"""
        import re
        
        text = f"{email.subject} {email.body_text}".lower()
        
        # Remove common stop words
        stop_words = {'the', 'is', 'at', 'which', 'on', 'and', 'or', 'but', 'in', 'with', 
                     'to', 'for', 'of', 'as', 'by', 'that', 'this', 'it', 'from', 'be',
                     'are', 'was', 'were', 'have', 'has', 'had', 'will', 'would', 'could'}
        
        # Extract words (3+ characters)
        words = re.findall(r'\b[a-zA-Z]{3,}\b', text)
        keywords = [word for word in words if word not in stop_words]
        
        # Count frequency and return top keywords
        from collections import Counter
        word_freq = Counter(keywords)
        return [word for word, count in word_freq.most_common(20)]
