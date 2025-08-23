# tests/unit/test_email_processing.py
import unittest
from unittest.mock import patch, MagicMock
import base64
from datetime import datetime

class TestEmailProcessing(unittest.TestCase):
    """Unit tests for email processing components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_raw_email = {
            'id': 'test_email_123',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'},
                    {'name': 'Subject', 'value': 'Test Email Subject'},
                    {'name': 'Date', 'value': 'Wed, 21 Aug 2025 10:00:00 +0000'}
                ],
                'body': {
                    'data': base64.b64encode(b'This is the email body content').decode()
                }
            }
        }
    
    def test_email_parsing(self):
        """Test parsing raw email data."""
        from mail_mind.email_parser import EmailParser
        
        parser = EmailParser()
        parsed_email = parser.parse_email(self.sample_raw_email)
        
        self.assertEqual(parsed_email['id'], 'test_email_123')
        self.assertEqual(parsed_email['from'], 'sender@example.com')
        self.assertEqual(parsed_email['to'], 'recipient@example.com')
        self.assertEqual(parsed_email['subject'], 'Test Email Subject')
        self.assertIn('body', parsed_email)
        self.assertIn('date', parsed_email)
    
    def test_body_extraction(self):
        """Test email body extraction and decoding."""
        from mail_mind.email_parser import EmailParser
        
        parser = EmailParser()
        body = parser.extract_body(self.sample_raw_email['payload'])
        
        self.assertEqual(body, 'This is the email body content')
    
    def test_attachment_detection(self):
        """Test attachment detection in emails."""
        from mail_mind.email_parser import EmailParser
        
        # Email with attachment
        email_with_attachment = {
            'payload': {
                'parts': [
                    {'filename': 'document.pdf', 'body': {'attachmentId': 'att_123'}},
                    {'filename': '', 'body': {'data': 'email_body_data'}}
                ]
            }
        }
        
        parser = EmailParser()
        attachments = parser.get_attachments(email_with_attachment)
        
        self.assertEqual(len(attachments), 1)
        self.assertEqual(attachments[0]['filename'], 'document.pdf')
    
    def test_email_validation(self):
        """Test email validation."""
        from mail_mind.email_validator import EmailValidator
        
        validator = EmailValidator()
        
        # Valid email
        valid_email = {
            'from': 'valid@example.com',
            'to': 'recipient@example.com',
            'subject': 'Valid Subject',
            'body': 'Valid body content'
        }
        self.assertTrue(validator.is_valid_email(valid_email))
        
        # Invalid email (missing required fields)
        invalid_email = {'from': 'invalid@example.com'}
        self.assertFalse(validator.is_valid_email(invalid_email))
    
    def test_email_sanitization(self):
        """Test email content sanitization."""
        from mail_mind.email_sanitizer import EmailSanitizer
        
        sanitizer = EmailSanitizer()
        
        dirty_content = '<script>alert("xss")</script><p>Clean content</p>'
        clean_content = sanitizer.sanitize_html(dirty_content)
        
        self.assertNotIn('<script>', clean_content)
        self.assertIn('Clean content', clean_content)
    
    def test_email_threading(self):
        """Test email thread detection and grouping."""
        from mail_mind.thread_manager import ThreadManager
        
        manager = ThreadManager()
        
        emails = [
            {'id': '1', 'subject': 'Original Subject', 'thread_id': 'thread_123'},
            {'id': '2', 'subject': 'Re: Original Subject', 'thread_id': 'thread_123'},
            {'id': '3', 'subject': 'Different Subject', 'thread_id': 'thread_456'}
        ]
        
        threads = manager.group_by_thread(emails)
        
        self.assertEqual(len(threads), 2)
        self.assertEqual(len(threads['thread_123']), 2)
        self.assertEqual(len(threads['thread_456']), 1)

if __name__ == '__main__':
    unittest.main()
