
# tests/integration/test_gmail_api.py
import unittest
from unittest.mock import patch, MagicMock
import json
from googleapiclient.errors import HttpError

class TestGmailAPI(unittest.TestCase):
    """Gmail API integration tests for Mail Mind."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'client_id': 'test_client_id',
            'client_secret': 'test_secret',
            'refresh_token': 'test_refresh_token'
        }
        
        self.sample_message = {
            'id': 'test_message_id',
            'threadId': 'test_thread_id',
            'payload': {
                'headers': [
                    {'name': 'From', 'value': 'sender@example.com'},
                    {'name': 'To', 'value': 'recipient@example.com'},
                    {'name': 'Subject', 'value': 'Test Email'}
                ],
                'body': {'data': 'VGVzdCBlbWFpbCBib2R5'}
            }
        }
    
    @patch('googleapiclient.discovery.build')
    def test_gmail_service_initialization(self, mock_build):
        """Test Gmail service initialization."""
        mock_service = MagicMock()
        mock_build.return_value = mock_service
        
        from src.email.gmail_client import GmailClient

        client = GmailClient(self.mock_credentials)
        
        self.assertIsNotNone(client)
        mock_build.assert_called_once()
    
    @patch('googleapiclient.discovery.build')
    def test_fetch_emails(self, mock_build):
        """Test fetching emails from Gmail API."""
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.return_value = {
            'messages': [{'id': 'msg1'}, {'id': 'msg2'}]
        }
        mock_service.users().messages().get().execute.return_value = self.sample_message
        mock_build.return_value = mock_service
        
        from src.email.gmail_client import GmailClient

        client = GmailClient(self.mock_credentials)
        emails = client.fetch_emails(max_results=10)
        
        self.assertIsInstance(emails, list)
        self.assertGreater(len(emails), 0)
    
    @patch('googleapiclient.discovery.build')
    def test_send_email(self, mock_build):
        """Test sending email via Gmail API."""
        mock_service = MagicMock()
        mock_service.users().messages().send().execute.return_value = {
            'id': 'sent_message_id',
            'threadId': 'thread_id'
        }
        mock_build.return_value = mock_service
        
        from src.email.gmail_client import GmailClient

        client = GmailClient(self.mock_credentials)
        
        result = client.send_email(
            to='recipient@example.com',
            subject='Test Subject',
            body='Test email body'
        )
        
        self.assertTrue(result['success'])
        self.assertEqual(result['message_id'], 'sent_message_id')
    
    @patch('googleapiclient.discovery.build')
    def test_gmail_api_error_handling(self, mock_build):
        """Test Gmail API error handling."""
        mock_service = MagicMock()
        mock_service.users().messages().list().execute.side_effect = HttpError(
            resp=MagicMock(status=403), 
            content=b'Quota exceeded'
        )
        mock_build.return_value = mock_service
        
        from src.email.gmail_client import GmailClient

        client = GmailClient(self.mock_credentials)
        
        with self.assertRaises(HttpError):
            client.fetch_emails()

if __name__ == '_main_':
    unittest.main()

