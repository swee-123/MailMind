# tests/unit/test_auth.py
import unittest
from unittest.mock import patch, MagicMock
import json
from datetime import datetime, timedelta

class TestAuth(unittest.TestCase):
    """Unit tests for authentication components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.mock_credentials = {
            'access_token': 'test_access_token',
            'refresh_token': 'test_refresh_token',
            'token_type': 'Bearer',
            'expires_at': (datetime.now() + timedelta(hours=1)).isoformat()
        }
    
    def test_token_validation(self):
        """Test OAuth token validation."""
        from mail_mind.auth import TokenValidator
        
        validator = TokenValidator()
        
        # Test valid token
        self.assertTrue(validator.is_valid(self.mock_credentials))
        
        # Test expired token
        expired_creds = self.mock_credentials.copy()
        expired_creds['expires_at'] = (datetime.now() - timedelta(hours=1)).isoformat()
        self.assertFalse(validator.is_valid(expired_creds))
    
    @patch('requests.post')
    def test_token_refresh(self, mock_post):
        """Test OAuth token refresh."""
        mock_response = MagicMock()
        mock_response.json.return_value = {
            'access_token': 'new_access_token',
            'expires_in': 3600
        }
        mock_response.status_code = 200
        mock_post.return_value = mock_response
        
        from mail_mind.auth import TokenManager
        manager = TokenManager()
        
        new_token = manager.refresh_token('old_refresh_token')
        
        self.assertEqual(new_token['access_token'], 'new_access_token')
        self.assertIn('expires_at', new_token)
    
    def test_credential_encryption(self):
        """Test credential encryption/decryption."""
        from mail_mind.auth import CredentialManager
        
        manager = CredentialManager()
        test_data = {'secret': 'sensitive_info'}
        
        # Test encryption
        encrypted = manager.encrypt_credentials(test_data)
        self.assertNotEqual(encrypted, test_data)
        
        # Test decryption
        decrypted = manager.decrypt_credentials(encrypted)
        self.assertEqual(decrypted, test_data)
    
    def test_oauth_flow_validation(self):
        """Test OAuth flow validation."""
        from mail_mind.auth import OAuthValidator
        
        validator = OAuthValidator()
        
        # Test valid flow
        valid_params = {
            'code': 'auth_code_123',
            'state': 'csrf_token_456'
        }
        self.assertTrue(validator.validate_oauth_response(valid_params))
        
        # Test invalid flow (missing code)
        invalid_params = {'state': 'csrf_token_456'}
        self.assertFalse(validator.validate_oauth_response(invalid_params))

if __name__ == '_main_':
    unittest.main()


