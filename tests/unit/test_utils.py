# tests/unit/test_utils.py
import unittest
from datetime import datetime, timedelta
import json

class TestUtils(unittest.TestCase):
    """Unit tests for utility functions."""
    
    def test_date_parsing(self):
        """Test date parsing utilities."""
        from mail_mind.utils import DateUtils
        
        date_util = DateUtils()
        
        # Test various date formats
        date_strings = [
            'Wed, 21 Aug 2025 10:00:00 +0000',
            '2025-08-21T10:00:00Z',
            'Aug 21, 2025 10:00 AM'
        ]
        
        for date_string in date_strings:
            parsed_date = date_util.parse_date(date_string)
            self.assertIsInstance(parsed_date, datetime)
    
    def test_text_cleaning(self):
        """Test text cleaning utilities."""
        from mail_mind.utils import TextUtils
        
        text_util = TextUtils()
        
        dirty_text = "  Hello   World!  \n\n  Extra   spaces  "
        clean_text = text_util.clean_text(dirty_text)
        
        self.assertEqual(clean_text, "Hello World! Extra spaces")
    
    def test_email_address_validation(self):
        """Test email address validation."""
        from mail_mind.utils import ValidationUtils
        
        validator = ValidationUtils()
        
        valid_emails = [
            'user@example.com',
            'test.email+tag@domain.co.uk',
            'user123@test-domain.org'
        ]
        
        invalid_emails = [
            'invalid-email',
            '@domain.com',
            'user@',
            'user@.com'
        ]
        
        for email in valid_emails:
            self.assertTrue(validator.is_valid_email(email))
        
        for email in invalid_emails:
            self.assertFalse(validator.is_valid_email(email))
    
    def test_json_serialization(self):
        """Test JSON serialization utilities."""
        from mail_mind.utils import JsonUtils
        
        json_util = JsonUtils()
        
        # Test datetime serialization
        data = {
            'timestamp': datetime.now(),
            'message': 'Test message',
            'count': 42
        }
        
        serialized = json_util.serialize(data)
        self.assertIsInstance(serialized, str)
        
        deserialized = json_util.deserialize(serialized)
        self.assertEqual(deserialized['message'], 'Test message')
        self.assertEqual(deserialized['count'], 42)
    
    def test_rate_limiting(self):
        """Test rate limiting utilities."""
        from mail_mind.utils import RateLimiter
        
        limiter = RateLimiter(max_requests=5, time_window=60)
        
        # Test within limits
        for i in range(5):
            self.assertTrue(limiter.allow_request('user1'))
        
        # Test rate limit exceeded
        self.assertFalse(limiter.allow_request('user1'))
    
    def test_cache_utilities(self):
        """Test caching utilities."""
        from mail_mind.utils import CacheManager
        
        cache = CacheManager()
        
        # Test cache set/get
        cache.set('test_key', 'test_value', ttl=300)
        self.assertEqual(cache.get('test_key'), 'test_value')
        
        # Test cache miss
        self.assertIsNone(cache.get('nonexistent_key'))
    
    def test_logging_utilities(self):
        """Test logging configuration."""
        from mail_mind.utils import LoggerConfig
        
        logger_config = LoggerConfig()
        logger = logger_config.get_logger('test_module')
        
        self.assertIsNotNone(logger)
        
        # Test log levels
        logger.info('Test info message')
        logger.warning('Test warning message')
        logger.error('Test error message')

if __name__ == '__main__':
    unittest.main()