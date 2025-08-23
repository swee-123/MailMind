# tests/unit/test_ai_integration.py
import unittest
from unittest.mock import patch, MagicMock
import json

class TestAIIntegration(unittest.TestCase):
    """Unit tests for AI integration components."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.sample_email = {
            'subject': 'Meeting Request',
            'body': 'Can we schedule a meeting for next week?',
            'from': 'colleague@company.com'
        }
    
    def test_email_preprocessing(self):
        """Test email preprocessing for AI analysis."""
        from mail_mind.preprocessor import EmailPreprocessor
        
        preprocessor = EmailPreprocessor()
        processed = preprocessor.preprocess(self.sample_email)
        
        self.assertIn('cleaned_text', processed)
        self.assertIn('metadata', processed)
        self.assertIsInstance(processed['cleaned_text'], str)
    
    def test_feature_extraction(self):
        """Test feature extraction from emails."""
        from mail_mind.feature_extractor import FeatureExtractor
        
        extractor = FeatureExtractor()
        features = extractor.extract_features(self.sample_email)
        
        self.assertIn('word_count', features)
        self.assertIn('has_attachments', features)
        self.assertIn('urgency_keywords', features)
        self.assertIsInstance(features['word_count'], int)
    
    @patch('mail_mind.ai_analyzer.AIAnalyzer')
    def test_batch_email_analysis(self, mock_analyzer):
        """Test batch processing of multiple emails."""
        mock_analyzer.return_value.analyze_batch.return_value = [
            {'email_id': '1', 'sentiment': 'positive', 'category': 'meeting'},
            {'email_id': '2', 'sentiment': 'neutral', 'category': 'update'}
        ]
        
        emails = [
            {'id': '1', 'subject': 'Meeting', 'body': 'Let\'s meet'},
            {'id': '2', 'subject': 'Update', 'body': 'Status update'}
        ]
        
        from mail_mind.batch_processor import BatchProcessor
        processor = BatchProcessor()
        results = processor.process_emails(emails)
        
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]['sentiment'], 'positive')

if _name_ == '_main_':
    unittest.main()


