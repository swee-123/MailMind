# tests/integration/test_groq_api.py
import unittest
from unittest.mock import patch, MagicMock
import json

class TestGroqAPI(unittest.TestCase):
    """Groq API integration tests for Mail Mind."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.api_key = 'test_groq_api_key'
        self.sample_email_content = """
        From: john@company.com
        Subject: Urgent: Project Deadline
        
        Hi team,
        
        We need to discuss the upcoming project deadline. 
        The client is expecting delivery by Friday.
        
        Please review the attached documents and prepare for tomorrow's meeting.
        
        Best regards,
        John
        """
    
    @patch('groq.Groq')
    def test_groq_client_initialization(self, mock_groq):
        """Test Groq client initialization."""
        mock_client = MagicMock()
        mock_groq.return_value = mock_client
        
        from mail_mind.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer(api_key=self.api_key)
        
        self.assertIsNotNone(analyzer)
        mock_groq.assert_called_once_with(api_key=self.api_key)
    
    @patch('groq.Groq')
    def test_email_sentiment_analysis(self, mock_groq):
        """Test email sentiment analysis using Groq."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            'sentiment': 'urgent',
            'confidence': 0.85,
            'key_emotions': ['concern', 'pressure']
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        from mail_mind.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer(api_key=self.api_key)
        
        result = analyzer.analyze_sentiment(self.sample_email_content)
        
        self.assertEqual(result['sentiment'], 'urgent')
        self.assertGreater(result['confidence'], 0.8)
        self.assertIn('concern', result['key_emotions'])
    
    @patch('groq.Groq')
    def test_email_categorization(self, mock_groq):
        """Test email categorization using Groq."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = json.dumps({
            'category': 'work',
            'subcategory': 'project_management',
            'priority': 'high',
            'action_required': True
        })
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        from mail_mind.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer(api_key=self.api_key)
        
        result = analyzer.categorize_email(self.sample_email_content)
        
        self.assertEqual(result['category'], 'work')
        self.assertEqual(result['subcategory'], 'project_management')
        self.assertEqual(result['priority'], 'high')
        self.assertTrue(result['action_required'])
    
    @patch('groq.Groq')
    def test_email_summary_generation(self, mock_groq):
        """Test email summary generation using Groq."""
        mock_client = MagicMock()
        mock_response = MagicMock()
        mock_response.choices[0].message.content = "Project deadline discussion needed. Client expects delivery by Friday. Team meeting tomorrow to review documents."
        mock_client.chat.completions.create.return_value = mock_response
        mock_groq.return_value = mock_client
        
        from mail_mind.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer(api_key=self.api_key)
        
        summary = analyzer.generate_summary(self.sample_email_content)
        
        self.assertIn('deadline', summary.lower())
        self.assertIn('friday', summary.lower())
        self.assertIn('meeting', summary.lower())
    
    @patch('groq.Groq')
    def test_groq_api_error_handling(self, mock_groq):
        """Test Groq API error handling."""
        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = Exception("API Rate limit exceeded")
        mock_groq.return_value = mock_client
        
        from mail_mind.ai_analyzer import AIAnalyzer
        analyzer = AIAnalyzer(api_key=self.api_key)
        
        with self.assertRaises(Exception):
            analyzer.analyze_sentiment(self.sample_email_content)

if __name__ == '_main_':
    unittest.main()
