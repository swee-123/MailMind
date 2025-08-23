# ai/summarizer.py
import json
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime
from .groq_client import GroqClient
from .prompt_manager import PromptManager, PromptType
import logging

class SummaryType:
    BRIEF = "brief"          # 1-2 sentences
    STANDARD = "standard"    # 3-5 sentences  
    DETAILED = "detailed"    # Comprehensive summary
    BULLET_POINTS = "bullet_points"  # Key points as bullets
    ACTION_FOCUSED = "action_focused"  # Focus on action items

class EmailSummarizer:
    """AI-powered email summarization and analysis"""
    
    def __init__(self, groq_client: GroqClient, prompt_manager: PromptManager):
        self.groq_client = groq_client
        self.prompt_manager = prompt_manager
        self.logger = logging.getLogger(__name__)
    
    def summarize_email(self, email_data: Dict[str, Any], 
                       summary_type: str = SummaryType.STANDARD,
                       focus_areas: List[str] = None) -> Dict[str, Any]:
        """Generate email summary with analysis"""
        try:
            # Prepare email content
            content = self._prepare_email_content(email_data)
            
            # Format prompt
            prompt_vars = {
                'sender': email_data.get('sender', ''),
                'recipients': ', '.join(email_data.get('recipients', [])),
                'subject': email_data.get('subject', ''),
                'date': email_data.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M'),
                'content': content
            }
            
            # Generate base summary
            messages = self.prompt_manager.format_prompt(
                PromptType.EMAIL_SUMMARY, **prompt_vars
            )
            
            message_dicts = []
            for msg in messages:
                message_dicts.append({
                    'role': 'system' if msg.__class__.__name__ == 'SystemMessage' else 'user',
                    'content': msg.content
                })
            
            response = self.groq_client.chat_completion(
                messages=message_dicts,
                temperature=0.3,  # Lower temperature for consistency
                max_tokens=512
            )
            
            base_summary = response['content'].strip()
            
            # Customize summary based on type
            final_summary = self._customize_summary(base_summary, summary_type, focus_areas)
            
            # Perform additional analysis
            sentiment = self._analyze_sentiment(content)
            actions = self._extract_actions(content, email_data)
            priority = self._analyze_priority(email_data, content)
            
            return {
                'summary': final_summary,
                'summary_type': summary_type,
                'sentiment_analysis': sentiment,
                'action_items': actions,
                'priority_analysis': priority,
                'word_count': len(content.split()),
                'key_entities': self._extract_entities(content),
                'metadata': {
                    'summarized_at': datetime.now().isoformat(),
                    'model_used': response.get('model', ''),
                    'tokens_used': response.get('usage', {})
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error summarizing email: {e}")
            return {
                'summary': self._generate_fallback_summary(email_data),
                'error': str(e)
            }
    
    def summarize_thread(self, thread_emails: List[Dict[str, Any]], 
                        include_timeline: bool = True) -> Dict[str, Any]:
        """Summarize an entire email thread"""
        try:
            # Prepare thread content
            thread_content = self._prepare_thread_content(thread_emails)
            
            prompt_vars = {
                'thread_emails': thread_content,
                'context': f"Thread with {len(thread_emails)} emails"
            }
            
            # Use thread analysis prompt
            prompt_text = self.prompt_manager.format_prompt(
                PromptType.THREAD_ANALYSIS, **prompt_vars
            )
            
            messages = [{'role': 'user', 'content': prompt_text}]
            
            response = self.groq_client.chat_completion(
                messages=messages,
                temperature=0.3,
                max_tokens=1024
            )
            
            # Parse the JSON response
            try:
                thread_analysis = json.loads(response['content'])
            except json.JSONDecodeError:
                # Fallback if JSON parsing fails
                thread_analysis = {
                    'thread_summary': response['content'],
                    'main_topic': 'Unable to parse',
                    'thread_status': 'unknown'
                }
            
            result = {
                'thread_summary': thread_analysis.get('thread_summary', ''),
                'main_topic': thread_analysis.get('main_topic', ''),
                'key_participants': thread_analysis.get('key_participants', []),
                'decisions_made': thread_analysis.get('decisions_made', []),
                'open_questions': thread_analysis.get('open_questions', []),
                'consolidated_actions': thread_analysis.get('action_items', []),
                'thread_status': thread_analysis.get('thread_status', 'ongoing'),
                'next_expected_action': thread_analysis.get('next_expected_action', ''),
                'email_count': len(thread_emails),
                'thread_span': self._calculate_thread_span(thread_emails)
            }
            
            if include_timeline:
                result['timeline'] = self._create_thread_timeline(thread_emails)
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error summarizing thread: {e}")
            return {
                'thread_summary': f"Thread with {len(thread_emails)} emails",
                'error': str(e)
            }
    
    def batch_summarize(self, emails: List[Dict[str, Any]], 
                       max_batch_size: int = 10) -> Dict[str, Any]:
        """Summarize multiple emails in batches"""
        summaries = []
        total_emails = len(emails)
        
        # Process emails in batches
        for i in range(0, total_emails, max_batch_size):
            batch = emails[i:i + max_batch_size]
            batch_summaries = []
            
            for email in batch:
                try:
                    summary = self.summarize_email(email, SummaryType.BRIEF)
                    batch_summaries.append({
                        'email_id': email.get('id', f'email_{i}'),
                        'subject': email.get('subject', ''),
                        'sender': email.get('sender', ''),
                        'summary': summary['summary'],
                        'priority': summary.get('priority_analysis', {}).get('priority_level', 'MEDIUM')
                    })
                except Exception as e:
                    self.logger.error(f"Error in batch summary: {e}")
                    batch_summaries.append({
                        'email_id': email.get('id', f'email_{i}'),
                        'summary': 'Error generating summary',
                        'error': str(e)
                    })
            
            summaries.extend(batch_summaries)
        
        return {
            'summaries': summaries,
            'total_processed': len(summaries),
            'batch_size': max_batch_size,
            'processed_at': datetime.now().isoformat()
        }
    
    def create_digest(self, emails: List[Dict[str, Any]], 
                     digest_type: str = 'daily') -> Dict[str, Any]:
        """Create a digest of multiple emails"""
        try:
            # Categorize emails
            categories = self._categorize_emails(emails)
            high_priority = [e for e in emails if self._is_high_priority(e)]
            
            # Generate digest content
            digest_sections = []
            
            # High priority section
            if high_priority:
                digest_sections.append(f"## High Priority ({len(high_priority)} emails)")
                for email in high_priority[:5]:  # Top 5 high priority
                    summary = self.summarize_email(email, SummaryType.BRIEF)
                    digest_sections.append(f"- **{email.get('subject', 'No Subject')}** from {email.get('sender', 'Unknown')}")
                    digest_sections.append(f"  {summary['summary']}")
            
            # Category sections
            for category, category_emails in categories.items():
                if category_emails and category != 'SPAM':
                    digest_sections.append(f"## {category.title()} ({len(category_emails)} emails)")
                    for email in category_emails[:3]:  # Top 3 per category
                        digest_sections.append(f"- {email.get('subject', 'No Subject')} from {email.get('sender', 'Unknown')}")
            
            digest_content = "\n".join(digest_sections)
            
            return {
                'digest_type': digest_type,
                'total_emails': len(emails),
                'high_priority_count': len(high_priority),
                'categories': {cat: len(emails) for cat, emails in categories.items()},
                'digest_content': digest_content,
                'generated_at': datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error creating digest: {e}")
            return {'digest_content': 'Error generating digest', 'error': str(e)}
    
    def _prepare_email_content(self, email_data: Dict[str, Any]) -> str:
        """Prepare email content for processing"""
        content = email_data.get('body_text', '') or email_data.get('content', '')
        
        # Truncate very long emails
        if len(content) > 4000:
            content = content[:4000] + "... [content truncated]"
        
        return content
    
    def _prepare_thread_content(self, thread_emails: List[Dict[str, Any]]) -> str:
        """Prepare thread content for analysis"""
        thread_parts = []
        
        for i, email in enumerate(thread_emails):
            timestamp = email.get('timestamp', datetime.now()).strftime('%Y-%m-%d %H:%M')
            sender = email.get('sender', 'Unknown')
            subject = email.get('subject', 'No Subject')
            content = self._prepare_email_content(email)
            
            thread_parts.append(f"""
EMAIL #{i+1} - {timestamp}
FROM: {sender}
SUBJECT: {subject}
CONTENT: {content}
---""")
        
        return "\n".join(thread_parts)
    
    def _customize_summary(self, base_summary: str, summary_type: str, 
                          focus_areas: List[str] = None) -> str:
        """Customize summary based on type and focus areas"""
        if summary_type == SummaryType.BRIEF:
            # Extract first sentence or two
            sentences = base_summary.split('.')
            return '.'.join(sentences[:2]) + '.' if len(sentences) > 1 else base_summary
        
        elif summary_type == SummaryType.BULLET_POINTS:
            # Convert to bullet points
            try:
                points_prompt = f"""Convert this summary into 3-5 bullet points:
                
                {base_summary}
                
                Format as:
                • Point 1
                • Point 2
                • Point 3"""
                
                messages = [{'role': 'user', 'content': points_prompt}]
                response = self.groq_client.chat_completion(messages=messages, max_tokens=256)
                return response['content'].strip()
            except:
                return base_summary
        
        elif summary_type == SummaryType.ACTION_FOCUSED:
            # Focus on action items
            if 'action' in base_summary.lower() or 'need' in base_summary.lower():
                return base_summary
            else:
                return f"Action Required: {base_summary}"
        
        return base_summary
    
    def _analyze_sentiment(self, content: str) -> Dict[str, Any]:
        """Analyze email sentiment"""
        try:
            prompt_text = self.prompt_manager.format_prompt(
                PromptType.SENTIMENT_ANALYSIS, content=content
            )
            
            messages = [{'role': 'user', 'content': prompt_text}]
            response = self.groq_client.chat_completion(
                messages=messages, temperature=0.1, max_tokens=256
            )
            
            try:
                sentiment_data = json.loads(response['content'])
                return sentiment_data
            except json.JSONDecodeError:
                return {
                    'sentiment': 'neutral',
                    'confidence': 0.5,
                    'tone': 'unknown'
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing sentiment: {e}")
            return {'sentiment': 'neutral', 'confidence': 0.0}
    
    def _extract_actions(self, content: str, email_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract action items from email"""
        try:
            context = f"Email from {email_data.get('sender', '')} with subject: {email_data.get('subject', '')}"
            
            prompt_text = self.prompt_manager.format_prompt(
                PromptType.ACTION_EXTRACTION, content=content, context=context
            )
            
            messages = [{'role': 'user', 'content': prompt_text}]
            response = self.groq_client.chat_completion(
                messages=messages, temperature=0.1, max_tokens=512
            )
            
            try:
                actions_data = json.loads(response['content'])
                return actions_data
            except json.JSONDecodeError:
                return {
                    'actions': [],
                    'requires_response': False,
                    'next_steps': 'Review email for action items'
                }
                
        except Exception as e:
            self.logger.error(f"Error extracting actions: {e}")
            return {'actions': [], 'requires_response': False}
    
    def _analyze_priority(self, email_data: Dict[str, Any], content: str) -> Dict[str, Any]:
        """Analyze email priority"""
        try:
            context = f"Email received at {email_data.get('timestamp', datetime.now())}"
            
            prompt_vars = {
                'sender': email_data.get('sender', ''),
                'subject': email_data.get('subject', ''),
                'content': content,
                'context': context
            }
            
            prompt_text = self.prompt_manager.format_prompt(
                PromptType.PRIORITY_ANALYSIS, **prompt_vars
            )
            
            messages = [{'role': 'user', 'content': prompt_text}]
            response = self.groq_client.chat_completion(
                messages=messages, temperature=0.1, max_tokens=256
            )
            
            try:
                priority_data = json.loads(response['content'])
                return priority_data
            except json.JSONDecodeError:
                return {
                    'priority_score': 5,
                    'priority_level': 'MEDIUM',
                    'reasoning': 'Unable to analyze priority'
                }
                
        except Exception as e:
            self.logger.error(f"Error analyzing priority: {e}")
            return {'priority_score': 5, 'priority_level': 'MEDIUM'}
    
    def _extract_entities(self, content: str) -> List[str]:
        """Extract key entities (names, dates, places, etc.)"""
        import re
        
        entities = []
        
        # Extract email addresses
        emails = re.findall(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b', content)
        entities.extend([f"Email: {email}" for email in emails])
        
        # Extract dates (simple patterns)
        dates = re.findall(r'\b\d{1,2}/\d{1,2}/\d{4}\b|\b\d{4}-\d{2}-\d{2}\b', content)
        entities.extend([f"Date: {date}" for date in dates])
        
        # Extract phone numbers (simple pattern)
        phones = re.findall(r'\b\d{3}-\d{3}-\d{4}\b|\b\(\d{3}\)\s*\d{3}-\d{4}\b', content)
        entities.extend([f"Phone: {phone}" for phone in phones])
        
        # Extract URLs
        urls = re.findall(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', content)
        entities.extend([f"URL: {url}" for url in urls])
        
        return entities[:10]  # Return top 10 entities
    
    def _categorize_emails(self, emails: List[Dict[str, Any]]) -> Dict[str, List[Dict[str, Any]]]:
        """Categorize emails by type"""
        categories = {
            'MEETING': [],
            'PROJECT': [],
            'SALES': [],
            'SUPPORT': [],
            'PERSONAL': [],
            'OTHER': []
        }
        
        for email in emails:
            try:
                content = self._prepare_email_content(email)
                subject = email.get('subject', '').lower()
                
                # Simple keyword-based categorization
                if any(word in subject for word in ['meeting', 'calendar', 'invite']):
                    categories['MEETING'].append(email)
                elif any(word in subject for word in ['project', 'task', 'deliverable']):
                    categories['PROJECT'].append(email)
                elif any(word in subject for word in ['sale', 'proposal', 'quote']):
                    categories['SALES'].append(email)
                elif any(word in subject for word in ['support', 'help', 'issue']):
                    categories['SUPPORT'].append(email)
                elif any(word in subject for word in ['personal', 'private']):
                    categories['PERSONAL'].append(email)
                else:
                    categories['OTHER'].append(email)
                    
            except Exception as e:
                self.logger.error(f"Error categorizing email: {e}")
                categories['OTHER'].append(email)
        
        return categories
    
    def _is_high_priority(self, email: Dict[str, Any]) -> bool:
        """Simple high priority detection"""
        subject = email.get('subject', '').lower()
        content = self._prepare_email_content(email).lower()
        
        urgent_keywords = ['urgent', 'asap', 'immediate', 'critical', 'emergency']
        return any(keyword in subject or keyword in content for keyword in urgent_keywords)
    
    def _calculate_thread_span(self, thread_emails: List[Dict[str, Any]]) -> Dict[str, str]:
        """Calculate time span of thread"""
        if not thread_emails:
            return {}
        
        timestamps = []
        for email in thread_emails:
            timestamp = email.get('timestamp')
            if timestamp:
                timestamps.append(timestamp)
        
        if not timestamps:
            return {}
        
        earliest = min(timestamps)
        latest = max(timestamps)
        duration = latest - earliest
        
        return {
            'start_date': earliest.isoformat(),
            'end_date': latest.isoformat(),
            'duration_hours': duration.total_seconds() / 3600
        }
    
    def _create_thread_timeline(self, thread_emails: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Create timeline of thread"""
        timeline = []
        
        for email in sorted(thread_emails, key=lambda x: x.get('timestamp', datetime.now())):
            timeline.append({
                'timestamp': email.get('timestamp', datetime.now()).isoformat(),
                'sender': email.get('sender', 'Unknown'),
                'subject': email.get('subject', ''),
                'summary': self.summarize_email(email, SummaryType.BRIEF)['summary']
            })
        
        return timeline
    
    def _generate_fallback_summary(self, email_data: Dict[str, Any]) -> str:
        """Generate fallback summary when AI fails"""
        subject = email_data.get('subject', 'No Subject')
        sender = email_data.get('sender', 'Unknown Sender')
        
        return f"Email from {sender} regarding: {subject}. Please review the full content for details."