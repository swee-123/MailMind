# email/prioritizer.py
import re
from datetime import datetime, timedelta
from typing import Dict, List
from dataclasses import dataclass
from .email_processor import ProcessedEmail
import logging

@dataclass
class PriorityRule:
    name: str
    condition: callable
    weight: float
    description: str

class EmailPrioritizer:
    """Intelligent email prioritization engine"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.rules = self._initialize_rules()
        self.vip_senders = set()
        self.keywords_urgent = {'urgent', 'asap', 'emergency', 'critical', 'immediate'}
        self.keywords_important = {'important', 'meeting', 'deadline', 'action required'}
        
    def _initialize_rules(self) -> List[PriorityRule]:
        """Initialize priority rules"""
        return [
            PriorityRule(
                name="VIP_SENDER",
                condition=self._is_vip_sender,
                weight=30.0,
                description="Email from VIP sender"
            ),
            PriorityRule(
                name="URGENT_KEYWORDS",
                condition=self._has_urgent_keywords,
                weight=25.0,
                description="Contains urgent keywords"
            ),
            PriorityRule(
                name="DIRECT_TO_ME",
                condition=self._is_direct_to_me,
                weight=20.0,
                description="Email sent directly to user"
            ),
            PriorityRule(
                name="IMPORTANT_KEYWORDS",
                condition=self._has_important_keywords,
                weight=15.0,
                description="Contains important keywords"
            ),
            PriorityRule(
                name="RECENT_EMAIL",
                condition=self._is_recent,
                weight=10.0,
                description="Recent email (within 2 hours)"
            ),
            PriorityRule(
                name="REPLY_THREAD",
                condition=self._is_reply_thread,
                weight=12.0,
                description="Part of ongoing conversation"
            ),
            PriorityRule(
                name="HAS_ATTACHMENTS",
                condition=self._has_attachments,
                weight=8.0,
                description="Email has attachments"
            ),
            PriorityRule(
                name="SHORT_EMAIL",
                condition=self._is_short_email,
                weight=5.0,
                description="Short email (likely quick read)"
            )
        ]
    
    def calculate_priority(self, email: ProcessedEmail, user_email: str = '') -> float:
        """Calculate priority score for email"""
        total_score = 0.0
        applied_rules = []
        
        for rule in self.rules:
            try:
                if rule.condition(email, user_email):
                    total_score += rule.weight
                    applied_rules.append(rule.name)
            except Exception as e:
                self.logger.error(f"Error applying rule {rule.name}: {e}")
        
        # Apply sender reputation multiplier
        sender_multiplier = self._get_sender_reputation(email.sender)
        total_score *= sender_multiplier
        
        # Cap the score at 100
        final_score = min(total_score, 100.0)
        
        self.logger.debug(f"Email {email.id} priority: {final_score:.1f} "
                         f"(rules: {', '.join(applied_rules)})")
        
        return final_score
    
    def _is_vip_sender(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if sender is VIP"""
        return email.sender.lower() in [vip.lower() for vip in self.vip_senders]
    
    def _has_urgent_keywords(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check for urgent keywords"""
        content = f"{email.subject} {email.body_text}".lower()
        return any(keyword in content for keyword in self.keywords_urgent)
    
    def _is_direct_to_me(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if email is sent directly to user"""
        if not user_email:
            return len(email.recipients) <= 2  # Assume direct if few recipients
        return user_email.lower() in [r.lower() for r in email.recipients[:1]]
    
    def _has_important_keywords(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check for important keywords"""
        content = f"{email.subject} {email.body_text}".lower()
        return any(keyword in content for keyword in self.keywords_important)
    
    def _is_recent(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if email is recent (within 2 hours)"""
        return datetime.now() - email.timestamp <= timedelta(hours=2)
    
    def _is_reply_thread(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if email is part of ongoing conversation"""
        subject_lower = email.subject.lower()
        return (subject_lower.startswith('re:') or 
                subject_lower.startswith('fwd:') or
                'reply' in subject_lower)
    
    def _has_attachments(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if email has attachments"""
        return len(email.attachments) > 0
    
    def _is_short_email(self, email: ProcessedEmail, user_email: str) -> bool:
        """Check if email is short (quick read)"""
        return len(email.body_text.split()) <= 50
    
    def _get_sender_reputation(self, sender: str) -> float:
        """Get sender reputation multiplier"""
        # This would typically be based on historical data
        # For now, return default multiplier
        domain = sender.split('@')[-1].lower() if '@' in sender else ''
        
        # Higher reputation for common business domains
        high_rep_domains = {'gmail.com', 'outlook.com', 'company.com'}
        if domain in high_rep_domains:
            return 1.1
            
        return 1.0
    
    def add_vip_sender(self, email_address: str):
        """Add sender to VIP list"""
        self.vip_senders.add(email_address.lower())
    
    def remove_vip_sender(self, email_address: str):
        """Remove sender from VIP list"""
        self.vip_senders.discard(email_address.lower())
    
    def get_priority_category(self, score: float) -> str:
        """Get priority category based on score"""
        if score >= 70:
            return "HIGH"
        elif score >= 40:
            return "MEDIUM"
        elif score >= 15:
            return "LOW"
        else:
            return "MINIMAL"