"""MailMind Email Processing Module"""

from .gmail_client import GmailClient
from .email_processor import EmailProcessor
from .prioritizer import EmailPrioritizer
from .attachment_handler import AttachmentHandler

__all__ = ['GmailClient', 'EmailProcessor', 'EmailPrioritizer', 'AttachmentHandler']