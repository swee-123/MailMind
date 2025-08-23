# src/ai/response_generator.py
import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from enum import Enum
import re

from src.ai.model_manager import ai_manager
from src.ai.prompt_manager import prompt_manager
from src.email.gmail_client import gmail_client
from src.utils.text_utils import clean_text, extract_key_phrases
from src.utils.cache_manager import cache_manager

logger = logging.getLogger(__name__)

class ReplyTone(Enum):
    PROFESSIONAL = "professional"
    CASUAL = "casual"
    FRIENDLY = "friendly"
    FORMAL = "formal"
    URGENT = "urgent"
    APOLOGETIC = "apologetic"
    ENTHUSIASTIC = "enthusiastic"

class ReplyType(Enum):
    ACKNOWLEDGMENT = "acknowledgment"
    ANSWER = "answer"
    FOLLOW_UP = "follow_up"
    DECLINE = "decline"
    SCHEDULE = "schedule"
    REQUEST_INFO = "request_info"
    THANK_YOU = "thank_you"

@dataclass
class ReplyContext:
    """Context information for generating replies"""
    original_email_id: str
    subject: str
    sender: str
    sender_email: str
    content: str
    thread_id: Optional[str] = None
    priority_level: str = "medium"
    urgency_indicators: List[str] = None
    action_items: List[str] = None
    deadline_mentioned: Optional[str] = None
    sender_relationship: str = "unknown"  # colleague, client, boss, vendor, etc.

@dataclass
class ReplyDraft:
    """Generated reply draft"""
    subject: str
    body: str
    tone: ReplyTone
    reply_type: ReplyType
    confidence: float
    key_points_addressed: List[str]
    suggestions: List[str]
    estimated_reading_time: int  # seconds
    word_count: int

class ResponseGenerator:
    """AI-powered email response generator"""
    
    def __init__(self):
        self.tone_templates = self._load_tone_templates()
        self.reply_patterns = self._load_reply_patterns()
        
    def _load_tone_templates(self) -> Dict[ReplyTone, Dict]:
        """Load tone-specific templates and guidelines"""
        return {
            ReplyTone.PROFESSIONAL: {
                "greeting": ["Dear {name}", "Hello {name}", "Hi {name}"],
                "closing": ["Best regards", "Kind regards", "Sincerely"],
                "style_guide": "Use formal language, complete sentences, proper grammar"
            },
            ReplyTone.CASUAL: {
                "greeting": ["Hi {name}", "Hey {name}", "Hello"],
                "closing": ["Thanks", "Best", "Cheers", "Talk soon"],
                "style_guide": "Use conversational language, contractions OK, keep it relaxed"
            },
            ReplyTone.FRIENDLY: {
                "greeting": ["Hi {name}!", "Hello {name}", "Hey there"],
                "closing": ["Thanks so much", "Best wishes", "Have a great day"],
                "style_guide": "Warm and approachable, use positive language"
            },
            ReplyTone.FORMAL: {
                "greeting": ["Dear Mr./Ms. {name}", "Dear {name}"],
                "closing": ["Respectfully", "Yours sincerely", "Kind regards"],
                "style_guide": "Very formal, avoid contractions, use titles"
            },
            ReplyTone.URGENT: {
                "greeting": ["Hi {name}", "{name}"],
                "closing": ["Thanks", "Regards"],
                "style_guide": "Direct and concise, emphasize urgency without being rude"
            },
            ReplyTone.APOLOGETIC: {
                "greeting": ["Dear {name}", "Hi {name}"],
                "closing": ["Apologies again", "Thank you for your patience"],
                "style_guide": "Acknowledge issues, take responsibility, offer solutions"
            }
        }
    
    def _load_reply_patterns(self) -> Dict[ReplyType, Dict]:
        """Load reply type patterns and structures"""
        return {
            ReplyType.ACKNOWLEDGMENT: {
                "structure": ["acknowledge_receipt", "next_steps", "closing"],
                "key_phrases": ["received your email", "will review", "get back to you"]
            },
            ReplyType.ANSWER: {
                "structure": ["address_question", "provide_details", "offer_help"],
                "key_phrases": ["regarding your question", "the answer is", "hope this helps"]
            },
            ReplyType.FOLLOW_UP: {
                "structure": ["reference_previous", "provide_update", "next_actions"],
                "key_phrases": ["following up on", "wanted to update you", "next steps"]
            },
            ReplyType.DECLINE: {
                "structure": ["acknowledge_request", "polite_decline", "alternative_offer"],
                "key_phrases": ["thank you for", "unfortunately", "however, I can"]
            },
            ReplyType.SCHEDULE: {
                "structure": ["acknowledge_request", "availability", "confirmation"],
                "key_phrases": ["happy to meet", "available times", "please confirm"]
            }
        }
    
    async def generate_reply(self, context: ReplyContext, 
                           preferred_tone: Optional[ReplyTone] = None,
                           reply_type: Optional[ReplyType] = None,
                           custom_instructions: str = "") -> ReplyDraft:
        """Generate a reply draft with AI assistance"""
        try:
            # Analyze the original email for context
            analysis = await self._analyze_original_email(context)
            
            # Determine appropriate tone and type if not specified
            if not preferred_tone:
                preferred_tone = self._suggest_tone(context, analysis)
            
            if not reply_type:
                reply_type = self._suggest_reply_type(context, analysis)
            
            # Generate the reply using AI
            reply_draft = await self._generate_ai_reply(
                context, preferred_tone, reply_type, custom_instructions, analysis
            )
            
            return reply_draft
            
        except Exception as e:
            logger.error(f"Reply generation failed: {str(e)}")
            return self._create_fallback_reply(context, preferred_tone or ReplyTone.PROFESSIONAL)
    
    async def _analyze_original_email(self, context: ReplyContext) -> Dict:
        """Analyze the original email for better context understanding"""
        analysis_prompt = f"""
        Analyze this email for context and reply requirements:
        
        Subject: {context.subject}
        From: {context.sender} ({context.sender_email})
        Content: {context.content[:1500]}
        
        Analyze for:
        1. Questions asked (list them)
        2. Requests made (list them) 
        3. Sentiment/tone of sender
        4. Urgency level
        5. Relationship context (formal/informal)
        6. Key information that needs addressing
        
        Respond in JSON format.
        """
        
        try:
            response = await ai_manager.generate_response(analysis_prompt, temperature=0.3)
            
            # Try to parse JSON response
            import json
            json_match = re.search(r'\{.*\}', response, re.DOTALL)
            if json_match:
                return json.loads(json_match.group())
            
            # Fallback analysis
            return {
                "questions": self._extract_questions(context.content),
                "requests": self._extract_requests(context.content),
                "sentiment": "neutral",
                "urgency": "medium",
                "relationship": "professional"
            }
            
        except Exception as e:
            logger.warning(f"Email analysis failed: {str(e)}")
            return {"questions": [], "requests": [], "sentiment": "neutral"}
    
    def _extract_questions(self, content: str) -> List[str]:
        """Extract questions from email content"""
        questions = []
        sentences = content.split('.')
        
        for sentence in sentences:
            sentence = sentence.strip()
            if '?' in sentence:
                questions.append(sentence + '?')
        
        return questions[:5]  # Limit to 5 questions
    
    def _extract_requests(self, content: str) -> List[str]:
        """Extract requests from email content"""
        request_patterns = [
            r'please\s+([^.!?]+)',
            r'could you\s+([^.!?]+)',
            r'can you\s+([^.!?]+)',
            r'would you\s+([^.!?]+)',
            r'i need\s+([^.!?]+)',
            r'we need\s+([^.!?]+)'
        ]
        
        requests = []
        content_lower = content.lower()
        
        for pattern in request_patterns:
            matches = re.findall(pattern, content_lower)
            requests.extend(matches)
        
        return requests[:5]  # Limit to 5 requests
    
    def _suggest_tone(self, context: ReplyContext, analysis: Dict) -> ReplyTone:
        """Suggest appropriate tone based on context"""
        # Check sender relationship
        if context.sender_relationship in ['boss', 'executive', 'client']:
            return ReplyTone.PROFESSIONAL
        
        # Check domain for formality
        if context.sender_email.endswith(('.gov', '.edu')):
            return ReplyTone.FORMAL
        
        # Check urgency
        if context.priority_level == 'critical' or 'urgent' in analysis.get('sentiment', ''):
            return ReplyTone.URGENT
        
        # Check for apology needed
        apology_indicators = ['complaint', 'issue', 'problem', 'error', 'mistake']
        if any(indicator in context.content.lower() for indicator in apology_indicators):
            return ReplyTone.APOLOGETIC
        
        # Default to professional
        return ReplyTone.PROFESSIONAL
    
    def _suggest_reply_type(self, context: ReplyContext, analysis: Dict) -> ReplyType:
        """Suggest appropriate reply type based on content"""
        questions = analysis.get('questions', [])
        requests = analysis.get('requests', [])
        content_lower = context.content.lower()
        
        # Check for scheduling requests
        if any(word in content_lower for word in ['meet', 'schedule', 'appointment', 'call']):
            return ReplyType.SCHEDULE
        
        # Check if there are questions to answer
        if questions:
            return ReplyType.ANSWER
        
        # Check if there are requests to decline/accept
        if requests and any(word in content_lower for word in ['decline', 'cannot', "can't"]):
            return ReplyType.DECLINE
        
        # Check for follow-up scenarios
        if any(word in content_lower for word in ['update', 'status', 'progress']):
            return ReplyType.FOLLOW_UP
        
        # Check for thank you scenarios
        if any(word in content_lower for word in ['thank', 'appreciate', 'grateful']):
            return ReplyType.THANK_YOU
        
        # Default acknowledgment
        return ReplyType.ACKNOWLEDGMENT
    
    async def _generate_ai_reply(self, context: ReplyContext, tone: ReplyTone, 
                               reply_type: ReplyType, custom_instructions: str,
                               analysis: Dict) -> ReplyDraft:
        """Generate reply using AI with specific tone and type"""
        
        # Build comprehensive prompt
        tone_guide = self.tone_templates[tone]["style_guide"]
        reply_structure = self.reply_patterns[reply_type]["structure"]
        
        generation_prompt = f"""
        Generate a professional email reply with the following specifications:
        
        ORIGINAL EMAIL:
        Subject: {context.subject}
        From: {context.sender} ({context.sender_email})
        Content: {context.content}
        
        REPLY REQUIREMENTS:
        - Tone: {tone.value} ({tone_guide})
        - Type: {reply_type.value}
        - Structure: {' -> '.join(reply_structure)}
        
        CONTEXT:
        - Sender relationship: {context.sender_relationship}
        - Priority level: {context.priority_level}
        - Questions to address: {analysis.get('questions', [])}
        - Requests to handle: {analysis.get('requests', [])}
        
        CUSTOM INSTRUCTIONS: {custom_instructions}
        
        Generate:
        1. Appropriate subject line (if replying, start with "Re:")
        2. Email body following the tone and structure requirements
        3. Address all questions and requests appropriately
        
        Keep the reply concise but complete. Use proper email formatting.
        """
        
        try:
            # Generate with AI
            ai_response = await ai_manager.generate_response(generation_prompt, temperature=0.4)
            
            # Parse the response
            subject, body = self._parse_ai_reply(ai_response, context.subject)
            
            # Calculate metrics
            word_count = len(body.split())
            reading_time = max(1, word_count // 200)  # Assume 200 words per minute
            
            # Extract key points addressed
            key_points = self._extract_addressed_points(body, analysis)
            
            # Generate suggestions for improvement
            suggestions = await self._generate_reply_suggestions(body, tone, reply_type)
            
            return ReplyDraft(
                subject=subject,
                body=body,
                tone=tone,
                reply_type=reply_type,
                confidence=0.8,
                key_points_addressed=key_points,
                suggestions=suggestions,
                estimated_reading_time=reading_time,
                word_count=word_count
            )
            
        except Exception as e:
            logger.error(f"AI reply generation failed: {str(e)}")
            return self._create_fallback_reply(context, tone)
    
    def _parse_ai_reply(self, ai_response: str, original_subject: str) -> Tuple[str, str]:
        """Parse AI response into subject and body"""
        lines = ai_response.strip().split('\n')
        
        subject = f"Re: {original_subject}"
        body = ai_response
        
        # Look for subject line in response
        for i, line in enumerate(lines):
            if line.strip().lower().startswith(('subject:', 'subj:')):
                subject = line.split(':', 1)[1].strip()
                body = '\n'.join(lines[i+1:]).strip()
                break
            elif line.strip().startswith('Re:'):
                subject = line.strip()
                body = '\n'.join(lines[i+1:]).strip()
                break
        
        # Clean up body
        body = body.strip()
        
        # Ensure subject starts with "Re:" if it's a reply
        if not subject.startswith('Re:') and not original_subject.startswith('Re:'):
            subject = f"Re: {subject}" if not subject.startswith(original_subject) else f"Re: {original_subject}"
        
        return subject, body
    
    def _extract_addressed_points(self, reply_body: str, analysis: Dict) -> List[str]:
        """Extract key points that were addressed in the reply"""
        addressed_points = []
        body_lower = reply_body.lower()
        
        # Check if questions were addressed
        questions = analysis.get('questions', [])
        for question in questions[:3]:
            question_keywords = extract_key_phrases(question, max_phrases=2)
            if any(keyword.lower() in body_lower for keyword in question_keywords):
                addressed_points.append(f"Addressed question about {question_keywords[0]}")
        
        # Check if requests were handled
        requests = analysis.get('requests', [])
        for request in requests[:3]:
            request_keywords = extract_key_phrases(request, max_phrases=2)
            if any(keyword.lower() in body_lower for keyword in request_keywords):
                addressed_points.append(f"Responded to request: {request_keywords[0]}")
        
        return addressed_points
    
    async def _generate_reply_suggestions(self, reply_body: str, tone: ReplyTone, 
                                        reply_type: ReplyType) -> List[str]:
        """Generate suggestions for improving the reply"""
        suggestions = []
        
        # Tone-specific suggestions
        if tone == ReplyTone.PROFESSIONAL:
            suggestions.append("Consider making it more formal")
        elif tone == ReplyTone.CASUAL:
            suggestions.append("Add more personal touch")
        elif tone == ReplyTone.URGENT:
            suggestions.append("Emphasize urgency more clearly")
        
        # Length suggestions
        word_count = len(reply_body.split())
        if word_count < 50:
            suggestions.append("Consider adding more details")
        elif word_count > 200:
            suggestions.append("Consider making it more concise")
        
        # Generic suggestions
        suggestions.extend([
            "Add a specific call-to-action",
            "Include a deadline if relevant",
            "Proofread for clarity"
        ])
        
        return suggestions[:3]  # Return top 3 suggestions
    
    def _create_fallback_reply(self, context: ReplyContext, tone: ReplyTone) -> ReplyDraft:
        """Create a basic fallback reply when AI generation fails"""
        tone_config = self.tone_templates[tone]
        greeting = tone_config["greeting"][0].format(name=context.sender.split()[0])
        closing = tone_config["closing"][0]
        
        body = f"""{greeting},

Thank you for your email regarding "{context.subject}".

I have received your message and will review it carefully. I'll get back to you with a detailed response shortly.

{closing}"""
        
        return ReplyDraft(
            subject=f"Re: {context.subject}",
            body=body,
            tone=tone,
            reply_type=ReplyType.ACKNOWLEDGMENT,
            confidence=0.5,
            key_points_addressed=["Acknowledged receipt"],
            suggestions=["Add specific details", "Set expectation for response time"],
            estimated_reading_time=1,
            word_count=len(body.split())
        )
    
    async def generate_multiple_reply_options(self, context: ReplyContext, 
                                            num_options: int = 3) -> List[ReplyDraft]:
        """Generate multiple reply options with different tones/approaches"""
        try:
            # Define different approaches
            approaches = [
                (ReplyTone.PROFESSIONAL, ReplyType.ANSWER),
                (ReplyTone.FRIENDLY, ReplyType.ANSWER),
                (ReplyTone.CASUAL, ReplyType.ACKNOWLEDGMENT)
            ]
            
            # Generate multiple options
            tasks = []
            for tone, reply_type in approaches[:num_options]:
                task = self.generate_reply(context, tone, reply_type)
                tasks.append(task)
            
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # Filter successful results
            valid_results = [r for r in results if isinstance(r, ReplyDraft)]
            
            return valid_results
            
        except Exception as e:
            logger.error(f"Multiple reply generation failed: {str(e)}")
            return [self._create_fallback_reply(context, ReplyTone.PROFESSIONAL)]
    
    async def refine_reply(self, original_draft: ReplyDraft, 
                         refinement_instructions: str) -> ReplyDraft:
        """Refine an existing reply based on user instructions"""
        try:
            refinement_prompt = f"""
            Refine this email reply based on the following instructions:
            
            ORIGINAL REPLY:
            Subject: {original_draft.subject}
            Body: {original_draft.body}
            
            REFINEMENT INSTRUCTIONS: {refinement_instructions}
            
            Maintain the same general tone ({original_draft.tone.value}) but apply the requested changes.
            Return the refined subject and body.
            """
            
            refined_response = await ai_manager.generate_response(refinement_prompt, temperature=0.3)
            
            # Parse refined response
            subject, body = self._parse_ai_reply(refined_response, original_draft.subject)
            
            # Create refined draft
            refined_draft = ReplyDraft(
                subject=subject,
                body=body,
                tone=original_draft.tone,
                reply_type=original_draft.reply_type,
                confidence=original_draft.confidence * 0.9,  # Slightly lower confidence
                key_points_addressed=original_draft.key_points_addressed,
                suggestions=["Review refinements", "Compare with original"],
                estimated_reading_time=max(1, len(body.split()) // 200),
                word_count=len(body.split())
            )
            
            return refined_draft
            
        except Exception as e:
            logger.error(f"Reply refinement failed: {str(e)}")
            return original_draft  # Return original if refinement fails

# Global response generator instance
response_generator = ResponseGenerator()