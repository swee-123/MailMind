import streamlit as st
import os
import json
import base64
import email
import re
import io
import mimetypes
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import pandas as pd
import tempfile
import uuid

# Graceful imports with fallbacks
LANGCHAIN_AVAILABLE = False
AI_PROVIDERS = {}

try:
    from langchain.prompts import PromptTemplate
    from langchain.schema import HumanMessage
    LANGCHAIN_AVAILABLE = True
    
    # Try importing AI providers
    try:
        from langchain_groq import ChatGroq
        AI_PROVIDERS['groq'] = ChatGroq
    except ImportError:
        pass
    
    try:
        from langchain_openai import ChatOpenAI
        AI_PROVIDERS['openai'] = ChatOpenAI
    except ImportError:
        pass
    
    try:
        from langchain_anthropic import ChatAnthropic
        AI_PROVIDERS['anthropic'] = ChatAnthropic
    except ImportError:
        pass
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        AI_PROVIDERS['google'] = ChatGoogleGenerativeAI
    except ImportError:
        pass
        
except ImportError:
    st.error("Please install LangChain: pip install langchain==0.3.15")

try:
    import markitdown
    from markitdown import MarkItDown
    MARKITDOWN_AVAILABLE = True
except ImportError:
    MARKITDOWN_AVAILABLE = False

# Try to import reportlab for PDF export
REPORTLAB_AVAILABLE = False
try:
    import reportlab
    REPORTLAB_AVAILABLE = True
except ImportError:
    pass

def generate_unique_key(base_key: str) -> str:
    """Generate a unique key for Streamlit components"""
    return f"{base_key}_{uuid.uuid4().hex[:8]}"

class SetupHelper:
    """Helper class for setup and dependency checking"""
    
    @staticmethod
    def show_installation_guide():
        """Show comprehensive installation guide"""
        st.markdown("""
        ## 🔧 **Installation Guide**
        
        ### **Step 1: Install Core Packages**
        ```bash
        pip install streamlit==1.28.0 pandas python-dateutil python-dotenv
        pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client
        pip install langchain==0.3.15
        ```
        
        ### **Step 2: Choose and Install AI Provider**
        
        #### **🚀 Option 1: Groq (Recommended - Free & Fast)**
        ```bash
        pip install langchain-groq
        ```
        - Get free API key from: [console.groq.com](https://console.groq.com/)
        - **Latest Models**: llama-3.3-70b-versatile, llama-3.1-8b-instant
        - Very fast inference, excellent for email processing
        
        #### **🧠 Option 2: OpenAI**
        ```bash
        pip install langchain-openai
        ```
        - Get API key from: [platform.openai.com](https://platform.openai.com/)
        - **Latest Models**: gpt-4o, gpt-4o-mini (cost-effective), gpt-4-turbo
        - Paid service (~$0.15-3.00 per 1M tokens depending on model)
        
        #### **🎯 Option 3: Anthropic Claude**
        ```bash
        pip install langchain-anthropic
        ```
        - Get API key from: [console.anthropic.com](https://console.anthropic.com/)
        - **Latest Models**: claude-3-5-sonnet-20241022, claude-3-5-haiku-20241022
        - Excellent for professional communication
        
        #### **🌟 Option 4: Google Gemini**
        ```bash
        pip install langchain-google-genai
        ```
        - Get API key from: [makersuite.google.com](https://makersuite.google.com/)
        - **Latest Models**: gemini-1.5-flash, gemini-1.5-pro
        - Free tier available with good performance
        
        ### **Step 3: Set API Key**
        
        #### **Method 1: Environment Variable**
        
        **Windows:**
        ```cmd
        set GROQ_API_KEY=your_api_key_here
        set OPENAI_API_KEY=your_api_key_here
        set ANTHROPIC_API_KEY=your_api_key_here
        set GOOGLE_API_KEY=your_api_key_here
        ```
        
        **Mac/Linux:**
        ```bash
        export GROQ_API_KEY=your_api_key_here
        export OPENAI_API_KEY=your_api_key_here
        export ANTHROPIC_API_KEY=your_api_key_here
        export GOOGLE_API_KEY=your_google_api_key_here
        ```
        
        #### **Method 2: .env File**
        Create a `.env` file in your project directory:
        ```env
        GROQ_API_KEY=your_groq_api_key_here
        OPENAI_API_KEY=your_openai_api_key_here
        ANTHROPIC_API_KEY=your_anthropic_api_key_here
        GOOGLE_API_KEY=your_google_api_key_here
        ```
        
        ### **Step 4: Optional Packages**
        ```bash
        pip install markitdown "Pillow<10"  # For attachment processing
        ```
        
        ### **Step 5: Gmail API Setup**
        
        1. **Go to Google Cloud Console**: https://console.cloud.google.com/
        2. **Create or select a project**
        3. **Enable Gmail API**
        4. **Create OAuth Credentials**
        5. **Download credentials.json** and place in credentials/ folder
        
        ### **Step 6: Restart Application**
        ```bash
        streamlit run streamlit_app.py
        ```
        """)

class ImprovedGmailAuth:
    """Improved Gmail authentication with better error handling"""
    
    def __init__(self):
        self.credentials_path = "credentials/credentials.json"
        self.token_path = "credentials/token.json"
        self._validate_setup()
    
    def _validate_setup(self):
        """Validate the authentication setup"""
        credentials_dir = Path("credentials")
        if not credentials_dir.exists():
            st.error("🔐 **Missing credentials folder!**")
            st.error("Please create a 'credentials' folder in your project directory.")
            return False
        
        if not os.path.exists(self.credentials_path):
            st.error("📄 **Missing credentials.json file!**")
            with st.expander("🔧 **Gmail Setup Guide**"):
                st.markdown("""
                ### **Gmail API Setup:**
                
                1. **Go to Google Cloud Console**: https://console.cloud.google.com/
                2. **Create or select a project**
                3. **Enable Gmail API**:
                   - Go to "APIs & Services" > "Library"
                   - Search for "Gmail API" and enable it
                4. **Create OAuth Credentials**:
                   - Go to "APIs & Services" > "Credentials"
                   - Click "Create Credentials" > "OAuth 2.0 Client IDs"
                   - Choose "Desktop application"
                   - Download the JSON file
                5. **Setup locally**:
                   - Rename downloaded file to `credentials.json`
                   - Place it in the `credentials/` folder in your project
                """)
            return False
        
        try:
            with open(self.credentials_path, 'r') as f:
                creds_data = json.load(f)
                
            if 'installed' not in creds_data and 'web' not in creds_data:
                st.error("❌ **Invalid credentials.json format!**")
                return False
                
        except json.JSONDecodeError:
            st.error("❌ **Corrupted credentials.json file!**")
            return False
        except Exception as e:
            st.error(f"❌ **Error reading credentials.json**: {str(e)}")
            return False
        
        return True
    
    def authenticate(self):
        """Authenticate with Gmail API"""
        try:
            from google.auth.transport.requests import Request
            from google.oauth2.credentials import Credentials
            from google_auth_oauthlib.flow import InstalledAppFlow
            from googleapiclient.discovery import build
            
            SCOPES = [
                'https://www.googleapis.com/auth/gmail.readonly',
                'https://www.googleapis.com/auth/gmail.modify',
                'https://www.googleapis.com/auth/gmail.send'
            ]
            
            creds = None
            
            if os.path.exists(self.token_path):
                try:
                    creds = Credentials.from_authorized_user_file(self.token_path, SCOPES)
                    st.info("📄 Found existing authentication token...")
                except Exception as e:
                    st.warning(f"⚠️ Existing token is invalid: {str(e)}")
                    if os.path.exists(self.token_path):
                        os.remove(self.token_path)
                    creds = None
            
            if not creds or not creds.valid:
                if creds and creds.expired and creds.refresh_token:
                    try:
                        st.info("📄 Refreshing expired token...")
                        creds.refresh(Request())
                        st.success("✅ Token refreshed successfully!")
                    except Exception:
                        creds = None
                
                if not creds:
                    st.info("🔍 Starting OAuth flow...")
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, SCOPES)
                        force_select = st.session_state.get('force_account_select', False)
                        prompt_type = 'select_account' if force_select else 'consent'
                        if 'force_account_select' in st.session_state:
                            del st.session_state['force_account_select']
                        creds = flow.run_local_server(
                            port=8080,
                            prompt=prompt_type,
                            authorization_prompt_message='Please visit this URL to authorize: {url}',
                            success_message='Authorization complete! You may close this window.',
                            open_browser=True
                        )
                        st.success("✅ OAuth flow completed successfully!")
                        
                    except OSError as e:
                        if "Address already in use" in str(e):
                            st.warning("📄 Port 8080 busy, trying port 8081...")
                            creds = flow.run_local_server(port=8081, prompt='consent', open_browser=True)
                        else:
                            st.error(f"❌ OAuth error: {str(e)}")
                            return None
                
                # Save credentials
                try:
                    os.makedirs(os.path.dirname(self.token_path), exist_ok=True)
                    with open(self.token_path, 'w') as token:
                        token.write(creds.to_json())
                    st.success("💾 Authentication token saved!")
                except Exception as e:
                    st.warning(f"⚠️ Could not save token: {str(e)}")
            
            # Build Gmail service
            service = build('gmail', 'v1', credentials=creds)
            profile = service.users().getProfile(userId='me').execute()
            email_address = profile.get('emailAddress', 'Unknown')
            
            st.success(f"🎉 **Successfully connected to Gmail!**")
            st.success(f"📧 **Account**: {email_address}")
            
            return service
            
        except ImportError:
            st.error("📦 **Missing Google API libraries!**")
            st.code("pip install google-auth google-auth-oauthlib google-auth-httplib2 google-api-python-client")
            return None
        except Exception as e:
            st.error(f"💥 **Authentication error**: {str(e)}")
            return None

class AIProvider:
    """Enhanced AI provider with better error handling"""
    
    def __init__(self):
        self.provider = None
        self.llm = None
        self._initialize_provider()
    
    def _initialize_provider(self):
        """Initialize AI provider based on available packages and API keys"""
        if not LANGCHAIN_AVAILABLE:
            st.error("❌ **LangChain not installed!**")
            st.code("pip install langchain==0.3.15")
            return
        
        # Check for API keys and available providers with updated models
        providers_to_try = [
            ('GROQ_API_KEY', 'groq', 'Groq', 'llama-3.3-70b-versatile'),
            ('OPENAI_API_KEY', 'openai', 'OpenAI', 'gpt-4o-mini'),
            ('ANTHROPIC_API_KEY', 'anthropic', 'Anthropic', 'claude-3-5-sonnet-20241022'),
            ('GOOGLE_API_KEY', 'google', 'Google Gemini', 'gemini-1.5-flash')
        ]
        
        for env_key, provider_key, provider_name, model in providers_to_try:
            api_key = os.getenv(env_key)
            if api_key and provider_key in AI_PROVIDERS:
                try:
                    if provider_key == 'groq':
                        self.llm = AI_PROVIDERS[provider_key](
                            api_key=api_key, 
                            model_name=model,
                            temperature=0.3
                        )
                    elif provider_key == 'openai':
                        self.llm = AI_PROVIDERS[provider_key](
                            api_key=api_key, 
                            model_name=model,
                            temperature=0.3
                        )
                    elif provider_key == 'anthropic':
                        self.llm = AI_PROVIDERS[provider_key](
                            api_key=api_key, 
                            model_name=model,
                            temperature=0.3
                        )
                    elif provider_key == 'google':
                        self.llm = AI_PROVIDERS[provider_key](
                            api_key=api_key, 
                            model=model,
                            temperature=0.3
                        )
                    
                    self.provider = provider_name
                    self.model_name = model
                    return
                    
                except Exception as e:
                    st.warning(f"Failed to initialize {provider_name} with {model}: {str(e)}")
                    # Try fallback models for Groq
                    if provider_key == 'groq':
                        fallback_models = ['llama-3.1-8b-instant', 'llama3-groq-70b-8192-tool-use-preview']
                        for fallback_model in fallback_models:
                            try:
                                self.llm = AI_PROVIDERS[provider_key](
                                    api_key=api_key, 
                                    model_name=fallback_model,
                                    temperature=0.3
                                )
                                self.provider = provider_name
                                self.model_name = fallback_model
                                return
                            except Exception as fallback_e:
                                continue
                    continue
        
        # Show what's missing
        missing_packages = []
        missing_keys = []
        
        for env_key, provider_key, provider_name, model in providers_to_try:
            if provider_key not in AI_PROVIDERS:
                missing_packages.append(f"langchain-{provider_key}")
            elif not os.getenv(env_key):
                missing_keys.append(env_key)
        
        if missing_packages:
            st.warning(f"📦 **Missing AI packages**: {', '.join(missing_packages)}")
        
        if missing_keys:
            st.error(f"🔐 **Missing API keys**: {', '.join(missing_keys)}")
    
    def is_available(self):
        """Check if AI provider is available"""
        return self.llm is not None
    
    def generate_response(self, prompt: str) -> str:
        """Generate response using available AI provider"""
        if not self.is_available():
            return "❌ AI provider not available"
        
        try:
            if hasattr(self.llm, 'invoke'):
                response = self.llm.invoke([HumanMessage(content=prompt)])
                return response.content
            else:
                # Fallback for different LangChain versions
                response = self.llm(prompt)
                return response
        except Exception as e:
            return f"❌ AI generation error: {str(e)}"

class EnhancedEmailProcessor:
    """Enhanced email processor with complete attachment and reply functionality"""
    
    def __init__(self, gmail_service, ai_provider=None):
        self.gmail_service = gmail_service
        self.ai_provider = ai_provider
        self.draft_folder = Path("drafts")
        self.attachments_folder = Path("downloads")
        self._ensure_folders()
    
    def _ensure_folders(self):
        """Ensure required folders exist"""
        self.draft_folder.mkdir(exist_ok=True)
        self.attachments_folder.mkdir(exist_ok=True)
    
    def fetch_emails(self, time_filter: str = "24 Hours", max_results: int = 50) -> List[Dict]:
        """Fetch emails from Gmail with time filtering and improved error handling"""
        try:
            now = datetime.now()
            time_filters = {
                "1 Hour": now - timedelta(hours=1),
                "6 Hours": now - timedelta(hours=6), 
                "24 Hours": now - timedelta(hours=24),
                "7 Days": now - timedelta(days=7),
                "30 Days": now - timedelta(days=30)
            }
            
            after_date = time_filters.get(time_filter, time_filters["24 Hours"])
            query = f'after:{after_date.strftime("%Y/%m/%d")}'
            
            # Ensure max_results is within reasonable bounds
            safe_max_results = min(max(max_results, 1), 500)  # Between 1 and 500
            
            results = self.gmail_service.users().messages().list(
                userId='me', 
                q=query,
                maxResults=safe_max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            if not messages:
                return emails
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process emails in smaller batches to avoid timeouts
            batch_size = 10
            for batch_start in range(0, len(messages), batch_size):
                batch_end = min(batch_start + batch_size, len(messages))
                batch_messages = messages[batch_start:batch_end]
                
                for i, message in enumerate(batch_messages):
                    try:
                        msg = self.gmail_service.users().messages().get(
                            userId='me', 
                            id=message['id']
                        ).execute()
                        
                        email_data = self._parse_email(msg)
                        if email_data:
                            emails.append(email_data)
                        
                        overall_progress = (batch_start + i + 1) / len(messages)
                        progress_bar.progress(overall_progress)
                        status_text.text(f"Processing email {batch_start + i + 1} of {len(messages)}")
                        
                    except Exception as e:
                        # Log individual email errors but continue
                        st.warning(f"Failed to process email {batch_start + i + 1}: {str(e)}")
                        continue
            
            progress_bar.empty()
            status_text.empty()
            
            return emails
            
        except Exception as e:
            st.error(f"Error fetching emails: {str(e)}")
            return []
    
    def _parse_email(self, msg: Dict) -> Optional[Dict]:
        """Parse Gmail message into structured data"""
        try:
            headers = msg['payload'].get('headers', [])
            header_dict = {h['name']: h['value'] for h in headers}
            
            body = self._extract_body(msg['payload'])
            attachments = self._extract_attachments_info(msg['payload'])
            
            email_data = {
                'id': msg['id'],
                'thread_id': msg.get('threadId', ''),
                'from': header_dict.get('From', ''),
                'to': header_dict.get('To', ''),
                'subject': header_dict.get('Subject', 'No Subject'),
                'date': header_dict.get('Date', ''),
                'body': body,
                'snippet': msg.get('snippet', ''),
                'labels': msg.get('labelIds', []),
                'attachments': attachments,
                'is_unread': 'UNREAD' in msg.get('labelIds', []),
                'is_important': 'IMPORTANT' in msg.get('labelIds', []),
                'raw_message': msg,
                'priority_score': self._calculate_improved_priority(header_dict, body, msg),
                'needs_reply': self._improved_reply_detection(body, header_dict),
                'ai_summary': msg.get('snippet', '')[:100] + '...'
            }
            
            return email_data
            
        except Exception as e:
            return None
    
    def _calculate_improved_priority(self, headers: Dict, body: str, msg: Dict) -> int:
        """Improved priority calculation that deprioritizes noreply emails"""
        score = 5  # Base score
        
        # Check sender for noreply patterns first
        sender = headers.get('From', '').lower()
        noreply_patterns = [
            'noreply', 'no-reply', 'donotreply', 'do-not-reply', 'no_reply',
            'notifications@', 'automated@', 'newsletter@', 'marketing@',
            'system@', 'bot@', 'mailer@', 'updates@', 'alerts@'
        ]
         
        # If it's a noreply email, set very low priority
        is_noreply = any(pattern in sender for pattern in noreply_patterns)
    
        if is_noreply:
            return 2  # Cap noreply emails at priority 2
    
        # Gmail labels influence
        if 'UNREAD' in msg.get('labelIds', []):
            score += 1

        if 'IMPORTANT' in msg.get('labelIds', []):
            score += 2

        # Subject line analysis
        subject = headers.get('Subject', '').lower()

        # Check if subject indicates automated/promotional content
        automated_subject_keywords = [
            'newsletter', 'unsubscribe', 'promotion', 'offer', 'sale',
            'marketing', 'advertisement', 'spam', 'bulk', 'automated',
            'notification', 'alert', 'reminder', 'receipt', 'invoice'
        ]

        # Reduce priority for automated emails
        if any(kw in subject for kw in automated_subject_keywords):
            score -= 2

        return min(max(score, 1), 10)
    
    def _improved_reply_detection(self, body: str, headers: Dict) -> bool:
        """Improved reply detection that excludes noreply emails"""
    
        # First check if it's a noreply email
        sender = headers.get('From', '').lower()
        noreply_patterns = [
            'noreply', 'no-reply', 'donotreply', 'do-not-reply', 'no_reply',
            'notifications@', 'automated@', 'newsletter@', 'marketing@',
            'system@', 'bot@', 'mailer@', 'updates@', 'alerts@'
        ]
    
        # If it's a noreply email, it never needs a reply
        if any(pattern in sender for pattern in noreply_patterns):
            return False
    
        content = f"{headers.get('Subject', '')} {body}".lower()
    
        # Check for automated/promotional content that doesn't need replies
        automated_content = [
            'unsubscribe', 'this is an automated', 'do not reply',
            'auto-generated', 'automated message', 'newsletter',
            'promotional', 'marketing', 'advertisement'
        ]
    
        if any(phrase in content for phrase in automated_content):
            return False
    
        # Direct question detection
        if '?' in content:
            return True
    
        # Action request patterns
        action_patterns = [
            'please', 'can you', 'could you', 'would you', 'will you',
            'let me know', 'get back to me', 'respond', 'reply', 
            'confirm', 'schedule', 'send me', 'provide', 'share',
            'need your', 'waiting for', 'expecting', 'required'
        ]
    
        for pattern in action_patterns:
            if pattern in content:
                return True
    
        # Meeting/appointment requests
        meeting_patterns = [
            'meeting', 'call', 'schedule', 'appointment', 'available',
            'free time', 'calendar', 'book', 'arrange'
        ]
    
        for pattern in meeting_patterns:
            if pattern in content:
                return True
    
        # No clear reply needed
        if any(phrase in content for phrase in ['fyi', 'for your information', 'just to let you know']):
            return False
    
        return False
    
    def _extract_body(self, payload: Dict) -> str:
        """Extract email body from payload"""
        body = ""
        if 'parts' in payload:
            for part in payload['parts']:
                if part['mimeType'] in ['text/plain', 'text/html']:
                    data = part['body'].get('data', '')
                    if data:
                        decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                        if part['mimeType'] == 'text/html':
                            import re
                            decoded = re.sub(r'<[^>]+>', '', decoded)
                        body += decoded
                if 'parts' in part:
                    body += self._extract_body(part)
        else:
            if payload['mimeType'] in ['text/plain', 'text/html']:
                data = payload['body'].get('data', '')
                if data:
                    decoded = base64.urlsafe_b64decode(data).decode('utf-8', errors='ignore')
                    if payload['mimeType'] == 'text/html':
                        import re
                        decoded = re.sub(r'<[^>]+>', '', decoded)
                    body = decoded

        return body.strip()
    
    def _extract_attachments_info(self, payload: Dict) -> List[Dict]:
        """Extract attachment information"""
        attachments = []
        
        def process_part(part):
            filename = part.get('filename', '')
            if filename:
                attachments.append({
                    'filename': filename,
                    'mimeType': part.get('mimeType', ''),
                    'size': part['body'].get('size', 0),
                    'attachmentId': part['body'].get('attachmentId', '')
                })
            
            if 'parts' in part:
                for subpart in part['parts']:
                    process_part(subpart)
        
        if 'parts' in payload:
            for part in payload['parts']:
                process_part(part)
        
        return attachments
    
    def download_attachment(self, email_id: str, attachment_id: str, filename: str) -> Optional[str]:
        """Download email attachment"""
        try:
            attachment = self.gmail_service.users().messages().attachments().get(
                userId='me',
                messageId=email_id,
                id=attachment_id
            ).execute()
            
            file_data = base64.urlsafe_b64decode(attachment['data'])
            
            # Ensure safe filename
            safe_filename = re.sub(r'[^\w\-_\.]', '_', filename)
            file_path = self.attachments_folder / safe_filename
            
            # Handle duplicate filenames
            counter = 1
            original_path = file_path
            while file_path.exists():
                stem = original_path.stem
                suffix = original_path.suffix
                file_path = original_path.parent / f"{stem}_{counter}{suffix}"
                counter += 1
            
            with open(file_path, 'wb') as f:
                f.write(file_data)
            
            return str(file_path)
            
        except Exception as e:
            st.error(f"Error downloading attachment: {str(e)}")
            return None
    
    def save_draft(self, subject: str, body: str, to_email: str = "", reply_to_id: str = None) -> str:
        """Save email as draft in Gmail"""
        try:
            # Create draft message
            draft_message = {
                'raw': self._create_message(to_email, subject, body)
            }
            
            if reply_to_id:
                draft_message['threadId'] = reply_to_id
            
            draft = self.gmail_service.users().drafts().create(
                userId='me',
                body={'message': draft_message}
            ).execute()
            
            return draft['id']
            
        except Exception as e:
            st.error(f"Error saving draft: {str(e)}")
            return None
    
    def send_email(self, to_email: str, subject: str, body: str, reply_to_id: str = None) -> bool:
        """Send email via Gmail"""
        try:
            message = {
                'raw': self._create_message(to_email, subject, body)
            }
            
            if reply_to_id:
                message['threadId'] = reply_to_id
            
            sent_message = self.gmail_service.users().messages().send(
                userId='me',
                body=message
            ).execute()
            
            return sent_message['id'] is not None
            
        except Exception as e:
            st.error(f"Error sending email: {str(e)}")
            return False
    
    def _create_message(self, to: str, subject: str, message_text: str) -> str:
        """Create email message in proper format"""
        import email.mime.text
        
        message = email.mime.text.MIMEText(message_text)
        message['to'] = to
        message['subject'] = subject
        
        # Get sender email
        try:
            profile = self.gmail_service.users().getProfile(userId='me').execute()
            sender_email = profile.get('emailAddress', 'unknown@gmail.com')
            message['from'] = sender_email
        except:
            message['from'] = 'unknown@gmail.com'
        
        raw_message = base64.urlsafe_b64encode(message.as_bytes()).decode()
        return raw_message
    
    def enhance_with_ai(self, emails: List[Dict]) -> List[Dict]:
        """Enhance emails with AI analysis if available"""
        if not self.ai_provider or not self.ai_provider.is_available():
            return emails
    
        enhanced_emails = []
    
        for email_data in emails:
            # Store the original priority score from local calculation
            original_priority = email_data.get('priority_score', 5)
        
            # Check if this is a noreply email first
            sender = email_data.get('from', '').lower()
            noreply_patterns = [
                'noreply', 'no-reply', 'donotreply', 'do-not-reply', 'no_reply',
                'notifications@', 'automated@', 'newsletter@', 'marketing@',
                'system@', 'bot@', 'mailer@', 'updates@', 'alerts@'
            ]
        
            is_noreply = any(pattern in sender for pattern in noreply_patterns)
        
            # Only use AI priority scoring for non-noreply emails
            if not is_noreply:
                # AI Priority scoring
                priority_prompt = f"""
                Analyze this email and rate its priority from 1-10 (10 being most urgent):
            
                Subject: {email_data['subject']}
                From: {email_data['from']}
                Snippet: {email_data['snippet']}
            
                Consider factors like:
                - Urgency indicators (urgent, asap, deadline)
                - Sender importance
                - Meeting requests
                - Action required
            
                Respond with just a number from 1-10.
                """
            
                try:
                    priority_response = self.ai_provider.generate_response(priority_prompt)
                    priority_match = re.search(r'\b(\d+)\b', priority_response)
                    if priority_match:
                        ai_priority = min(max(int(priority_match.group(1)), 1), 10)
                        # Use the lower of AI priority and original priority for safety
                        email_data['priority_score'] = min(ai_priority, original_priority)
                except:
                    # Keep original priority if AI fails
                    email_data['priority_score'] = original_priority
            else:
                # For noreply emails, force low priority regardless of AI
                email_data['priority_score'] = min(original_priority, 2)
        
            # AI Summary
            summary_prompt = f"""
            Provide a concise 1-2 sentence summary of this email:
        
            Subject: {email_data['subject']}
            From: {email_data['from']}
            Body: {email_data['body'][:500]}...
        
            Focus on key action items or main points.
            """
        
            try:
                summary = self.ai_provider.generate_response(summary_prompt)
                email_data['ai_summary'] = summary[:200] + "..." if len(summary) > 200 else summary
            except:
                pass
        
            # Reply detection - but don't let AI override noreply detection
            if not is_noreply:
                reply_prompt = f"""
                Does this email require a reply? Answer with just "YES" or "NO".
            
                Subject: {email_data['subject']}
                Body: {email_data['body'][:300]}...
            
                Consider:
                - Questions asked
                - Requests made
                - Meeting invitations
                - Action items for you
                """
            
                try:
                    reply_response = self.ai_provider.generate_response(reply_prompt)
                    email_data['needs_reply'] = 'yes' in reply_response.lower()
                except:
                    pass
            else:
                # Noreply emails never need replies
                email_data['needs_reply'] = False
        
            enhanced_emails.append(email_data)
    
        return enhanced_emails

class MailMindApp:
    """Main MailMind application with complete functionality"""
    
    def __init__(self):
        self.setup_page_config()
        self.initialize_session_state()
        self.check_dependencies()
    
    def setup_page_config(self):
        """Configure Streamlit page"""
        st.set_page_config(
            page_title="MailMind - Smart Email Prioritizer",
            page_icon="📧",
            layout="wide",
            initial_sidebar_state="collapsed"
        )
    
    def initialize_session_state(self):
        """Initialize session state variables"""
        defaults = {
            'gmail_service': None,
            'authenticated': False,
            'emails': [],
            'processed_emails': [],
            'selected_email': None,
            'setup_complete': False,
            'ai_provider': None,
            'current_view': 'dashboard',
            'email_processor': None,
            'connected_accounts': []  # Track multiple accounts
        }
        
        for key, value in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = value
    
    def check_dependencies(self):
        """Check if all dependencies are installed"""
        missing_deps = []
        
        try:
            import google.auth
        except ImportError:
            missing_deps.append("Google API libraries")
        
        if not LANGCHAIN_AVAILABLE:
            missing_deps.append("LangChain")
        
        if missing_deps:
            st.error("❌ **Missing Dependencies**")
            for dep in missing_deps:
                st.error(f"- {dep}")
            
            SetupHelper.show_installation_guide()
            st.session_state.setup_complete = False
        else:
            st.session_state.setup_complete = True
    
    def render_setup_page(self):
        """Render setup and dependency check page"""
        st.title("🔧 MailMind - Setup Required")
        
        st.markdown("### 📋 Dependency Check")
        
        # Check packages
        checks = [
            ("Core Streamlit packages", True),
            ("Google API libraries", self._check_google_apis()),
            ("LangChain", LANGCHAIN_AVAILABLE),
            ("AI Provider packages", len(AI_PROVIDERS) > 0),
        ]
        
        all_good = True
        for name, status in checks:
            icon = "✅" if status else "❌"
            st.markdown(f"{icon} {name}")
            if not status:
                all_good = False
        
        st.markdown("### 🔐 API Key Check")
        
        # Check API keys
        api_keys = {
            'GROQ_API_KEY': 'Groq (Recommended)',
            'OPENAI_API_KEY': 'OpenAI', 
            'ANTHROPIC_API_KEY': 'Anthropic Claude',
            'GOOGLE_API_KEY': 'Google Gemini'
        }
        
        available_keys = []
        for key, name in api_keys.items():
            if os.getenv(key):
                st.markdown(f"✅ {name}")
                available_keys.append(name)
            else:
                st.markdown(f"❌ {name}")
        
        if not available_keys:
            st.error("🔐 **No AI API keys found!**")
            all_good = False
        
        st.markdown("---")
        
        if not all_good:
            st.error("⚠️ **Setup incomplete!** Please follow the installation guide below.")
            SetupHelper.show_installation_guide()
            
            if st.button("🔄 Recheck Dependencies", key="recheck_deps"):
                st.rerun()
        else:
            st.success("🎉 **All dependencies satisfied!** Ready to proceed.")
            if st.button("▶️ Continue to MailMind", type="primary", key="continue_to_app"):
                st.session_state.setup_complete = True
                st.rerun()
    
    def _check_google_apis(self) -> bool:
        """Check if Google API libraries are available"""
        try:
            import google.auth
            import google.oauth2.credentials
            import googleapiclient.discovery
            return True
        except ImportError:
            return False
    
    def render_authentication(self):
        """Enhanced authentication interface supporting multiple accounts"""
        st.title("📧 MailMind - Smart Email Prioritizer & AI Assistant")
        
        # Initialize components
        auth = ImprovedGmailAuth()
        ai_provider = AIProvider()
        
        # Show connected accounts if any
        if st.session_state.connected_accounts:
            st.markdown("### 📱 Connected Accounts")
            for account in st.session_state.connected_accounts:
                st.success(f"✅ {account}")
            
            if st.button("➕ Add Another Account", key="add_another_account"):
                # Clear current session for new account
                if os.path.exists("credentials/token.json"):
                    os.remove("credentials/token.json")
                st.info("Please authenticate with a new account")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            # Check prerequisites
            creds_folder_exists = Path("credentials").exists()
            creds_file_exists = os.path.exists("credentials/credentials.json")
            
            if not all([creds_folder_exists, creds_file_exists]):
                st.error("⚠️ **Gmail setup incomplete!**")
                return None
            
            if not ai_provider.is_available():
                st.warning("⚠️ **AI features limited without API key**")
                st.info("The app will work with basic email management, but AI features will be disabled.")
            
            # Account selection options
            st.markdown("### 📧 Account Selection")
            account_option = st.radio(
                "Choose account option:",
                ["📄 Use existing account", "➕ Add new account", "🔀 Switch account"],
                horizontal=True,
                key="account_selection_radio"
            )

            # Connection button with dynamic behavior
            if account_option == "➕ Add new account":
                button_text = "➕ Add New Gmail Account"
                force_account_chooser = True
            elif account_option == "🔀 Switch account":
                button_text = "🔀 Switch Gmail Account" 
                force_account_chooser = True
            else:
                button_text = "🔗 Connect Gmail Account"
                force_account_chooser = False
            
            if st.button(button_text, type="primary", use_container_width=True, key="connect_gmail_btn"):
                with st.spinner("🔍 Authenticating with Gmail..."):
                    if force_account_chooser and os.path.exists("credentials/token.json"):
                        os.remove("credentials/token.json")
                        st.session_state['force_account_select'] = True
    
                    service = auth.authenticate()
                    
                    if service:
                        try:
                            profile = service.users().getProfile(userId='me').execute()
                            email_address = profile.get('emailAddress', 'Unknown')
                            
                            # Add to connected accounts if not already present
                            if email_address not in st.session_state.connected_accounts:
                                st.session_state.connected_accounts.append(email_address)
                            
                            st.session_state.gmail_service = service
                            st.session_state.authenticated = True
                            st.session_state.ai_provider = ai_provider
                            st.session_state.email_processor = EnhancedEmailProcessor(service, ai_provider)
                            st.rerun()
                        except Exception as e:
                            st.error(f"Failed to get profile: {str(e)}")
                    else:
                        st.error("Authentication failed!")
        
        return None
    
    def render_main_dashboard(self):
        """Render main email dashboard"""
        email_processor = st.session_state.email_processor or EnhancedEmailProcessor(
            st.session_state.gmail_service, 
            st.session_state.ai_provider
        )
        
        st.title("📧 MailMind Dashboard")
        
        # Connection status and controls
        col1, col2, col3, col4 = st.columns([2, 2, 2, 1])
        
        with col1:
            try:
                profile = st.session_state.gmail_service.users().getProfile(userId='me').execute()
                st.success(f"✅ Connected: {profile.get('emailAddress', 'Unknown')}")
            except:
                st.error("❌ Connection issue")
        
        with col2:
            time_filter = st.selectbox(
                "📅 Time Filter",
                ["1 Hour", "6 Hours", "24 Hours", "7 Days", "30 Days"],
                index=2,
                key="time_filter_select"
            )
        
        with col3:
            max_emails = st.number_input(
                "📊 Max Emails",
                min_value=10,
                max_value=500,  # Increased limit but with safety checks
                value=50,
                key="max_emails_input"
            )
        
        with col4:
            if st.button("🔄 Refresh", key="refresh_emails_btn"):
                st.session_state.emails = []
                st.session_state.processed_emails = []
                st.rerun()
        
        # Navigation tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📥 Inbox", "⚡ Priority", "📊 Analytics", "⚙️ Settings"])
        
        with tab1:
            self.render_inbox_tab(email_processor, time_filter, max_emails)
        
        with tab2:
            self.render_priority_tab()
        
        with tab3:
            self.render_analytics_tab()
        
        with tab4:
            self.render_settings_tab()
    
    def render_inbox_tab(self, email_processor, time_filter, max_emails):
        """Render the main inbox tab"""
        
        # Load emails if not already loaded
        if not st.session_state.emails:
            with st.spinner("📧 Fetching emails..."):
                try:
                    emails = email_processor.fetch_emails(time_filter, max_emails)
                    st.session_state.emails = emails
                    
                    if emails:
                        # Enhance with AI if available (limit to first 20 for performance)
                        if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
                            with st.spinner("🤖 Enhancing with AI..."):
                                enhanced_emails = email_processor.enhance_with_ai(emails[:20])
                                for i, enhanced in enumerate(enhanced_emails):
                                    if i < len(st.session_state.emails):
                                        st.session_state.emails[i].update(enhanced)
                    else:
                        st.info("📭 No emails found in the selected time range.")
                        return
                        
                except Exception as e:
                    st.error(f"❌ Failed to fetch emails: {str(e)}")
                    if "quota" in str(e).lower():
                        st.error("Gmail API quota exceeded. Please try again later or reduce the number of emails.")
                    return
        
        emails = st.session_state.emails
        
        if not emails:
            st.info("📭 No emails found in the selected time range.")
            return
        
        # Email summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("📧 Total Emails", len(emails))
        
        with col2:
            unread_count = sum(1 for email in emails if email.get('is_unread', False))
            st.metric("📩 Unread", unread_count)
        
        with col3:
            high_priority = sum(1 for email in emails if email.get('priority_score', 5) >= 8)
            st.metric("🔥 High Priority", high_priority)
        
        with col4:
            needs_reply = sum(1 for email in emails if email.get('needs_reply', False))
            st.metric("↩️ Needs Reply", needs_reply)
        
        st.markdown("---")
        
        # Filters
        col1, col2, col3 = st.columns(3)
        
        with col1:
            show_unread_only = st.checkbox("📩 Unread only", key="filter_unread_only")
        
        with col2:
            show_high_priority = st.checkbox("🔥 High priority only", key="filter_high_priority")
        
        with col3:
            show_needs_reply = st.checkbox("↩️ Needs reply only", key="filter_needs_reply")
        
        # Filter emails
        filtered_emails = emails
        
        if show_unread_only:
            filtered_emails = [e for e in filtered_emails if e.get('is_unread', False)]
        
        if show_high_priority:
            filtered_emails = [e for e in filtered_emails if e.get('priority_score', 5) >= 8]
        
        if show_needs_reply:
            filtered_emails = [e for e in filtered_emails if e.get('needs_reply', False)]
        
        # Sort by priority score
        filtered_emails.sort(key=lambda x: x.get('priority_score', 5), reverse=True)
        
        # Email list
        st.markdown("### 📧 Email List")
        
        if not filtered_emails:
            st.info("📭 No emails match the selected filters.")
            return
        
        for i, email in enumerate(filtered_emails):
            self.render_email_card(email, i)
    
    def render_email_card(self, email, index):
        """Render individual email card with enhanced functionality"""
        
        # Priority color coding
        priority = email.get('priority_score', 5)
        if priority >= 8:
            border_color = "🔴"
            priority_color = "red"
        elif priority >= 6:
            border_color = "🟡"
            priority_color = "orange"
        else:
            border_color = "🟢"
            priority_color = "green"
        
        # Create unique key base for this email
        email_key = f"{email['id']}_{index}"
        
        # Create expandable email card
        with st.expander(
            f"{border_color} **{email.get('subject', 'No Subject')}** - {email.get('from', 'Unknown Sender')[:50]}",
            expanded=False
        ):
            
            # Email metadata
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.markdown(f"**📅 Date:** {email.get('date', 'Unknown')}")
                st.markdown(f"**📧 From:** {email.get('from', 'Unknown')}")
            
            with col2:
                st.markdown(f"**📊 Priority:** :{priority_color}[{priority}/10]")
                st.markdown(f"**📩 Status:** {'Unread' if email.get('is_unread') else 'Read'}")
            
            with col3:
                st.markdown(f"**↩️ Needs Reply:** {'Yes' if email.get('needs_reply') else 'No'}")
                if email.get('attachments'):
                    st.markdown(f"**📎 Attachments:** {len(email['attachments'])}")
            
            # AI Summary (if available)
            if email.get('ai_summary'):
                st.markdown("**🤖 AI Summary:**")
                st.info(email['ai_summary'])
            
            # Email body preview
            st.markdown("**📄 Content Preview:**")
            body = email.get('body', email.get('snippet', 'No content available'))
            if len(body) > 500:
                st.text_area(
                    "", 
                    body[:500] + "...", 
                    height=100, 
                    disabled=True, 
                    key=f"preview_{email_key}"
                )
                
                if st.button("📖 View Full Email", key=f"full_{email_key}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'full_email'
                    st.rerun()
            else:
                st.text_area(
                    "", 
                    body, 
                    height=100, 
                    disabled=True, 
                    key=f"preview_{email_key}"
                )
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📖 Full View", key=f"view_{email_key}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'full_email'
                    st.rerun()
            
            with col2:
                if st.button("↩️ Reply", key=f"reply_{email_key}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'reply'
                    st.rerun()
            
            with col3:
                if email.get('is_unread') and st.button("✅ Mark Read", key=f"read_{email_key}"):
                    self.mark_as_read(email['id'])
            
            with col4:
                if email.get('attachments') and st.button("📎 Downloads", key=f"attach_{email_key}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'attachments'
                    st.rerun()

    def render_priority_tab(self):
        """Render priority analysis tab"""
        st.markdown("### 🔥 Priority Analysis")
        
        emails = st.session_state.emails
        if not emails:
            st.info("📭 No emails loaded. Please check the Inbox tab first.")
            return
        
        # Priority distribution
        priority_counts = {}
        for email in emails:
            priority = email.get('priority_score', 5)
            if priority >= 8:
                priority_range = "High (8-10)"
            elif priority >= 6:
                priority_range = "Medium (6-7)"
            else:
                priority_range = "Low (1-5)"
            
            priority_counts[priority_range] = priority_counts.get(priority_range, 0) + 1
        
        # Create DataFrame for visualization
        if priority_counts:
            df = pd.DataFrame(list(priority_counts.items()), columns=['Priority', 'Count'])
            st.bar_chart(df.set_index('Priority'))
        
        # Create tabs for different priority views
        tab1, tab2, tab3 = st.tabs(["🔴 High Priority", "↩️ Needs Reply", "📝 Auto Drafts"])
        
        with tab1:
            self.render_high_priority_tab(emails)
        
        with tab2:
            self.render_needs_reply_tab(emails)
        
        with tab3:
            self.render_auto_drafts_tab(emails)

    def render_high_priority_tab(self, emails):
        """Render high priority emails"""
        high_priority_emails = [e for e in emails if e.get('priority_score', 5) >= 8]
        st.markdown(f"### 🔴 High Priority Emails ({len(high_priority_emails)})")
        
        if high_priority_emails:
            for i, email in enumerate(high_priority_emails[:10]):
                email_key = f"{email['id']}_{i}_high_priority"
                
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{email.get('subject', 'No Subject')}**")
                        st.caption(f"From: {email.get('from', 'Unknown')}")
                    
                    with col2:
                        st.metric("Priority", f"{email.get('priority_score', 5)}/10")
                    
                    with col3:
                        if st.button("View", key=f"priority_view_{email_key}"):
                            st.session_state.selected_email = email
                            st.session_state.current_view = 'full_email'
                            st.rerun()
                    
                    st.markdown("---")
        else:
            st.info("🎉 No high priority emails found!")

    def render_needs_reply_tab(self, emails):
        """Render emails that need replies"""
        needs_reply_emails = [e for e in emails if e.get('needs_reply', False)]
        
        st.markdown(f"### ↩️ Emails Needing Reply ({len(needs_reply_emails)})")
        
        if not needs_reply_emails:
            st.info("🎉 No emails need replies!")
            return
        
        needs_reply_emails.sort(key=lambda x: x.get('priority_score', 5), reverse=True)
        
        for i, email in enumerate(needs_reply_emails):
            email_key = f"{email['id']}_{i}_needs_reply"
            
            with st.expander(
                f"📧 **{email.get('subject', 'No Subject')}** - Priority: {email.get('priority_score', 5)}/10",
                expanded=False
            ):
                col1, col2 = st.columns([2, 1])
                
                with col1:
                    st.markdown(f"**From:** {email.get('from', 'Unknown')}")
                    st.markdown(f"**Date:** {email.get('date', 'Unknown')}")
                    
                    if email.get('ai_summary'):
                        st.markdown("**🤖 AI Summary:**")
                        st.info(email['ai_summary'])
                    
                    if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
                        if st.button("📋 Extract Key Points", key=f"extract_{email_key}"):
                            with st.spinner("Extracting key points..."):
                                key_points = self.extract_key_points(email)
                                st.session_state[f"key_points_{email['id']}"] = key_points
                                st.rerun()
                        
                        if f"key_points_{email['id']}" in st.session_state:
                            st.markdown("**📋 Key Points to Address:**")
                            st.success(st.session_state[f"key_points_{email['id']}"])
                
                with col2:
                    st.metric("Priority", f"{email.get('priority_score', 5)}/10")
                    
                    if st.button("📝 Generate Draft", key=f"draft_{email_key}", type="primary"):
                        st.session_state.selected_email = email
                        st.session_state.current_view = 'auto_reply'
                        st.rerun()
                    
                    if st.button("👀 Full View", key=f"full_view_{email_key}"):
                        st.session_state.selected_email = email
                        st.session_state.current_view = 'full_email'
                        st.rerun()

    def render_auto_drafts_tab(self, emails):
        """Render auto-generated drafts tab"""
        st.markdown("### 📝 Auto-Generated Drafts")
        
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.warning("⚠️ AI provider required for auto-draft generation")
            return
        
        needs_reply_emails = [e for e in emails if e.get('needs_reply', False)]
        
        if not needs_reply_emails:
            st.info("📭 No emails need replies!")
            return
        
        # Batch draft generation
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.info("💡 Generate drafts for all emails that need replies automatically")
        
        with col2:
            if st.button("🚀 Generate All Drafts", type="primary", key="generate_all_drafts"):
                self.generate_batch_drafts(needs_reply_emails[:5])
        
        # Show existing drafts
        st.markdown("#### 📄 Generated Drafts")
        
        for i, email in enumerate(needs_reply_emails[:10]):
            email_key = f"{email['id']}_{i}_draft"
            draft_key = f"auto_draft_{email['id']}"
            
            if draft_key in st.session_state:
                with st.expander(f"📝 Draft for: {email.get('subject', 'No Subject')}", expanded=False):
                    
                    # Original email context
                    st.markdown("**📧 Original Email:**")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.caption(f"From: {email.get('from', 'Unknown')}")
                        st.caption(f"Date: {email.get('date', 'Unknown')}")
                    with col2:
                        st.caption(f"Priority: {email.get('priority_score', 5)}/10")
                    
                    st.text_area(
                        "Original Content", 
                        email.get('body', '')[:300] + "..." if len(email.get('body', '')) > 300 else email.get('body', ''),
                        height=100,
                        disabled=True,
                        key=f"orig_preview_{email_key}"
                    )
                    
                    # Generated draft
                    st.markdown("**📝 Generated Draft:**")
                    draft_content = st.text_area(
                        "Draft Reply",
                        st.session_state[draft_key],
                        height=200,
                        key=f"draft_edit_{email_key}"
                    )
                    
                    if draft_content != st.session_state[draft_key]:
                        st.session_state[draft_key] = draft_content
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("🔄 Regenerate", key=f"regen_{email_key}"):
                            self.generate_single_draft(email)
                            st.rerun()
                    
                    with col2:
                        if st.button("💾 Save Draft", key=f"save_{email_key}"):
                            self.save_to_gmail_drafts(email, draft_content)
                    
                    with col3:
                        if st.button("📤 Send Now", key=f"send_{email_key}", type="primary"):
                            self.send_reply_email(email, draft_content)
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{email_key}"):
                            del st.session_state[draft_key]
                            st.rerun()

    def render_analytics_tab(self):
        """Render analytics and insights tab"""
        st.markdown("### 📊 Email Analytics & Insights")
        
        emails = st.session_state.emails
        if not emails:
            st.info("📭 No emails loaded. Please check the Inbox tab first.")
            return
        
        # Overall statistics
        col1, col2, col3, col4 = st.columns(4)
        
        total_emails = len(emails)
        unread_emails = sum(1 for e in emails if e.get('is_unread', False))
        high_priority = sum(1 for e in emails if e.get('priority_score', 5) >= 8)
        needs_reply = sum(1 for e in emails if e.get('needs_reply', False))
        
        with col1:
            st.metric("📧 Total Emails", total_emails)
        with col2:
            st.metric("📩 Unread", unread_emails, f"{(unread_emails/total_emails*100):.1f}%" if total_emails > 0 else "0%")
        with col3:
            st.metric("🔥 High Priority", high_priority, f"{(high_priority/total_emails*100):.1f}%" if total_emails > 0 else "0%")
        with col4:
            st.metric("↩️ Needs Reply", needs_reply, f"{(needs_reply/total_emails*100):.1f}%" if total_emails > 0 else "0%")
        
        st.markdown("---")
        
        # Create analysis tabs
        tab1, tab2, tab3, tab4 = st.tabs(["📈 Trends", "👤 Sender Analysis", "🏷️ Category Analysis", "📊 Export"])
        
        with tab1:
            self.render_trends_analysis(emails)
        
        with tab2:
            self.render_sender_analysis(emails)
        
        with tab3:
            self.render_category_analysis(emails)
        
        with tab4:
            self.render_export_options(emails)

    def render_trends_analysis(self, emails):
        """Render email trends analysis"""
        st.markdown("#### 📈 Email Volume Trends")
        
        if not emails:
            st.info("No data available for trends analysis")
            return
        
        # Parse dates and create timeline
        email_dates = []
        for email in emails:
            try:
                date_str = email.get('date', '')
                if date_str:
                    # Parse email date (simplified)
                    import dateutil.parser
                    parsed_date = dateutil.parser.parse(date_str)
                    email_dates.append(parsed_date.date())
            except:
                continue
        
        if email_dates:
            # Count emails by date
            from collections import Counter
            date_counts = Counter(email_dates)
            
            # Create DataFrame for visualization
            df = pd.DataFrame(list(date_counts.items()), columns=['Date', 'Count'])
            df = df.sort_values('Date')
            
            st.line_chart(df.set_index('Date'))
            
            # Show peak days
            if len(date_counts) > 0:
                peak_date = max(date_counts, key=date_counts.get)
                peak_count = date_counts[peak_date]
                st.info(f"📅 Peak day: {peak_date} with {peak_count} emails")
        
        # Priority distribution over time
        st.markdown("#### 🔥 Priority Distribution")
        priority_data = {}
        for email in emails:
            priority = email.get('priority_score', 5)
            if priority >= 8:
                level = "High"
            elif priority >= 6:
                level = "Medium"
            else:
                level = "Low"
            priority_data[level] = priority_data.get(level, 0) + 1
        
        if priority_data:
            df_priority = pd.DataFrame(list(priority_data.items()), columns=['Priority', 'Count'])
            st.bar_chart(df_priority.set_index('Priority'))

    def render_sender_analysis(self, emails):
        """Render sender analysis"""
        st.markdown("#### 👤 Top Senders Analysis")
        
        # Count emails by sender
        sender_counts = {}
        sender_priorities = {}
        
        for email in emails:
            sender = email.get('from', 'Unknown')
            # Extract email address from sender field
            email_match = re.search(r'<([^>]+)>', sender)
            if email_match:
                sender_email = email_match.group(1)
            else:
                sender_email = sender
            
            sender_counts[sender_email] = sender_counts.get(sender_email, 0) + 1
            
            # Track average priority
            if sender_email not in sender_priorities:
                sender_priorities[sender_email] = []
            sender_priorities[sender_email].append(email.get('priority_score', 5))
        
        # Create top senders DataFrame
        top_senders = []
        for sender, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:15]:
            avg_priority = sum(sender_priorities[sender]) / len(sender_priorities[sender])
            needs_reply_count = sum(1 for email in emails 
                                  if sender in email.get('from', '') and email.get('needs_reply', False))
            
            top_senders.append({
                'Sender': sender[:50],  # Truncate long emails
                'Count': count,
                'Avg Priority': f"{avg_priority:.1f}",
                'Needs Reply': needs_reply_count
            })
        
        if top_senders:
            df_senders = pd.DataFrame(top_senders)
            st.dataframe(df_senders, use_container_width=True)
            
            # Show sender distribution chart
            st.markdown("#### 📊 Email Distribution by Sender")
            sender_chart_data = df_senders.head(10).set_index('Sender')['Count']
            st.bar_chart(sender_chart_data)

    def render_category_analysis(self, emails):
        """Render email category analysis"""
        st.markdown("#### 🏷️ Email Categories")
        
        # Categorize emails based on content patterns
        categories = {
            'Newsletters': 0,
            'Notifications': 0,
            'Work/Business': 0,
            'Personal': 0,
            'Automated': 0,
            'Marketing': 0,
            'Other': 0
        }
        
        for email in emails:
            subject = email.get('subject', '').lower()
            sender = email.get('from', '').lower()
            body = email.get('body', '').lower()
            
            # Simple categorization logic
            if any(word in sender for word in ['newsletter', 'news', 'digest']):
                categories['Newsletters'] += 1
            elif any(word in sender for word in ['notification', 'alert', 'system']):
                categories['Notifications'] += 1
            elif any(word in sender for word in ['noreply', 'automated', 'bot', 'mailer']):
                categories['Automated'] += 1
            elif any(word in subject for word in ['sale', 'offer', 'promotion', 'deal', 'marketing']):
                categories['Marketing'] += 1
            elif any(word in subject for word in ['meeting', 'project', 'work', 'business']):
                categories['Work/Business'] += 1
            elif '@gmail.com' in sender or '@yahoo.com' in sender or '@hotmail.com' in sender:
                categories['Personal'] += 1
            else:
                categories['Other'] += 1
        
        # Display category distribution
        df_categories = pd.DataFrame(list(categories.items()), columns=['Category', 'Count'])
        df_categories = df_categories[df_categories['Count'] > 0]
        
        if not df_categories.empty:
            st.bar_chart(df_categories.set_index('Category'))
            
            # Show category details
            st.markdown("#### 📋 Category Breakdown")
            for category, count in categories.items():
                if count > 0:
                    percentage = (count / len(emails)) * 100
                    st.write(f"**{category}:** {count} emails ({percentage:.1f}%)")

    def render_export_options(self, emails):
        """Render export options"""
        st.markdown("#### 📊 Export & Reports")
        
        if not emails:
            st.info("No data to export")
            return
        
        # Export format selection
        export_format = st.radio(
            "Choose export format:",
            ["📄 CSV", "📋 Excel", "📝 Summary Report"],
            horizontal=True,
            key="export_format_radio"
        )
        
        # Filter options for export
        st.markdown("##### 🔍 Export Filters")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            export_unread_only = st.checkbox("📩 Unread only", key="export_unread")
        with col2:
            export_high_priority = st.checkbox("🔥 High priority only", key="export_high_priority")
        with col3:
            export_needs_reply = st.checkbox("↩️ Needs reply only", key="export_needs_reply")
        
        # Filter emails for export
        export_emails = emails
        if export_unread_only:
            export_emails = [e for e in export_emails if e.get('is_unread', False)]
        if export_high_priority:
            export_emails = [e for e in export_emails if e.get('priority_score', 5) >= 8]
        if export_needs_reply:
            export_emails = [e for e in export_emails if e.get('needs_reply', False)]
        
        st.info(f"📊 {len(export_emails)} emails will be exported")
        
        # Export buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📥 Download CSV", key="download_csv"):
                csv_data = self.create_csv_export(export_emails)
                st.download_button(
                    label="💾 Download CSV File",
                    data=csv_data,
                    file_name=f"mailmind_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col2:
            if st.button("📥 Download Excel", key="download_excel"):
                excel_data = self.create_excel_export(export_emails)
                st.download_button(
                    label="💾 Download Excel File",
                    data=excel_data,
                    file_name=f"mailmind_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
        
        with col3:
            if st.button("📥 Generate Report", key="generate_report"):
                report = self.create_summary_report(export_emails)
                st.download_button(
                    label="💾 Download Report",
                    data=report,
                    file_name=f"mailmind_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain"
                )

    def render_settings_tab(self):
        """Render settings and configuration tab"""
        st.markdown("### ⚙️ Settings & Configuration")
        
        # Account management
        st.markdown("#### 📱 Account Management")
        
        if st.session_state.connected_accounts:
            for i, account in enumerate(st.session_state.connected_accounts):
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.info(f"✅ {account}")
                with col2:
                    if st.button("🔌 Disconnect", key=f"disconnect_{i}"):
                        self.disconnect_account(account)
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🔄 Refresh Connection", key="refresh_connection"):
                st.session_state.authenticated = False
                st.rerun()
        
        with col2:
            if st.button("🗑️ Clear All Data", key="clear_all_data"):
                self.clear_all_data()
        
        st.markdown("---")
        
        # AI Settings
        st.markdown("#### 🤖 AI Configuration")
        
        if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
            st.success(f"✅ AI Provider: {st.session_state.ai_provider.provider}")
            st.info(f"📊 Model: {getattr(st.session_state.ai_provider, 'model_name', 'Unknown')}")
        else:
            st.error("❌ No AI provider available")
        
        # AI Enhancement settings
        st.markdown("##### 🎛️ AI Enhancement Options")
        col1, col2 = st.columns(2)
        
        with col1:
            auto_enhance = st.checkbox("🤖 Auto-enhance emails with AI", 
                                     value=st.session_state.get('auto_enhance', True),
                                     key="auto_enhance_checkbox")
            st.session_state['auto_enhance'] = auto_enhance
        
        with col2:
            batch_size = st.slider("📊 AI processing batch size", 
                                 min_value=5, max_value=50, 
                                 value=st.session_state.get('ai_batch_size', 20),
                                 key="ai_batch_size_slider")
            st.session_state['ai_batch_size'] = batch_size
        
        st.markdown("---")
        
        # Display Settings
        st.markdown("#### 🎨 Display Settings")
        
        col1, col2 = st.columns(2)
        with col1:
            theme = st.selectbox("🎨 Theme", 
                               ["Auto", "Light", "Dark"], 
                               key="theme_select")
        
        with col2:
            emails_per_page = st.number_input("📧 Emails per page", 
                                            min_value=10, max_value=200, 
                                            value=50,
                                            key="emails_per_page")
        
        # Priority Settings
        st.markdown("#### 🔥 Priority Settings")
        col1, col2 = st.columns(2)
        
        with col1:
            high_priority_threshold = st.slider("🔴 High priority threshold", 
                                               min_value=6, max_value=10, 
                                               value=8,
                                               key="high_priority_threshold")
        
        with col2:
            low_priority_threshold = st.slider("🟢 Low priority threshold", 
                                              min_value=1, max_value=5, 
                                              value=3,
                                              key="low_priority_threshold")
        
        st.markdown("---")
        
        # Data & Privacy
        st.markdown("#### 🔐 Data & Privacy")
        
        st.info("💡 **Privacy Notice:** All email data is processed locally. No email content is stored permanently.")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("🧹 Clear Cache", key="clear_cache"):
                self.clear_cache()
                st.success("Cache cleared!")
        
        with col2:
            if st.button("📁 Open Downloads Folder", key="open_downloads"):
                self.open_downloads_folder()
        
        with col3:
            if st.button("📋 Export Settings", key="export_settings"):
                self.export_settings()

    def render_full_email_view(self):
        """Render full email view"""
        email = st.session_state.selected_email
        if not email:
            st.error("No email selected")
            if st.button("🔙 Back to Dashboard"):
                st.session_state.current_view = 'dashboard'
                st.rerun()
            return
        
        st.title("📧 Full Email View")
        
        if st.button("🔙 Back to Dashboard", key="back_to_dashboard"):
            st.session_state.current_view = 'dashboard'
            st.rerun()
        
        st.markdown("---")
        
        # Email header
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.markdown(f"### {email.get('subject', 'No Subject')}")
            st.markdown(f"**From:** {email.get('from', 'Unknown')}")
            st.markdown(f"**To:** {email.get('to', 'Unknown')}")
            st.markdown(f"**Date:** {email.get('date', 'Unknown')}")
        
        with col2:
            priority = email.get('priority_score', 5)
            st.metric("Priority", f"{priority}/10")
            st.markdown(f"**Status:** {'Unread' if email.get('is_unread') else 'Read'}")
            st.markdown(f"**Needs Reply:** {'Yes' if email.get('needs_reply') else 'No'}")
        
        # AI Summary
        if email.get('ai_summary'):
            st.markdown("### 🤖 AI Summary")
            st.info(email['ai_summary'])
        
        # Email body
        st.markdown("### 📄 Email Content")
        st.text_area(
            "",
            email.get('body', 'No content available'),
            height=400,
            disabled=True,
            key="full_email_body"
        )
        
        # Attachments
        if email.get('attachments'):
            st.markdown("### 📎 Attachments")
            for i, attachment in enumerate(email['attachments']):
                col1, col2, col3 = st.columns([2, 1, 1])
                
                with col1:
                    st.write(f"📄 {attachment['filename']}")
                    st.caption(f"Type: {attachment.get('mimeType', 'Unknown')}")
                
                with col2:
                    size = attachment.get('size', 0)
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} bytes"
                    st.write(size_str)
                
                with col3:
                    if st.button("📥 Download", key=f"download_attachment_{i}"):
                        self.download_attachment_handler(email, attachment)
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("↩️ Reply", key="full_view_reply"):
                st.session_state.current_view = 'reply'
                st.rerun()
        
        with col2:
            if st.button("📝 AI Reply", key="full_view_ai_reply"):
                st.session_state.current_view = 'auto_reply'
                st.rerun()
        
        with col3:
            if email.get('is_unread') and st.button("✅ Mark Read", key="full_view_mark_read"):
                self.mark_as_read(email['id'])
                st.rerun()
        
        with col4:
            if st.button("🗑️ Archive", key="full_view_archive"):
                self.archive_email(email['id'])

    def render_auto_reply_view(self):
        """Render AI-generated reply view"""
        email = st.session_state.selected_email
        if not email:
            st.error("No email selected")
            if st.button("🔙 Back to Dashboard"):
                st.session_state.current_view = 'dashboard'
                st.rerun()
            return
        
        st.title("🤖 AI-Generated Reply")
        
        if st.button("🔙 Back to Full View", key="back_to_full_from_ai"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.markdown("---")
        
        # Check if draft already exists
        draft_key = f"auto_draft_{email['id']}"
        
        if draft_key not in st.session_state:
            # Generate new draft if AI is available
            if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
                with st.spinner("🤖 Generating AI reply..."):
                    self.generate_single_draft(email)
            else:
                st.error("AI provider not available")
                return
        
        if draft_key in st.session_state:
            # Original email context
            with st.expander("📧 Original Email", expanded=False):
                st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
                st.markdown(f"**From:** {email.get('from', 'Unknown')}")
                st.text_area(
                    "Original Content",
                    email.get('body', '')[:400] + "..." if len(email.get('body', '')) > 400 else email.get('body', ''),
                    height=120,
                    disabled=True,
                    key="ai_original_preview"
                )
            
            # AI-generated reply
            st.markdown("### 🤖 Generated Reply")
            
            # Extract sender for reply
            sender = email.get('from', '')
            sender_match = re.search(r'<([^>]+)>', sender)
            to_email = sender_match.group(1) if sender_match else sender
            
            # Reply subject
            subject = email.get('subject', '')
            if not subject.startswith('Re:'):
                subject = f"Re: {subject}"
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                ai_reply_to = st.text_input("To:", value=to_email, key="ai_reply_to")
            
            with col2:
                ai_reply_subject = st.text_input("Subject:", value=subject, key="ai_reply_subject")
            
            # Editable AI reply
            ai_reply_body = st.text_area(
                "AI Generated Reply (editable):",
                value=st.session_state[draft_key],
                height=300,
                key="ai_reply_body"
            )
            
            # Update the stored draft if edited
            if ai_reply_body != st.session_state[draft_key]:
                st.session_state[draft_key] = ai_reply_body
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🔄 Regenerate", key="regenerate_ai_reply"):
                    self.generate_single_draft(email)
                    st.rerun()
            
            with col2:
                if st.button("💾 Save Draft", key="save_ai_draft"):
                    self.save_manual_draft(ai_reply_to, ai_reply_subject, ai_reply_body, email)
            
            with col3:
                if st.button("📤 Send Now", type="primary", key="send_ai_reply"):
                    self.send_manual_reply(ai_reply_to, ai_reply_subject, ai_reply_body, email)
            
            with col4:
                if st.button("✏️ Manual Edit", key="switch_to_manual"):
                    st.session_state['reply_body_input'] = ai_reply_body
                    st.session_state.current_view = 'reply'
                    st.rerun()

    def render_reply_view(self):
        """Render manual reply composition view with enhanced AI assist"""
        email = st.session_state.selected_email
        if not email:
            st.error("No email selected")
            if st.button("🔙 Back to Dashboard"):
                st.session_state.current_view = 'dashboard'
                st.rerun()
            return
        
        st.title("↩️ Reply to Email")
        
        if st.button("🔙 Back to Full View", key="back_to_full_view"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.markdown("---")
        
        # Original email context
        with st.expander("📧 Original Email", expanded=False):
            st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
            st.markdown(f"**From:** {email.get('from', 'Unknown')}")
            st.text_area(
                "Original Content",
                email.get('body', '')[:500] + "..." if len(email.get('body', '')) > 500 else email.get('body', ''),
                height=150,
                disabled=True,
                key="original_email_preview"
            )
        
        # Reply composition
        st.markdown("### ✏️ Compose Reply")
        
        # Extract sender for reply
        sender = email.get('from', '')
        sender_match = re.search(r'<([^>]+)>', sender)
        to_email = sender_match.group(1) if sender_match else sender
        
        # Reply subject
        subject = email.get('subject', '')
        if not subject.startswith('Re:'):
            subject = f"Re: {subject}"
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            reply_to = st.text_input("To:", value=to_email, key="reply_to_input")
        
        with col2:
            reply_subject = st.text_input("Subject:", value=subject, key="reply_subject_input")
        
        # Initialize reply body in session state if not exists
        reply_body_key = f"reply_body_{email['id']}"
        if reply_body_key not in st.session_state:
            st.session_state[reply_body_key] = ""
        
        # Reply body with proper session state management
        reply_body = st.text_area(
            "Reply Message:",
            value=st.session_state[reply_body_key],
            height=300,
            placeholder="Type your reply here...",
            key="reply_body_textarea"
        )
        
        # Update session state when text changes
        st.session_state[reply_body_key] = reply_body
        
        # AI Assist Section
        if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
            st.markdown("---")
            st.markdown("### 🤖 AI Assistant Options")
            
            with st.expander("🎯 AI Assist Settings", expanded=False):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    tone_options = [
                        "Professional", "Friendly", "Casual", "Formal", 
                        "Enthusiastic", "Apologetic", "Grateful", "Concise"
                    ]
                    selected_tone = st.selectbox(
                        "🎭 Reply Tone:",
                        tone_options,
                        index=0,
                        key="ai_tone_select"
                    )
                
                with col2:
                    reply_types = [
                        "Standard Reply", "Acknowledgment", "Request Information", 
                        "Schedule Meeting", "Decline Politely", "Accept Invitation",
                        "Provide Update", "Ask for Clarification"
                    ]
                    selected_type = st.selectbox(
                        "📝 Reply Type:",
                        reply_types,
                        index=0,
                        key="ai_type_select"
                    )
                
                with col3:
                    length_options = ["Brief", "Medium", "Detailed"]
                    selected_length = st.selectbox(
                        "📏 Reply Length:",
                        length_options,
                        index=1,
                        key="ai_length_select"
                    )
            
            # AI Action Buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("🤖 Generate Reply", key="ai_generate_new", type="primary"):
                    generated_reply = self.generate_ai_reply_with_options(
                        email, selected_tone, selected_type, selected_length
                    )
                    if generated_reply:
                        st.session_state[reply_body_key] = generated_reply
                        st.rerun()
            
            with col2:
                if st.button("✨ Improve Current", key="ai_improve_current"):
                    if st.session_state[reply_body_key].strip():
                        improved_reply = self.improve_reply_with_ai_options(
                            st.session_state[reply_body_key], email, selected_tone, selected_type
                        )
                        if improved_reply:
                            st.session_state[reply_body_key] = improved_reply
                            st.rerun()
                    else:
                        st.warning("Please write some content first to improve")
            
            with col3:
                if st.button("🎯 Suggest Points", key="ai_suggest_points"):
                    suggestions = self.get_reply_suggestions(email, selected_type)
                    if suggestions:
                        st.session_state['ai_suggestions'] = suggestions
            
            with col4:
                if st.button("📝 Auto-Complete", key="ai_complete"):
                    if st.session_state[reply_body_key].strip():
                        completed_reply = self.complete_reply_with_ai(
                            st.session_state[reply_body_key], email, selected_tone
                        )
                        if completed_reply:
                            st.session_state[reply_body_key] = completed_reply
                            st.rerun()
        
        # Show AI suggestions if available
        if 'ai_suggestions' in st.session_state:
            st.markdown("#### 💡 AI Suggestions:")
            st.info(st.session_state['ai_suggestions'])
            if st.button("❌ Clear Suggestions", key="clear_suggestions"):
                del st.session_state['ai_suggestions']
                st.rerun()
        
        # Action buttons
        st.markdown("---")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("💾 Save as Draft", key="save_reply_draft"):
                self.save_manual_draft(reply_to, reply_subject, st.session_state[reply_body_key], email)
        
        with col2:
            if st.button("📤 Send Reply", type="primary", key="send_manual_reply"):
                if st.session_state[reply_body_key].strip():
                    self.send_manual_reply(reply_to, reply_subject, st.session_state[reply_body_key], email)
                else:
                    st.error("Please enter a reply message")
        
        with col3:
            if st.button("🗑️ Clear Draft", key="clear_draft"):
                st.session_state[reply_body_key] = ""
                st.rerun()

    def render_attachments_view(self):
        """Render attachments download view"""
        email = st.session_state.selected_email
        if not email:
            st.error("No email selected")
            if st.button("🔙 Back to Dashboard"):
                st.session_state.current_view = 'dashboard'
                st.rerun()
            return
        
        st.title("📎 Email Attachments")
        
        if st.button("🔙 Back to Full View", key="back_from_attachments"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.markdown("---")
        
        attachments = email.get('attachments', [])
        
        if not attachments:
            st.info("📭 This email has no attachments")
            return
        
        st.markdown(f"### 📎 {len(attachments)} Attachment(s) Found")
        
        # Bulk download options
        if len(attachments) > 1:
            if st.button("📥 Download All Attachments", type="primary", key="download_all_attachments"):
                self.download_all_attachments(email, attachments)
        
        st.markdown("---")
        
        # Individual attachments
        for i, attachment in enumerate(attachments):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 2, 1, 1])
                
                with col1:
                    st.markdown(f"**📄 {attachment['filename']}**")
                    st.caption(f"Type: {attachment.get('mimeType', 'Unknown')}")
                
                with col2:
                    size = attachment.get('size', 0)
                    if size > 1024 * 1024:
                        size_str = f"{size / (1024 * 1024):.1f} MB"
                    elif size > 1024:
                        size_str = f"{size / 1024:.1f} KB"
                    else:
                        size_str = f"{size} bytes"
                    st.write(f"📊 {size_str}")
                
                with col3:
                    if st.button("📥 Download", key=f"download_single_{i}"):
                        self.download_attachment_handler(email, attachment)
                
                with col4:
                    if MARKITDOWN_AVAILABLE and self.is_previewable(attachment):
                        if st.button("👁️ Preview", key=f"preview_{i}"):
                            self.preview_attachment(email, attachment)
                
                st.markdown("---")

    # Helper methods for email processing
    def mark_as_read(self, email_id: str):
        """Mark email as read in Gmail"""
        try:
            st.session_state.gmail_service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            st.success("✅ Email marked as read")
            
            # Update local state
            for email in st.session_state.emails:
                if email['id'] == email_id:
                    email['is_unread'] = False
                    break
        except Exception as e:
            st.error(f"❌ Failed to mark as read: {str(e)}")

    def extract_key_points(self, email):
        """Extract key points from email using AI"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return "AI not available for key point extraction"
        
        prompt = f"""
        Extract the key points and action items from this email:
        
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        
        Content: {email.get('body', '')[:1000]}...
        
        Please provide:
        1. Main topics discussed
        2. Action items or requests
        3. Deadlines mentioned
        4. Questions that need answers
        
        Keep it concise and bullet-pointed.
        """
        
        return st.session_state.ai_provider.generate_response(prompt)

    def generate_single_draft(self, email):
        """Generate a single draft reply using AI"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.error("AI provider not available")
            return
        
        draft_key = f"auto_draft_{email['id']}"
        
        prompt = f"""
        Write a professional email reply to this message:
        
        Original Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        
        Original Message:
        {email.get('body', '')[:1000]}
        
        Please write a appropriate, professional response that:
        1. Acknowledges the original message
        2. Addresses any questions or requests
        3. Is concise and clear
        4. Uses a professional but friendly tone
        
        Start directly with the email content (no "Subject:" or "Dear" unless needed).
        """
        
        with st.spinner("🤖 Generating draft reply..."):
            draft = st.session_state.ai_provider.generate_response(prompt)
            st.session_state[draft_key] = draft
        
        st.success("✅ Draft generated successfully!")

    def generate_batch_drafts(self, emails):
        """Generate drafts for multiple emails"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.error("AI provider not available")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, email in enumerate(emails):
            status_text.text(f"Generating draft {i+1} of {len(emails)}")
            self.generate_single_draft(email)
            progress_bar.progress((i + 1) / len(emails))
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Generated {len(emails)} drafts successfully!")

    def save_to_gmail_drafts(self, email, draft_content):
        """Save draft to Gmail drafts folder"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        # Extract sender email for reply
        sender = email.get('from', '')
        sender_match = re.search(r'<([^>]+)>', sender)
        to_email = sender_match.group(1) if sender_match else sender
        
        # Create reply subject
        subject = email.get('subject', '')
        if not subject.startswith('Re:'):
            subject = f"Re: {subject}"
        
        draft_id = processor.save_draft(subject, draft_content, to_email, email.get('thread_id'))
        
        if draft_id:
            st.success("💾 Draft saved to Gmail!")
        else:
            st.error("❌ Failed to save draft")

    def send_reply_email(self, email, draft_content):
        """Send reply email directly"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        # Extract sender email for reply
        sender = email.get('from', '')
        sender_match = re.search(r'<([^>]+)>', sender)
        to_email = sender_match.group(1) if sender_match else sender
        
        # Create reply subject
        subject = email.get('subject', '')
        if not subject.startswith('Re:'):
            subject = f"Re: {subject}"
        
        if processor.send_email(to_email, subject, draft_content, email.get('thread_id')):
            st.success("📤 Email sent successfully!")
            # Remove from needs reply
            email['needs_reply'] = False
        else:
            st.error("❌ Failed to send email")

    # AI Helper Methods
    def generate_ai_reply_with_options(self, email, tone, reply_type, length):
        """Generate AI reply with specific tone and type options"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.error("AI provider not available")
            return None
        
        # Create detailed prompt based on options
        tone_instructions = {
            "Professional": "Use professional business language, formal tone, and proper email etiquette",
            "Friendly": "Use warm, approachable language while maintaining professionalism", 
            "Casual": "Use relaxed, conversational tone appropriate for informal communication",
            "Formal": "Use very formal, traditional business language and structure",
            "Enthusiastic": "Show excitement and positive energy in the response",
            "Apologetic": "Express sincere apologies and take responsibility where appropriate",
            "Grateful": "Express genuine appreciation and thankfulness",
            "Concise": "Keep the response brief, direct, and to the point"
        }
        
        type_instructions = {
            "Standard Reply": "Provide a standard response addressing the main points",
            "Acknowledgment": "Acknowledge receipt and understanding of the message",
            "Request Information": "Ask for additional details or clarification politely",
            "Schedule Meeting": "Propose meeting times and coordinate scheduling",
            "Decline Politely": "Politely decline the request while offering alternatives if possible",
            "Accept Invitation": "Accept the invitation enthusiastically and confirm details",
            "Provide Update": "Give a status update or progress report on relevant matters",
            "Ask for Clarification": "Request clarification on specific points mentioned"
        }
        
        length_instructions = {
            "Brief": "Keep the response to 2-3 sentences maximum",
            "Medium": "Write a balanced response of 1-2 short paragraphs", 
            "Detailed": "Provide a comprehensive response with full explanations"
        }
        
        prompt = f"""
        Write a {tone.lower()} email reply with the following specifications:
        
        TONE: {tone_instructions.get(tone, '')}
        TYPE: {type_instructions.get(reply_type, '')}
        LENGTH: {length_instructions.get(length, '')}
        
        Original Email Details:
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        Content: {email.get('body', '')[:1000]}
        
        Key Requirements:
        1. Match the specified tone: {tone}
        2. Follow the reply type: {reply_type}
        3. Keep the length: {length}
        4. Address the main points from the original email
        5. Include appropriate greeting and closing
        6. Be contextually relevant and helpful
        
        Write only the email body content (no subject line or metadata).
        """
        
        with st.spinner(f"🤖 Generating {tone.lower()} {reply_type.lower()} reply..."):
            try:
                reply = st.session_state.ai_provider.generate_response(prompt)
                st.success(f"✅ Generated {tone.lower()} reply successfully!")
                return reply.strip()
            except Exception as e:
                st.error(f"❌ AI generation failed: {str(e)}")
                return None

    def improve_reply_with_ai_options(self, current_reply, email, tone, reply_type):
        """Improve existing reply with AI using specific options"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.error("AI provider not available")
            return None
        
        prompt = f"""
        Improve this email reply to make it more {tone.lower()} and better suited for a {reply_type.lower()}:
        
        Original Email Context:
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        
        Current Reply:
        {current_reply}
        
        Improvement Goals:
        1. Adjust tone to be more {tone.lower()}
        2. Optimize for {reply_type.lower()} purpose
        3. Improve clarity and professionalism
        4. Fix any grammatical issues
        5. Enhance overall effectiveness
        
        Maintain the original intent while making these improvements.
        Return only the improved email content.
        """
        
        with st.spinner(f"✨ Improving reply with {tone.lower()} tone..."):
            try:
                improved = st.session_state.ai_provider.generate_response(prompt)
                st.success("✅ Reply improved successfully!")
                return improved.strip()
            except Exception as e:
                st.error(f"❌ AI improvement failed: {str(e)}")
                return None

    def get_reply_suggestions(self, email, reply_type):
        """Get AI suggestions for reply content"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return None
        
        prompt = f"""
        Analyze this email and provide 3-5 key points that should be addressed in a {reply_type.lower()}:
        
        Original Email:
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        Content: {email.get('body', '')[:800]}
        
        Provide specific, actionable suggestions for what to include in the reply.
        Format as bullet points for easy reading.
        """
        
        with st.spinner("🎯 Getting AI suggestions..."):
            try:
                suggestions = st.session_state.ai_provider.generate_response(prompt)
                return suggestions
            except Exception as e:
                st.error(f"❌ Failed to get suggestions: {str(e)}")
                return None

    def complete_reply_with_ai(self, partial_reply, email, tone):
        """Auto-complete a partially written reply"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return None
        
        prompt = f"""
        Complete this partially written email reply in a {tone.lower()} tone:
        
        Original Email Context:
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        
        Partial Reply:
        {partial_reply}
        
        Please complete the reply by:
        1. Maintaining the {tone.lower()} tone established
        2. Adding any missing important points
        3. Including proper closing if needed
        4. Ensuring the reply is complete and professional
        
        Return the complete email reply.
        """
        
        with st.spinner("📝 Auto-completing reply..."):
            try:
                completed = st.session_state.ai_provider.generate_response(prompt)
                st.success("✅ Reply completed successfully!")
                return completed.strip()
            except Exception as e:
                st.error(f"❌ Auto-completion failed: {str(e)}")
                return None

    # Attachment helper methods
    def download_attachment_handler(self, email, attachment):
        """Handle single attachment download"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        with st.spinner(f"📥 Downloading {attachment['filename']}..."):
            file_path = processor.download_attachment(
                email['id'],
                attachment['attachmentId'],
                attachment['filename']
            )
            
            if file_path:
                st.success(f"✅ Downloaded: {file_path}")
            else:
                st.error("❌ Download failed")

    def download_all_attachments(self, email, attachments):
        """Download all attachments"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        downloaded = 0
        
        for i, attachment in enumerate(attachments):
            status_text.text(f"Downloading {attachment['filename']}...")
            
            file_path = processor.download_attachment(
                email['id'],
                attachment['attachmentId'],
                attachment['filename']
            )
            
            if file_path:
                downloaded += 1
            
            progress_bar.progress((i + 1) / len(attachments))
        
        progress_bar.empty()
        status_text.empty()
        
        st.success(f"✅ Downloaded {downloaded} of {len(attachments)} attachments")

    def is_previewable(self, attachment):
        """Check if attachment can be previewed"""
        mime_type = attachment.get('mimeType', '').lower()
        previewable_types = [
            'text/plain', 'text/html', 'text/csv',
            'application/pdf', 'application/json',
            'image/jpeg', 'image/png', 'image/gif'
        ]
        return mime_type in previewable_types

    def preview_attachment(self, email, attachment):
        """Preview attachment content"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        # Download to temp location for preview
        temp_path = processor.download_attachment(
            email['id'],
            attachment['attachmentId'],
            attachment['filename']
        )
        
        if not temp_path:
            st.error("Failed to download for preview")
            return
        
        mime_type = attachment.get('mimeType', '').lower()
        
        with st.expander(f"👁️ Preview: {attachment['filename']}", expanded=True):
            try:
                if mime_type.startswith('text/'):
                    with open(temp_path, 'r', encoding='utf-8', errors='ignore') as f:
                        content = f.read()[:2000]  # Limit preview size
                        st.text_area("Content Preview", content, height=300, disabled=True)
                
                elif mime_type.startswith('image/'):
                    st.image(temp_path, caption=attachment['filename'])
                
                elif mime_type == 'application/json':
                    with open(temp_path, 'r') as f:
                        json_data = json.load(f)
                        st.json(json_data)
                
                else:
                    st.info("Preview not available for this file type")
                    
            except Exception as e:
                st.error(f"Preview failed: {str(e)}")

    def save_manual_draft(self, to_email, subject, body, original_email):
        """Save manually composed draft"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        draft_id = processor.save_draft(subject, body, to_email, original_email.get('thread_id'))
        
        if draft_id:
            st.success("💾 Draft saved to Gmail!")
        else:
            st.error("❌ Failed to save draft")

    def send_manual_reply(self, to_email, subject, body, original_email):
        """Send manually composed reply"""
        processor = st.session_state.email_processor
        if not processor:
            st.error("Email processor not available")
            return
        
        if processor.send_email(to_email, subject, body, original_email.get('thread_id')):
            st.success("📤 Reply sent successfully!")
            # Mark original email as replied
            original_email['needs_reply'] = False
            # Go back to dashboard
            st.session_state.current_view = 'dashboard'
            st.rerun()
        else:
            st.error("❌ Failed to send reply")

    def archive_email(self, email_id):
        """Archive email in Gmail"""
        try:
            st.session_state.gmail_service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['INBOX']}
            ).execute()
            st.success("🗃️ Email archived successfully!")
            
            # Update local state
            for email in st.session_state.emails:
                if email['id'] == email_id:
                    email['archived'] = True
                    break
                    
        except Exception as e:
            st.error(f"❌ Failed to archive: {str(e)}")

    # Export helper methods
    def create_csv_export(self, emails):
        """Create CSV export data"""
        export_data = []
        for email in emails:
            export_data.append({
                'Subject': email.get('subject', ''),
                'From': email.get('from', ''),
                'Date': email.get('date', ''),
                'Priority Score': email.get('priority_score', ''),
                'Is Unread': email.get('is_unread', False),
                'Needs Reply': email.get('needs_reply', False),
                'AI Summary': email.get('ai_summary', ''),
                'Attachments': len(email.get('attachments', [])),
                'Snippet': email.get('snippet', '')[:200]
            })
        
        df = pd.DataFrame(export_data)
        return df.to_csv(index=False)

    def create_excel_export(self, emails):
        """Create Excel export data"""
        export_data = []
        for email in emails:
            export_data.append({
                'Subject': email.get('subject', ''),
                'From': email.get('from', ''),
                'Date': email.get('date', ''),
                'Priority Score': email.get('priority_score', ''),
                'Is Unread': email.get('is_unread', False),
                'Needs Reply': email.get('needs_reply', False),
                'AI Summary': email.get('ai_summary', ''),
                'Attachments': len(email.get('attachments', [])),
                'Body': email.get('body', '')[:1000]  # Limit body length
            })
        
        df = pd.DataFrame(export_data)
        output = io.BytesIO()
        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Emails')
        return output.getvalue()

    def create_summary_report(self, emails):
        """Create summary report"""
        total_emails = len(emails)
        unread = sum(1 for e in emails if e.get('is_unread', False))
        high_priority = sum(1 for e in emails if e.get('priority_score', 5) >= 8)
        needs_reply = sum(1 for e in emails if e.get('needs_reply', False))
        
        report = f"""
MailMind Email Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

SUMMARY STATISTICS:
==================
Total Emails: {total_emails}
Unread Emails: {unread} ({(unread/total_emails*100):.1f}%)
High Priority: {high_priority} ({(high_priority/total_emails*100):.1f}%)
Needs Reply: {needs_reply} ({(needs_reply/total_emails*100):.1f}%)

TOP SENDERS:
============
"""
        
        # Add top senders
        sender_counts = {}
        for email in emails:
            sender = email.get('from', 'Unknown')
            sender_counts[sender] = sender_counts.get(sender, 0) + 1
        
        for sender, count in sorted(sender_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
            report += f"{sender[:50]}: {count} emails\n"
        
        report += f"""

HIGH PRIORITY EMAILS:
====================
"""
        
        high_priority_emails = [e for e in emails if e.get('priority_score', 5) >= 8]
        for email in high_priority_emails[:10]:
            report += f"- {email.get('subject', 'No Subject')} (Priority: {email.get('priority_score', 5)}/10)\n"
        
        return report

    # Settings helper methods
    def disconnect_account(self, account):
        """Disconnect an account"""
        if account in st.session_state.connected_accounts:
            st.session_state.connected_accounts.remove(account)
        
        # Clear session if it's the current account
        try:
            profile = st.session_state.gmail_service.users().getProfile(userId='me').execute()
            current_account = profile.get('emailAddress', '')
            if current_account == account:
                st.session_state.authenticated = False
                st.session_state.gmail_service = None
        except:
            pass
        
        st.success(f"✅ Disconnected from {account}")
        st.rerun()

    def clear_all_data(self):
        """Clear all application data"""
        # Clear session state
        for key in ['emails', 'processed_emails', 'selected_email']:
            if key in st.session_state:
                del st.session_state[key]
        
        # Clear cached drafts
        keys_to_remove = [k for k in st.session_state.keys() if k.startswith('auto_draft_')]
        for key in keys_to_remove:
            del st.session_state[key]
        
        st.success("🗑️ All data cleared!")
        st.rerun()

    def clear_cache(self):
        """Clear application cache"""
        st.session_state.emails = []
        st.session_state.processed_emails = []

    def open_downloads_folder(self):
        """Open downloads folder"""
        downloads_path = Path("downloads")
        if downloads_path.exists():
            st.info(f"📁 Downloads folder: {downloads_path.absolute()}")
        else:
            st.warning("📁 Downloads folder doesn't exist yet")

    def export_settings(self):
        """Export current settings"""
        settings = {
            'auto_enhance': st.session_state.get('auto_enhance', True),
            'ai_batch_size': st.session_state.get('ai_batch_size', 20),
            'high_priority_threshold': 8,
            'low_priority_threshold': 3
        }
        
        settings_json = json.dumps(settings, indent=2)
        st.download_button(
            "💾 Download Settings",
            settings_json,
            file_name=f"mailmind_settings_{datetime.now().strftime('%Y%m%d')}.json",
            mime="application/json"
        )

    def run(self):
        """Main application runner"""
        # Check if setup is complete
        if not st.session_state.setup_complete:
            self.render_setup_page()
            return
        
        # Check authentication
        if not st.session_state.authenticated:
            self.render_authentication()
            return
        
        # Handle different views
        if st.session_state.current_view == 'full_email':
            self.render_full_email_view()
        elif st.session_state.current_view == 'reply':
            self.render_reply_view()
        elif st.session_state.current_view == 'auto_reply':
            self.render_auto_reply_view()
        elif st.session_state.current_view == 'attachments':
            self.render_attachments_view()
        else:
            self.render_main_dashboard()


# Main application entry point
def main():
    """Main application entry point"""
    try:
        app = MailMindApp()
        app.run()
    except Exception as e:
        st.error(f"💥 **Application Error**: {str(e)}")
        st.error("Please check your setup and try refreshing the page.")
        
        with st.expander("🔧 Debug Information"):
            import traceback
            st.code(traceback.format_exc())

if __name__ == "__main__":
    main()