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
        export GOOGLE_API_KEY=your_api_key_here
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
            st.error("🔒 **Missing credentials folder!**")
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
                    st.info("🔐 Starting OAuth flow...")
                    
                    try:
                        flow = InstalledAppFlow.from_client_secrets_file(
                            self.credentials_path, SCOPES)
                        
                        creds = flow.run_local_server(
                            port=8080,
                            prompt='consent',
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
                    st.success(f"🚀 **AI Provider**: {self.provider} ({model}) initialized successfully!")
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
                                st.success(f"🚀 **AI Provider**: {self.provider} ({fallback_model}) initialized successfully!")
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
            st.error(f"🔑 **Missing API keys**: {', '.join(missing_keys)}")
    
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
        """Fetch emails from Gmail with time filtering"""
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
            
            results = self.gmail_service.users().messages().list(
                userId='me', 
                q=query,
                maxResults=max_results
            ).execute()
            
            messages = results.get('messages', [])
            emails = []
            
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            for i, message in enumerate(messages):
                try:
                    msg = self.gmail_service.users().messages().get(
                        userId='me', 
                        id=message['id']
                    ).execute()
                    
                    email_data = self._parse_email(msg)
                    if email_data:
                        emails.append(email_data)
                    
                    progress = (i + 1) / len(messages)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing email {i+1} of {len(messages)}")
                    
                except Exception as e:
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
                'priority_score': self._calculate_basic_priority(header_dict, body, msg),
                'needs_reply': self._basic_reply_detection(body, header_dict),
                'ai_summary': msg.get('snippet', '')[:100] + '...'
            }
            
            return email_data
            
        except Exception as e:
            return None
    
    def _calculate_basic_priority(self, headers: Dict, body: str, msg: Dict) -> int:
        """Calculate basic priority without AI"""
        score = 5
        
        if 'UNREAD' in msg.get('labelIds', []):
            score += 1
        
        if 'IMPORTANT' in msg.get('labelIds', []):
            score += 2
        
        content = f"{headers.get('Subject', '')} {body}".lower()
        urgent_keywords = ['urgent', 'asap', 'important', 'deadline', 'meeting', 'call']
        
        for keyword in urgent_keywords:
            if keyword in content:
                score += 1
                break
        
        return min(max(score, 1), 10)
    
    def _basic_reply_detection(self, body: str, headers: Dict) -> bool:
        """Basic reply detection without AI"""
        content = f"{headers.get('Subject', '')} {body}".lower()
        
        if '?' in content:
            return True
        
        request_patterns = [
            'please', 'can you', 'could you', 'would you', 'let me know',
            'get back to me', 'respond', 'reply', 'confirm', 'schedule'
        ]
        
        return any(pattern in content for pattern in request_patterns)
    
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
                    email_data['priority_score'] = min(max(int(priority_match.group(1)), 1), 10)
            except:
                pass
            
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
            
            # Reply detection
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
            'email_processor': None
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
        
        st.markdown("### 🔑 API Key Check")
        
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
            st.error("🔑 **No AI API keys found!**")
            all_good = False
        
        st.markdown("---")
        
        if not all_good:
            st.error("⚠️ **Setup incomplete!** Please follow the installation guide below.")
            SetupHelper.show_installation_guide()
            
            if st.button("🔄 Recheck Dependencies"):
                st.rerun()
        else:
            st.success("🎉 **All dependencies satisfied!** Ready to proceed.")
            if st.button("▶️ Continue to MailMind", type="primary"):
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
        """Render authentication interface"""
        st.title("📧 MailMind - Smart Email Prioritizer & AI Assistant")
        
        # Initialize components
        auth = ImprovedGmailAuth()
        ai_provider = AIProvider()
        
        # Pre-flight checks
        st.markdown("### ✅ Pre-flight Checklist")
        
        creds_folder_exists = Path("credentials").exists()
        st.markdown(f"{'✅' if creds_folder_exists else '❌'} Credentials folder exists")
        
        creds_file_exists = os.path.exists("credentials/credentials.json")
        st.markdown(f"{'✅' if creds_file_exists else '❌'} credentials.json file present")
        
        valid_json = False
        if creds_file_exists:
            try:
                with open("credentials/credentials.json", 'r') as f:
                    json.load(f)
                valid_json = True
            except:
                pass
        st.markdown(f"{'✅' if valid_json else '❌'} Valid JSON format")
        
        ai_available = ai_provider.is_available()
        provider_info = f"{ai_provider.provider} ({getattr(ai_provider, 'model_name', 'Unknown Model')})" if ai_available else "None"
        st.markdown(f"{'✅' if ai_available else '❌'} AI Provider: {provider_info}")
        
        st.markdown("---")
        
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            if not all([creds_folder_exists, creds_file_exists, valid_json]):
                st.error("⚠️ **Gmail setup incomplete!**")
                return None
            
            if not ai_available:
                st.warning("⚠️ **AI features limited without API key**")
                st.info("The app will work with basic email management, but AI features will be disabled.")
            
            st.success("🎉 **Ready to connect!**")
            
            if st.button("🔗 Connect Gmail", type="primary", use_container_width=True):
                with st.spinner("🔐 Authenticating with Gmail..."):
                    service = auth.authenticate()
                    
                    if service:
                        st.session_state.gmail_service = service
                        st.session_state.authenticated = True
                        st.session_state.ai_provider = ai_provider
                        st.session_state.email_processor = EnhancedEmailProcessor(service, ai_provider)
                        st.rerun()
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
                index=2
            )
        
        with col3:
            max_emails = st.number_input(
                "📊 Max Emails",
                min_value=10,
                max_value=200,
                value=50
            )
        
        with col4:
            if st.button("🔄 Refresh"):
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
                emails = email_processor.fetch_emails(time_filter, max_emails)
                st.session_state.emails = emails
                
                # Enhance with AI if available
                if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
                    with st.spinner("🤖 Enhancing with AI..."):
                        enhanced_emails = email_processor.enhance_with_ai(emails[:10])
                        for i, enhanced in enumerate(enhanced_emails):
                            if i < len(st.session_state.emails):
                                st.session_state.emails[i].update(enhanced)
        
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
            show_unread_only = st.checkbox("📩 Unread only")
        
        with col2:
            show_high_priority = st.checkbox("🔥 High priority only")
        
        with col3:
            show_needs_reply = st.checkbox("↩️ Needs reply only")
        
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
                st.text_area("", body[:500] + "...", height=100, disabled=True, key=f"preview_{email['id']}")
                
                if st.button("📖 View Full Email", key=f"full_{email['id']}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'full_email'
                    st.rerun()
            else:
                st.text_area("", body, height=100, disabled=True, key=f"preview_{email['id']}")
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📖 Full View", key=f"view_{email['id']}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'full_email'
                    st.rerun()
            
            with col2:
                if st.button("↩️ Reply", key=f"reply_{email['id']}"):
                    st.session_state.selected_email = email
                    st.session_state.current_view = 'reply'
                    st.rerun()
            
            with col3:
                if email.get('is_unread') and st.button("✅ Mark Read", key=f"read_{email['id']}"):
                    self.mark_as_read(email['id'])
            
            with col4:
                if email.get('attachments') and st.button("📎 Downloads", key=f"attach_{email['id']}"):
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
            for email in high_priority_emails[:10]:
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{email.get('subject', 'No Subject')}**")
                        st.caption(f"From: {email.get('from', 'Unknown')}")
                    
                    with col2:
                        st.metric("Priority", f"{email.get('priority_score', 5)}/10")
                    
                    with col3:
                        if st.button("View", key=f"priority_view_{email['id']}"):
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
        
        for email in needs_reply_emails:
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
                        if st.button("📋 Extract Key Points", key=f"extract_{email['id']}"):
                            with st.spinner("Extracting key points..."):
                                key_points = self.extract_key_points(email)
                                st.session_state[f"key_points_{email['id']}"] = key_points
                        
                        if f"key_points_{email['id']}" in st.session_state:
                            st.markdown("**📋 Key Points to Address:**")
                            st.success(st.session_state[f"key_points_{email['id']}"])
                
                with col2:
                    st.metric("Priority", f"{email.get('priority_score', 5)}/10")
                    
                    if st.button("📝 Generate Draft", key=f"draft_{email['id']}", type="primary"):
                        st.session_state.selected_email = email
                        st.session_state.current_view = 'auto_reply'
                        st.rerun()
                    
                    if st.button("👀 Full View", key=f"full_view_{email['id']}"):
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
            if st.button("🚀 Generate All Drafts", type="primary"):
                self.generate_batch_drafts(needs_reply_emails[:5])
        
        # Show existing drafts
        st.markdown("#### 📄 Generated Drafts")
        
        for email in needs_reply_emails[:10]:
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
                        key=f"orig_preview_{email['id']}"
                    )
                    
                    # Generated draft
                    st.markdown("**📝 Generated Draft:**")
                    draft_content = st.text_area(
                        "Draft Reply",
                        st.session_state[draft_key],
                        height=200,
                        key=f"draft_edit_{email['id']}"
                    )
                    
                    if draft_content != st.session_state[draft_key]:
                        st.session_state[draft_key] = draft_content
                    
                    # Action buttons
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if st.button("🔄 Regenerate", key=f"regen_{email['id']}"):
                            with st.spinner("Regenerating draft..."):
                                new_draft = self.generate_reply_draft(email)
                                st.session_state[draft_key] = new_draft
                                st.rerun()
                    
                    with col2:
                        if st.button("📤 Use Draft", key=f"use_{email['id']}"):
                            st.session_state.selected_email = email
                            st.session_state['reply_body'] = draft_content
                            st.session_state.current_view = 'reply'
                            st.rerun()
                    
                    with col3:
                        if st.button("💾 Save to Gmail", key=f"save_{email['id']}"):
                            self.save_draft_to_gmail(email, draft_content)
                    
                    with col4:
                        if st.button("🗑️ Delete", key=f"delete_{email['id']}"):
                            del st.session_state[draft_key]
                            st.rerun()
            else:
                # Show option to generate individual draft
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        st.markdown(f"**{email.get('subject', 'No Subject')}**")
                        st.caption(f"From: {email.get('from', 'Unknown')}")
                    
                    with col2:
                        st.metric("Priority", f"{email.get('priority_score', 5)}/10")
                    
                    with col3:
                        if st.button("📝 Generate", key=f"gen_{email['id']}"):
                            with st.spinner("Generating draft..."):
                                draft = self.generate_reply_draft(email)
                                st.session_state[draft_key] = draft
                                st.rerun()
                    
                    st.markdown("---")
    
    def extract_key_points(self, email):
        """Extract key points from email that need to be addressed in reply"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return "AI provider not available"
        
        prompt = f"""
        Analyze this email and extract the key points that need to be addressed in a reply:
        
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        Content: {email.get('body', '')[:1000]}
        
        Please identify:
        1. Questions that need answers
        2. Requests that need responses
        3. Action items for the recipient
        4. Important information that should be acknowledged
        
        Format as a concise bullet-point list.
        """
        
        try:
            response = st.session_state.ai_provider.generate_response(prompt)
            return response
        except Exception as e:
            return f"Error extracting key points: {str(e)}"
    
    def generate_reply_draft(self, email):
        """Generate a comprehensive reply draft for an email"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return "AI provider not available for draft generation"
        
        prompt = f"""
        Generate a professional email reply draft for the following email:
        
        Original Subject: {email.get('subject', '')}
        Original From: {email.get('from', '')}
        Original Date: {email.get('date', '')}
        Original Content: {email.get('body', '')[:1500]}
        
        Please create a reply that:
        1. Acknowledges receipt of the original email
        2. Addresses any questions or requests made
        3. Provides helpful responses or next steps
        4. Maintains a professional and courteous tone
        5. Is concise but comprehensive
        6. Includes appropriate closing
        
        Format as a complete email ready to send.
        """
        
        try:
            response = st.session_state.ai_provider.generate_response(prompt)
            return response
        except Exception as e:
            return f"Error generating draft: {str(e)}"
    
    def generate_batch_drafts(self, emails):
        """Generate drafts for multiple emails in batch"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            st.error("AI provider required for batch draft generation")
            return
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, email in enumerate(emails):
            try:
                status_text.text(f"Generating draft {i+1} of {len(emails)}: {email.get('subject', 'No Subject')[:50]}...")
                
                draft = self.generate_reply_draft(email)
                st.session_state[f"auto_draft_{email['id']}"] = draft
                
                progress = (i + 1) / len(emails)
                progress_bar.progress(progress)
                
            except Exception as e:
                st.error(f"Failed to generate draft for email {i+1}: {str(e)}")
                continue
        
        progress_bar.empty()
        status_text.empty()
        st.success(f"✅ Generated {len(emails)} drafts successfully!")
        st.rerun()
    
    def save_draft_to_gmail(self, email, draft_content):
        """Save draft to Gmail"""
        try:
            from_email = email.get('from', '')
            # Extract email address from "Name <email@domain.com>" format
            import re
            email_match = re.search(r'<(.+?)>', from_email)
            to_email = email_match.group(1) if email_match else from_email
            
            subject = f"Re: {email.get('subject', 'No Subject')}"
            
            draft_id = st.session_state.email_processor.save_draft(
                subject=subject,
                body=draft_content,
                to_email=to_email,
                reply_to_id=email.get('thread_id')
            )
            
            if draft_id:
                st.success("✅ Draft saved to Gmail!")
            else:
                st.error("❌ Failed to save draft to Gmail")
                
        except Exception as e:
            st.error(f"Error saving draft: {str(e)}")
    
    def render_analytics_tab(self):
        """Render analytics and insights tab"""
        st.markdown("### 📊 Email Analytics")
        
        emails = st.session_state.emails
        if not emails:
            st.info("📭 No emails loaded. Please check the Inbox tab first.")
            return
        
        # Time-based analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("#### 📅 Email Volume Over Time")
            dates = []
            for email in emails:
                try:
                    from email.utils import parsedate_to_datetime
                    date_obj = parsedate_to_datetime(email.get('date', ''))
                    dates.append(date_obj.date())
                except:
                    continue
            
            if dates:
                date_counts = pd.Series(dates).value_counts().sort_index()
                st.line_chart(date_counts)
        
        with col2:
            st.markdown("#### 👥 Top Senders")
            senders = {}
            for email in emails:
                sender = email.get('from', 'Unknown')
                if '<' in sender:
                    sender = sender.split('<')[1].split('>')[0]
                senders[sender] = senders.get(sender, 0) + 1
            
            if senders:
                top_senders = sorted(senders.items(), key=lambda x: x[1], reverse=True)[:10]
                for sender, count in top_senders:
                    st.markdown(f"**{sender}**: {count} emails")
        
        # Response analysis
        st.markdown("#### ↩️ Response Requirements")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            needs_reply_count = sum(1 for e in emails if e.get('needs_reply', False))
            st.metric("Needs Reply", needs_reply_count)
        
        with col2:
            avg_priority = sum(e.get('priority_score', 5) for e in emails) / len(emails) if emails else 0
            st.metric("Avg Priority", f"{avg_priority:.1f}/10")
        
        with col3:
            unread_count = sum(1 for e in emails if e.get('is_unread', False))
            st.metric("Unread Rate", f"{unread_count/len(emails)*100:.1f}%")
    
    def render_settings_tab(self):
        """Render settings and configuration tab"""
        st.markdown("### ⚙️ Settings & Configuration")
        
        # AI Provider settings
        st.markdown("#### 🤖 AI Provider Status")
        
        ai_provider = st.session_state.ai_provider
        if ai_provider and ai_provider.is_available():
            st.success(f"✅ Active: {ai_provider.provider}")
            st.info(f"Model: {getattr(ai_provider, 'model_name', 'Unknown')}")
        else:
            st.error("❌ No AI provider available")
            st.info("AI features like smart prioritization and summaries are disabled.")
        
        # Gmail connection
        st.markdown("#### 📧 Gmail Connection")
        
        try:
            profile = st.session_state.gmail_service.users().getProfile(userId='me').execute()
            st.success(f"✅ Connected: {profile.get('emailAddress', 'Unknown')}")
            
            if st.button("🔌 Disconnect Gmail"):
                token_path = "credentials/token.json"
                if os.path.exists(token_path):
                    os.remove(token_path)
                st.session_state.gmail_service = None
                st.session_state.authenticated = False
                st.success("Disconnected successfully!")
                st.rerun()
                
        except:
            st.error("❌ Connection issue")
        
        # Data management
        st.markdown("#### 🗂️ Data Management")
        
        if st.button("🗑️ Clear Cached Emails"):
            st.session_state.emails = []
            st.session_state.processed_emails = []
            # Clear all draft data
            keys_to_remove = [k for k in st.session_state.keys() if k.startswith(('auto_draft_', 'key_points_'))]
            for key in keys_to_remove:
                del st.session_state[key]
            st.success("Email cache and drafts cleared!")
        
        # Export functionality
        if st.session_state.emails:
            st.markdown("#### 📤 Export Data")
            
            if st.button("📊 Export to CSV"):
                self.export_emails_to_csv()
        
        # Download folders info
        st.markdown("#### 📁 Folders")
        st.info("📎 **Downloads**: ./downloads/ (email attachments)")
        st.info("📝 **Drafts**: ./drafts/ (local draft backups)")
    
    def render_full_email_view(self):
        """Render full email view"""
        if not st.session_state.selected_email:
            st.error("No email selected")
            return
        
        email = st.session_state.selected_email
        
        # Back button
        if st.button("← Back to Inbox"):
            st.session_state.current_view = 'dashboard'
            st.rerun()
        
        st.title("📧 Full Email View")
        
        # Email header
        st.markdown("### 📄 Email Details")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
            st.markdown(f"**From:** {email.get('from', 'Unknown')}")
            st.markdown(f"**To:** {email.get('to', 'Unknown')}")
            st.markdown(f"**Date:** {email.get('date', 'Unknown')}")
        
        with col2:
            priority = email.get('priority_score', 5)
            st.markdown(f"**Priority:** {priority}/10")
            st.markdown(f"**Status:** {'Unread' if email.get('is_unread') else 'Read'}")
            st.markdown(f"**Needs Reply:** {'Yes' if email.get('needs_reply') else 'No'}")
            
            if email.get('attachments'):
                st.markdown(f"**Attachments:** {len(email['attachments'])}")
        
        # AI Summary
        if email.get('ai_summary'):
            st.markdown("### 🤖 AI Summary")
            st.info(email['ai_summary'])
        
        # Full email body
        st.markdown("### 📄 Email Content")
        body = email.get('body', email.get('snippet', 'No content available'))
        st.text_area("", body, height=400, disabled=True)
        
        # Actions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("↩️ Reply to Email"):
                st.session_state.current_view = 'reply'
                st.rerun()
        
        with col2:
            if email.get('is_unread') and st.button("✅ Mark as Read"):
                self.mark_as_read(email['id'])
        
        with col3:
            if email.get('attachments') and st.button("📎 View Attachments"):
                st.session_state.current_view = 'attachments'
                st.rerun()
        
        with col4:
            if st.button("🤖 Generate Reply"):
                st.session_state.current_view = 'auto_reply'
                st.rerun()
    
    def render_attachments_view(self):
        """Render attachments download interface"""
        if not st.session_state.selected_email:
            st.error("No email selected")
            return
        
        email = st.session_state.selected_email
        
        # Back button
        if st.button("← Back to Email"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.title("📎 Email Attachments")
        
        # Email context
        st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
        st.markdown(f"**From:** {email.get('from', 'Unknown')}")
        
        attachments = email.get('attachments', [])
        
        if not attachments:
            st.info("📭 No attachments found in this email.")
            return
        
        st.markdown(f"### 📎 Found {len(attachments)} attachment(s)")
        
        for i, attachment in enumerate(attachments):
            with st.container():
                col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
                
                with col1:
                    st.markdown(f"**📁 {attachment['filename']}**")
                    st.caption(f"Type: {attachment['mimeType']}")
                
                with col2:
                    size_mb = attachment['size'] / (1024 * 1024)
                    if size_mb < 1:
                        size_kb = attachment['size'] / 1024
                        st.markdown(f"**{size_kb:.1f} KB**")
                    else:
                        st.markdown(f"**{size_mb:.1f} MB**")
                
                with col3:
                    if attachment['attachmentId']:
                        if st.button("📥 Download", key=f"download_{i}"):
                            with st.spinner(f"Downloading {attachment['filename']}..."):
                                file_path = st.session_state.email_processor.download_attachment(
                                    email['id'],
                                    attachment['attachmentId'],
                                    attachment['filename']
                                )
                                if file_path:
                                    st.success(f"✅ Downloaded to: {file_path}")
                                else:
                                    st.error("❌ Download failed")
                    else:
                        st.markdown("*No ID*")
                
                with col4:
                    # File type icon
                    file_ext = attachment['filename'].split('.')[-1].lower()
                    if file_ext in ['jpg', 'jpeg', 'png', 'gif']:
                        st.markdown("🖼️")
                    elif file_ext in ['pdf']:
                        st.markdown("📄")
                    elif file_ext in ['doc', 'docx']:
                        st.markdown("📝")
                    elif file_ext in ['xls', 'xlsx']:
                        st.markdown("📊")
                    elif file_ext in ['zip', 'rar']:
                        st.markdown("🗜️")
                    else:
                        st.markdown("📁")
                
                st.markdown("---")
        
        # Batch download
        st.markdown("### 📥 Batch Operations")
        
        if st.button("📥 Download All Attachments"):
            downloaded_count = 0
            progress_bar = st.progress(0)
            
            for i, attachment in enumerate(attachments):
                if attachment['attachmentId']:
                    with st.spinner(f"Downloading {attachment['filename']}..."):
                        file_path = st.session_state.email_processor.download_attachment(
                            email['id'],
                            attachment['attachmentId'],
                            attachment['filename']
                        )
                        if file_path:
                            downloaded_count += 1
                
                progress = (i + 1) / len(attachments)
                progress_bar.progress(progress)
            
            progress_bar.empty()
            st.success(f"✅ Downloaded {downloaded_count} of {len(attachments)} attachments!")
    
    def render_reply_interface(self):
        """Enhanced reply interface with Gmail integration"""
        if not st.session_state.selected_email:
            st.error("No email selected")
            return
        
        email = st.session_state.selected_email
        
        # Back button
        if st.button("← Back to Email"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.title("↩️ Reply to Email")
        
        # Original email context
        with st.expander("📧 Original Email", expanded=False):
            st.markdown(f"**From:** {email.get('from', 'Unknown')}")
            st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
            st.text_area("Original Content", email.get('body', '')[:500] + "...", height=150, disabled=True)
        
        # Reply form
        st.markdown("### ✍️ Compose Reply")
        
        # Extract recipient email
        from_email = email.get('from', '')
        import re
        email_match = re.search(r'<(.+?)>', from_email)
        to_email = email_match.group(1) if email_match else from_email.split()[0]
        
        # Form fields
        recipient = st.text_input("To", value=to_email)
        subject = st.text_input("Subject", value=f"Re: {email.get('subject', '')}")
        
        # AI suggestion button
        if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.info("💡 Need help composing? Use AI to generate a professional reply.")
            
            with col2:
                if st.button("🤖 AI Suggest"):
                    with st.spinner("Generating suggestion..."):
                        suggestion = self.generate_reply_draft(email)
                        st.session_state['reply_suggestion'] = suggestion
        
        # Show AI suggestion if available
        if 'reply_suggestion' in st.session_state:
            st.markdown("### 🤖 AI Suggestion")
            with st.expander("View AI Suggestion", expanded=False):
                st.text_area("AI Generated Reply", st.session_state['reply_suggestion'], height=200, disabled=True)
                if st.button("📝 Use This Suggestion"):
                    st.session_state['reply_body'] = st.session_state['reply_suggestion']
                    st.rerun()
        
        # Reply body
        reply_body = st.text_area(
            "Reply Message", 
            value=st.session_state.get('reply_body', ''),
            height=300,
            placeholder="Type your reply here...",
            key="reply_textarea"
        )
        
        # Update session state
        st.session_state['reply_body'] = reply_body
        
        # Action buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📤 Send Reply", type="primary"):
                if reply_body.strip():
                    with st.spinner("Sending email..."):
                        success = st.session_state.email_processor.send_email(
                            to_email=recipient,
                            subject=subject,
                            body=reply_body,
                            reply_to_id=email.get('thread_id')
                        )
                    
                    if success:
                        st.success("✅ Email sent successfully!")
                        # Clear the reply body
                        st.session_state['reply_body'] = ''
                        # Go back to inbox
                        st.session_state.current_view = 'dashboard'
                        st.rerun()
                    else:
                        st.error("❌ Failed to send email")
                else:
                    st.error("❌ Please enter a reply message")
        
        with col2:
            if st.button("💾 Save as Draft"):
                if reply_body.strip():
                    with st.spinner("Saving draft..."):
                        draft_id = st.session_state.email_processor.save_draft(
                            subject=subject,
                            body=reply_body,
                            to_email=recipient,
                            reply_to_id=email.get('thread_id')
                        )
                    
                    if draft_id:
                        st.success("✅ Draft saved to Gmail!")
                    else:
                        st.error("❌ Failed to save draft")
                else:
                    st.error("❌ Please enter a message to save")
        
        with col3:
            if st.button("🗑️ Clear"):
                st.session_state['reply_body'] = ''
                if 'reply_suggestion' in st.session_state:
                    del st.session_state['reply_suggestion']
                st.rerun()
    
    def render_auto_reply_interface(self):
        """Enhanced auto-reply interface"""
        if not st.session_state.selected_email:
            st.error("No email selected")
            return
        
        email = st.session_state.selected_email
        
        # Back button
        if st.button("← Back to Email"):
            st.session_state.current_view = 'full_email'
            st.rerun()
        
        st.title("🤖 AI-Powered Reply Assistant")
        
        # Original email context
        with st.expander("📧 Original Email Context", expanded=True):
            col1, col2 = st.columns(2)
            
            with col1:
                st.markdown(f"**From:** {email.get('from', 'Unknown')}")
                st.markdown(f"**Subject:** {email.get('subject', 'No Subject')}")
                st.markdown(f"**Date:** {email.get('date', 'Unknown')}")
            
            with col2:
                st.markdown(f"**Priority:** {email.get('priority_score', 5)}/10")
                st.markdown(f"**Needs Reply:** {'Yes' if email.get('needs_reply') else 'No'}")
            
            # AI Summary
            if email.get('ai_summary'):
                st.markdown("**🤖 Email Summary:**")
                st.info(email['ai_summary'])
            
            # Original content preview
            st.text_area(
                "Original Email Content", 
                email.get('body', '')[:500] + "..." if len(email.get('body', '')) > 500 else email.get('body', ''),
                height=150,
                disabled=True
            )
        
        # Reply customization options
        st.markdown("### 🎛️ Reply Customization")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            reply_tone = st.selectbox(
                "📝 Reply Tone",
                ["Professional", "Friendly", "Formal", "Casual"],
                index=0
            )
        
        with col2:
            reply_length = st.selectbox(
                "📏 Reply Length",
                ["Concise", "Standard", "Detailed"],
                index=1
            )
        
        with col3:
            include_action_items = st.checkbox("✅ Include Action Items", value=True)
        
        # Generate reply button
        if st.button("🚀 Generate AI Reply", type="primary"):
            if st.session_state.ai_provider and st.session_state.ai_provider.is_available():
                with st.spinner("🤖 Generating personalized reply..."):
                    draft = self.generate_advanced_reply_draft(
                        email, reply_tone, reply_length, include_action_items
                    )
                    st.session_state[f"ai_generated_reply_{email['id']}"] = draft
                    st.success("✅ AI reply generated!")
            else:
                st.error("❌ AI provider required for reply generation")
        
        # Show generated reply
        if f"ai_generated_reply_{email['id']}" in st.session_state:
            st.markdown("### 📄 Generated Reply")
            
            generated_reply = st.text_area(
                "AI Generated Reply (Editable)",
                st.session_state[f"ai_generated_reply_{email['id']}"],
                height=400,
                help="You can edit the AI-generated reply before sending"
            )
            
            # Update if edited
            if generated_reply != st.session_state[f"ai_generated_reply_{email['id']}"]:
                st.session_state[f"ai_generated_reply_{email['id']}"] = generated_reply
            
            # Action buttons
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                if st.button("📤 Send Reply"):
                    # Extract recipient email
                    from_email = email.get('from', '')
                    import re
                    email_match = re.search(r'<(.+?)>', from_email)
                    to_email = email_match.group(1) if email_match else from_email
                    subject = f"Re: {email.get('subject', 'No Subject')}"
                    
                    with st.spinner("Sending reply..."):
                        success = st.session_state.email_processor.send_email(
                            to_email=to_email,
                            subject=subject,
                            body=generated_reply,
                            reply_to_id=email.get('thread_id')
                        )
                    
                    if success:
                        st.success("✅ Reply sent successfully!")
                        st.session_state.current_view = 'dashboard'
                        st.rerun()
                    else:
                        st.error("❌ Failed to send reply")
            
            with col2:
                if st.button("💾 Save Draft"):
                    from_email = email.get('from', '')
                    import re
                    email_match = re.search(r'<(.+?)>', from_email)
                    to_email = email_match.group(1) if email_match else from_email
                    subject = f"Re: {email.get('subject', 'No Subject')}"
                    
                    with st.spinner("Saving draft..."):
                        draft_id = st.session_state.email_processor.save_draft(
                            subject=subject,
                            body=generated_reply,
                            to_email=to_email,
                            reply_to_id=email.get('thread_id')
                        )
                    
                    if draft_id:
                        st.success("✅ Draft saved to Gmail!")
                    else:
                        st.error("❌ Failed to save draft")
            
            with col3:
                if st.button("✏️ Manual Edit"):
                    st.session_state['reply_body'] = generated_reply
                    st.session_state.current_view = 'reply'
                    st.rerun()
            
            with col4:
                if st.button("🔄 Regenerate"):
                    with st.spinner("Regenerating reply..."):
                        new_draft = self.generate_advanced_reply_draft(
                            email, reply_tone, reply_length, include_action_items
                        )
                        st.session_state[f"ai_generated_reply_{email['id']}"] = new_draft
                        st.rerun()
    
    def generate_advanced_reply_draft(self, email, tone="Professional", length="Standard", include_action_items=True):
        """Generate an advanced reply draft with specific parameters"""
        if not st.session_state.ai_provider or not st.session_state.ai_provider.is_available():
            return "AI provider not available for draft generation"
        
        tone_instructions = {
            "Professional": "Use formal, business-appropriate language with proper etiquette",
            "Friendly": "Use warm, approachable language while maintaining professionalism",
            "Formal": "Use very formal, traditional business language with complete sentences",
            "Casual": "Use relaxed, conversational tone while remaining respectful"
        }
        
        length_instructions = {
            "Concise": "Keep the reply brief and to the point, maximum 2-3 short paragraphs",
            "Standard": "Provide a comprehensive but balanced response, 3-4 paragraphs",
            "Detailed": "Provide thorough, detailed responses with explanations, 4-5 paragraphs"
        }
        
        action_items_instruction = (
            "Include clear action items and next steps where appropriate" 
            if include_action_items else 
            "Focus on acknowledgment and responses without specific action items"
        )
        
        prompt = f"""
        Generate a professional email reply with the following specifications:
        
        ORIGINAL EMAIL:
        Subject: {email.get('subject', '')}
        From: {email.get('from', '')}
        Date: {email.get('date', '')}
        Content: {email.get('body', '')[:1500]}
        
        REPLY REQUIREMENTS:
        - Tone: {tone} - {tone_instructions.get(tone, '')}
        - Length: {length} - {length_instructions.get(length, '')}
        - Action Items: {action_items_instruction}
        
        REPLY STRUCTURE:
        1. Appropriate greeting and acknowledgment
        2. Address all questions and requests from the original email
        3. Provide helpful information or next steps
        4. {action_items_instruction}
        5. Professional closing
        
        IMPORTANT:
        - Address the sender by name if possible
        - Reference specific points from their email
        - Be helpful and solution-oriented
        - Include proper email signature placeholder
        
        Generate the complete email reply:
        """
        
        try:
            response = st.session_state.ai_provider.generate_response(prompt)
            return response
        except Exception as e:
            return f"Error generating advanced draft: {str(e)}"
    
    def mark_as_read(self, email_id):
        """Mark email as read"""
        try:
            st.session_state.gmail_service.users().messages().modify(
                userId='me',
                id=email_id,
                body={'removeLabelIds': ['UNREAD']}
            ).execute()
            st.success("✅ Email marked as read!")
            
            # Update local cache
            for email in st.session_state.emails:
                if email['id'] == email_id:
                    email['is_unread'] = False
                    break
            
        except Exception as e:
            st.error(f"❌ Error marking as read: {str(e)}")
    
    def export_emails_to_csv(self):
        """Export emails to CSV"""
        try:
            emails_data = []
            for email in st.session_state.emails:
                emails_data.append({
                    'Subject': email.get('subject', ''),
                    'From': email.get('from', ''),
                    'Date': email.get('date', ''),
                    'Priority': email.get('priority_score', 5),
                    'Needs Reply': email.get('needs_reply', False),
                    'Unread': email.get('is_unread', False),
                    'Summary': email.get('ai_summary', '')[:100],
                    'Attachments': len(email.get('attachments', []))
                })
            
            df = pd.DataFrame(emails_data)
            csv = df.to_csv(index=False)
            
            st.download_button(
                label="📥 Download CSV",
                data=csv,
                file_name=f"mailmind_export_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
            
        except Exception as e:
            st.error(f"❌ Export error: {str(e)}")
    
    def run(self):
        """Main application runner with enhanced routing"""
        
        # Check if setup is complete
        if not st.session_state.setup_complete:
            self.render_setup_page()
            return
        
        # Check authentication
        if not st.session_state.authenticated:
            self.render_authentication()
            return
        
        # Route to appropriate view
        if st.session_state.current_view == 'full_email':
            self.render_full_email_view()
        elif st.session_state.current_view == 'reply':
            self.render_reply_interface()
        elif st.session_state.current_view == 'auto_reply':
            self.render_auto_reply_interface()
        elif st.session_state.current_view == 'attachments':
            self.render_attachments_view()
        else:
            self.render_main_dashboard()

def main():
    """Main function to run the enhanced MailMind application"""
    app = MailMindApp()
    app.run()

if __name__ == "__main__":
    main()