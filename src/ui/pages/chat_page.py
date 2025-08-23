import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import time

def show_chat_page():
    """Main function to display the chat page"""
    
    st.title("💬 Email Assistant Chat")
    st.markdown("---")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
        # Add welcome message
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': "Hello! I'm your Email Assistant. I can help you with email management, drafting responses, analyzing email patterns, and more. How can I assist you today?",
            'timestamp': datetime.now()
        })
    
    if 'current_conversation' not in st.session_state:
        st.session_state.current_conversation = []
    
    # Sidebar with chat features
    with st.sidebar:
        st.header("💬 Chat Features")
        
        # Quick actions
        st.subheader("🚀 Quick Actions")
        if st.button("📧 Help with Email Draft"):
            quick_prompt = "I need help drafting a professional email. Can you guide me through the process?"
            handle_user_input(quick_prompt)
        
        if st.button("📊 Analyze My Emails"):
            quick_prompt = "Can you help me analyze patterns in my email data and suggest improvements?"
            handle_user_input(quick_prompt)
        
        if st.button("⚡ Email Tips"):
            quick_prompt = "Give me some tips for better email management and productivity."
            handle_user_input(quick_prompt)
        
        if st.button("🔧 Troubleshooting"):
            quick_prompt = "I'm having issues with my email system. Can you help me troubleshoot?"
            handle_user_input(quick_prompt)
        
        # Conversation management
        st.markdown("---")
        st.subheader("🗂️ Conversations")
        
        if st.button("🗑️ Clear Chat"):
            st.session_state.chat_history = [{
                'role': 'assistant',
                'content': "Chat cleared! How can I help you today?",
                'timestamp': datetime.now()
            }]
            st.rerun()
        
        if st.button("💾 Save Conversation"):
            save_conversation()
            st.success("Conversation saved!")
        
        # Chat settings
        st.markdown("---")
        st.subheader("⚙️ Settings")
        
        response_style = st.selectbox(
            "Response Style",
            ["Professional", "Casual", "Detailed", "Concise"]
        )
        
        auto_suggestions = st.checkbox("Auto Suggestions", value=True)
        typing_indicator = st.checkbox("Typing Indicator", value=True)
        
        # Export options
        st.markdown("---")
        st.subheader("📤 Export")
        
        if st.button("📄 Export as Text"):
            export_chat_as_text()
        
        if st.button("📊 Export as JSON"):
            export_chat_as_json()
    
    # Main chat interface
    chat_container = st.container()
    
    with chat_container:
        # Display chat messages
        for message in st.session_state.chat_history:
            display_message(message)
    
    # Chat input
    st.markdown("---")
    
    # Quick suggestion buttons
    if auto_suggestions and len(st.session_state.chat_history) > 1:
        st.subheader("💡 Suggested Questions")
        suggestions = get_smart_suggestions()
        
        cols = st.columns(len(suggestions))
        for i, suggestion in enumerate(suggestions):
            with cols[i]:
                if st.button(suggestion, key=f"suggestion_{i}"):
                    handle_user_input(suggestion)
    
    # Main input area
    col1, col2 = st.columns([5, 1])
    
    with col1:
        user_input = st.text_area(
            "Type your message here...",
            height=100,
            key="chat_input",
            placeholder="Ask me anything about email management, drafting, analysis, or general help..."
        )
    
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        
        if st.button("📤 Send", type="primary", use_container_width=True):
            if user_input.strip():
                handle_user_input(user_input)
        
        if st.button("🎤 Voice", use_container_width=True):
            st.info("Voice input feature coming soon!")
        
        if st.button("📎 Attach", use_container_width=True):
            uploaded_file = st.file_uploader(
                "Upload file for analysis",
                type=['txt', 'csv', 'xlsx', 'pdf'],
                key="chat_file_upload"
            )
            if uploaded_file:
                handle_file_upload(uploaded_file)
    
    # File upload area
    if st.session_state.get('show_file_upload', False):
        st.markdown("---")
        uploaded_file = st.file_uploader(
            "Upload a file for analysis (emails, CSV data, etc.)",
            type=['txt', 'csv', 'xlsx', 'pdf', 'eml']
        )
        if uploaded_file:
            handle_file_upload(uploaded_file)


def display_message(message):
    """Display a chat message with appropriate styling"""
    
    if message['role'] == 'user':
        # User message (right-aligned)
        st.markdown(f"""
        <div style='text-align: right; margin: 10px 0;'>
            <div style='display: inline-block; background-color: #007bff; color: white; 
                       padding: 10px 15px; border-radius: 18px; max-width: 70%; 
                       text-align: left; margin-left: 30%;'>
                {message['content']}
            </div>
            <div style='font-size: 0.8em; color: #666; margin-top: 5px;'>
                You • {message['timestamp'].strftime('%H:%M')}
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    else:
        # Assistant message (left-aligned)
        with st.chat_message("assistant", avatar="🤖"):
            st.markdown(message['content'])
            st.caption(f"Assistant • {message['timestamp'].strftime('%H:%M')}")
            
            # Add action buttons for assistant messages
            col1, col2, col3 = st.columns([1, 1, 4])
            with col1:
                if st.button("👍", key=f"like_{message['timestamp']}", help="Helpful"):
                    st.session_state[f"liked_{message['timestamp']}"] = True
            with col2:
                if st.button("📋", key=f"copy_{message['timestamp']}", help="Copy"):
                    st.session_state[f"copied_{message['timestamp']}"] = message['content']
                    st.toast("Message copied!")


def handle_user_input(user_input):
    """Process user input and generate response"""
    
    # Clear the input
    if 'chat_input' in st.session_state:
        st.session_state.chat_input = ""
    
    # Add user message to chat history
    st.session_state.chat_history.append({
        'role': 'user',
        'content': user_input,
        'timestamp': datetime.now()
    })
    
    # Show typing indicator
    if st.session_state.get('typing_indicator', True):
        with st.spinner("Assistant is typing..."):
            time.sleep(1)  # Simulate thinking time
    
    # Generate response based on user input
    response = generate_response(user_input)
    
    # Add assistant response to chat history
    st.session_state.chat_history.append({
        'role': 'assistant',
        'content': response,
        'timestamp': datetime.now()
    })
    
    st.rerun()


def generate_response(user_input):
    """Generate appropriate response based on user input"""
    
    user_input_lower = user_input.lower()
    
    # Email drafting help
    if any(keyword in user_input_lower for keyword in ['draft', 'write email', 'compose', 'reply']):
        return """I'd be happy to help you draft an email! Here's how we can approach this:

**1. What type of email are you writing?**
- Professional business email
- Follow-up email  
- Meeting request
- Apology or complaint response
- Thank you email

**2. Key information I need:**
- Who is the recipient?
- What's the main purpose/message?
- What tone do you want? (formal, casual, urgent, friendly)
- Any specific details to include?

**3. I can help with:**
- Structuring your email properly
- Choosing the right tone and words
- Proofreading and suggestions
- Email templates

Just tell me more about your specific situation and I'll guide you through creating an effective email!"""

    # Email analysis
    elif any(keyword in user_input_lower for keyword in ['analyze', 'analysis', 'pattern', 'insight']):
        return """I can help you analyze your email patterns and provide insights! Here's what I can do:

**📊 Email Analytics I can provide:**
- Response time patterns
- Most active senders/recipients  
- Email volume trends
- Priority distribution analysis
- Time-based activity patterns

**🎯 Insights I can offer:**
- Identify email overload periods
- Suggest optimal response times
- Highlight important contacts
- Recommend email management strategies
- Detect communication bottlenecks

**📈 To get started:**
1. Upload your email data (CSV, Excel, or text format)
2. Or tell me about specific email challenges you're facing
3. I'll provide personalized recommendations

What specific aspect of your email patterns would you like to explore?"""

    # Email management tips
    elif any(keyword in user_input_lower for keyword in ['tips', 'productivity', 'manage', 'organize']):
        return """Here are my top email productivity and management tips:

**⚡ Quick Wins:**
- Use the 2-minute rule: If it takes less than 2 minutes, do it now
- Set specific times for checking email (not all day)
- Use templates for common responses
- Enable email notifications only for truly urgent emails

**🗂️ Organization Strategies:**
- Implement a folder system (Action, Waiting, Archive)
- Use filters and rules to auto-sort emails
- Flag important emails that need follow-up
- Archive rather than delete old emails

**✍️ Writing Efficiency:**
- Keep emails concise and scannable
- Use bullet points for multiple items
- Put action items at the beginning
- Use clear, specific subject lines

**📱 Advanced Tips:**
- Use keyboard shortcuts to speed up navigation
- Set up auto-responses for common questions
- Batch similar tasks together
- Regular inbox cleanup sessions

Would you like me to elaborate on any of these strategies or help you implement them?"""

    # Troubleshooting
    elif any(keyword in user_input_lower for keyword in ['problem', 'issue', 'troubleshoot', 'help', 'error']):
        return """I'm here to help you troubleshoot! Let me know what specific issues you're experiencing:

**🔧 Common Email Issues I can help with:**

**Technical Problems:**
- Email not sending/receiving
- Attachment issues
- Synchronization problems
- Login/authentication issues

**Organization Problems:**
- Inbox overload management
- Missing important emails
- Difficulty finding old emails
- Poor email workflow

**Communication Issues:**
- Getting better responses
- Managing email tone
- Reducing back-and-forth emails
- Professional email etiquette

**Productivity Problems:**
- Too much time on email
- Constant interruptions
- Missing deadlines
- Poor prioritization

Please describe your specific situation and I'll provide targeted solutions and step-by-step guidance!"""

    # General help or greeting
    elif any(keyword in user_input_lower for keyword in ['hello', 'hi', 'help', 'what can you do']):
        return """Hello! I'm your Email Assistant, and I'm here to make your email experience better! 🚀

**Here's how I can help you:**

**📝 Email Composition:**
- Draft professional emails
- Improve email tone and clarity
- Create templates for common emails
- Proofread and edit your messages

**📊 Email Analysis:**
- Analyze your email patterns
- Identify productivity opportunities  
- Generate insights from your email data
- Track response times and trends

**⚡ Productivity & Organization:**
- Email management strategies
- Inbox organization tips
- Time-saving techniques
- Workflow optimization

**🔧 Problem Solving:**
- Troubleshoot email issues
- Answer email etiquette questions
- Provide best practices guidance
- Help with specific challenges

**Just ask me anything like:**
- "Help me write a professional follow-up email"
- "Analyze my email response patterns"
- "Give me tips for managing email overload"
- "I'm having trouble with [specific issue]"

What would you like to work on today?"""

    # Default response
    else:
        return f"""I understand you're asking about: "{user_input}"

Let me help you with that! Based on your question, I can provide assistance with:

**If you're looking for email help:**
- Email drafting and composition
- Response strategies and templates
- Professional communication advice

**If you need technical support:**
- Troubleshooting email issues
- Setup and configuration guidance
- Best practices for email management

**If you want productivity tips:**
- Time management strategies
- Inbox organization methods
- Workflow optimization

Could you provide a bit more detail about what specifically you'd like help with? I'm here to give you the most relevant and useful assistance!"""


def get_smart_suggestions():
    """Generate smart suggestions based on chat history"""
    
    if len(st.session_state.chat_history) < 2:
        return [
            "Help me write a professional email",
            "Analyze my email patterns",
            "Give me productivity tips"
        ]
    
    last_message = st.session_state.chat_history[-1]['content'].lower()
    
    if 'draft' in last_message or 'write' in last_message:
        return [
            "Show me email templates",
            "Help with email tone",
            "What's a good subject line?"
        ]
    elif 'analyze' in last_message or 'pattern' in last_message:
        return [
            "Show me response time tips",
            "How to prioritize emails?",
            "Email organization strategies"
        ]
    else:
        return [
            "Help with email etiquette",
            "Troubleshoot email issues",
            "Time management tips"
        ]


def handle_file_upload(uploaded_file):
    """Handle file uploads and provide analysis"""
    
    if uploaded_file is not None:
        file_details = {
            "filename": uploaded_file.name,
            "filetype": uploaded_file.type,
            "filesize": uploaded_file.size
        }
        
        st.session_state.chat_history.append({
            'role': 'user',
            'content': f"I've uploaded a file: {uploaded_file.name}",
            'timestamp': datetime.now()
        })
        
        # Analyze the file based on its type
        if uploaded_file.type == 'text/csv':
            response = f"""📊 I've received your CSV file "{uploaded_file.name}". 

I can help you analyze this email data! Here's what I can do:

**Available Analysis:**
- Email volume trends over time
- Top senders and recipients
- Response time analysis  
- Priority distribution
- Peak activity periods

**Next Steps:**
1. The file contains {uploaded_file.size} bytes of data
2. I'll process the email data and generate insights
3. You'll get visualizations and actionable recommendations

Would you like me to start with a specific type of analysis, or should I provide a comprehensive overview of your email patterns?"""

        elif uploaded_file.type in ['text/plain', 'application/vnd.ms-outlook']:
            response = f"""📧 I've received your email file "{uploaded_file.name}".

I can help you with:

**Email Content Analysis:**
- Tone and sentiment analysis
- Professional writing suggestions
- Response recommendations
- Template creation

**What would you like me to focus on:**
- Improving the email's clarity and impact
- Analyzing communication patterns
- Creating a template based on this email
- Providing response suggestions

Let me know how you'd like to proceed!"""

        else:
            response = f"""📎 I've received your file "{uploaded_file.name}" ({uploaded_file.type}).

While I can see the file, I work best with:
- CSV files for email data analysis
- Text files for email content review
- Email files (.eml) for message analysis

If this contains email data, could you try converting it to CSV format? Or let me know what specific help you need with this file!"""
        
        st.session_state.chat_history.append({
            'role': 'assistant',
            'content': response,
            'timestamp': datetime.now()
        })
        
        st.rerun()


def save_conversation():
    """Save the current conversation"""
    
    conversation_data = {
        'timestamp': datetime.now().isoformat(),
        'messages': st.session_state.chat_history,
        'session_id': f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    }
    
    # In a real app, this would save to a database or file
    if 'saved_conversations' not in st.session_state:
        st.session_state.saved_conversations = []
    
    st.session_state.saved_conversations.append(conversation_data)


def export_chat_as_text():
    """Export chat history as text"""
    
    text_content = "Email Assistant Conversation\n"
    text_content += "=" * 50 + "\n\n"
    
    for message in st.session_state.chat_history:
        role = "You" if message['role'] == 'user' else "Assistant"
        timestamp = message['timestamp'].strftime('%Y-%m-%d %H:%M')
        text_content += f"[{timestamp}] {role}:\n{message['content']}\n\n"
    
    st.download_button(
        label="📄 Download Chat as Text",
        data=text_content,
        file_name=f"email_assistant_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        mime="text/plain"
    )


def export_chat_as_json():
    """Export chat history as JSON"""
    
    chat_data = {
        'export_timestamp': datetime.now().isoformat(),
        'total_messages': len(st.session_state.chat_history),
        'chat_history': [
            {
                'role': msg['role'],
                'content': msg['content'],
                'timestamp': msg['timestamp'].isoformat()
            }
            for msg in st.session_state.chat_history
        ]
    }
    
    json_content = json.dumps(chat_data, indent=2)
    
    st.download_button(
        label="📊 Download Chat as JSON",
        data=json_content,
        file_name=f"email_assistant_chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        mime="application/json"
    )


if __name__ == "__main__":
    show_chat_page()