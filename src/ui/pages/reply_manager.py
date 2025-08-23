import streamlit as st
import pandas as pd
from datetime import datetime, timedelta
import json
import re

def show_reply_manager():
    """Main function to display the reply management page"""
    
    st.title("✉️ Reply Manager")
    st.markdown("---")
    
    # Initialize session state
    if 'draft_replies' not in st.session_state:
        st.session_state.draft_replies = []
    if 'templates' not in st.session_state:
        st.session_state.templates = get_default_templates()
    if 'sent_replies' not in st.session_state:
        st.session_state.sent_replies = []
    
    # Main tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📝 Compose Reply", "📄 Templates", "📋 Drafts", "📤 Sent"])
    
    with tab1:
        show_compose_reply()
    
    with tab2:
        show_templates_manager()
    
    with tab3:
        show_drafts_manager()
    
    with tab4:
        show_sent_manager()


def show_compose_reply():
    """Display the compose reply interface"""
    st.subheader("📝 Compose New Reply")
    
    # Email selection or manual input
    col1, col2 = st.columns([2, 1])
    
    with col1:
        reply_mode = st.radio(
            "Reply Mode",
            ["Reply to Existing Email", "Compose New Email"],
            horizontal=True
        )
    
    with col2:
        if st.button("🤖 AI Assistant", help="Get AI help with your reply"):
            st.session_state.show_ai_helper = True
    
    if reply_mode == "Reply to Existing Email":
        # Mock email selection
        selected_email = st.selectbox(
            "Select Email to Reply To",
            ["john.doe@company.com - Urgent: Project deadline approaching",
             "sarah.smith@client.com - Meeting reschedule request",
             "mike.jones@partner.org - Budget approval needed"]
        )
        
        if selected_email:
            # Show original email context
            with st.expander("📧 Original Email", expanded=True):
                st.write("**From:** john.doe@company.com")
                st.write("**Subject:** Urgent: Project deadline approaching")
                st.write("**Received:** 2025-01-15 09:30")
                st.markdown("---")
                st.write("""
                Hi there,
                
                I wanted to follow up on the project deadline we discussed last week. 
                It looks like we might need to push the delivery date by a few days due to 
                some technical complications that have arisen.
                
                Could we schedule a quick call to discuss the revised timeline?
                
                Best regards,
                John
                """)
    
    # Reply composition form
    with st.form("compose_reply_form"):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            to_email = st.text_input("To:", value="john.doe@company.com" if reply_mode == "Reply to Existing Email" else "")
            cc_email = st.text_input("CC:", "")
            bcc_email = st.text_input("BCC:", "")
        
        with col2:
            priority = st.selectbox("Priority", ["Normal", "High", "Low"])
            
        subject = st.text_input("Subject:", value="Re: Project deadline approaching" if reply_mode == "Reply to Existing Email" else "")
        
        # Template selection
        template_options = ["None"] + list(st.session_state.templates.keys())
        selected_template = st.selectbox("Use Template:", template_options)
        
        # Email body
        if selected_template != "None":
            template_content = st.session_state.templates[selected_template]['content']
            default_body = template_content
        else:
            default_body = ""
        
        email_body = st.text_area("Email Body:", value=default_body, height=300)
        
        # AI suggestions (if enabled)
        if st.session_state.get('show_ai_helper', False):
            st.markdown("---")
            st.subheader("🤖 AI Suggestions")
            
            suggestion_type = st.selectbox(
                "Get AI help with:",
                ["Improve tone", "Make more professional", "Make more concise", "Add empathy"]
            )
            
            if st.button("Generate Suggestion"):
                suggestion = generate_ai_suggestion(email_body, suggestion_type)
                st.text_area("AI Suggestion:", value=suggestion, height=150)
                if st.button("Use This Suggestion"):
                    email_body = suggestion
                    st.rerun()
        
        # Attachment upload
        uploaded_files = st.file_uploader("Attachments:", accept_multiple_files=True)
        
        # Form submission buttons
        col1, col2, col3 = st.columns(3)
        
        with col1:
            send_now = st.form_submit_button("📤 Send Now", type="primary")
        with col2:
            save_draft = st.form_submit_button("💾 Save Draft")
        with col3:
            schedule_send = st.form_submit_button("⏰ Schedule Send")
        
        if send_now:
            if to_email and subject and email_body:
                reply_data = {
                    'id': f'reply_{len(st.session_state.sent_replies) + 1}',
                    'to': to_email,
                    'cc': cc_email,
                    'bcc': bcc_email,
                    'subject': subject,
                    'body': email_body,
                    'priority': priority,
                    'sent_at': datetime.now(),
                    'attachments': [f.name for f in uploaded_files] if uploaded_files else []
                }
                st.session_state.sent_replies.append(reply_data)
                st.success("✅ Email sent successfully!")
            else:
                st.error("Please fill in all required fields (To, Subject, Body)")
        
        elif save_draft:
            if subject or email_body:
                draft_data = {
                    'id': f'draft_{len(st.session_state.draft_replies) + 1}',
                    'to': to_email,
                    'cc': cc_email,
                    'bcc': bcc_email,
                    'subject': subject,
                    'body': email_body,
                    'priority': priority,
                    'saved_at': datetime.now(),
                    'attachments': [f.name for f in uploaded_files] if uploaded_files else []
                }
                st.session_state.draft_replies.append(draft_data)
                st.success("💾 Draft saved successfully!")
        
        elif schedule_send:
            scheduled_time = st.datetime_input("Schedule for:", min_value=datetime.now())
            st.info(f"⏰ Email scheduled for {scheduled_time}")


def show_templates_manager():
    """Display and manage email templates"""
    st.subheader("📄 Email Templates")
    
    # Template actions
    col1, col2 = st.columns([3, 1])
    with col2:
        if st.button("➕ Create New Template"):
            st.session_state.show_template_form = True
    
    # Show existing templates
    if st.session_state.templates:
        for template_name, template_data in st.session_state.templates.items():
            with st.expander(f"📄 {template_name}", expanded=False):
                col1, col2 = st.columns([4, 1])
                
                with col1:
                    st.write(f"**Category:** {template_data['category']}")
                    st.write(f"**Description:** {template_data['description']}")
                    st.code(template_data['content'], language=None)
                
                with col2:
                    if st.button("✏️ Edit", key=f"edit_{template_name}"):
                        st.session_state.edit_template = template_name
                    if st.button("🗑️ Delete", key=f"delete_{template_name}"):
                        del st.session_state.templates[template_name]
                        st.rerun()
                    if st.button("📋 Use", key=f"use_{template_name}"):
                        st.info("Template selected! Go to Compose tab to use it.")
    
    # Template creation form
    if st.session_state.get('show_template_form', False):
        st.markdown("---")
        st.subheader("Create New Template")
        
        with st.form("new_template_form"):
            template_name = st.text_input("Template Name:")
            template_category = st.selectbox(
                "Category:",
                ["General", "Meeting", "Follow-up", "Apology", "Thank You", "Formal"]
            )
            template_description = st.text_input("Description:")
            template_content = st.text_area("Template Content:", height=200, 
                                          placeholder="Dear [NAME],\n\nI hope this email finds you well...\n\nBest regards,\n[YOUR_NAME]")
            
            col1, col2 = st.columns(2)
            with col1:
                if st.form_submit_button("💾 Save Template"):
                    if template_name and template_content:
                        st.session_state.templates[template_name] = {
                            'category': template_category,
                            'description': template_description,
                            'content': template_content
                        }
                        st.session_state.show_template_form = False
                        st.success("Template created successfully!")
                        st.rerun()
            with col2:
                if st.form_submit_button("❌ Cancel"):
                    st.session_state.show_template_form = False
                    st.rerun()


def show_drafts_manager():
    """Display and manage draft emails"""
    st.subheader("📋 Draft Emails")
    
    if not st.session_state.draft_replies:
        st.info("No drafts saved yet.")
        return
    
    for draft in st.session_state.draft_replies:
        with st.expander(
            f"📝 {draft['subject'] or 'No Subject'} - To: {draft['to']}"
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**To:** {draft['to']}")
                if draft['cc']:
                    st.write(f"**CC:** {draft['cc']}")
                st.write(f"**Subject:** {draft['subject']}")
                st.write(f"**Saved:** {draft['saved_at'].strftime('%Y-%m-%d %H:%M')}")
                st.write("**Body:**")
                st.text_area("", value=draft['body'], height=150, disabled=True, key=f"draft_body_{draft['id']}")
                
                if draft['attachments']:
                    st.write("**Attachments:**", ", ".join(draft['attachments']))
            
            with col2:
                if st.button("✏️ Continue Editing", key=f"edit_draft_{draft['id']}"):
                    st.info("Loading draft for editing...")
                if st.button("📤 Send Now", key=f"send_draft_{draft['id']}"):
                    # Move to sent items
                    draft['sent_at'] = datetime.now()
                    st.session_state.sent_replies.append(draft)
                    st.session_state.draft_replies.remove(draft)
                    st.success("Draft sent!")
                    st.rerun()
                if st.button("🗑️ Delete", key=f"delete_draft_{draft['id']}"):
                    st.session_state.draft_replies.remove(draft)
                    st.success("Draft deleted!")
                    st.rerun()


def show_sent_manager():
    """Display sent emails history"""
    st.subheader("📤 Sent Emails")
    
    if not st.session_state.sent_replies:
        st.info("No emails sent yet.")
        return
    
    # Filters
    col1, col2, col3 = st.columns(3)
    with col1:
        date_filter = st.date_input("Filter by date:", value=None)
    with col2:
        recipient_filter = st.text_input("Filter by recipient:")
    with col3:
        subject_filter = st.text_input("Filter by subject:")
    
    # Display sent emails
    filtered_sent = st.session_state.sent_replies.copy()
    
    if date_filter:
        filtered_sent = [email for email in filtered_sent 
                        if email['sent_at'].date() == date_filter]
    
    if recipient_filter:
        filtered_sent = [email for email in filtered_sent 
                        if recipient_filter.lower() in email['to'].lower()]
    
    if subject_filter:
        filtered_sent = [email for email in filtered_sent 
                        if subject_filter.lower() in email['subject'].lower()]
    
    for email in filtered_sent:
        with st.expander(
            f"📧 {email['subject']} - To: {email['to']} ({email['sent_at'].strftime('%Y-%m-%d %H:%M')})"
        ):
            col1, col2 = st.columns([3, 1])
            
            with col1:
                st.write(f"**To:** {email['to']}")
                if email['cc']:
                    st.write(f"**CC:** {email['cc']}")
                st.write(f"**Subject:** {email['subject']}")
                st.write(f"**Sent:** {email['sent_at'].strftime('%Y-%m-%d %H:%M')}")
                st.write(f"**Priority:** {email['priority']}")
                st.write("**Body:**")
                st.text_area("", value=email['body'], height=150, disabled=True, key=f"sent_body_{email['id']}")
                
                if email['attachments']:
                    st.write("**Attachments:**", ", ".join(email['attachments']))
            
            with col2:
                if st.button("↩️ Reply Again", key=f"reply_again_{email['id']}"):
                    st.info("Creating new reply based on this email...")
                if st.button("📄 Create Template", key=f"template_{email['id']}"):
                    st.info("Creating template from this email...")


def get_default_templates():
    """Return default email templates"""
    return {
        "Meeting Request": {
            "category": "Meeting",
            "description": "Request a meeting with someone",
            "content": """Dear [NAME],

I hope this email finds you well. I would like to schedule a meeting with you to discuss [TOPIC].

Would you be available for a [DURATION] meeting sometime next week? I'm flexible with timing and can work around your schedule.

Please let me know what works best for you.

Best regards,
[YOUR_NAME]"""
        },
        "Follow-up": {
            "category": "Follow-up",
            "description": "Follow up on a previous conversation or request",
            "content": """Dear [NAME],

I wanted to follow up on our previous conversation regarding [TOPIC].

[SPECIFIC_DETAILS]

Please let me know if you need any additional information from my side.

Looking forward to your response.

Best regards,
[YOUR_NAME]"""
        },
        "Thank You": {
            "category": "Thank You",
            "description": "Express gratitude",
            "content": """Dear [NAME],

Thank you very much for [REASON]. I really appreciate [SPECIFIC_APPRECIATION].

[ADDITIONAL_DETAILS]

Thanks again for your time and assistance.

Best regards,
[YOUR_NAME]"""
        },
        "Apology": {
            "category": "Apology",
            "description": "Apologize for an issue or mistake",
            "content": """Dear [NAME],

I sincerely apologize for [ISSUE]. I understand that this may have caused [IMPACT], and I take full responsibility.

To address this situation, I will [ACTION_PLAN].

Thank you for your patience and understanding.

Best regards,
[YOUR_NAME]"""
        }
    }


def generate_ai_suggestion(email_body, suggestion_type):
    """Generate AI suggestions for email improvement (mock function)"""
    suggestions = {
        "Improve tone": f"Here's a more polished version:\n\n{email_body}\n\n(Note: This would be an AI-improved version with better tone)",
        "Make more professional": f"Professional version:\n\n{email_body}\n\n(Note: This would be a more formal, professional version)",
        "Make more concise": f"Concise version:\n\n{email_body[:len(email_body)//2]}...\n\n(Note: This would be a shortened version)",
        "Add empathy": f"Empathetic version:\n\n{email_body}\n\n(Note: This would include more empathetic language)"
    }
    return suggestions.get(suggestion_type, email_body)


if __name__ == "__main__":
    show_reply_manager()