import streamlit as st
import requests
import json
import uuid
from datetime import datetime
import time

# Configuration
API_BASE_URL = "http://localhost:8000"

# Set page config
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .chat-message {
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
        border-left: 4px solid #1f77b4;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left-color: #2196f3;
    }
    .bot-message {
        background-color: #f3e5f5;
        border-left-color: #9c27b0;
    }
    .status-success {
        color: #4caf50;
        font-weight: bold;
    }
    .status-error {
        color: #f44336;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "session_id" not in st.session_state:
    st.session_state.session_id = str(uuid.uuid4())
if "messages" not in st.session_state:
    st.session_state.messages = []
if "api_status" not in st.session_state:
    st.session_state.api_status = None

def check_api_health():
    """Check if the API is running."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API returned status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def send_message(message, model="gemini-pro"):
    """Send a message to the chatbot API."""
    try:
        payload = {
            "question": message,
            "session_id": st.session_state.session_id,
            "model": model
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=30)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def upload_document(uploaded_file):
    """Upload a document to the API."""
    try:
        files = {"file": (uploaded_file.name, uploaded_file, uploaded_file.type)}
        response = requests.post(f"{API_BASE_URL}/upload-doc", files=files, timeout=60)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Upload error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def get_documents():
    """Get list of uploaded documents."""
    try:
        response = requests.get(f"{API_BASE_URL}/list-docs", timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def delete_document(file_id):
    """Delete a document."""
    try:
        payload = {"file_id": file_id}
        response = requests.post(f"{API_BASE_URL}/delete-doc", json=payload, timeout=10)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"Delete error: {response.status_code} - {response.text}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

def get_stats():
    """Get system statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            return True, response.json()
        else:
            return False, f"API error: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return False, f"Connection error: {str(e)}"

# Main app
st.markdown('<h1 class="main-header">🤖 RAG Chatbot with Google Gemini AI</h1>', unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.header("🔧 Configuration")
    
    # API Status
    st.subheader("📡 API Status")
    if st.button("Check API Status"):
        is_healthy, health_data = check_api_health()
        if is_healthy:
            st.markdown('<p class="status-success">✅ API is running</p>', unsafe_allow_html=True)
            st.json(health_data)
            st.session_state.api_status = "healthy"
        else:
            st.markdown('<p class="status-error">❌ API is not available</p>', unsafe_allow_html=True)
            st.error(health_data)
            st.session_state.api_status = "error"
    
    # Model selection
    st.subheader("🧠 Model Settings")
    model_choice = st.selectbox(
        "Select Model",
        ["gemini-pro", "gemini-pro-vision"],
        index=0,
        help="Choose the Gemini model for conversation"
    )
    
    # Session management
    st.subheader("💬 Session")
    st.text(f"Session ID: {st.session_state.session_id[:8]}...")
    if st.button("New Session"):
        st.session_state.session_id = str(uuid.uuid4())
        st.session_state.messages = []
        st.rerun()
    
    # Clear chat
    if st.button("Clear Chat"):
        st.session_state.messages = []
        st.rerun()
    
    # Document management
    st.subheader("📄 Document Management")
    
    # Upload documents
    uploaded_file = st.file_uploader(
        "Upload Document",
        type=['pdf', 'docx', 'html', 'txt'],
        help="Upload a document to add to the knowledge base"
    )
    
    if uploaded_file is not None:
        if st.button("Upload & Index"):
            with st.spinner("Uploading and indexing document..."):
                success, result = upload_document(uploaded_file)
                if success:
                    st.success(result["message"])
                else:
                    st.error(result)
    
    # List documents
    if st.button("Refresh Documents"):
        success, docs = get_documents()
        if success:
            st.session_state.documents = docs
        else:
            st.error(f"Failed to get documents: {docs}")
    
    # Show documents
    if "documents" in st.session_state:
        st.subheader("📚 Indexed Documents")
        for doc in st.session_state.documents:
            with st.expander(f"📄 {doc['filename']}"):
                st.write(f"**ID:** {doc['id']}")
                st.write(f"**Uploaded:** {doc['upload_timestamp']}")
                if st.button(f"Delete", key=f"delete_{doc['id']}"):
                    success, result = delete_document(doc['id'])
                    if success:
                        st.success("Document deleted successfully!")
                        # Refresh the documents list
                        success, docs = get_documents()
                        if success:
                            st.session_state.documents = docs
                        st.rerun()
                    else:
                        st.error(f"Failed to delete: {result}")
    
    # System stats
    if st.button("Show Stats"):
        success, stats = get_stats()
        if success:
            st.json(stats)
        else:
            st.error(f"Failed to get stats: {stats}")

# Main chat interface
st.subheader("💬 Chat Interface")

# Display chat messages
for message in st.session_state.messages:
    if message["role"] == "user":
        st.markdown(f'''
        <div class="chat-message user-message">
            <strong>You:</strong> {message["content"]}
        </div>
        ''', unsafe_allow_html=True)
    else:
        st.markdown(f'''
        <div class="chat-message bot-message">
            <strong>Assistant:</strong> {message["content"]}
        </div>
        ''', unsafe_allow_html=True)

# Chat input
with st.form("chat_form", clear_on_submit=True):
    user_input = st.text_area(
        "Type your message:",
        placeholder="Ask me anything about your uploaded documents...",
        height=100,
        key="user_input"
    )
    col1, col2 = st.columns([1, 4])
    with col1:
        submit = st.form_submit_button("Send 📤")
    with col2:
        if st.session_state.api_status == "error":
            st.warning("⚠️ API is not available. Please check the API status in the sidebar.")

if submit and user_input.strip():
    # Add user message to chat
    st.session_state.messages.append({"role": "user", "content": user_input})
    
    # Show spinner while processing
    with st.spinner("Thinking..."):
        success, response = send_message(user_input, model_choice)
        
        if success:
            # Add assistant response to chat
            st.session_state.messages.append({
                "role": "assistant", 
                "content": response["answer"]
            })
        else:
            # Add error message
            st.session_state.messages.append({
                "role": "assistant", 
                "content": f"Sorry, I encountered an error: {response}"
            })
    
    # Rerun to update the chat display
    st.rerun()

# Instructions
with st.expander("ℹ️ How to use this chatbot"):
    st.markdown("""
    ### Getting Started:
    1. **Check API Status**: Use the sidebar to verify the API is running
    2. **Upload Documents**: Add PDF, DOCX, HTML, or TXT files to the knowledge base
    3. **Start Chatting**: Ask questions about your uploaded documents
    
    ### Features:
    - 🤖 **Google Gemini AI**: Powered by Google's advanced language model
    - 📚 **Document RAG**: Ask questions about your uploaded documents
    - 💾 **Session Memory**: Maintains conversation context
    - 🔄 **Real-time Updates**: Upload and delete documents dynamically
    
    ### Tips:
    - Upload relevant documents before asking specific questions
    - Use clear, specific questions for better responses
    - Check the document list to see what's in your knowledge base
    - Start a new session to clear conversation history
    """)

# Footer
st.markdown("---")
st.markdown("🚀 **RAG Chatbot** - Powered by Google Gemini AI & Custom RAG System")

# Auto-refresh documents on load
if "documents" not in st.session_state:
    success, docs = get_documents()
    if success:
        st.session_state.documents = docs