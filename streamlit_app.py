"""
Streamlit Web Interface for RAG System - Claude AI Style
"""

import streamlit as st
import os
import tempfile
import traceback
from dotenv import load_dotenv
from datetime import datetime

# Load environment variables
load_dotenv()

# Try to import the RAG system
try:
    from rag_system import SemanticFinancialRAG
except ImportError as e:
    st.error(f"❌ Failed to import RAG system: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Claude AI inspired styling
st.markdown("""
    <style>
    /* Main container */
    .main {
        padding: 0;
        background-color: #f7f7f8;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background-color: #ffffff;
        border-right: 1px solid #e5e5e5;
    }
    
    /* Chat history items */
    .chat-history-item {
        padding: 12px 16px;
        margin: 4px 0;
        border-radius: 8px;
        cursor: pointer;
        background-color: #f7f7f8;
        border: 1px solid transparent;
        transition: all 0.2s;
    }
    
    .chat-history-item:hover {
        background-color: #ebebeb;
        border-color: #d0d0d0;
    }
    
    .chat-history-item.active {
        background-color: #e8e8e8;
        border-color: #c0c0c0;
    }
    
    /* Message containers */
    .user-message {
        background-color: #ffffff;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    .assistant-message {
        background-color: #f7f7f8;
        border: 1px solid #e5e5e5;
        border-radius: 12px;
        padding: 20px;
        margin: 16px 0;
        box-shadow: 0 1px 2px rgba(0,0,0,0.05);
    }
    
    /* Input area */
    .stTextArea textarea {
        border-radius: 12px;
        border: 1px solid #d0d0d0;
        padding: 12px;
        font-size: 15px;
    }
    
    .stTextArea textarea:focus {
        border-color: #ab7c5f;
        box-shadow: 0 0 0 2px rgba(171, 124, 95, 0.1);
    }
    
    /* Buttons */
    .stButton button {
        border-radius: 8px;
        font-weight: 500;
        padding: 8px 16px;
        transition: all 0.2s;
    }
    
    .stButton button[kind="primary"] {
        background-color: #ab7c5f;
        border: none;
    }
    
    .stButton button[kind="primary"]:hover {
        background-color: #9a6d52;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background-color: #ffffff;
        border: 2px dashed #d0d0d0;
        border-radius: 12px;
        padding: 20px;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        color: #ab7c5f;
        font-size: 24px;
        font-weight: 600;
    }
    
    /* Headers */
    h1 {
        color: #2d2d2d;
        font-weight: 600;
    }
    
    h2 {
        color: #2d2d2d;
        font-weight: 600;
        font-size: 20px;
    }
    
    h3 {
        color: #2d2d2d;
        font-weight: 600;
        font-size: 16px;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Scrollbar */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: #f7f7f8;
    }
    
    ::-webkit-scrollbar-thumb {
        background: #d0d0d0;
        border-radius: 4px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: #b0b0b0;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'current_chat_index' not in st.session_state:
    st.session_state.current_chat_index = None
if 'show_upload' not in st.session_state:
    st.session_state.show_upload = False


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system - API key should be in secrets."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except:
                return None, "ANTHROPIC_API_KEY not found in secrets."
        
        if not api_key:
            return None, "API key is empty"
        
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        rag = SemanticFinancialRAG(
            model_name="intfloat/e5-large-v2",
            collection_name="financial_documents",
            persist_directory="./chroma_db"
        )
        return rag, None
        
    except Exception as e:
        error_msg = f"Error: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temp location."""
    try:
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path, None
    except Exception as e:
        return None, f"Error saving file: {str(e)}"


# Auto-initialize on first load
if st.session_state.rag_system is None:
    with st.spinner("Initializing system..."):
        rag, error = initialize_rag_system()
        if rag:
            st.session_state.rag_system = rag
        else:
            st.error(f"❌ Initialization failed: {error}")
            st.stop()

# ===========================
# SIDEBAR - Chat History
# ===========================
with st.sidebar:
    # Header
    st.markdown("### 💬 Conversations")
    
    # New chat button
    col1, col2 = st.columns([3, 1])
    with col1:
        if st.button("➕ New Chat", use_container_width=True, type="primary"):
            st.session_state.current_chat_index = None
            st.rerun()
    with col2:
        if st.button("📤", help="Upload Documents"):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()
    
    st.markdown("---")
    
    # Chat history list
    if st.session_state.chat_history:
        st.markdown("**Recent Chats**")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            actual_idx = len(st.session_state.chat_history) - 1 - idx
            
            # Truncate question for display
            question_preview = chat['question'][:50] + "..." if len(chat['question']) > 50 else chat['question']
            timestamp = chat.get('timestamp', '')
            
            # Create button for each chat
            if st.button(
                f"💭 {question_preview}",
                key=f"chat_{actual_idx}",
                help=f"Asked: {timestamp}",
                use_container_width=True
            ):
                st.session_state.current_chat_index = actual_idx
                st.rerun()
    else:
        st.info("No conversations yet.\nStart by asking a question!")
    
    st.markdown("---")
    
    # System stats
    st.markdown("**📊 System Stats**")
    if st.session_state.rag_system:
        try:
            stats = st.session_state.rag_system.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", stats.get('total_chunks', 0))
            with col2:
                st.metric("Docs", len(st.session_state.uploaded_files_list))
        except:
            pass
    
    st.markdown("---")
    
    # Settings
    with st.expander("⚙️ Settings"):
        n_results = st.slider(
            "Chunks to retrieve",
            min_value=20,
            max_value=200,
            value=100,
            step=10
        )
    
    # Clear all button
    if st.button("🗑️ Clear All", type="secondary", use_container_width=True):
        if st.session_state.rag_system:
            try:
                st.session_state.rag_system.client.delete_collection(
                    st.session_state.rag_system.collection.name
                )
            except:
                pass
        
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.session_state.current_chat_index = None
        st.success("Cleared!")
        st.rerun()

# ===========================
# MAIN AREA
# ===========================

# Upload Modal (shown when upload button clicked)
if st.session_state.show_upload:
    st.markdown("## 📤 Upload Documents")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files to analyze",
        type=['pdf'],
        accept_multiple_files=True,
        help="Upload financial reports, earnings documents, etc."
    )
    
    col1, col2 = st.columns([1, 5])
    with col1:
        if st.button("✅ Done"):
            st.session_state.show_upload = False
            st.rerun()
    
    if uploaded_files and st.button("📥 Process Files", type="primary"):
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        file_paths = []
        errors = []
        
        for uploaded_file in uploaded_files:
            file_path, error = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append((uploaded_file.name, file_path))
            else:
                errors.append(f"{uploaded_file.name}: {error}")
        
        if errors:
            for error in errors:
                st.error(error)
        
        if file_paths:
            try:
                status_text.text(f"Processing {len(file_paths)} files...")
                paths = [fp[1] for fp in file_paths]
                
                st.session_state.rag_system.ingest_documents(paths, clear_existing=False)
                
                for file_name, _ in file_paths:
                    if file_name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(file_name)
                
                stats = st.session_state.rag_system.get_stats()
                total = stats.get('total_chunks', 0)
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"✅ Processed {len(file_paths)} files! Total: {total} chunks")
                
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
            finally:
                for _, file_path in file_paths:
                    try:
                        os.remove(file_path)
                        os.rmdir(os.path.dirname(file_path))
                    except:
                        pass
    
    # Show uploaded files
    if st.session_state.uploaded_files_list:
        st.markdown("### 📚 Uploaded Documents")
        for idx, file_name in enumerate(st.session_state.uploaded_files_list, 1):
            st.markdown(f"**{idx}.** {file_name}")
    
    st.markdown("---")

# Main chat interface
if not st.session_state.show_upload:
    # Show current conversation or new chat
    if st.session_state.current_chat_index is not None:
        # Display selected conversation
        chat = st.session_state.chat_history[st.session_state.current_chat_index]
        
        st.markdown("## 💬 Conversation")
        
        # User message
        st.markdown(f"""
        <div class="user-message">
            <strong>You</strong>
            <p style="margin-top: 8px; color: #2d2d2d;">{chat['question']}</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="assistant-message">
            <strong>🤖 Assistant</strong>
        </div>
        """, unsafe_allow_html=True)
        st.markdown(chat['answer'])
        
        # Sources expander
        if chat.get('sources'):
            with st.expander(f"📚 View {len(chat['sources'])} Source Chunks"):
                sources_by_file = {}
                for source in chat['sources']:
                    file_name = source['metadata'].get('source', 'Unknown')
                    if file_name not in sources_by_file:
                        sources_by_file[file_name] = []
                    sources_by_file[file_name].append(source)
                
                for file_name, sources in sources_by_file.items():
                    st.markdown(f"**📄 {file_name}** ({len(sources)} chunks)")
                    
                    for i, source in enumerate(sources[:5], 1):
                        try:
                            page = source['metadata'].get('page', '?')
                            text = source.get('text', '')
                            st.markdown(f"*Page {page}*")
                            st.text(text[:200] + "..." if len(text) > 200 else text)
                            st.markdown("---")
                        except:
                            pass
    
    else:
        # New chat interface
        st.markdown("## 🤖 AI Document Assistant")
        st.markdown("Ask questions about your uploaded documents")
        
        if st.session_state.uploaded_files_list:
            st.info(f"📚 {len(st.session_state.uploaded_files_list)} documents loaded")
        else:
            st.warning("⚠️ No documents uploaded. Click the 📤 button to upload.")
    
    # Chat input (always at bottom)
    st.markdown("---")
    
    query = st.text_area(
        "Type your question here...",
        height=100,
        placeholder="e.g., What is the employee headcount for all companies?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        send_button = st.button("🚀 Send", type="primary", use_container_width=True)
    with col2:
        if st.session_state.current_chat_index is not None:
            if st.button("🔄 New Chat", use_container_width=True):
                st.session_state.current_chat_index = None
                st.rerun()
    
    # Process query
    if send_button and query and query.strip():
        if not st.session_state.uploaded_files_list:
            st.error("Please upload documents first!")
        else:
            with st.spinner(f"🤔 Analyzing {n_results} chunks..."):
                try:
                    answer = st.session_state.rag_system.generate_answer(
                        query, 
                        n_results=n_results
                    )
                    
                    chunks = st.session_state.rag_system.retrieve(query, n_results=n_results)
                    
                    # Add to history
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': chunks,
                        'n_results': n_results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                    # Set as current chat
                    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #999; font-size: 13px; padding: 20px;'>
        Powered by Claude Sonnet 4 & ChromaDB | Hybrid Search
    </div>
""", unsafe_allow_html=True)