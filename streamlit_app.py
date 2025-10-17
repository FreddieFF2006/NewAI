"""
Streamlit Web Interface for RAG System - Professional Enterprise Style
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
    st.error(f"Failed to import RAG system: {e}")
    st.code(traceback.format_exc())
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Document Intelligence Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Professional Enterprise Styling
st.markdown("""
    <style>
    /* Import professional font */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    /* Global styles */
    * {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }
    
    /* Main container */
    .main {
        padding: 0;
        background: linear-gradient(135deg, #f5f7fa 0%, #e8ecef 100%);
    }
    
    /* Remove default padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 100%;
    }
    
    /* Sidebar styling - Professional dark theme */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1d24 0%, #252932 100%);
        border-right: none;
        box-shadow: 4px 0 24px rgba(0,0,0,0.12);
    }
    
    [data-testid="stSidebar"] * {
        color: #e4e6eb !important;
    }
    
    [data-testid="stSidebar"] h1,
    [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3 {
        color: #ffffff !important;
        font-weight: 600;
        letter-spacing: -0.02em;
    }
    
    /* Sidebar buttons */
    [data-testid="stSidebar"] .stButton button {
        background: rgba(255,255,255,0.08);
        border: 1px solid rgba(255,255,255,0.12);
        color: #ffffff !important;
        border-radius: 8px;
        font-weight: 500;
        padding: 10px 16px;
        transition: all 0.2s ease;
        backdrop-filter: blur(10px);
    }
    
    [data-testid="stSidebar"] .stButton button:hover {
        background: rgba(255,255,255,0.15);
        border-color: rgba(255,255,255,0.2);
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(0,0,0,0.15);
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: #ffffff !important;
    }
    
    [data-testid="stSidebar"] .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #63408a 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.4);
    }
    
    /* Chat history items */
    [data-testid="stSidebar"] .stButton button[key^="chat_"] {
        text-align: left;
        font-size: 13px;
        padding: 12px 14px;
        margin: 4px 0;
        background: rgba(255,255,255,0.05);
        white-space: normal;
        height: auto;
        line-height: 1.4;
    }
    
    [data-testid="stSidebar"] .stButton button[key^="chat_"]:hover {
        background: rgba(255,255,255,0.12);
    }
    
    /* Message containers - Professional card design */
    .user-message {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        position: relative;
    }
    
    .user-message::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px 0 0 12px;
    }
    
    .assistant-message {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 12px;
        padding: 24px 28px;
        margin: 20px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.04);
        position: relative;
    }
    
    .assistant-message::before {
        content: '';
        position: absolute;
        left: 0;
        top: 0;
        bottom: 0;
        width: 4px;
        background: linear-gradient(180deg, #11998e 0%, #38ef7d 100%);
        border-radius: 12px 0 0 12px;
    }
    
    .message-header {
        display: flex;
        align-items: center;
        margin-bottom: 12px;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        color: #586069;
    }
    
    .message-content {
        color: #24292e;
        font-size: 15px;
        line-height: 1.7;
        font-weight: 400;
    }
    
    /* Input area - Professional design */
    .stTextArea textarea {
        border-radius: 12px;
        border: 2px solid #e1e4e8;
        padding: 16px 20px;
        font-size: 15px;
        line-height: 1.6;
        background: #ffffff;
        transition: all 0.2s ease;
        font-family: 'Inter', sans-serif;
    }
    
    .stTextArea textarea:focus {
        border-color: #667eea;
        box-shadow: 0 0 0 4px rgba(102, 126, 234, 0.1);
        outline: none;
    }
    
    /* Main content buttons */
    .main .stButton button {
        border-radius: 10px;
        font-weight: 600;
        padding: 12px 24px;
        transition: all 0.2s ease;
        font-size: 14px;
        letter-spacing: 0.3px;
        border: 2px solid transparent;
    }
    
    .main .stButton button[kind="primary"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border: none;
        color: #ffffff;
        box-shadow: 0 4px 14px rgba(102, 126, 234, 0.4);
    }
    
    .main .stButton button[kind="primary"]:hover {
        background: linear-gradient(135deg, #5568d3 0%, #63408a 100%);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.5);
    }
    
    .main .stButton button[kind="secondary"] {
        background: #ffffff;
        border: 2px solid #e1e4e8;
        color: #24292e;
    }
    
    .main .stButton button[kind="secondary"]:hover {
        border-color: #667eea;
        background: #f6f8ff;
        transform: translateY(-1px);
    }
    
    /* File uploader - Professional */
    [data-testid="stFileUploader"] {
        background: #ffffff;
        border: 2px dashed #d1d5db;
        border-radius: 16px;
        padding: 32px;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: #f6f8ff;
    }
    
    /* Metrics - Modern cards */
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.1);
        padding: 12px;
        border-radius: 10px;
        border: 1px solid rgba(255,255,255,0.15);
    }
    
    [data-testid="stMetricValue"] {
        color: #ffffff !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }
    
    [data-testid="stMetricLabel"] {
        color: rgba(255,255,255,0.7) !important;
        font-size: 12px !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        font-weight: 500 !important;
    }
    
    /* Headers - Professional typography */
    h1 {
        color: #1a1d24;
        font-weight: 700;
        font-size: 32px;
        letter-spacing: -0.03em;
        margin-bottom: 8px;
    }
    
    h2 {
        color: #24292e;
        font-weight: 600;
        font-size: 24px;
        letter-spacing: -0.02em;
        margin-top: 32px;
        margin-bottom: 16px;
    }
    
    h3 {
        color: #24292e;
        font-weight: 600;
        font-size: 18px;
        letter-spacing: -0.01em;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px;
        border: none;
        padding: 16px 20px;
        font-size: 14px;
        line-height: 1.6;
    }
    
    /* Success */
    .stSuccess {
        background: linear-gradient(135deg, #d4fc79 0%, #96e6a1 100%);
        color: #1e4620;
    }
    
    /* Warning */
    .stWarning {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        color: #7c2d12;
    }
    
    /* Error */
    .stError {
        background: linear-gradient(135deg, #ff9a9e 0%, #fecfef 100%);
        color: #7f1d1d;
    }
    
    /* Info */
    .stInfo {
        background: linear-gradient(135deg, #a1c4fd 0%, #c2e9fb 100%);
        color: #1e3a8a;
    }
    
    /* Expander - Clean design */
    .streamlit-expanderHeader {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 10px;
        padding: 14px 18px;
        font-weight: 500;
        color: #24292e;
        transition: all 0.2s ease;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea;
        background: #f6f8ff;
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 10px;
    }
    
    /* Slider */
    [data-testid="stSidebar"] .stSlider {
        padding: 10px 0;
    }
    
    [data-testid="stSidebar"] .stSlider > div > div > div {
        background: rgba(255,255,255,0.2);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Scrollbar - Minimal design */
    ::-webkit-scrollbar {
        width: 10px;
        height: 10px;
    }
    
    ::-webkit-scrollbar-track {
        background: transparent;
    }
    
    ::-webkit-scrollbar-thumb {
        background: rgba(0,0,0,0.2);
        border-radius: 10px;
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: rgba(0,0,0,0.3);
        border: 2px solid transparent;
        background-clip: padding-box;
    }
    
    /* Badge styling */
    .badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        letter-spacing: 0.3px;
        margin-left: 8px;
    }
    
    .badge-primary {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }
    
    .badge-success {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
    }
    
    /* Card styling */
    .card {
        background: #ffffff;
        border: 1px solid #e1e4e8;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        box-shadow: 0 2px 12px rgba(0,0,0,0.04);
        transition: all 0.3s ease;
    }
    
    .card:hover {
        box-shadow: 0 8px 24px rgba(0,0,0,0.08);
        transform: translateY(-2px);
    }
    
    /* Divider */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent 0%, #e1e4e8 50%, transparent 100%);
        margin: 32px 0;
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
    """Initialize the RAG system."""
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


# Auto-initialize
if st.session_state.rag_system is None:
    with st.spinner("Initializing system..."):
        rag, error = initialize_rag_system()
        if rag:
            st.session_state.rag_system = rag
        else:
            st.error(f"Initialization failed: {error}")
            st.stop()

# ===========================
# SIDEBAR
# ===========================
with st.sidebar:
    st.markdown("### CONVERSATIONS")
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("New Chat", use_container_width=True, type="primary", key="new_chat_btn"):
            st.session_state.current_chat_index = None
            st.session_state.show_upload = False
            st.rerun()
    with col2:
        if st.button("Upload", use_container_width=True, key="upload_btn"):
            st.session_state.show_upload = not st.session_state.show_upload
            st.rerun()
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Chat history
    if st.session_state.chat_history:
        st.markdown("**Recent**")
        for idx, chat in enumerate(reversed(st.session_state.chat_history)):
            actual_idx = len(st.session_state.chat_history) - 1 - idx
            
            question_preview = chat['question'][:60] + "..." if len(chat['question']) > 60 else chat['question']
            
            if st.button(
                question_preview,
                key=f"chat_{actual_idx}",
                use_container_width=True
            ):
                st.session_state.current_chat_index = actual_idx
                st.session_state.show_upload = False
                st.rerun()
    else:
        st.markdown("*No conversations yet*")
    
    st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Stats
    st.markdown("### SYSTEM")
    if st.session_state.rag_system:
        try:
            stats = st.session_state.rag_system.get_stats()
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Chunks", f"{stats.get('total_chunks', 0):,}")
            with col2:
                st.metric("Docs", len(st.session_state.uploaded_files_list))
        except:
            st.metric("Chunks", "0")
            st.metric("Docs", "0")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Settings
    with st.expander("Settings"):
        n_results = st.slider(
            "Retrieval depth",
            min_value=20,
            max_value=200,
            value=100,
            step=10
        )
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Action buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Data", type="secondary", use_container_width=True):
            if st.session_state.rag_system:
                try:
                    st.session_state.rag_system.client.delete_collection(
                        st.session_state.rag_system.collection_name
                    )
                    # Recreate immediately
                    st.session_state.rag_system.collection = st.session_state.rag_system.client.create_collection(
                        name=st.session_state.rag_system.collection_name,
                        metadata={"hnsw:space": "cosine"}
                    )
                    st.success("Data cleared")
                except Exception as e:
                    st.error(f"Error: {e}")
            
            st.session_state.uploaded_files_list = []
            st.session_state.chat_history = []
            st.session_state.current_chat_index = None
            st.rerun()
    
    with col2:
        if st.button("Restart", use_container_width=True):
            # Force complete restart
            initialize_rag_system.clear()
            st.session_state.rag_system = None
            st.session_state.uploaded_files_list = []
            st.session_state.chat_history = []
            st.session_state.current_chat_index = None
            st.session_state.show_upload = False
            st.rerun()

# ===========================
# MAIN CONTENT
# ===========================

# Upload interface
if st.session_state.show_upload:
    st.markdown("# Document Upload")
    st.markdown("Upload financial reports, earnings documents, and other PDFs for analysis")
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Drag and drop files here",
        type=['pdf'],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([1, 1, 4])
    with col1:
        if st.button("Close", use_container_width=True):
            st.session_state.show_upload = False
            st.rerun()
    with col2:
        process_btn = uploaded_files and st.button("Process", type="primary", use_container_width=True)
    
    if process_btn:
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        file_paths = []
        for uploaded_file in uploaded_files:
            file_path, error = save_uploaded_file(uploaded_file)
            if file_path:
                file_paths.append((uploaded_file.name, file_path))
        
        if file_paths:
            try:
                status_text.text(f"Processing {len(file_paths)} documents...")
                paths = [fp[1] for fp in file_paths]
                
                st.session_state.rag_system.ingest_documents(paths, clear_existing=False)
                
                for file_name, _ in file_paths:
                    if file_name not in st.session_state.uploaded_files_list:
                        st.session_state.uploaded_files_list.append(file_name)
                
                stats = st.session_state.rag_system.get_stats()
                total = stats.get('total_chunks', 0)
                
                status_text.empty()
                progress_bar.empty()
                
                st.success(f"Successfully processed {len(file_paths)} documents ({total:,} total chunks)")
                
            except Exception as e:
                st.error(f"Processing error: {str(e)}")
            finally:
                for _, file_path in file_paths:
                    try:
                        os.remove(file_path)
                        os.rmdir(os.path.dirname(file_path))
                    except:
                        pass
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    # Document list
    if st.session_state.uploaded_files_list:
        st.markdown("### Uploaded Documents")
        st.markdown('<div class="card">', unsafe_allow_html=True)
        for idx, file_name in enumerate(st.session_state.uploaded_files_list, 1):
            st.markdown(f"**{idx}.** {file_name}")
        st.markdown('</div>', unsafe_allow_html=True)

# Chat interface
else:
    # Display conversation
    if st.session_state.current_chat_index is not None:
        chat = st.session_state.chat_history[st.session_state.current_chat_index]
        
        # User message
        st.markdown(f"""
        <div class="user-message">
            <div class="message-header">YOU</div>
            <div class="message-content">{chat['question']}</div>
        </div>
        """, unsafe_allow_html=True)
        
        # Assistant message
        st.markdown(f"""
        <div class="assistant-message">
            <div class="message-header">ASSISTANT</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown(f'<div class="message-content">{chat["answer"]}</div>', unsafe_allow_html=True)
        
        # Sources
        if chat.get('sources'):
            st.markdown("<br>", unsafe_allow_html=True)
            with st.expander(f"📚 View {len(chat['sources'])} source references"):
                sources_by_file = {}
                for source in chat['sources']:
                    file_name = source['metadata'].get('source', 'Unknown')
                    if file_name not in sources_by_file:
                        sources_by_file[file_name] = []
                    sources_by_file[file_name].append(source)
                
                for file_name, sources in sources_by_file.items():
                    st.markdown(f"**{file_name}** — {len(sources)} references")
                    
                    for i, source in enumerate(sources[:5], 1):
                        page = source['metadata'].get('page', '?')
                        text = source.get('text', '')
                        st.markdown(f"*Page {page}*")
                        st.text(text[:250] + "..." if len(text) > 250 else text)
                        if i < len(sources[:5]):
                            st.markdown("---")
    
    # New chat
    else:
        st.markdown("# Document Intelligence Platform")
        st.markdown("Ask questions about your uploaded documents using advanced AI")
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        if st.session_state.uploaded_files_list:
            st.info(f"📚 {len(st.session_state.uploaded_files_list)} documents loaded and ready for analysis")
        else:
            st.warning("⚠️ No documents uploaded. Click 'Upload' to begin.")
        
        st.markdown("<br><br>", unsafe_allow_html=True)
    
    # Input area
    st.markdown("---")
    
    query = st.text_area(
        "Ask a question about your documents",
        height=120,
        placeholder="Example: What is the total revenue across all companies?",
        label_visibility="collapsed"
    )
    
    col1, col2, col3 = st.columns([2, 2, 8])
    with col1:
        send_button = st.button("Send", type="primary", use_container_width=True)
    with col2:
        if st.session_state.current_chat_index is not None:
            if st.button("New Chat", use_container_width=True):
                st.session_state.current_chat_index = None
                st.rerun()
    
    # Process query
    if send_button and query and query.strip():
        if not st.session_state.uploaded_files_list:
            st.error("Please upload documents before asking questions")
        else:
            with st.spinner(f"Analyzing {n_results} document chunks..."):
                try:
                    answer = st.session_state.rag_system.generate_answer(query, n_results=n_results)
                    chunks = st.session_state.rag_system.retrieve(query, n_results=n_results)
                    
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': chunks,
                        'n_results': n_results,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M")
                    })
                    
                    st.session_state.current_chat_index = len(st.session_state.chat_history) - 1
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("Technical details"):
                        st.code(traceback.format_exc())

# Footer
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("""
    <div style='text-align: center; color: #6b7280; font-size: 13px; padding: 20px; letter-spacing: 0.3px;'>
        Powered by Claude Sonnet 4 • ChromaDB • Hybrid Semantic Search
    </div>
""", unsafe_allow_html=True)