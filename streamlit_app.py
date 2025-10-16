"""
Streamlit Web Interface for RAG System
"""

import streamlit as st
import os
import tempfile
import traceback
from dotenv import load_dotenv

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

# Custom CSS
st.markdown("""
    <style>
    .main { padding: 0rem 1rem; }
    .stAlert { padding: 1rem; margin: 1rem 0; }
    </style>
""", unsafe_allow_html=True)

# Initialize session state
if 'rag_system' not in st.session_state:
    st.session_state.rag_system = None
if 'uploaded_files_list' not in st.session_state:
    st.session_state.uploaded_files_list = []
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with API key."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        
        if not api_key:
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("CLAUDE_API_KEY")
            except:
                return None, "No API key found. Set ANTHROPIC_API_KEY in environment or secrets."
        
        if not api_key:
            return None, "API key is empty"
        
        # Set the API key
        os.environ["ANTHROPIC_API_KEY"] = api_key
        
        # Initialize RAG system
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


# Header
st.title("🤖 AI Document Assistant")
st.markdown("### Upload documents and ask questions - powered by Claude AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input
    api_key_input = st.text_input(
        "Anthropic API Key (optional)", 
        type="password",
        help="Enter API key or set ANTHROPIC_API_KEY in .env"
    )
    
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing..."):
            initialize_rag_system.clear()
            rag, error = initialize_rag_system()
            
            if rag:
                st.session_state.rag_system = rag
                st.session_state.initialization_error = None
                st.success("✅ System initialized!")
            else:
                st.session_state.rag_system = None
                st.session_state.initialization_error = error
                st.error(f"❌ Failed to initialize")
    
    # Show error details
    if st.session_state.initialization_error:
        with st.expander("❌ Error Details"):
            st.code(st.session_state.initialization_error)
    
    st.markdown("---")
    
    # System status
    st.header("📊 Status")
    if st.session_state.rag_system:
        st.success("🟢 Active")
        try:
            stats = st.session_state.rag_system.get_stats()
            st.metric("Chunks", stats.get('total_chunks', 0))
            st.metric("Documents", len(st.session_state.uploaded_files_list))
        except:
            pass
    else:
        st.warning("🟡 Not Initialized")
    
    st.markdown("---")
    
    # Settings
    st.header("🎛️ Settings")
    n_results = st.slider("Chunks to retrieve", 5, 50, 30)
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear All"):
        if st.session_state.rag_system:
            try:
                st.session_state.rag_system.client.delete_collection(
                    st.session_state.rag_system.collection.name
                )
            except:
                pass
        
        initialize_rag_system.clear()
        st.session_state.rag_system = None
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.session_state.initialization_error = None
        st.success("Cleared!")
        st.rerun()

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Initialize system first")
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if uploaded_files and st.button("📥 Process", type="primary"):
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
                    with st.expander("Details"):
                        st.code(traceback.format_exc())
                finally:
                    for _, file_path in file_paths:
                        try:
                            os.remove(file_path)
                            os.rmdir(os.path.dirname(file_path))
                        except:
                            pass
        
        if st.session_state.uploaded_files_list:
            st.markdown("### 📚 Uploaded")
            for file_name in st.session_state.uploaded_files_list:
                st.write(f"✓ {file_name}")

with col2:
    st.header("💬 Ask")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Initialize system first")
    elif not st.session_state.uploaded_files_list:
        st.info("ℹ️ Upload documents first")
    else:
        query = st.text_area(
            "Your question:",
            height=100,
            placeholder="What would you like to know?"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col_btn2:
            clear_chat = st.button("🗑️ Clear")
        
        if clear_chat:
            st.session_state.chat_history = []
            st.success("Cleared!")
            st.rerun()
        
        if ask_button and query and query.strip():
            with st.spinner("🤔 Thinking..."):
                try:
                    answer = st.session_state.rag_system.generate_answer(
                        query, 
                        n_results=n_results
                    )
                    
                    chunks = st.session_state.rag_system.retrieve(query, n_results=n_results)
                    
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': chunks
                    })
                    
                except Exception as e:
                    st.error(f"Error: {str(e)}")
                    with st.expander("Details"):
                        st.code(traceback.format_exc())

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("💬 History")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"### ❓ Question {len(st.session_state.chat_history) - idx}")
            st.info(chat['question'])
            
            st.markdown(f"### 🤖 Answer")
            st.success(chat['answer'])
            
            if chat.get('sources'):
                with st.expander(f"📚 Sources ({len(chat['sources'])})"):
                    for i, source in enumerate(chat['sources'], 1):
                        try:
                            st.markdown(f"**Source {i}**")
                            st.markdown(f"*File:* `{source['metadata'].get('source', '?')}`")
                            st.markdown(f"*Page:* {source['metadata'].get('page', '?')}")
                            st.markdown(f"*Distance:* {source.get('distance', 0):.4f}")
                            text = source.get('text', '')
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.markdown("---")
                        except:
                            pass
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by Claude AI & ChromaDB</p>
    </div>
""", unsafe_allow_html=True)