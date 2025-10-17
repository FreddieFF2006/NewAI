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


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system - API key should be in secrets."""
    try:
        # API key should be in Streamlit secrets or environment
        api_key = os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            try:
                api_key = st.secrets["ANTHROPIC_API_KEY"]
            except:
                return None, "ANTHROPIC_API_KEY not found in secrets. Please add it in Streamlit settings."
        
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


# Auto-initialize on first load
if st.session_state.rag_system is None:
    with st.spinner("Initializing system..."):
        rag, error = initialize_rag_system()
        if rag:
            st.session_state.rag_system = rag
            st.success("✅ System initialized!", icon="🚀")
        else:
            st.error(f"❌ Initialization failed: {error}")
            st.stop()

# Header
st.title("🤖 AI Document Assistant")
st.markdown("### Upload documents and ask questions - powered by Claude AI")

# Sidebar
with st.sidebar:
    # System status
    st.header("📊 System Status")
    if st.session_state.rag_system:
        st.success("🟢 Active")
        try:
            stats = st.session_state.rag_system.get_stats()
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Documents Uploaded", len(st.session_state.uploaded_files_list))
        except:
            pass
    else:
        st.error("🔴 Not Initialized")
    
    st.markdown("---")
    
    # Settings
    st.header("🎛️ Settings")
    
    n_results = st.slider(
        "Number of chunks to retrieve",
        min_value=20,
        max_value=200,
        value=100,
        step=10,
        help="More chunks = better coverage but slower. Recommended: 100-150"
    )
    
    st.markdown("---")
    
    # Clear button
    if st.button("🗑️ Clear All Data", type="secondary"):
        if st.session_state.rag_system:
            try:
                st.session_state.rag_system.client.delete_collection(
                    st.session_state.rag_system.collection.name
                )
                st.success("Collection cleared!")
            except Exception as e:
                st.error(f"Error clearing: {e}")
        
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.rerun()
    
    # Restart system button
    if st.button("🔄 Restart System"):
        initialize_rag_system.clear()
        st.session_state.rag_system = None
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.rerun()

# Main area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ System not initialized")
    else:
        uploaded_files = st.file_uploader(
            "Choose PDF files",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload financial reports, earnings documents, etc."
        )
        
        if uploaded_files and st.button("📥 Process Documents", type="primary"):
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            file_paths = []
            errors = []
            
            # Save files
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
                    
                    st.success(f"✅ Successfully processed {len(file_paths)} files!")
                    st.info(f"📊 Total chunks in database: {total}")
                    
                except Exception as e:
                    st.error(f"❌ Processing error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())
                finally:
                    # Cleanup temp files
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
                st.write(f"{idx}. {file_name}")

with col2:
    st.header("💬 Ask Questions")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ System not initialized")
    elif not st.session_state.uploaded_files_list:
        st.info("ℹ️ Upload some documents to start asking questions")
    else:
        query = st.text_area(
            "Your question:",
            height=100,
            placeholder="e.g., What is the employee headcount for all companies?"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col_btn2:
            clear_chat = st.button("🗑️ Clear Chat")
        
        if clear_chat:
            st.session_state.chat_history = []
            st.success("Chat cleared!")
            st.rerun()
        
        if ask_button and query and query.strip():
            with st.spinner(f"🤔 Analyzing {n_results} chunks..."):
                try:
                    answer = st.session_state.rag_system.generate_answer(
                        query, 
                        n_results=n_results
                    )
                    
                    chunks = st.session_state.rag_system.retrieve(query, n_results=n_results)
                    
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': answer,
                        'sources': chunks,
                        'n_results': n_results
                    })
                    
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    with st.expander("Error Details"):
                        st.code(traceback.format_exc())

# Chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("💬 Conversation History")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"### ❓ Question {len(st.session_state.chat_history) - idx}")
            st.info(chat['question'])
            
            st.markdown(f"### 🤖 Answer")
            chunks_used = chat.get('n_results', 'Unknown')
            st.caption(f"📊 Retrieved {chunks_used} chunks from {len(set(s['metadata'].get('source', '?') for s in chat.get('sources', [])))} documents")
            st.success(chat['answer'])
            
            # Sources
            if chat.get('sources'):
                with st.expander(f"📚 View {len(chat['sources'])} Source Chunks"):
                    # Group by source
                    sources_by_file = {}
                    for source in chat['sources']:
                        file_name = source['metadata'].get('source', 'Unknown')
                        if file_name not in sources_by_file:
                            sources_by_file[file_name] = []
                        sources_by_file[file_name].append(source)
                    
                    for file_name, sources in sources_by_file.items():
                        st.markdown(f"**📄 {file_name}** ({len(sources)} chunks)")
                        
                        for i, source in enumerate(sources[:10], 1):  # Show first 10 per file
                            try:
                                page = source['metadata'].get('page', '?')
                                chunk_type = source['metadata'].get('type', '?')
                                distance = source.get('distance', 0)
                                text = source.get('text', '')
                                
                                st.markdown(f"*Chunk {i} - Page {page} ({chunk_type}) - Distance: {distance:.3f}*")
                                st.text(text[:250] + "..." if len(text) > 250 else text)
                                st.markdown("---")
                            except:
                                pass
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>🤖 Powered by Claude Sonnet 4 & ChromaDB | 🔍 Hybrid Search (Semantic + Keyword)</p>
        <p>📊 Retrieves up to 200 chunks for comprehensive answers</p>
    </div>
""", unsafe_allow_html=True)