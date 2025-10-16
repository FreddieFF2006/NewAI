"""
Streamlit Web Interface for RAG System
A user-friendly web app for document upload and AI-powered querying
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from rag_system import DocumentRAGSystem
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="AI Document Assistant",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        padding: 1rem;
        margin: 1rem 0;
    }
    .upload-section {
        background-color: #f0f2f6;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    .query-section {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 10px;
        margin: 1rem 0;
        border: 1px solid #e0e0e0;
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


def initialize_rag_system():
    """Initialize the RAG system with API key."""
    api_key = os.getenv("CLAUDE_API_KEY")
    
    if not api_key:
        # Try to get from Streamlit secrets (for deployment)
        try:
            api_key = st.secrets["CLAUDE_API_KEY"]
        except:
            return None
    
    if api_key:
        try:
            rag = DocumentRAGSystem(claude_api_key=api_key)
            return rag
        except Exception as e:
            st.error(f"Error initializing RAG system: {e}")
            return None
    return None


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path."""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the file
        with open(file_path, 'wb') as f:
            f.write(uploaded_file.getbuffer())
        
        return file_path
    except Exception as e:
        st.error(f"Error saving file: {e}")
        return None


# Header
st.title("🤖 AI Document Assistant")
st.markdown("### Upload documents and ask questions - powered by Claude AI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # API Key input (for local development)
    api_key_input = st.text_input(
        "Claude API Key (optional if using .env)", 
        type="password",
        help="Enter your Claude API key or set it in .env file"
    )
    
    if api_key_input:
        os.environ["CLAUDE_API_KEY"] = api_key_input
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing RAG system..."):
            st.session_state.rag_system = initialize_rag_system()
            if st.session_state.rag_system:
                st.success("✅ System initialized successfully!")
            else:
                st.error("❌ Failed to initialize. Please check your API key.")
    
    st.markdown("---")
    
    # System status
    st.header("📊 System Status")
    if st.session_state.rag_system:
        st.success("🟢 System Active")
        try:
            stats = st.session_state.rag_system.get_collection_stats()
            st.metric("Total Chunks", stats['total_vectors'])
            st.metric("Documents Uploaded", len(st.session_state.uploaded_files_list))
        except:
            pass
    else:
        st.warning("🟡 System Not Initialized")
    
    st.markdown("---")
    
    # Settings
    st.header("🎛️ Settings")
    top_k = st.slider("Number of relevant chunks", 1, 10, 5)
    max_tokens = st.slider("Max response tokens", 500, 4000, 2000)
    
    st.markdown("---")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        st.session_state.rag_system = None
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.success("All data cleared!")
        st.rerun()

# Main content area
col1, col2 = st.columns([1, 1])

with col1:
    st.header("📤 Upload Documents")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the system first using the sidebar.")
    else:
        uploaded_files = st.file_uploader(
            "Choose text files",
            type=['txt', 'md'],
            accept_multiple_files=True,
            help="Upload one or more text files (.txt or .md)"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                total_chunks = 0
                processed_files = []
                
                for idx, uploaded_file in enumerate(uploaded_files):
                    status_text.text(f"Processing {uploaded_file.name}...")
                    
                    # Save file temporarily
                    file_path = save_uploaded_file(uploaded_file)
                    
                    if file_path:
                        try:
                            # Upload to RAG system
                            chunks = st.session_state.rag_system.upload_file(file_path)
                            total_chunks += chunks
                            processed_files.append({
                                'name': uploaded_file.name,
                                'chunks': chunks
                            })
                            
                            # Update uploaded files list
                            if uploaded_file.name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(uploaded_file.name)
                            
                        except Exception as e:
                            st.error(f"Error processing {uploaded_file.name}: {e}")
                        
                        # Clean up temp file
                        try:
                            os.remove(file_path)
                        except:
                            pass
                    
                    progress_bar.progress((idx + 1) / len(uploaded_files))
                
                status_text.empty()
                progress_bar.empty()
                
                # Show success message
                st.success(f"✅ Processed {len(processed_files)} files with {total_chunks} total chunks!")
                
                # Show details
                with st.expander("📋 Processing Details"):
                    for file_info in processed_files:
                        st.write(f"• **{file_info['name']}**: {file_info['chunks']} chunks")
        
        # Show uploaded files
        if st.session_state.uploaded_files_list:
            st.markdown("### 📚 Uploaded Documents")
            for file_name in st.session_state.uploaded_files_list:
                st.write(f"✓ {file_name}")

with col2:
    st.header("💬 Ask Questions")
    
    if not st.session_state.rag_system:
        st.warning("⚠️ Please initialize the system and upload documents first.")
    elif not st.session_state.uploaded_files_list:
        st.info("ℹ️ Upload some documents to start asking questions!")
    else:
        # Query input
        query = st.text_area(
            "Enter your question:",
            height=100,
            placeholder="What would you like to know about your documents?"
        )
        
        col_btn1, col_btn2 = st.columns([1, 4])
        with col_btn1:
            ask_button = st.button("🔍 Ask", type="primary")
        with col_btn2:
            clear_chat = st.button("🗑️ Clear Chat")
        
        if clear_chat:
            st.session_state.chat_history = []
            st.success("Chat history cleared!")
            st.rerun()
        
        if ask_button and query:
            with st.spinner("🤔 Thinking..."):
                try:
                    result = st.session_state.rag_system.query(
                        query, 
                        top_k=top_k,
                        max_tokens=max_tokens
                    )
                    
                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': query,
                        'answer': result['answer'],
                        'sources': result['sources']
                    })
                    
                except Exception as e:
                    st.error(f"Error processing query: {e}")

# Display chat history
if st.session_state.chat_history:
    st.markdown("---")
    st.header("💬 Conversation History")
    
    for idx, chat in enumerate(reversed(st.session_state.chat_history)):
        with st.container():
            st.markdown(f"### ❓ Question {len(st.session_state.chat_history) - idx}")
            st.info(chat['question'])
            
            st.markdown(f"### 🤖 Answer")
            st.success(chat['answer'])
            
            # Show sources in expander
            with st.expander(f"📚 View Sources ({len(chat['sources'])} sources)"):
                for i, source in enumerate(chat['sources'], 1):
                    st.markdown(f"**Source {i}** (Relevance: {source['score']:.3f})")
                    st.markdown(f"*File:* `{source['file']}`")
                    st.markdown(f"*Chunk:* {source['chunk']}")
                    st.text(source['text_preview'])
                    st.markdown("---")
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by Claude AI, Qdrant, and E5-large-v2 embeddings</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)
