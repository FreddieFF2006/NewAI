"""
Streamlit Web Interface for RAG System
A user-friendly web app for document upload and AI-powered querying
"""

import streamlit as st
import os
from pathlib import Path
import tempfile
from rag_system import SemanticFinancialRAG
from dotenv import load_dotenv
import traceback

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
if 'initialization_error' not in st.session_state:
    st.session_state.initialization_error = None


@st.cache_resource
def initialize_rag_system():
    """Initialize the RAG system with API key."""
    try:
        api_key = os.getenv("ANTHROPIC_API_KEY") or os.getenv("CLAUDE_API_KEY")
        
        if not api_key:
            # Try to get from Streamlit secrets (for deployment)
            try:
                api_key = st.secrets.get("ANTHROPIC_API_KEY") or st.secrets.get("CLAUDE_API_KEY")
            except:
                return None, "No API key found. Please set ANTHROPIC_API_KEY in environment or secrets."
        
        if api_key:
            # Set the API key in environment
            os.environ["ANTHROPIC_API_KEY"] = api_key
            
            # Initialize the RAG system
            rag = SemanticFinancialRAG(
                model_name="intfloat/e5-large-v2",
                collection_name="financial_documents",
                persist_directory="./chroma_db"
            )
            return rag, None
        
        return None, "API key is empty or invalid"
        
    except Exception as e:
        error_msg = f"Error initializing RAG system: {str(e)}\n\n{traceback.format_exc()}"
        return None, error_msg


def save_uploaded_file(uploaded_file):
    """Save uploaded file to temporary location and return path."""
    try:
        # Create a temporary directory
        temp_dir = tempfile.mkdtemp()
        file_path = os.path.join(temp_dir, uploaded_file.name)
        
        # Write the file
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
    
    # API Key input (for local development)
    api_key_input = st.text_input(
        "Anthropic API Key (optional if using .env)", 
        type="password",
        help="Enter your Anthropic API key or set it in .env file as ANTHROPIC_API_KEY"
    )
    
    if api_key_input:
        os.environ["ANTHROPIC_API_KEY"] = api_key_input
    
    # Initialize button
    if st.button("🚀 Initialize System", type="primary"):
        with st.spinner("Initializing RAG system..."):
            # Clear cache and reinitialize
            initialize_rag_system.clear()
            rag, error = initialize_rag_system()
            
            if rag:
                st.session_state.rag_system = rag
                st.session_state.initialization_error = None
                st.success("✅ System initialized successfully!")
            else:
                st.session_state.rag_system = None
                st.session_state.initialization_error = error
                st.error(f"❌ Failed to initialize.\n\n{error}")
    
    # Show persistent error if exists
    if st.session_state.initialization_error:
        with st.expander("❌ Initialization Error Details"):
            st.code(st.session_state.initialization_error)
    
    st.markdown("---")
    
    # System status
    st.header("📊 System Status")
    if st.session_state.rag_system:
        st.success("🟢 System Active")
        try:
            stats = st.session_state.rag_system.get_stats()
            st.metric("Total Chunks", stats.get('total_chunks', 0))
            st.metric("Documents Uploaded", len(st.session_state.uploaded_files_list))
        except Exception as e:
            st.warning(f"Could not get stats: {e}")
    else:
        st.warning("🟡 System Not Initialized")
    
    st.markdown("---")
    
    # Settings
    st.header("🎛️ Settings")
    n_results = st.slider("Number of relevant chunks", 5, 50, 30)
    
    st.markdown("---")
    
    # Clear data button
    if st.button("🗑️ Clear All Data"):
        if st.session_state.rag_system:
            try:
                st.session_state.rag_system.client.delete_collection(
                    st.session_state.rag_system.collection.name
                )
            except Exception as e:
                st.error(f"Error clearing collection: {e}")
        
        initialize_rag_system.clear()
        st.session_state.rag_system = None
        st.session_state.uploaded_files_list = []
        st.session_state.chat_history = []
        st.session_state.initialization_error = None
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
            "Choose PDF files to upload",
            type=['pdf'],
            accept_multiple_files=True,
            help="Upload PDF documents for analysis"
        )
        
        if uploaded_files:
            if st.button("📥 Process Documents", type="primary"):
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                file_paths = []
                errors = []
                
                # Save all files first
                for uploaded_file in uploaded_files:
                    file_path, error = save_uploaded_file(uploaded_file)
                    if file_path:
                        file_paths.append((uploaded_file.name, file_path))
                    else:
                        errors.append(f"{uploaded_file.name}: {error}")
                
                if errors:
                    for error in errors:
                        st.error(error)
                
                # Process all files together
                if file_paths:
                    try:
                        status_text.text(f"Processing {len(file_paths)} files...")
                        
                        # Get just the paths
                        paths = [fp[1] for fp in file_paths]
                        
                        # Ingest all documents
                        st.session_state.rag_system.ingest_documents(
                            paths, 
                            clear_existing=False
                        )
                        
                        # Update the uploaded files list
                        for file_name, _ in file_paths:
                            if file_name not in st.session_state.uploaded_files_list:
                                st.session_state.uploaded_files_list.append(file_name)
                        
                        # Get total count
                        stats = st.session_state.rag_system.get_stats()
                        total_chunks = stats.get('total_chunks', 0)
                        
                        status_text.empty()
                        progress_bar.empty()
                        
                        st.success(f"✅ Processed {len(file_paths)} files! Total chunks in database: {total_chunks}")
                        
                    except Exception as e:
                        st.error(f"❌ Error processing files: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())
                    finally:
                        # Clean up temp files
                        for _, file_path in file_paths:
                            try:
                                os.remove(file_path)
                                # Try to remove parent directory
                                os.rmdir(os.path.dirname(file_path))
                            except Exception as e:
                                st.warning(f"Could not clean up temp file: {e}")
        
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
            if not query.strip():
                st.warning("Please enter a valid question")
            else:
                with st.spinner("🤔 Analyzing documents..."):
                    try:
                        # Use the RAG system to generate answer
                        answer = st.session_state.rag_system.generate_answer(
                            query, 
                            n_results=n_results
                        )
                        
                        # Get the chunks that were used
                        chunks = st.session_state.rag_system.retrieve(query, n_results=n_results)
                        
                        # Add to chat history
                        st.session_state.chat_history.append({
                            'question': query,
                            'answer': answer,
                            'sources': chunks
                        })
                        
                    except Exception as e:
                        st.error(f"Error processing query: {str(e)}")
                        with st.expander("Error Details"):
                            st.code(traceback.format_exc())

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
            if chat.get('sources'):
                with st.expander(f"📚 View Sources ({len(chat['sources'])} chunks)"):
                    for i, source in enumerate(chat['sources'], 1):
                        try:
                            st.markdown(f"**Source {i}**")
                            st.markdown(f"*File:* `{source['metadata'].get('source', 'Unknown')}`")
                            st.markdown(f"*Page:* {source['metadata'].get('page', 'Unknown')}")
                            st.markdown(f"*Type:* {source['metadata'].get('type', 'Unknown')}")
                            st.markdown(f"*Distance:* {source.get('distance', 0):.4f}")
                            text = source.get('text', '')
                            st.text(text[:300] + "..." if len(text) > 300 else text)
                            st.markdown("---")
                        except Exception as e:
                            st.warning(f"Could not display source {i}: {e}")
            
            st.markdown("---")

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Powered by Claude AI, ChromaDB, and E5-large-v2 embeddings</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
""", unsafe_allow_html=True)