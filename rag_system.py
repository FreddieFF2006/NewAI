import os
import anthropic
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from sentence_transformers import SentenceTransformer
import uuid
from typing import List, Dict
import numpy as np
import traceback

# File extraction libraries
try:
    from PyPDF2 import PdfReader
except ImportError:
    PdfReader = None

try:
    from docx import Document
except ImportError:
    Document = None

try:
    from pptx import Presentation
except ImportError:
    Presentation = None

try:
    import openpyxl
    import pandas as pd
except ImportError:
    openpyxl = None
    pd = None

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    # dotenv not installed, will use system environment variables
    pass

class DocumentRAGSystem:
    """
    RAG (Retrieval-Augmented Generation) system using:
    - Qdrant vector database with cosine similarity
    - Claude API for generation
    - E5-large-v2 for embeddings
    - Overlapping chunks for context awareness
    """
    
    def __init__(self, claude_api_key: str, collection_name: str = "documents"):
        """
        Initialize the RAG system.
        
        Args:
            claude_api_key: Your Anthropic Claude API key
            collection_name: Name for the Qdrant collection
        """
        # Initialize Claude client
        self.claude_client = anthropic.Anthropic(api_key=claude_api_key)
        
        # Initialize Qdrant client (in-memory for simplicity, can be changed to server)
        self.qdrant_client = QdrantClient(":memory:")
        
        # Initialize embedding model
        print("Loading E5-large-v2 model...")
        self.embedding_model = SentenceTransformer('intfloat/e5-large-v2')
        self.embedding_dim = 1024  # E5-large-v2 produces 1024-dimensional vectors
        
        # Collection name
        self.collection_name = collection_name
        
        # Create collection
        self._create_collection()
        
        print("RAG system initialized successfully!")
    
    def _create_collection(self):
        """Create Qdrant collection with cosine distance."""
        try:
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=self.embedding_dim,
                    distance=Distance.COSINE
                )
            )
            print(f"Created collection '{self.collection_name}' with cosine distance")
        except Exception as e:
            print(f"Collection might already exist: {e}")
    
    def _extract_text_from_file(self, file_path: str) -> str:
        """
        Extract text from various file types.
        
        Args:
            file_path: Path to the file
            
        Returns:
            Extracted text as string
        """
        file_extension = os.path.splitext(file_path)[1].lower()
        
        if file_extension == '.pdf':
            return self._extract_from_pdf(file_path)
        elif file_extension == '.docx':
            return self._extract_from_docx(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return self._extract_from_excel(file_path)
        elif file_extension == '.pptx':
            return self._extract_from_pptx(file_path)
        elif file_extension in ['.txt', '.md']:
            return self._extract_from_txt(file_path)
        else:
            raise ValueError(f"Unsupported file type: {file_extension}")
    
    def _extract_from_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        if PdfReader is None:
            raise ImportError("PyPDF2 not installed. Install with: pip install PyPDF2")
        
        try:
            reader = PdfReader(file_path)
            text = []
            for page in reader.pages:
                text.append(page.extract_text())
            return "\n\n".join(text)
        except Exception as e:
            raise Exception(f"Error reading PDF: {e}")
    
    def _extract_from_docx(self, file_path: str) -> str:
        """Extract text from Word document."""
        if Document is None:
            raise ImportError("python-docx not installed. Install with: pip install python-docx")
        
        try:
            doc = Document(file_path)
            text = []
            for paragraph in doc.paragraphs:
                text.append(paragraph.text)
            return "\n".join(text)
        except Exception as e:
            raise Exception(f"Error reading DOCX: {e}")
    
    def _extract_from_excel(self, file_path: str) -> str:
        """Extract text from Excel file."""
        if pd is None or openpyxl is None:
            raise ImportError("pandas and openpyxl not installed. Install with: pip install pandas openpyxl")
        
        try:
            # Read all sheets
            xls = pd.ExcelFile(file_path)
            text = []
            
            for sheet_name in xls.sheet_names:
                df = pd.read_excel(xls, sheet_name=sheet_name)
                text.append(f"Sheet: {sheet_name}\n")
                text.append(df.to_string())
                text.append("\n\n")
            
            return "\n".join(text)
        except Exception as e:
            raise Exception(f"Error reading Excel: {e}")
    
    def _extract_from_pptx(self, file_path: str) -> str:
        """Extract text from PowerPoint presentation."""
        if Presentation is None:
            raise ImportError("python-pptx not installed. Install with: pip install python-pptx")
        
        try:
            prs = Presentation(file_path)
            text = []
            
            for slide_num, slide in enumerate(prs.slides, 1):
                text.append(f"Slide {slide_num}:\n")
                for shape in slide.shapes:
                    if hasattr(shape, "text"):
                        text.append(shape.text)
                text.append("\n")
            
            return "\n".join(text)
        except Exception as e:
            raise Exception(f"Error reading PPTX: {e}")
    
    def _extract_from_txt(self, file_path: str) -> str:
        """Extract text from plain text file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            # Try with different encoding
            with open(file_path, 'r', encoding='latin-1') as f:
                return f.read()
    
    def _chunk_text(self, text: str, chunk_size: int = 500, overlap: int = 100) -> List[str]:
        """
        Split text into overlapping chunks for better context awareness.
        
        Args:
            text: Input text to chunk
            chunk_size: Size of each chunk in characters
            overlap: Number of overlapping characters between chunks
            
        Returns:
            List of text chunks
        """
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            
            # Only add non-empty chunks
            if chunk.strip():
                chunks.append(chunk)
            
            # Move start position with overlap
            start = end - overlap
            
            # Break if we've reached the end
            if end >= len(text):
                break
        
        return chunks
    
    def _embed_text(self, text: str) -> List[float]:
        """
        Create embedding for text using E5-large-v2.
        E5 models require a prefix for better performance.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        # E5 models work better with "query: " or "passage: " prefixes
        prefixed_text = f"passage: {text}"
        embedding = self.embedding_model.encode(prefixed_text, normalize_embeddings=True)
        return embedding.tolist()
    
    def upload_file(self, file_path: str, metadata: Dict = None) -> int:
        """
        Upload a file to the vector database.
        Supports: .txt, .md, .pdf, .docx, .xlsx, .xls, .pptx
        
        Args:
            file_path: Path to the file to upload
            metadata: Optional metadata to associate with the file
            
        Returns:
            Number of chunks created
        """
        file_name = os.path.basename(file_path)
        file_extension = os.path.splitext(file_path)[1].lower()
        print(f"Processing file: {file_name} (type: {file_extension})")
        
        # Extract text based on file type
        try:
            text = self._extract_text_from_file(file_path)
            print(f"Extracted {len(text)} characters from {file_name}")
        except Exception as e:
            print(f"Error extracting text from {file_name}: {e}")
            import traceback
            traceback.print_exc()
            return 0
        
        if not text or not text.strip():
            print(f"Warning: No text extracted from {file_name}")
            return 0
        
        # Create chunks with overlap
        chunks = self._chunk_text(text, chunk_size=500, overlap=100)
        print(f"Created {len(chunks)} overlapping chunks from {file_name}")
        
        # Prepare points for Qdrant
        points = []
        for idx, chunk in enumerate(chunks):
            # Create embedding
            embedding = self._embed_text(chunk)
            
            # Prepare metadata
            point_metadata = {
                "text": chunk,
                "file_path": file_path,
                "chunk_index": idx,
                "total_chunks": len(chunks)
            }
            
            # Add user metadata if provided
            if metadata:
                point_metadata.update(metadata)
            
            # Create point
            point = PointStruct(
                id=str(uuid.uuid4()),
                vector=embedding,
                payload=point_metadata
            )
            points.append(point)
        
        # Upload to Qdrant
        self.qdrant_client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        
        print(f"Uploaded {len(chunks)} chunks from {file_path}")
        return len(chunks)
    
    def upload_multiple_files(self, file_paths: List[str], metadata: Dict = None) -> Dict:
        """
        Upload multiple files to the vector database.
        
        Args:
            file_paths: List of file paths to upload
            metadata: Optional metadata to associate with all files
            
        Returns:
            Dictionary with upload statistics
        """
        total_chunks = 0
        results = {}
        
        for file_path in file_paths:
            chunks = self.upload_file(file_path, metadata)
            results[file_path] = chunks
            total_chunks += chunks
        
        print(f"\nTotal: Uploaded {total_chunks} chunks from {len(file_paths)} files")
        return results
    
    def search(self, query: str, top_k: int = 5) -> List[Dict]:
        """
        Search for relevant chunks in the vector database.
        
        Args:
            query: Search query
            top_k: Number of results to return
            
        Returns:
            List of search results with text and metadata
        """
        # Create query embedding with "query: " prefix for E5
        query_embedding = self.embedding_model.encode(
            f"query: {query}", 
            normalize_embeddings=True
        ).tolist()
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_embedding,
            limit=top_k
        )
        
        # Format results
        results = []
        for result in search_results:
            results.append({
                "text": result.payload["text"],
                "score": result.score,
                "file_path": result.payload["file_path"],
                "chunk_index": result.payload["chunk_index"],
                "metadata": result.payload
            })
        
        return results
    
    def query(self, question: str, top_k: int = 5, max_tokens: int = 2000) -> Dict:
        """
        Query the system with RAG: retrieve relevant chunks and generate answer with Claude.
        
        Args:
            question: User's question
            top_k: Number of chunks to retrieve
            max_tokens: Maximum tokens for Claude's response
            
        Returns:
            Dictionary with answer and sources
        """
        print(f"\nProcessing query: {question}")
        
        # Search for relevant chunks
        search_results = self.search(question, top_k=top_k)
        
        if not search_results:
            return {
                "answer": "I couldn't find any relevant information in the uploaded documents.",
                "sources": []
            }
        
        # Prepare context from search results
        context_parts = []
        for idx, result in enumerate(search_results, 1):
            context_parts.append(
                f"[Source {idx}] (from {result['file_path']}, chunk {result['chunk_index']}, "
                f"relevance: {result['score']:.3f}):\n{result['text']}\n"
            )
        
        context = "\n".join(context_parts)
        
        # Create prompt for Claude
        prompt = f"""You are a helpful AI assistant. Answer the following question based on the provided context.

Context from documents:
{context}

Question: {question}

Please provide a clear and accurate answer based on the context above. If the context doesn't contain enough information to fully answer the question, please say so."""

        # Query Claude
        print("Generating answer with Claude...")
        message = self.claude_client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=max_tokens,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = message.content[0].text
        
        # Prepare sources
        sources = [
            {
                "file": result["file_path"],
                "chunk": result["chunk_index"],
                "score": result["score"],
                "text_preview": result["text"][:200] + "..."
            }
            for result in search_results
        ]
        
        return {
            "answer": answer,
            "sources": sources
        }
    
    def get_collection_stats(self) -> Dict:
        """Get statistics about the collection."""
        collection_info = self.qdrant_client.get_collection(self.collection_name)
        return {
            "total_vectors": collection_info.points_count,
            "vector_dimension": self.embedding_dim,
            "distance_metric": "cosine"
        }


def main():
    """Example usage of the RAG system."""
    
    # Get API key from environment variable (loaded from .env if available)
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    
    if not claude_api_key:
        print("Error: Please set CLAUDE_API_KEY environment variable")
        print("\nOption 1 - Create .env file:")
        print("  Create a file named '.env' with: CLAUDE_API_KEY=your-api-key-here")
        print("\nOption 2 - Export environment variable:")
        print("  export CLAUDE_API_KEY='your-api-key-here'")
        return
    
    # Initialize the system
    rag = DocumentRAGSystem(claude_api_key=claude_api_key)
    
    # Example: Upload multiple files
    # Replace these with your actual file paths
    example_files = [
        "document1.txt",
        "document2.txt",
        "document3.txt"
    ]
    
    # Uncomment to upload files
    # rag.upload_multiple_files(example_files)
    
    # Example: Query the system
    # question = "What is the main topic discussed in the documents?"
    # result = rag.query(question)
    # print(f"\nAnswer: {result['answer']}")
    # print(f"\nSources used: {len(result['sources'])}")
    
    # Get collection stats
    stats = rag.get_collection_stats()
    print(f"\nCollection statistics: {stats}")


if __name__ == "__main__":
    main()