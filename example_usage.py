"""
RAG System with Qdrant, Claude API, and E5-large-v2 Embeddings
================================================================

This is a complete example showing how to use the RAG system.

Setup Instructions:
-------------------
1. Install dependencies:
   pip install -r requirements.txt
   pip install python-dotenv

2. Set your Claude API key:
   Option A - Create .env file (recommended):
   Create a file named '.env' in the same directory with:
   CLAUDE_API_KEY=sk-ant-your-actual-key-here
   
   Option B - Export environment variable:
   export CLAUDE_API_KEY='your-api-key-here'

3. Run the example below

"""

import os

# Load .env file BEFORE importing rag_system
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from rag_system import DocumentRAGSystem

def example_usage():
    """Comprehensive example of using the RAG system."""
    
    # Get API key from environment (loaded from .env file if present)
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    
    if not claude_api_key:
        print("Error: CLAUDE_API_KEY not found!")
        print("Please create a .env file with: CLAUDE_API_KEY=your-api-key-here")
        return
    
    # Initialize the system
    print("=" * 60)
    print("Initializing RAG System...")
    print("=" * 60)
    rag = DocumentRAGSystem(claude_api_key=claude_api_key)
    
    # Example 1: Create sample documents
    print("\n" + "=" * 60)
    print("Creating Sample Documents...")
    print("=" * 60)
    
    # Create sample document 1
    with open("sample_doc1.txt", "w") as f:
        f.write("""
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) is the simulation of human intelligence by machines.
        Machine Learning (ML) is a subset of AI that enables systems to learn from data.
        Deep Learning is a subset of ML that uses neural networks with multiple layers.
        
        Common applications include:
        - Natural Language Processing for understanding text
        - Computer Vision for image recognition
        - Recommendation systems for personalized content
        - Autonomous vehicles for self-driving cars
        
        The field has grown exponentially since 2010, with major breakthroughs in
        transformers, large language models, and generative AI.
        """)
    
    # Create sample document 2
    with open("sample_doc2.txt", "w") as f:
        f.write("""
        Vector Databases and Embeddings
        
        Vector databases store data as high-dimensional vectors, enabling semantic search.
        Unlike traditional databases that use exact keyword matching, vector databases
        understand the meaning and context of queries.
        
        Popular vector databases include:
        - Qdrant: High-performance vector search engine
        - Pinecone: Managed vector database service
        - Weaviate: Open-source vector search engine
        - Milvus: Scalable vector database
        
        Embeddings are numerical representations of text that capture semantic meaning.
        Models like E5-large-v2 create embeddings that enable similarity search.
        Cosine similarity is commonly used to measure vector similarity.
        """)
    
    # Create sample document 3
    with open("sample_doc3.txt", "w") as f:
        f.write("""
        Claude API and Language Models
        
        Claude is an AI assistant created by Anthropic. It excels at:
        - Natural conversations and question answering
        - Analysis and reasoning tasks
        - Code generation and debugging
        - Creative writing and content creation
        
        The Claude API provides programmatic access to Claude's capabilities.
        It uses a messages-based API similar to chat interfaces.
        
        RAG (Retrieval-Augmented Generation) combines:
        1. Information retrieval from a knowledge base
        2. Language model generation based on retrieved context
        
        This approach reduces hallucinations and provides source attribution.
        """)
    
    # Example 2: Upload multiple files
    print("\n" + "=" * 60)
    print("Uploading Documents...")
    print("=" * 60)
    
    files_to_upload = [
        "sample_doc1.txt",
        "sample_doc2.txt",
        "sample_doc3.txt"
    ]
    
    upload_results = rag.upload_multiple_files(
        files_to_upload,
        metadata={"source": "example", "category": "technical"}
    )
    
    # Example 3: Check collection statistics
    print("\n" + "=" * 60)
    print("Collection Statistics...")
    print("=" * 60)
    stats = rag.get_collection_stats()
    print(f"Total vectors: {stats['total_vectors']}")
    print(f"Vector dimension: {stats['vector_dimension']}")
    print(f"Distance metric: {stats['distance_metric']}")
    
    # Example 4: Perform searches
    print("\n" + "=" * 60)
    print("Searching Documents...")
    print("=" * 60)
    
    search_query = "What are vector databases?"
    search_results = rag.search(search_query, top_k=3)
    
    print(f"\nSearch query: '{search_query}'")
    print(f"Found {len(search_results)} results:\n")
    
    for idx, result in enumerate(search_results, 1):
        print(f"Result {idx}:")
        print(f"  Score: {result['score']:.4f}")
        print(f"  File: {result['file_path']}")
        print(f"  Chunk: {result['chunk_index']}")
        print(f"  Text preview: {result['text'][:150]}...")
        print()
    
    # Example 5: Query with RAG
    print("\n" + "=" * 60)
    print("Querying with RAG (Retrieval + Generation)...")
    print("=" * 60)
    
    questions = [
        "What is machine learning and what are its applications?",
        "Explain how vector databases work",
        "What is RAG and how does it reduce hallucinations?"
    ]
    
    for question in questions:
        print(f"\n{'=' * 60}")
        print(f"Question: {question}")
        print(f"{'=' * 60}")
        
        result = rag.query(question, top_k=3)
        
        print(f"\nAnswer:\n{result['answer']}")
        
        print(f"\n\nSources used ({len(result['sources'])}):")
        for idx, source in enumerate(result['sources'], 1):
            print(f"\n  {idx}. {source['file']} (chunk {source['chunk']}, "
                  f"relevance: {source['score']:.3f})")
            print(f"     Preview: {source['text_preview']}")
        
        print("\n" + "-" * 60)
    
    print("\n" + "=" * 60)
    print("Example Complete!")
    print("=" * 60)


def interactive_mode():
    """Interactive mode for querying the RAG system."""
    
    claude_api_key = os.getenv("CLAUDE_API_KEY")
    
    if not claude_api_key:
        print("Error: Please set CLAUDE_API_KEY environment variable")
        return
    
    rag = DocumentRAGSystem(claude_api_key=claude_api_key)
    
    print("\n" + "=" * 60)
    print("Interactive RAG System")
    print("=" * 60)
    print("\nCommands:")
    print("  upload <file1> <file2> ... - Upload files")
    print("  search <query> - Search for relevant chunks")
    print("  query <question> - Ask a question")
    print("  stats - Show collection statistics")
    print("  quit - Exit")
    print()
    
    while True:
        try:
            user_input = input("\n> ").strip()
            
            if not user_input:
                continue
            
            parts = user_input.split(maxsplit=1)
            command = parts[0].lower()
            
            if command == "quit":
                print("Goodbye!")
                break
            
            elif command == "upload":
                if len(parts) < 2:
                    print("Usage: upload <file1> <file2> ...")
                    continue
                
                files = parts[1].split()
                rag.upload_multiple_files(files)
            
            elif command == "search":
                if len(parts) < 2:
                    print("Usage: search <query>")
                    continue
                
                query = parts[1]
                results = rag.search(query, top_k=5)
                
                print(f"\nFound {len(results)} results:")
                for idx, result in enumerate(results, 1):
                    print(f"\n{idx}. Score: {result['score']:.4f}")
                    print(f"   File: {result['file_path']}")
                    print(f"   {result['text'][:200]}...")
            
            elif command == "query":
                if len(parts) < 2:
                    print("Usage: query <question>")
                    continue
                
                question = parts[1]
                result = rag.query(question)
                
                print(f"\nAnswer:\n{result['answer']}")
                print(f"\nSources: {len(result['sources'])}")
            
            elif command == "stats":
                stats = rag.get_collection_stats()
                print(f"\nCollection Statistics:")
                print(f"  Total vectors: {stats['total_vectors']}")
                print(f"  Vector dimension: {stats['vector_dimension']}")
                print(f"  Distance metric: {stats['distance_metric']}")
            
            else:
                print(f"Unknown command: {command}")
        
        except KeyboardInterrupt:
            print("\nGoodbye!")
            break
        except Exception as e:
            print(f"Error: {e}")


if __name__ == "__main__":
    # Run the example
    example_usage()
    
    # Uncomment to use interactive mode
    # interactive_mode()
