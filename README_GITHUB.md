# RAG System with Qdrant, Claude API, and E5-large-v2

A powerful Retrieval-Augmented Generation (RAG) system that combines:
- **Qdrant** vector database with cosine similarity
- **Claude API** for intelligent responses
- **E5-large-v2** for high-quality embeddings
- **Overlapping chunks** for enhanced context awareness

## Features

✅ **Multiple File Upload**: Upload and process multiple documents at once
✅ **Overlapping Chunks**: Creates overlapping text chunks for better context preservation
✅ **Semantic Search**: Uses E5-large-v2 embeddings with cosine similarity
✅ **RAG Pipeline**: Retrieves relevant context and generates accurate answers
✅ **Source Attribution**: Tracks and displays sources for each answer
✅ **Flexible API**: Easy-to-use Python interface

## Architecture

```
┌─────────────────┐
│   Documents     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│ Text Chunking   │ ← Overlapping chunks (500 chars, 100 overlap)
│   (Overlap)     │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│   E5-large-v2   │ ← 1024-dimensional embeddings
│   Embeddings    │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│     Qdrant      │ ← Cosine similarity search
│  Vector Store   │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Claude API     │ ← Generate answer with context
│   Generation    │
└─────────────────┘
```

## Installation

### Prerequisites
- Python 3.8 or higher
- Claude API key from Anthropic

### Setup

1. **Clone or download the files**

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Set your Claude API key**:
```bash
export CLAUDE_API_KEY='your-api-key-here'
```

Or in Python:
```python
import os
os.environ['CLAUDE_API_KEY'] = 'your-api-key-here'
```

## Quick Start

```python
from rag_system import DocumentRAGSystem

# Initialize the system
rag = DocumentRAGSystem(claude_api_key='your-api-key')

# Upload multiple files
files = ['doc1.txt', 'doc2.txt', 'doc3.txt']
rag.upload_multiple_files(files)

# Query the system
result = rag.query("What is machine learning?")
print(result['answer'])
print(f"Sources: {len(result['sources'])}")
```

## Detailed Usage

### 1. Initialize the System

```python
from rag_system import DocumentRAGSystem

# Initialize with your Claude API key
rag = DocumentRAGSystem(
    claude_api_key='your-claude-api-key',
    collection_name='my_documents'  # Optional: custom collection name
)
```

### 2. Upload Documents

**Single file:**
```python
chunks = rag.upload_file('document.txt')
print(f"Created {chunks} chunks")
```

**Multiple files:**
```python
files = ['doc1.txt', 'doc2.txt', 'doc3.txt']
results = rag.upload_multiple_files(files)
```

**With metadata:**
```python
metadata = {
    'category': 'technical',
    'author': 'John Doe',
    'date': '2024-01-01'
}
rag.upload_file('document.txt', metadata=metadata)
```

### 3. Search Documents

```python
# Search for relevant chunks
results = rag.search("machine learning applications", top_k=5)

for result in results:
    print(f"Score: {result['score']:.4f}")
    print(f"Text: {result['text']}")
    print(f"File: {result['file_path']}")
```

### 4. Query with RAG

```python
# Ask a question with full RAG pipeline
result = rag.query(
    question="What are the benefits of vector databases?",
    top_k=5,  # Number of chunks to retrieve
    max_tokens=2000  # Max response length
)

print(f"Answer: {result['answer']}")
print(f"\nSources used:")
for source in result['sources']:
    print(f"  - {source['file']} (chunk {source['chunk']})")
```

### 5. Check Collection Statistics

```python
stats = rag.get_collection_stats()
print(f"Total vectors: {stats['total_vectors']}")
print(f"Vector dimension: {stats['vector_dimension']}")
print(f"Distance metric: {stats['distance_metric']}")
```

## Configuration Options

### Chunking Parameters

The system uses overlapping chunks for better context:

```python
# In the _chunk_text method (modify in rag_system.py)
chunk_size = 500   # Characters per chunk
overlap = 100      # Overlapping characters between chunks
```

**Why overlapping chunks?**
- Preserves context across chunk boundaries
- Prevents information loss at split points
- Improves retrieval accuracy

### Embedding Model

The system uses `intfloat/e5-large-v2`:
- **Dimension**: 1024
- **Performance**: State-of-the-art semantic similarity
- **Prefix**: Automatically adds "query:" and "passage:" prefixes

### Vector Database

Qdrant configuration:
- **Distance metric**: Cosine similarity
- **Storage**: In-memory (can be changed to persistent)
- **Collection**: Customizable name

## Example Output

```
Processing query: What is machine learning?

Answer: Machine Learning (ML) is a subset of artificial intelligence 
that enables systems to learn from data without being explicitly 
programmed. It involves algorithms that improve their performance 
over time through experience...

Sources used (3):
  1. sample_doc1.txt (chunk 0, relevance: 0.856)
     Preview: Artificial Intelligence (AI) is the simulation of 
     human intelligence by machines. Machine Learning (ML) is a 
     subset of AI...
  
  2. sample_doc1.txt (chunk 1, relevance: 0.823)
     Preview: Common applications include: Natural Language Processing 
     for understanding text...
```

## Advanced Features

### Custom Distance Metrics

To change from cosine to other metrics, modify the collection creation:

```python
# In _create_collection method
Distance.COSINE    # Current (default)
Distance.EUCLID    # Euclidean distance
Distance.DOT       # Dot product
Distance.MANHATTAN # Manhattan distance
```

### Persistent Storage

To use persistent storage instead of in-memory:

```python
# In __init__ method, change:
self.qdrant_client = QdrantClient(":memory:")

# To:
self.qdrant_client = QdrantClient(path="./qdrant_data")

# Or for remote server:
self.qdrant_client = QdrantClient(
    url="http://localhost:6333",
    api_key="your-api-key"  # If using cloud
)
```

### Filtering Search Results

Add filters to your searches:

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Search with filters
search_results = self.qdrant_client.search(
    collection_name=self.collection_name,
    query_vector=query_embedding,
    query_filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="technical")
            )
        ]
    ),
    limit=top_k
)
```

## Running the Examples

### Basic Example
```bash
python example_usage.py
```

This will:
1. Create sample documents
2. Upload them to Qdrant
3. Perform searches
4. Query with RAG
5. Display results with sources

### Interactive Mode

Uncomment the last line in `example_usage.py` to enable interactive mode:

```python
if __name__ == "__main__":
    interactive_mode()
```

Then run:
```bash
python example_usage.py
```

Commands:
- `upload file1.txt file2.txt` - Upload files
- `search your query here` - Search documents
- `query your question here` - Ask a question
- `stats` - Show collection statistics
- `quit` - Exit

## Troubleshooting

### Issue: "CLAUDE_API_KEY not set"
**Solution**: Export your API key:
```bash
export CLAUDE_API_KEY='sk-ant-...'
```

### Issue: Model download is slow
**Solution**: The E5-large-v2 model is ~1.3GB. First download takes time, but it's cached for future use.

### Issue: Out of memory
**Solution**: 
- Reduce chunk size
- Process fewer files at once
- Use persistent storage instead of in-memory

### Issue: Poor search results
**Solution**:
- Increase chunk overlap
- Adjust chunk size
- Upload more relevant documents
- Use more specific queries

## Performance Tips

1. **Batch upload**: Upload multiple files at once for efficiency
2. **Chunk size**: Balance between context (larger) and precision (smaller)
3. **Overlap**: 15-20% overlap is usually optimal
4. **Top-k**: Start with 3-5 results for queries
5. **Metadata**: Add rich metadata for better filtering

## API Reference

### DocumentRAGSystem Class

#### `__init__(claude_api_key, collection_name='documents')`
Initialize the RAG system.

#### `upload_file(file_path, metadata=None)`
Upload a single file. Returns number of chunks created.

#### `upload_multiple_files(file_paths, metadata=None)`
Upload multiple files. Returns dictionary with upload statistics.

#### `search(query, top_k=5)`
Search for relevant chunks. Returns list of results with scores.

#### `query(question, top_k=5, max_tokens=2000)`
Full RAG pipeline: search + generate answer. Returns answer and sources.

#### `get_collection_stats()`
Get collection statistics. Returns dictionary with counts and configuration.

## Dependencies

- **anthropic**: Claude API client
- **qdrant-client**: Vector database client
- **sentence-transformers**: For E5-large-v2 embeddings
- **torch**: Required by sentence-transformers
- **numpy**: Numerical operations

## License

This project is provided as-is for educational and commercial use.

## Contributing

Feel free to modify and extend this system for your needs:
- Add new file formats (PDF, DOCX, etc.)
- Implement hybrid search (keyword + semantic)
- Add conversation memory
- Create a web interface
- Integrate with other LLMs

## Support

For issues with:
- **Claude API**: https://docs.anthropic.com
- **Qdrant**: https://qdrant.tech/documentation
- **E5 embeddings**: https://huggingface.co/intfloat/e5-large-v2

## Acknowledgments

- Anthropic for Claude API
- Qdrant for the vector database
- Microsoft for E5-large-v2 embeddings
