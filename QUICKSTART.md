# Quick Start Guide - RAG System

## Get Started in 5 Minutes

### Step 1: Install Dependencies (2 minutes)
```bash
pip install -r requirements.txt
```

### Step 2: Set Your API Key (30 seconds)
```bash
export CLAUDE_API_KEY='your-anthropic-api-key-here'
```

### Step 3: Test Installation (30 seconds)
```bash
python test_setup.py
```

### Step 4: Run Example (2 minutes)
```bash
python example_usage.py
```

## Minimal Working Example

Copy this into a file called `my_first_rag.py`:

```python
import os
from rag_system import DocumentRAGSystem

# Set your API key
os.environ['CLAUDE_API_KEY'] = 'your-api-key-here'

# Initialize
rag = DocumentRAGSystem(claude_api_key=os.getenv('CLAUDE_API_KEY'))

# Create a test document
with open('test.txt', 'w') as f:
    f.write("Python is a high-level programming language. "
            "It's known for readability and simplicity. "
            "Python is widely used in data science, web development, "
            "and artificial intelligence applications.")

# Upload the document
rag.upload_file('test.txt')

# Query it
result = rag.query("What is Python used for?")
print(result['answer'])
```

Run it:
```bash
python my_first_rag.py
```

## What You Get

✅ **rag_system.py** - Main RAG system (complete implementation)
✅ **example_usage.py** - Full examples with sample documents  
✅ **test_setup.py** - Verify your installation
✅ **requirements.txt** - All dependencies
✅ **README.md** - Complete documentation

## Key Features

1. **Qdrant Vector Database** with cosine similarity
2. **Claude API** for intelligent responses
3. **E5-large-v2** embeddings (1024-dimensional)
4. **Overlapping chunks** for context preservation
5. **Multiple file upload** support
6. **Source attribution** in answers

## Common Commands

```python
# Upload multiple files
rag.upload_multiple_files(['doc1.txt', 'doc2.txt'])

# Search documents
results = rag.search("your query", top_k=5)

# Ask questions
result = rag.query("What is...?")
print(result['answer'])

# Check stats
stats = rag.get_collection_stats()
```

## Troubleshooting

**Issue**: Package installation fails  
**Fix**: `pip install --upgrade pip` then retry

**Issue**: API key not working  
**Fix**: Get your key from https://console.anthropic.com/

**Issue**: Model download is slow  
**Fix**: First download is ~1.3GB, then cached

## Next Steps

1. Read the full **README.md** for advanced features
2. Customize chunk size and overlap in `rag_system.py`
3. Add PDF/DOCX support using libraries like PyPDF2
4. Deploy to production with persistent Qdrant storage

## Support

- **Claude API docs**: https://docs.anthropic.com
- **Qdrant docs**: https://qdrant.tech/documentation
- **E5 model**: https://huggingface.co/intfloat/e5-large-v2

---

**You're all set! Start uploading documents and querying them with AI. 🚀**
