"""
Semantic Financial RAG System
Using Semchunk for intelligent chunking and ChromaDB for vector storage
"""

import pdfplumber
from semchunk import chunkerify
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict
import anthropic


class SemanticFinancialRAG:
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        collection_name: str = "financial_documents",
        persist_directory: str = "./chroma_db"
    ):
        """Initialize the RAG system with Semchunk and ChromaDB"""
        
        # Initialize sentence transformer for embeddings
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        # Initialize Semchunk
        self.chunker = chunkerify(
            self.embedder,
            chunk_size=1000,        # Large enough for tables/metrics
            similarity_percentile=80,    # Adaptive splitting
            skip_window=2                # Look ahead for context
        )
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        # Create or get collection
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
        except:
            self.collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            print(f"Created new collection: {collection_name}")
        
        # Initialize Anthropic client
        self.claude = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))
    
    def _format_table(self, table: List[List]) -> str:
        """Format table as structured text"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            # Filter out empty rows
            if any(cells):
                lines.append(" | ".join(cells))
        
        return "\n".join(lines)
    
    def _extract_tables_with_context(self, page, page_num: int) -> List[Dict]:
        """Extract tables with surrounding context"""
        tables = page.extract_tables()
        table_chunks = []
        
        for table_idx, table in enumerate(tables):
            if not table or not any(table):
                continue
            
            # Format table
            table_text = self._format_table(table)
            
            if table_text:
                # Try to get context from page text near the table
                full_page_text = page.extract_text() or ""
                
                # Find the section header before the table
                lines = full_page_text.split('\n')
                context = ""
                
                # Look for headers near the table content
                for i, line in enumerate(lines):
                    if any(cell in line for row in table[:2] for cell in row if cell):
                        # Found table location, grab previous 2 lines as context
                        context_lines = lines[max(0, i-2):i]
                        context = "\n".join(context_lines)
                        break
                
                # Combine context with table
                full_table_text = f"{context}\n\n{table_text}" if context else table_text
                
                table_chunks.append({
                    'text': full_table_text,
                    'page': page_num,
                    'type': 'table',
                    'table_index': table_idx
                })
        
        return table_chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """
        Extract and semantically chunk a PDF
        Returns list of chunks with metadata
        """
        print(f"\nProcessing PDF: {pdf_path}")
        all_chunks = []
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_num}/{total_pages}...", end='\r')
                
                # 1. Extract tables first (keep intact)
                table_chunks = self._extract_tables_with_context(page, page_num)
                all_chunks.extend([{
                    **chunk,
                    'source': os.path.basename(pdf_path)
                } for chunk in table_chunks])
                
                # 2. Extract text
                page_text = page.extract_text()
                
                if page_text:
                    # Add page marker
                    page_text = f"[Page {page_num}]\n{page_text}"
                    
                    # 3. Semantic chunking on text
                    try:
                        text_chunks = self.chunker(page_text)
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
                            # Skip very short chunks
                            if len(chunk_text.strip()) < 50:
                                continue
                            
                            all_chunks.append({
                                'text': chunk_text,
                                'page': page_num,
                                'type': 'text',
                                'chunk_index': chunk_idx,
                                'source': os.path.basename(pdf_path)
                            })
                    except Exception as e:
                        print(f"\nWarning: Could not chunk page {page_num}: {e}")
                        # Fallback: use entire page as one chunk
                        all_chunks.append({
                            'text': page_text,
                            'page': page_num,
                            'type': 'text',
                            'chunk_index': 0,
                            'source': os.path.basename(pdf_path)
                        })
        
        print(f"\nCreated {len(all_chunks)} chunks from {pdf_path}")
        return all_chunks
    
    def ingest_documents(self, pdf_paths: List[str], clear_existing: bool = False):
        """
        Ingest multiple PDFs into ChromaDB
        """
        if clear_existing:
            print("Clearing existing collection...")
            self.client.delete_collection(name=self.collection.name)
            self.collection = self.client.create_collection(
                name=self.collection.name,
                metadata={"hnsw:space": "cosine"}
            )
        
        for pdf_path in pdf_paths:
            if not os.path.exists(pdf_path):
                print(f"Warning: File not found: {pdf_path}")
                continue
            
            # Process PDF
            chunks = self.process_pdf(pdf_path)
            
            # Prepare for ChromaDB
            documents = []
            metadatas = []
            ids = []
            
            for i, chunk in enumerate(chunks):
                documents.append(chunk['text'])
                metadatas.append({
                    'source': chunk['source'],
                    'page': chunk['page'],
                    'type': chunk['type'],
                })
                
                # Create unique ID
                chunk_id = f"{chunk['source']}_page{chunk['page']}_{chunk['type']}_{i}"
                ids.append(chunk_id)
            
            # Batch add to ChromaDB (more efficient)
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch_docs = documents[i:i+batch_size]
                batch_meta = metadatas[i:i+batch_size]
                batch_ids = ids[i:i+batch_size]
                
                # Generate embeddings
                embeddings = self.embedder.encode(batch_docs).tolist()
                
                self.collection.add(
                    documents=batch_docs,
                    metadatas=batch_meta,
                    ids=batch_ids,
                    embeddings=embeddings
                )
            
            print(f"✓ Ingested {len(chunks)} chunks from {pdf_path}")
        
        print(f"\n✓ Total documents in collection: {self.collection.count()}")
    
    def retrieve(self, query: str, n_results: int = 20) -> List[Dict]:
        """
        Retrieve relevant chunks for a query
        """
        # Generate query embedding
        query_embedding = self.embedder.encode(query).tolist()
        
        # Query ChromaDB
        results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=['documents', 'metadatas', 'distances']
        )
        
        # Format results
        retrieved_chunks = []
        for i in range(len(results['documents'][0])):
            retrieved_chunks.append({
                'text': results['documents'][0][i],
                'metadata': results['metadatas'][0][i],
                'distance': results['distances'][0][i]
            })
        
        return retrieved_chunks
    
    def generate_answer(self, query: str, n_results: int = 30) -> str:
        """
        Retrieve relevant chunks and generate answer using Claude
        """
        print(f"\nQuery: {query}")
        print(f"Retrieving top {n_results} chunks...")
        
        # Retrieve relevant chunks
        chunks = self.retrieve(query, n_results=n_results)
        
        if not chunks:
            return "No relevant information found in the documents."
        
        # Build context from chunks
        context = "\n\n---\n\n".join([
            f"[Source: {chunk['metadata']['source']}, Page: {chunk['metadata']['page']}, Type: {chunk['metadata']['type']}]\n{chunk['text']}"
            for chunk in chunks
        ])
        
        print(f"Retrieved {len(chunks)} chunks")
        print(f"Context length: {len(context)} characters")
        
        # Create prompt for Claude
        prompt = f"""You are a financial analyst assistant. Answer the user's question based on the provided context from earnings reports.

Context from financial documents:
{context}

User question: {query}

Instructions:
- Answer the question directly and concisely
- If you find specific numbers (like employee headcount), cite the source and page number
- If information is missing from the context, say so
- For numerical data, present it in a clear table format if multiple companies are involved

Answer:"""
        
        # Call Claude
        print("Generating answer with Claude...")
        message = self.claude.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[
                {"role": "user", "content": prompt}
            ]
        )
        
        answer = message.content[0].text
        return answer
    
    def debug_chunks(self, query: str, n_results: int = 10):
        """
        Debug: Show what chunks are retrieved for a query
        """
        chunks = self.retrieve(query, n_results=n_results)
        
        print(f"\n{'='*80}")
        print(f"DEBUG: Retrieved chunks for query: '{query}'")
        print(f"{'='*80}\n")
        
        for i, chunk in enumerate(chunks, 1):
            print(f"Chunk {i}:")
            print(f"  Source: {chunk['metadata']['source']}")
            print(f"  Page: {chunk['metadata']['page']}")
            print(f"  Type: {chunk['metadata']['type']}")
            print(f"  Distance: {chunk['distance']:.4f}")
            print(f"  Text preview: {chunk['text'][:200]}...")
            print()


def main():
    """Example usage"""
    # Initialize RAG system
    rag = SemanticFinancialRAG(
        model_name="intfloat/e5-large-v2",
        collection_name="earnings_q1_2025",
        persist_directory="./chroma_db"
    )
    
    # List of PDF files to process
    pdf_files = [
        "/mnt/user-data/uploads/2025q1-alphabet-earnings-release.pdf",
        "/mnt/user-data/uploads/AMZN-Q1-2025-Earnings-Release.pdf",
        "/mnt/user-data/uploads/Coca-Cola_2025_Q1_Earnings_Release_Full_Release_4_29_25.pdf",
        "/mnt/user-data/uploads/Meta-Reports-First-Quarter-2025-Results-2025.pdf"
    ]
    
    # Ingest documents (set clear_existing=True to start fresh)
    print("\n" + "="*80)
    print("INGESTING DOCUMENTS")
    print("="*80)
    rag.ingest_documents(pdf_files, clear_existing=True)
    
    # Test queries
    print("\n" + "="*80)
    print("TESTING QUERIES")
    print("="*80)
    
    queries = [
        "Give me the employee headcount for all companies",
        "What is Amazon's employee count?",
        "How many employees does each company have?",
        "Compare the employee headcount across all four companies"
    ]
    
    for query in queries:
        print("\n" + "-"*80)
        answer = rag.generate_answer(query, n_results=30)
        print(f"\n{answer}")
        print("-"*80)
    
    # Optional: Debug to see what chunks are retrieved
    print("\n" + "="*80)
    print("DEBUG MODE")
    print("="*80)
    rag.debug_chunks("employee headcount Amazon", n_results=10)


if __name__ == "__main__":
    main()


# Backward compatibility alias - ADD THIS AT THE VERY END
DocumentRAGSystem = SemanticFinancialRAG