"""
Semantic Financial RAG System
Using simple overlapping chunking and ChromaDB for vector storage
"""

import pdfplumber
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import os
from typing import List, Dict
import anthropic
import traceback


class SemanticFinancialRAG:
    def __init__(
        self,
        model_name: str = "intfloat/e5-large-v2",
        collection_name: str = "financial_documents",
        persist_directory: str = "./chroma_db"
    ):
        """Initialize the RAG system with ChromaDB"""
        
        # Initialize sentence transformer for embeddings
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        # Simple chunking parameters
        self.chunk_size = 1000
        self.chunk_overlap = 200
        
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
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY not found in environment")
        self.claude = anthropic.Anthropic(api_key=api_key)
    
    def _simple_chunk_text(self, text: str) -> List[str]:
        """Simple overlapping text chunking"""
        if not text or not text.strip():
            return []
        
        chunks = []
        start = 0
        text_len = len(text)
        
        while start < text_len:
            end = start + self.chunk_size
            chunk = text[start:end]
            
            if end < text_len:
                last_break = max(
                    chunk.rfind('. '),
                    chunk.rfind('? '),
                    chunk.rfind('! '),
                    chunk.rfind('\n')
                )
                if last_break > self.chunk_size // 2:
                    chunk = text[start:start + last_break + 1]
                    end = start + last_break + 1
            
            if chunk.strip():
                chunks.append(chunk.strip())
            
            start = end - self.chunk_overlap
            if start >= text_len:
                break
        
        return chunks if chunks else [text]
    
    def _format_table(self, table: List[List]) -> str:
        """Format table as structured text"""
        if not table:
            return ""
        
        lines = []
        for row in table:
            if not row:
                continue
            cells = [str(cell).strip() if cell is not None else "" for cell in row]
            if any(cells):
                lines.append(" | ".join(cells))
        
        return "\n".join(lines)
    
    def _extract_tables_with_context(self, page, page_num: int) -> List[Dict]:
        """Extract tables with surrounding context"""
        table_chunks = []
        
        try:
            tables = page.extract_tables()
            if not tables:
                return table_chunks
            
            for table_idx, table in enumerate(tables):
                if not table or not any(table):
                    continue
                
                table_text = self._format_table(table)
                if not table_text:
                    continue
                
                try:
                    full_page_text = page.extract_text() or ""
                    lines = full_page_text.split('\n')
                    context = ""
                    
                    for i, line in enumerate(lines):
                        if any(str(cell) in line for row in table[:2] for cell in row if cell):
                            context_lines = lines[max(0, i-2):i]
                            context = "\n".join(context_lines)
                            break
                    
                    full_table_text = f"{context}\n\n{table_text}" if context else table_text
                except:
                    full_table_text = table_text
                
                table_chunks.append({
                    'text': full_table_text,
                    'page': page_num,
                    'type': 'table',
                    'table_index': table_idx
                })
        except Exception as e:
            print(f"Warning: Error extracting tables from page {page_num}: {e}")
        
        return table_chunks
    
    def process_pdf(self, pdf_path: str) -> List[Dict]:
        """Extract and chunk a PDF"""
        print(f"\nProcessing PDF: {pdf_path}")
        all_chunks = []
        
        if not os.path.exists(pdf_path):
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")
        
        with pdfplumber.open(pdf_path) as pdf:
            total_pages = len(pdf.pages)
            print(f"Total pages: {total_pages}")
            
            for page_num, page in enumerate(pdf.pages, start=1):
                print(f"Processing page {page_num}/{total_pages}...", end='\r')
                
                # Extract tables
                table_chunks = self._extract_tables_with_context(page, page_num)
                all_chunks.extend([{
                    **chunk,
                    'source': os.path.basename(pdf_path)
                } for chunk in table_chunks])
                
                # Extract text
                page_text = page.extract_text()
                if page_text and page_text.strip():
                    page_text_with_marker = f"[Page {page_num}]\n{page_text}"
                    
                    try:
                        text_chunks = self._simple_chunk_text(page_text_with_marker)
                        
                        for chunk_idx, chunk_text in enumerate(text_chunks):
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
                        print(f"\nWarning: Error chunking page {page_num}: {e}")
                        all_chunks.append({
                            'text': page_text_with_marker,
                            'page': page_num,
                            'type': 'text',
                            'chunk_index': 0,
                            'source': os.path.basename(pdf_path)
                        })
        
        print(f"\nCreated {len(all_chunks)} chunks")
        return all_chunks
    
    def ingest_documents(self, pdf_paths: List[str], clear_existing: bool = False):
        """Ingest multiple PDFs into ChromaDB"""
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
            
            try:
                chunks = self.process_pdf(pdf_path)
                
                if not chunks:
                    print(f"Warning: No chunks from {pdf_path}")
                    continue
                
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
                    chunk_id = f"{chunk['source']}_page{chunk['page']}_{chunk['type']}_{i}"
                    ids.append(chunk_id)
                
                # Batch add
                batch_size = 100
                for i in range(0, len(documents), batch_size):
                    batch_docs = documents[i:i+batch_size]
                    batch_meta = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    
                    embeddings = self.embedder.encode(batch_docs).tolist()
                    
                    self.collection.add(
                        documents=batch_docs,
                        metadatas=batch_meta,
                        ids=batch_ids,
                        embeddings=embeddings
                    )
                
                print(f"✓ Ingested {len(chunks)} chunks from {pdf_path}")
                
            except Exception as e:
                print(f"Error processing {pdf_path}: {e}")
                traceback.print_exc()
        
        print(f"\n✓ Total chunks: {self.collection.count()}")
    
    def retrieve(self, query: str, n_results: int = 20) -> List[Dict]:
        """Retrieve relevant chunks"""
        if not query or not query.strip():
            return []
        
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=n_results,
                include=['documents', 'metadatas', 'distances']
            )
            
            if not results or not results.get('documents') or not results['documents'][0]:
                return []
            
            retrieved_chunks = []
            for i in range(len(results['documents'][0])):
                retrieved_chunks.append({
                    'text': results['documents'][0][i],
                    'metadata': results['metadatas'][0][i],
                    'distance': results['distances'][0][i]
                })
            
            return retrieved_chunks
            
        except Exception as e:
            print(f"Error retrieving: {e}")
            return []
    
    def generate_answer(self, query: str, n_results: int = 30) -> str:
        """Generate answer using Claude"""
        print(f"\nQuery: {query}")
        
        if not query or not query.strip():
            return "Please provide a valid question."
        
        try:
            chunks = self.retrieve(query, n_results=n_results)
            
            if not chunks:
                return "No relevant information found. Please upload documents first."
            
            print(f"Retrieved {len(chunks)} chunks")
            
            context_parts = []
            for chunk in chunks:
                source = chunk['metadata'].get('source', 'Unknown')
                page = chunk['metadata'].get('page', 'Unknown')
                chunk_type = chunk['metadata'].get('type', 'Unknown')
                text = chunk.get('text', '')
                
                context_parts.append(
                    f"[Source: {source}, Page: {page}, Type: {chunk_type}]\n{text}"
                )
            
            context = "\n\n---\n\n".join(context_parts)
            
            prompt = f"""You are a financial analyst assistant. Answer based on the context from earnings reports.

Context:
{context}

Question: {query}

Instructions:
- Answer directly and concisely
- Cite source and page for numbers
- Use tables for multiple companies
- Say if info is missing

Answer:"""
            
            print("Calling Claude...")
            message = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=2000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get collection stats"""
        try:
            return {
                'total_chunks': self.collection.count(),
                'collection_name': self.collection.name
            }
        except Exception as e:
            return {
                'total_chunks': 0,
                'collection_name': self.collection.name,
                'error': str(e)
            }


# Backward compatibility
DocumentRAGSystem = SemanticFinancialRAG