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
import warnings
import logging

warnings.filterwarnings('ignore', message='.*FontBBox.*')
logging.getLogger('pdfminer').setLevel(logging.ERROR)
logging.getLogger('pdfplumber').setLevel(logging.ERROR)


class SemanticFinancialRAG:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        collection_name: str = "financial_documents",
        persist_directory: str = "./chroma_db"
    ):
        """Initialize the RAG system with ChromaDB"""
        
        # Initialize sentence transformer for embeddings
        print(f"Loading embedding model: {model_name}...")
        self.embedder = SentenceTransformer(model_name)
        
        # Simple chunking parameters - smaller for better granularity
        self.chunk_size = 500
        self.chunk_overlap = 100
        
        # Initialize ChromaDB
        print("Initializing ChromaDB...")
        self.client = chromadb.PersistentClient(
            path=persist_directory,
            settings=Settings(
                anonymized_telemetry=False,
                allow_reset=True
            )
        )
        
        self.collection_name = collection_name
        
        # Create or get collection - with better error handling
        try:
            self.collection = self.client.get_collection(name=collection_name)
            print(f"Loaded existing collection: {collection_name}")
            # Verify collection is accessible
            try:
                self.collection.count()
            except Exception as e:
                print(f"Collection exists but is corrupted, recreating: {e}")
                self.client.delete_collection(name=collection_name)
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {collection_name}")
        except Exception as e:
            print(f"Collection does not exist, creating new one: {e}")
            try:
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Created new collection: {collection_name}")
            except Exception as create_error:
                print(f"Failed to create collection, trying to reset: {create_error}")
                # Last resort - delete all and recreate
                try:
                    self.client.delete_collection(name=collection_name)
                except:
                    pass
                self.collection = self.client.create_collection(
                    name=collection_name,
                    metadata={"hnsw:space": "cosine"}
                )
                print(f"Reset and created collection: {collection_name}")
        
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
        """Ingest multiple PDFs into ChromaDB with optimized batching"""
        if clear_existing:
            print("Clearing existing collection...")
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Verify collection exists and is accessible
            try:
                self.collection.count()
            except Exception as e:
                print(f"Collection not accessible, recreating: {e}")
                try:
                    self.client.delete_collection(name=self.collection_name)
                except:
                    pass
                self.collection = self.client.create_collection(
                    name=self.collection_name,
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
                
                # Optimized batch processing
                batch_size = 50
                total_batches = (len(documents) + batch_size - 1) // batch_size
                
                print(f"Generating embeddings for {len(documents)} chunks...")
                
                for i in range(0, len(documents), batch_size):
                    batch_num = i // batch_size + 1
                    print(f"  Processing batch {batch_num}/{total_batches}...", end='\r')
                    
                    batch_docs = documents[i:i+batch_size]
                    batch_meta = metadatas[i:i+batch_size]
                    batch_ids = ids[i:i+batch_size]
                    
                    # Generate embeddings
                    embeddings = self.embedder.encode(
                        batch_docs,
                        batch_size=32,
                        show_progress_bar=False,
                        normalize_embeddings=True,
                        convert_to_tensor=False
                    )
                    
                    # Ensure embeddings are in list format
                    if not isinstance(embeddings, list):
                        if hasattr(embeddings, 'tolist'):
                            embeddings = embeddings.tolist()
                        else:
                            embeddings = [
                                emb.tolist() if hasattr(emb, 'tolist') else list(emb) 
                                for emb in embeddings
                            ]
                    
                    # Verify collection before adding
                    try:
                        self.collection.count()
                    except Exception as e:
                        print(f"\nCollection lost during processing, recreating: {e}")
                        try:
                            self.collection = self.client.get_collection(name=self.collection_name)
                        except:
                            self.collection = self.client.create_collection(
                                name=self.collection_name,
                                metadata={"hnsw:space": "cosine"}
                            )
                    
                    # Add to ChromaDB
                    try:
                        self.collection.add(
                            documents=batch_docs,
                            metadatas=batch_meta,
                            ids=batch_ids,
                            embeddings=embeddings
                        )
                    except Exception as e:
                        print(f"\nError adding batch to collection: {e}")
                        # Try to get fresh reference
                        try:
                            self.collection = self.client.get_collection(name=self.collection_name)
                            self.collection.add(
                                documents=batch_docs,
                                metadatas=batch_meta,
                                ids=batch_ids,
                                embeddings=embeddings
                            )
                        except Exception as retry_error:
                            print(f"Failed to add batch even after retry: {retry_error}")
                            raise
                
                print(f"\n✓ Ingested {len(chunks)} chunks from {os.path.basename(pdf_path)}")
                
            except Exception as e:
                print(f"\nError processing {pdf_path}: {e}")
                traceback.print_exc()
        
        try:
            print(f"\n✓ Total chunks in collection: {self.collection.count()}")
        except Exception as e:
            print(f"Warning: Could not get collection count: {e}")
    
    def search_all_files(self, keyword: str) -> List[Dict]:
        """Search for a specific keyword across all files"""
        try:
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            matches = []
            for i, doc in enumerate(all_results['documents']):
                if keyword.lower() in doc.lower():
                    matches.append({
                        'text': doc,
                        'metadata': all_results['metadatas'][i],
                        'distance': 0.0
                    })
            
            print(f"Found {len(matches)} chunks containing '{keyword}'")
            return matches
            
        except Exception as e:
            print(f"Error searching: {e}")
            return []
    
    def retrieve_with_keywords(self, query: str, n_results: int = 100) -> List[Dict]:
        """Retrieve using both semantic search and keyword matching"""
        if not query or not query.strip():
            return []
        
        try:
            # Get all chunks
            all_results = self.collection.get(
                include=['documents', 'metadatas']
            )
            
            if not all_results or not all_results.get('documents'):
                return []
            
            # Extract keywords from query
            keywords = query.lower().replace('?', '').replace(',', '').split()
            # Remove common words
            stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'what', 'how', 'when', 'where', 'who', 'give', 'me', 'for', 'all', 'each', 'of', 'to', 'in', 'and', 'or', 'tell'}
            keywords = [k for k in keywords if k not in stopwords and len(k) > 2]
            
            # Score chunks by keyword matches
            keyword_matches = []
            for i, doc in enumerate(all_results['documents']):
                score = sum(1 for keyword in keywords if keyword in doc.lower())
                if score > 0:
                    keyword_matches.append({
                        'text': doc,
                        'metadata': all_results['metadatas'][i],
                        'keyword_score': score,
                        'distance': 1.0 - (score / max(len(keywords), 1))
                    })
            
            # Sort by keyword score
            keyword_matches.sort(key=lambda x: x['keyword_score'], reverse=True)
            
            # Get semantic matches
            semantic_chunks = self.retrieve(query, n_results=n_results // 2)
            
            # Combine and deduplicate
            seen_texts = set()
            combined = []
            
            # Add keyword matches first (they're often more precise)
            for chunk in keyword_matches[:n_results // 2]:
                text_key = chunk['text'][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    combined.append(chunk)
            
            # Add semantic matches
            for chunk in semantic_chunks:
                text_key = chunk['text'][:100]
                if text_key not in seen_texts:
                    seen_texts.add(text_key)
                    combined.append(chunk)
            
            print(f"Hybrid retrieval: {len(keyword_matches)} keyword matches + {len(semantic_chunks)} semantic matches = {len(combined)} total")
            return combined[:n_results]
            
        except Exception as e:
            print(f"Error in hybrid retrieval: {e}")
            traceback.print_exc()
            return self.retrieve(query, n_results)
    
    def retrieve(self, query: str, n_results: int = 100) -> List[Dict]:
        """Retrieve relevant chunks"""
        if not query or not query.strip():
            return []
        
        try:
            query_embedding = self.embedder.encode(query).tolist()
            
            total_chunks = self.collection.count()
            actual_n = min(n_results, total_chunks)
            
            results = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=actual_n,
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
    
    def generate_answer(self, query: str, n_results: int = 100) -> str:
        """Generate answer using Claude with hybrid retrieval"""
        print(f"\nQuery: {query}")
        
        if not query or not query.strip():
            return "Please provide a valid question."
        
        try:
            # Use hybrid retrieval (semantic + keyword)
            chunks = self.retrieve_with_keywords(query, n_results=n_results)
            
            if not chunks:
                return "No relevant information found. Please upload documents first."
            
            print(f"Retrieved {len(chunks)} total chunks")
            
            # Group chunks by source
            chunks_by_source = {}
            for chunk in chunks:
                source = chunk['metadata'].get('source', 'Unknown')
                if source not in chunks_by_source:
                    chunks_by_source[source] = []
                chunks_by_source[source].append(chunk)
            
            print(f"Chunks organized across {len(chunks_by_source)} documents")
            
            # Build context organized by source
            context_parts = []
            for source, source_chunks in chunks_by_source.items():
                context_parts.append(f"\n=== Document: {source} ===")
                # Take top chunks per source
                for chunk in source_chunks[:50]:
                    page = chunk['metadata'].get('page', '?')
                    chunk_type = chunk['metadata'].get('type', '?')
                    text = chunk.get('text', '')
                    
                    context_parts.append(
                        f"[Page {page}, Type: {chunk_type}]\n{text}"
                    )
            
            context = "\n\n---\n\n".join(context_parts)
            
            # Enhanced prompt
            prompt = f"""You are a financial analyst assistant. Answer the question based on the context from earnings reports.

CRITICAL INSTRUCTIONS:
1. The context below contains information from MULTIPLE companies
2. Search through ALL documents and ALL pages provided
3. Answer for EVERY company that has relevant data
4. Create a comprehensive table comparing all companies
5. Cite the source filename and page number for each data point
6. If a company's data is not in the context, state "Data not found in documents"

Context from all documents:
{context}

Question: {query}

Provide a thorough answer that covers all companies found in the context.

Answer:"""
            
            print("Calling Claude...")
            message = self.claude.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=4000,
                messages=[{"role": "user", "content": prompt}]
            )
            
            return message.content[0].text
            
        except Exception as e:
            print(f"Error: {e}")
            traceback.print_exc()
            return f"Error: {str(e)}"
    
    def get_stats(self) -> Dict:
        """Get collection stats with error handling"""
        try:
            count = self.collection.count()
            return {
                'total_chunks': count,
                'collection_name': self.collection_name
            }
        except Exception as e:
            print(f"Error getting stats, recreating collection: {e}")
            # Collection is broken, recreate it
            try:
                self.client.delete_collection(name=self.collection_name)
            except:
                pass
            self.collection = self.client.create_collection(
                name=self.collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            return {
                'total_chunks': 0,
                'collection_name': self.collection_name,
                'status': 'recreated'
            }


# Backward compatibility
DocumentRAGSystem = SemanticFinancialRAG