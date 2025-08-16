from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import os
import uuid
import tempfile
import shutil
import logging
from datetime import datetime
import sqlite3
import re
import math
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="RAG Chatbot API",
    description="A production-ready RAG chatbot using Google Gemini AI",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class ModelName:
    GEMINI_PRO = "gemini-pro"
    GEMINI_PRO_VISION = "gemini-pro-vision"

class QueryInput(BaseModel):
    question: str
    session_id: Optional[str] = None
    model: str = "gemini-pro"

class QueryResponse(BaseModel):
    answer: str
    session_id: str
    model: str

class DocumentInfo(BaseModel):
    id: int
    filename: str
    upload_timestamp: datetime

class DeleteFileRequest(BaseModel):
    file_id: int

class HealthCheck(BaseModel):
    status: str
    message: str

# Enhanced Search System with Smart Chunking
class AdvancedSearchEngine:
    def __init__(self):
        self.stopwords = {
            'a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from',
            'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the',
            'to', 'was', 'were', 'will', 'with', 'the', 'this', 'but', 'they',
            'have', 'had', 'what', 'said', 'each', 'which', 'their', 'time',
            'if', 'up', 'out', 'many', 'then', 'them', 'these', 'so', 'some',
            'her', 'would', 'make', 'like', 'into', 'him', 'two', 'more',
            'very', 'when', 'can', 'get', 'no', 'may', 'new', 'years', 'way',
            'could', 'there', 'use', 'your', 'work', 'life', 'only', 'his',
            'also', 'back', 'after', 'first', 'well', 'just', 'being', 'now',
            'made', 'before', 'here', 'through', 'how', 'much', 'should', 'our'
        }
        
    def preprocess_text(self, text):
        """Clean and preprocess text for analysis."""
        if not text:
            return ""
        
        # Convert to lowercase and remove special characters
        text = re.sub(r'[^\w\s]', ' ', text.lower())
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    
    def tokenize(self, text):
        """Tokenize text and remove stopwords."""
        if not text:
            return []
        
        words = self.preprocess_text(text).split()
        # Remove stopwords and short words
        return [word for word in words if word not in self.stopwords and len(word) > 2]
    
    def create_smart_chunks(self, content, filename, chunk_size=800, overlap=200):
        """Create smart overlapping chunks from long documents."""
        if not content or len(content.strip()) < 100:
            return [{'text': content, 'filename': filename, 'chunk_id': 0, 'start_pos': 0}]
        
        chunks = []
        
        # Method 1: Split by sections/headings first
        sections = self.detect_sections(content)
        
        if len(sections) > 1:
            print(f"📄 Found {len(sections)} sections in {filename}")
            for i, section in enumerate(sections):
                if len(section['content']) > chunk_size:
                    # Further split large sections
                    sub_chunks = self.split_by_words(section['content'], chunk_size, overlap)
                    for j, chunk_text in enumerate(sub_chunks):
                        chunks.append({
                            'text': chunk_text,
                            'filename': filename,
                            'chunk_id': f"section_{i}_chunk_{j}",
                            'start_pos': section['start_pos'],
                            'section_title': section['title']
                        })
                else:
                    chunks.append({
                        'text': section['content'],
                        'filename': filename,
                        'chunk_id': f"section_{i}",
                        'start_pos': section['start_pos'],
                        'section_title': section['title']
                    })
        else:
            # Method 2: Split by words with overlap
            word_chunks = self.split_by_words(content, chunk_size, overlap)
            for i, chunk_text in enumerate(word_chunks):
                chunks.append({
                    'text': chunk_text,
                    'filename': filename,
                    'chunk_id': f"chunk_{i}",
                    'start_pos': i * (chunk_size - overlap) * 6  # Rough estimate
                })
        
        print(f"📊 Created {len(chunks)} chunks for {filename}")
        return chunks
    
    def detect_sections(self, content):
        """Detect sections in content based on headings, page markers, etc."""
        sections = []
        lines = content.split('\n')
        current_section = {'title': 'Introduction', 'content': '', 'start_pos': 0}
        
        for i, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
                
            # Detect headings and section markers
            is_heading = (
                line.startswith('---') or  # Page markers
                (len(line) < 100 and line.isupper() and len(line.split()) <= 10) or  # ALL CAPS headings
                re.match(r'^\d+\.?\s', line) and len(line) < 80 or  # Numbered sections
                re.match(r'^Chapter \d+', line, re.IGNORECASE) or  # Chapters
                re.match(r'^Section \d+', line, re.IGNORECASE) or  # Sections
                re.match(r'^[A-Z][A-Za-z\s]+:$', line) or  # Title with colon
                (line.startswith('PAGE ') and line.replace('PAGE ', '').strip().isdigit())  # Page markers
            )
            
            if is_heading and len(current_section['content']) > 200:
                # Save current section
                sections.append(current_section)
                # Start new section
                current_section = {
                    'title': line,
                    'content': '',
                    'start_pos': content.find(line)
                }
            else:
                current_section['content'] += line + '\n'
        
        # Add the last section
        if current_section['content']:
            sections.append(current_section)
        
        return sections if len(sections) > 1 else [{'title': 'Document', 'content': content, 'start_pos': 0}]
    
    def split_by_words(self, text, chunk_size=800, overlap=200):
        """Split text into overlapping word-based chunks."""
        words = text.split()
        if len(words) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + chunk_size, len(words))
            chunk_words = words[start:end]
            chunk_text = ' '.join(chunk_words)
            chunks.append(chunk_text)
            
            if end >= len(words):
                break
                
            start = end - overlap  # Create overlap
        
        return chunks
    
    def calculate_tf_idf_chunks(self, chunks, query_terms):
        """Calculate TF-IDF scores for chunks instead of whole documents."""
        chunk_scores = {}
        total_chunks = len(chunks)
        
        # Calculate IDF for query terms
        idf_scores = {}
        for term in query_terms:
            chunk_freq = sum(1 for chunk in chunks if term in self.preprocess_text(chunk['text']))
            if chunk_freq > 0:
                idf_scores[term] = math.log(total_chunks / chunk_freq)
            else:
                idf_scores[term] = 0
        
        # Calculate TF-IDF for each chunk
        for i, chunk in enumerate(chunks):
            content = chunk['text']
            if not content or len(content.strip()) < 10:
                continue
                
            content_tokens = self.tokenize(content)
            if not content_tokens:
                continue
                
            # Calculate term frequency
            tf_counter = Counter(content_tokens)
            doc_length = len(content_tokens)
            
            tf_idf_score = 0
            matched_terms = 0
            
            for term in query_terms:
                if term in tf_counter:
                    tf = tf_counter[term] / doc_length
                    tf_idf = tf * idf_scores.get(term, 0)
                    tf_idf_score += tf_idf
                    matched_terms += 1
            
            if matched_terms > 0:
                # Normalize by number of matched terms
                chunk_scores[i] = {
                    'score': tf_idf_score / matched_terms,
                    'matched_terms': matched_terms,
                    'chunk': chunk
                }
        
        return chunk_scores
    
    def fuzzy_match_score(self, query, text, threshold=0.6):
        """Calculate fuzzy matching score for handling typos."""
        if not query or not text:
            return 0
        
        query_words = self.tokenize(query)
        text_words = self.tokenize(text)
        
        if not query_words or not text_words:
            return 0
        
        total_score = 0
        matched_words = 0
        
        for q_word in query_words:
            best_match_score = 0
            for t_word in text_words:
                similarity = SequenceMatcher(None, q_word, t_word).ratio()
                if similarity > best_match_score:
                    best_match_score = similarity
            
            if best_match_score >= threshold:
                total_score += best_match_score
                matched_words += 1
        
        return total_score / len(query_words) if query_words else 0
    
    def semantic_similarity(self, query, text):
        """Simple semantic similarity based on word co-occurrence."""
        if not query or not text:
            return 0
        
        query_words = set(self.tokenize(query))
        text_words = set(self.tokenize(text))
        
        if not query_words or not text_words:
            return 0
        
        # Jaccard similarity
        intersection = len(query_words.intersection(text_words))
        union = len(query_words.union(text_words))
        
        jaccard = intersection / union if union > 0 else 0
        
        # Boost score if query words appear close together in text
        proximity_boost = self.calculate_proximity_score(query, text)
        
        return jaccard + (proximity_boost * 0.3)
    
    def calculate_proximity_score(self, query, text):
        """Calculate how close query terms appear in the text."""
        query_words = self.tokenize(query)
        text_words = self.tokenize(text)
        
        if len(query_words) < 2 or not text_words:
            return 0
        
        # Find positions of query words in text
        word_positions = defaultdict(list)
        for i, word in enumerate(text_words):
            if word in query_words:
                word_positions[word].append(i)
        
        # Calculate proximity scores
        proximity_scores = []
        for i in range(len(query_words) - 1):
            word1, word2 = query_words[i], query_words[i + 1]
            if word1 in word_positions and word2 in word_positions:
                min_distance = float('inf')
                for pos1 in word_positions[word1]:
                    for pos2 in word_positions[word2]:
                        distance = abs(pos2 - pos1)
                        min_distance = min(min_distance, distance)
                
                if min_distance < float('inf'):
                    # Closer words get higher scores
                    proximity_scores.append(1.0 / (1.0 + min_distance))
        
        return sum(proximity_scores) / len(proximity_scores) if proximity_scores else 0
    
    def extract_relevant_passages(self, query, content, max_passages=2):
        """Extract the most relevant passages from content."""
        if not content or len(content.strip()) < 50:
            return [content] if content else []
        
        # Split content into sentences
        sentences = re.split(r'[.!?]+\s+', content)
        if len(sentences) <= 3:
            return [content]
        
        # Score each sentence
        sentence_scores = []
        for i, sentence in enumerate(sentences):
            if len(sentence.strip()) > 20:
                score = self.semantic_similarity(query, sentence)
                sentence_scores.append((score, i, sentence))
        
        # Sort by score and get top sentences
        sentence_scores.sort(reverse=True)
        
        # Extract passages around top sentences
        passages = []
        used_indices = set()
        
        for score, idx, sentence in sentence_scores[:max_passages * 2]:
            if idx in used_indices:
                continue
                
            # Create passage with context
            start_idx = max(0, idx - 1)
            end_idx = min(len(sentences), idx + 2)
            
            passage_sentences = sentences[start_idx:end_idx]
            passage = '. '.join(passage_sentences) + '.'
            
            if len(passage.strip()) > 30:
                passages.append(passage)
                used_indices.update(range(start_idx, end_idx))
                
                if len(passages) >= max_passages:
                    break
        
        return passages if passages else [content[:500] + "..." if len(content) > 500 else content]
    
    def hybrid_search_with_chunks(self, query, documents):
        """Perform hybrid search with smart chunking for long documents."""
        if not query or not documents:
            return []
        
        query_terms = self.tokenize(query)
        if not query_terms:
            return []
        
        print(f"🔍 Smart chunking search for: '{query[:50]}...'")
        
        # Step 1: Create chunks for all documents
        all_chunks = []
        for filename, content in documents:
            if content and len(content.strip()) > 10:
                chunks = self.create_smart_chunks(content, filename)
                all_chunks.extend(chunks)
        
        print(f"📊 Created {len(all_chunks)} total chunks from {len(documents)} documents")
        
        if not all_chunks:
            return []
        
        # Step 2: Calculate TF-IDF scores for chunks
        tfidf_scores = self.calculate_tf_idf_chunks(all_chunks, query_terms)
        
        # Step 3: Process chunks with multiple scoring methods
        results = []
        
        for chunk_idx, chunk_data in tfidf_scores.items():
            chunk = chunk_data['chunk']
            content = chunk['text']
            
            scores = {
                'tfidf': chunk_data['score'],
                'semantic': self.semantic_similarity(query, content),
                'fuzzy': self.fuzzy_match_score(query, content),
                'exact_match': self.exact_phrase_match(query, content)
            }
            
            # Weighted combination of scores
            final_score = (
                scores['tfidf'] * 0.35 +
                scores['semantic'] * 0.35 +
                scores['fuzzy'] * 0.2 +
                scores['exact_match'] * 0.1
            )
            
            if final_score > 0.01:  # Lower threshold for chunks
                # Extract relevant passages from this chunk
                relevant_passages = self.extract_relevant_passages(query, content)
                
                results.append({
                    'filename': chunk['filename'],
                    'content': content,
                    'score': final_score,
                    'passages': relevant_passages,
                    'matched_terms': chunk_data['matched_terms'],
                    'chunk_id': chunk['chunk_id'],
                    'section_title': chunk.get('section_title', 'Content'),
                    'individual_scores': scores
                })
        
        # Sort by final score
        results.sort(key=lambda x: x['score'], reverse=True)
        
        print(f"🎯 Found {len(results)} relevant chunks")
        return results[:10]  # Return top 10 chunks
    
    def exact_phrase_match(self, query, text):
        """Check for exact phrase matches."""
        if not query or not text:
            return 0
        
        query_clean = self.preprocess_text(query)
        text_clean = self.preprocess_text(text)
        
        # Check for exact phrase match
        if query_clean in text_clean:
            return 1.0
        
        # Check for partial phrase matches
        query_words = query_clean.split()
        if len(query_words) > 1:
            for i in range(len(query_words) - 1):
                phrase = ' '.join(query_words[i:i+2])
                if phrase in text_clean:
                    return 0.5
        
        return 0

# Initialize the search engine
search_engine = AdvancedSearchEngine()

# Database functions
def init_database():
    """Initialize SQLite database with proper schema."""
    conn = sqlite3.connect("rag_app.db")
    
    # Create tables
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  session_id TEXT,
                  user_query TEXT,
                  gpt_response TEXT,
                  model TEXT,
                  created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                 (id INTEGER PRIMARY KEY AUTOINCREMENT,
                  filename TEXT,
                  content TEXT,
                  upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    
    # Check if content column exists, if not add it (for existing databases)
    cursor = conn.cursor()
    cursor.execute("PRAGMA table_info(document_store)")
    columns = [column[1] for column in cursor.fetchall()]
    if 'content' not in columns:
        print("🔧 Adding content column to existing database...")
        cursor.execute('ALTER TABLE document_store ADD COLUMN content TEXT')
        print("✅ Content column added successfully!")
    
    conn.commit()
    conn.close()
    print("📚 Database initialized with Smart Chunking RAG support")

def insert_document_record(filename, content=""):
    """Insert document record with content and return ID."""
    try:
        conn = sqlite3.connect("rag_app.db")
        cursor = conn.cursor()
        cursor.execute('INSERT INTO document_store (filename, content) VALUES (?, ?)', (filename, content))
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        print(f"💾 Saved document: {filename} with {len(content)} chars of content")
        return file_id
    except Exception as e:
        print(f"❌ Database insert error: {e}")
        raise e

def get_all_documents():
    """Get all documents."""
    conn = sqlite3.connect("rag_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
    documents = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return documents

def search_documents(query):
    """Enhanced search with smart chunking for long documents."""
    try:
        conn = sqlite3.connect("rag_app.db")
        cursor = conn.cursor()
        cursor.execute('SELECT filename, content FROM document_store WHERE content IS NOT NULL AND content != ""')
        results = cursor.fetchall()
        conn.close()

        print(f"🔍 Smart chunking search for: '{query[:50]}...'")
        print(f"📚 Processing {len(results)} documents with smart chunking")

        if not results:
            print("📄 No documents with content found")
            return ""

        # Use the enhanced search engine with smart chunking
        search_results = search_engine.hybrid_search_with_chunks(query, results)
        
        if not search_results:
            print("🔍 No relevant content found with smart chunking search")
            return ""

        print(f"🎯 Found {len(search_results)} relevant chunks")

        # Format results for the chat system
        formatted_content = []
        
        for i, result in enumerate(search_results[:5]):  # Top 5 results
            filename = result['filename']
            score = result['score']
            passages = result['passages']
            matched_terms = result['matched_terms']
            chunk_id = result['chunk_id']
            section_title = result['section_title']
            
            print(f"📄 Chunk {i+1}: {filename} - {section_title} (Score: {score:.3f}, Terms: {matched_terms})")
            
            # Use the best passages, fallback to truncated content
            if passages:
                content_preview = '\n---\n'.join(passages)
            else:
                content_preview = result['content'][:600] + "..." if len(result['content']) > 600 else result['content']
            
            # Add score and relevance info for debugging
            debug_info = f"[Score: {score:.3f}, Section: {section_title}, Chunk: {chunk_id}]"
            
            formatted_content.append(
                f"📄 **From '{filename}'** - {section_title} {debug_info}\n{content_preview}\n"
            )

        final_content = "\n" + "="*60 + "\n" + "\n".join(formatted_content)
        
        print(f"🎯 Returning {len(final_content)} chars of smart chunking search results")
        return final_content

    except Exception as e:
        print(f"❌ Smart chunking search error: {e}")
        return ""

def delete_document_record(file_id):
    """Delete document record."""
    conn = sqlite3.connect("rag_app.db")
    cursor = conn.cursor()
    cursor.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
    conn.commit()
    conn.close()
    return True

def get_document_by_id(file_id):
    """Get document by ID."""
    conn = sqlite3.connect("rag_app.db")
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute('SELECT id, filename, upload_timestamp FROM document_store WHERE id = ?', (file_id,))
    result = cursor.fetchone()
    conn.close()
    return dict(result) if result else None

# Initialize database
init_database()

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting Smart Chunking RAG Chatbot API...")
    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak")
        
        # Try different model names
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'models/gemini-1.5-flash', 'gemini-pro']
        success = False
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Hello")
                logger.info(f"✅ Google Gemini AI connection successful with model: {model_name}")
                success = True
                break
            except Exception as model_error:
                logger.warning(f"⚠️ Model {model_name} failed: {model_error}")
                continue
        
        if not success:
            logger.error("❌ All Gemini models failed, but API will continue running")
    except Exception as e:
        logger.error(f"❌ Google Gemini AI setup failed: {e} - API will continue running")

# Root endpoint
@app.get("/")
def read_root():
    """Root endpoint with API information."""
    return {
        "message": "🤖 Smart Chunking RAG Chatbot API with Google Gemini AI",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health",
        "status": "✅ RUNNING",
        "rag_status": "📚 Upload ANY length document and ask questions about ANY page!",
        "search_engine": "🧠 Smart Chunking + Hybrid Search (TF-IDF + Semantic + Fuzzy)",
        "endpoints": ["/health", "/chat", "/upload-doc", "/list-docs", "/searchable-docs", "/delete-doc", "/stats"],
        "features": [
            "✅ Document upload (PDF, DOCX, TXT, HTML)",
            "🧠 Smart chunking for long documents",
            "📄 Section-based and overlapping chunks",
            "🔍 Searches ALL pages and sections",
            "✅ Hybrid search (TF-IDF + Semantic + Fuzzy)",
            "✅ Smart passage extraction",
            "✅ Multi-strategy ranking fusion",
            "✅ Chat with uploaded documents",
            "✅ Google Gemini AI integration"
        ]
    }

# Simple test endpoint
@app.get("/test")
def test_endpoint():
    """Simple test endpoint to verify server is running."""
    return {
        "status": "✅ SERVER IS RUNNING",
        "message": "FastAPI server is working perfectly!",
        "timestamp": datetime.now().isoformat()
    }

# Debug endpoint to check document content
@app.get("/debug-docs")
def debug_documents():
    """Debug endpoint to see what content is stored in documents."""
    try:
        conn = sqlite3.connect("rag_app.db")
        cursor = conn.cursor()
        cursor.execute('''SELECT filename,
                         CASE 
                           WHEN content IS NULL THEN 'NULL'
                           WHEN content = '' THEN 'EMPTY STRING'
                           WHEN length(content) < 50 THEN content
                           ELSE substr(content, 1, 100) || '...'
                         END as content_preview,
                         length(content) as content_length
                         FROM document_store 
                         ORDER BY upload_timestamp DESC''')
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "filename": row[0],
                "content_preview": row[1],
                "content_length": row[2] if row[2] else 0
            })
        
        conn.close()
        
        return {
            "total_documents": len(results),
            "documents": results,
            "debug_info": "This shows what content is actually stored in the database",
            "search_engine_status": "🧠 Smart Chunking Search Engine Active"
        }
    except Exception as e:
        return {"error": str(e)}

# Health check endpoint
@app.get("/health", response_model=HealthCheck)
def health_check():
    """Check the health of the API and its dependencies."""
    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak")
        
        # Try different model names
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'models/gemini-1.5-flash']
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content("Test")
                return HealthCheck(status="healthy", message=f"API is running. Google Gemini AI working with {model_name}. Smart chunking search engine active for long documents.")
            except:
                continue
        
        return HealthCheck(status="degraded", message="API is running but Gemini AI connection failed")
    except Exception as e:
        return HealthCheck(status="degraded", message=f"API is running but Gemini AI failed: {str(e)}")

# Chat endpoint with enhanced search
@app.post("/chat", response_model=QueryResponse)
def chat(query_input: QueryInput):
    """Process a chat query using Google Gemini AI with smart chunking RAG."""
    session_id = query_input.session_id or str(uuid.uuid4())
    logger.info(f"💬 SMART CHUNKING CHAT START: {query_input.question}")

    try:
        import google.generativeai as genai
        genai.configure(api_key="AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak")

        # 🔍 STEP 1: Smart chunking search through uploaded documents
        print(f"🧠 SMART CHUNKING SEARCH FOR: '{query_input.question[:50]}...'")
        relevant_content = search_documents(query_input.question)

        # 🧠 STEP 2: Create enhanced prompt based on search results
        if relevant_content and len(relevant_content.strip()) > 20:
            print(f"📚 FOUND CONTENT: {len(relevant_content)} chars - USING SMART CHUNKING RAG MODE")
            
            # Create enhanced RAG prompt with better context handling
            prompt = f"""You are a helpful AI assistant with access to specific documents. I've used smart chunking to search through ALL pages and sections of the uploaded documents. Answer the question based on the provided context.

The search used intelligent chunking to break down long documents and find the most relevant sections from ANY page.

CONTEXT FROM SMART CHUNKING SEARCH (ALL PAGES COVERED):
{relevant_content}

QUESTION: {query_input.question}

INSTRUCTIONS:
- Provide a thorough answer based on the chunked content from all document sections
- Reference specific documents and sections when citing information
- If you find relevant information from multiple chunks/sections, synthesize them coherently
- Include specific details and examples from the documents
- Be specific about which documents or sections contain different pieces of information

COMPREHENSIVE ANSWER:"""
        else:
            print("📄 NO RELEVANT CONTENT - USING GENERAL MODE")
            
            # No relevant documents found, use general prompt
            prompt = f"""I performed a smart chunking search through all uploaded documents (covering every page and section) but didn't find specific information relevant to your question.

Question: {query_input.question}

I'll provide a general answer based on my knowledge. For specific information from your documents, please ensure you have uploaded relevant files.

General answer:"""

        # 🤖 STEP 3: Get response from Gemini AI
        model_names = ['gemini-1.5-flash', 'gemini-1.5-pro', 'models/gemini-1.5-flash']
        answer = None
        
        for model_name in model_names:
            try:
                model = genai.GenerativeModel(model_name)
                response = model.generate_content(prompt)
                answer = response.text
                print(f"✅ SMART CHUNKING GEMINI RESPONSE: {len(answer)} chars using {model_name}")
                break
            except Exception as model_error:
                print(f"⚠️ MODEL {model_name} FAILED: {model_error}")
                continue

        if not answer:
            answer = f"I'm having trouble connecting to Google Gemini AI. Please try again."

        # 🏷️ STEP 4: Enhanced labeling for user understanding
        if relevant_content and len(relevant_content.strip()) > 20:
            final_answer = f"🧠 **Smart Chunking Document Analysis:**\n\n{answer}\n\n---\n*Analysis based on smart chunking search through ALL pages and sections using intelligent document splitting, TF-IDF scoring, semantic analysis, and section-aware chunking.*"
            print("🎯 FINAL: SMART CHUNKING RAG ANSWER DELIVERED")
        else:
            final_answer = f"💡 **General Knowledge Answer** (no relevant documents found):\n\n{answer}\n\n💡 *Note: Smart chunking search was performed across all uploaded documents and pages. Upload relevant documents for comprehensive document-based answers.*"
            print("🎯 FINAL: GENERAL ANSWER DELIVERED")

        return QueryResponse(
            answer=final_answer,
            session_id=session_id,
            model=query_input.model
        )

    except Exception as e:
        print(f"❌ SMART CHUNKING CHAT ERROR: {e}")
        error_response = f"I apologize, but I encountered an error: {str(e)}"
        return QueryResponse(
            answer=error_response,
            session_id=session_id,
            model=query_input.model
        )

# Document upload endpoint
@app.post("/upload-doc")
async def upload_and_index_document(file: UploadFile = File(...)):
    """Upload and index a document with smart chunking support."""
    logger.info(f"📁 SMART CHUNKING UPLOAD START: {file.filename}")
    
    try:
        # Check file type
        allowed_extensions = ['.pdf', '.docx', '.html', '.txt']
        file_extension = os.path.splitext(file.filename)[1].lower()
        
        if file_extension not in allowed_extensions:
            logger.error(f"❌ UNSUPPORTED FILE: {file_extension}")
            raise HTTPException(
                status_code=400, 
                detail=f"Unsupported file type '{file_extension}'. Allowed: {', '.join(allowed_extensions)}"
            )
        
        logger.info(f"✅ FILE TYPE OK: {file_extension}")
        
        # Read file content
        try:
            content = await file.read()
            logger.info(f"📖 FILE READ: {len(content)} bytes")
        except Exception as e:
            logger.error(f"❌ FILE READ ERROR: {e}")
            raise HTTPException(status_code=400, detail=f"Could not read file: {str(e)}")
        
        # Create temporary file for processing
        temp_file_path = None
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
                temp_file_path = temp_file.name
                temp_file.write(content)
                temp_file.flush()
                logger.info(f"💾 TEMP FILE CREATED: {temp_file_path}")
                
                # Extract content BEFORE database insert so we can save it
                content_text = ""
                try:
                    print(f"🔍 EXTRACTING CONTENT FROM: {file_extension}")
                    
                    if file_extension == '.txt':
                        with open(temp_file_path, 'r', encoding='utf-8') as f:
                            content_text = f.read()
                        print(f"📄 TXT CONTENT EXTRACTED: {len(content_text)} chars")
                        
                    elif file_extension == '.pdf':
                        try:
                            import pypdf
                            with open(temp_file_path, 'rb') as f:
                                reader = pypdf.PdfReader(f)
                                print(f"📄 PDF HAS {len(reader.pages)} PAGES")
                                for i, page in enumerate(reader.pages):
                                    page_text = page.extract_text()
                                    # Add page markers for better chunking
                                    content_text += f"\n--- PAGE {i+1} ---\n{page_text}\n"
                                    print(f"📄 PAGE {i+1}: {len(page_text)} chars")
                            print(f"📄 PDF TOTAL CONTENT: {len(content_text)} chars")
                        except Exception as pdf_error:
                            print(f"⚠️ PDF EXTRACTION FAILED: {pdf_error}")
                            content_text = f"PDF file uploaded: {file.filename}\nContent extraction failed: {str(pdf_error)}"
                            
                    elif file_extension == '.docx':
                        try:
                            import docx2txt
                            content_text = docx2txt.process(temp_file_path)
                            print(f"📄 DOCX CONTENT EXTRACTED: {len(content_text)} chars")
                        except Exception as docx_error:
                            print(f"⚠️ DOCX EXTRACTION FAILED: {docx_error}")
                            content_text = f"DOCX file uploaded: {file.filename}\nContent extraction failed: {str(docx_error)}"
                            
                    elif file_extension == '.html':
                        with open(temp_file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        # Simple HTML tag removal
                        import re
                        content_text = re.sub('<[^<]+?>', '', html_content)
                        print(f"📄 HTML CONTENT EXTRACTED: {len(content_text)} chars")
                        
                    else:
                        content_text = f"Document uploaded: {file.filename}\nFile type: {file_extension}"
                        print(f"📄 GENERIC CONTENT: {len(content_text)} chars")
                    
                    # Ensure we have meaningful content
                    if len(content_text.strip()) < 10:
                        content_text = f"File uploaded: {file.filename}\nWarning: Very little text content extracted."
                        print("⚠️ WARNING: Very little content extracted")
                    else:
                        print(f"✅ CONTENT READY: {len(content_text)} chars")
                    
                    # Show first 100 chars for debugging
                    preview = content_text[:100].replace('\n', ' ')
                    print(f"📖 CONTENT PREVIEW: '{preview}...'")
                    
                    # Test chunking
                    test_chunks = search_engine.create_smart_chunks(content_text, file.filename)
                    print(f"🧠 Created {len(test_chunks)} smart chunks for search")
                    
                except Exception as extract_error:
                    print(f"❌ CONTENT EXTRACTION ERROR: {extract_error}")
                    content_text = f"File uploaded: {file.filename}\nContent extraction failed: {str(extract_error)}"
                
                # Insert document record with content
                try:
                    file_id = insert_document_record(file.filename, content_text)
                    logger.info(f"✅ DATABASE INSERT: file_id={file_id}, content_length={len(content_text)}")
                except Exception as e:
                    logger.error(f"❌ DATABASE ERROR: {e}")
                    raise HTTPException(status_code=500, detail=f"Database error: {str(e)}")
                
                # Success response
                logger.info(f"🎉 UPLOAD SUCCESS: {file.filename} (ID: {file_id}) - Smart chunking ready!")
                
                return {
                    "message": f"✅ Successfully uploaded '{file.filename}' with smart chunking!",
                    "file_id": file_id,
                    "filename": file.filename,
                    "content_length": len(content_text),
                    "chunks_created": len(test_chunks),
                    "status": "success",
                    "search_features": "🧠 Smart chunking with section detection + TF-IDF + Semantic + Fuzzy matching",
                    "note": "📚 You can now ask questions about ANY page or section of this document!"
                }
                
        except HTTPException:
            # Don't catch HTTPExceptions, let them pass through
            raise
        except Exception as e:
            logger.error(f"❌ PROCESSING ERROR: {e}")
            # Try to clean up database record if it was created
            if 'file_id' in locals():
                try:
                    delete_document_record(file_id)
                    logger.info(f"🧹 CLEANED UP DATABASE RECORD: {file_id}")
                except:
                    pass
            raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")
            
    except HTTPException:
        # Let HTTPExceptions pass through
        raise
    except Exception as e:
        logger.error(f"❌ UNEXPECTED ERROR: {e}")
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")
    finally:
        # Always clean up temp file
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.remove(temp_file_path)
                logger.info(f"🧹 TEMP FILE CLEANED: {temp_file_path}")
            except Exception as cleanup_error:
                logger.warning(f"⚠️ CLEANUP ERROR: {cleanup_error}")

# List documents endpoint
@app.get("/list-docs", response_model=List[DocumentInfo])
def list_documents():
    """Get a list of all indexed documents."""
    try:
        documents = get_all_documents()
        logger.info(f"Retrieved {len(documents)} documents")
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to retrieve documents: {str(e)}")

# Delete document endpoint
@app.post("/delete-doc")
def delete_document(request: DeleteFileRequest):
    """Delete a document from the system."""
    logger.info(f"Document deletion request - File ID: {request.file_id}")
    
    try:
        # Check if document exists
        document = get_document_by_id(request.file_id)
        if not document:
            raise HTTPException(status_code=404, detail=f"Document with ID {request.file_id} not found")
        
        # Delete from database
        db_delete_success = delete_document_record(request.file_id)
        
        if db_delete_success:
            logger.info(f"Document deleted successfully - ID: {request.file_id}")
            return {
                "message": f"Successfully deleted document '{document['filename']}' (ID: {request.file_id})"
            }
        else:
            raise HTTPException(status_code=500, detail=f"Failed to delete document ID {request.file_id}")
            
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {e}")
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# Get statistics
@app.get("/stats")
def get_stats():
    """Get system statistics."""
    try:
        documents = get_all_documents()
        
        # Count documents with content
        conn = sqlite3.connect("rag_app.db")
        cursor = conn.cursor()
        cursor.execute('SELECT COUNT(*) FROM document_store WHERE content IS NOT NULL AND content != ""')
        docs_with_content = cursor.fetchone()[0]
        conn.close()
        
        return {
            "total_documents": len(documents),
            "documents_with_searchable_content": docs_with_content,
            "api_status": "running",
            "search_engine": "🧠 Smart Chunking Hybrid Search Active",
            "rag_status": "✅ Ready - Upload ANY length document and ask questions about ANY page!",
            "features": [
                "🧠 Smart chunking for long documents",
                "📄 Section detection and splitting", 
                "🔍 Overlapping chunks for complete coverage",
                "📊 TF-IDF scoring for chunk relevance",
                "💡 Semantic similarity matching",
                "🔤 Fuzzy matching for typos",
                "📝 Smart passage extraction",
                "⚡ Multi-strategy ranking fusion"
            ],
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error getting stats: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get statistics: {str(e)}")

# New endpoint to show searchable content
@app.get("/searchable-docs")
def get_searchable_docs():
    """Get list of documents with searchable content."""
    try:
        conn = sqlite3.connect("rag_app.db")
        cursor = conn.cursor()
        cursor.execute('''SELECT filename,
                         CASE 
                           WHEN length(content) > 100 THEN substr(content, 1, 100) || '...'
                           ELSE content
                         END as preview,
                         length(content) as content_length
                         FROM document_store 
                         WHERE content IS NOT NULL AND content != ""
                         ORDER BY upload_timestamp DESC''')
        
        results = []
        for row in cursor.fetchall():
            # Estimate chunks that would be created
            content_length = row[2]
            estimated_chunks = max(1, content_length // 600)  # Rough estimate
            
            results.append({
                "filename": row[0],
                "content_preview": row[1],
                "content_length": content_length,
                "estimated_chunks": estimated_chunks,
                "search_ready": "🧠 Smart chunking enabled"
            })
        
        conn.close()
        
        return {
            "searchable_documents": results,
            "count": len(results),
            "search_engine_info": "🧠 Smart chunking hybrid search with section detection",
            "message": "📚 These documents are indexed with smart chunking for comprehensive search!"
        }
    except Exception as e:
        logger.error(f"Error getting searchable docs: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to get searchable documents: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main_working:app", host="0.0.0.0", port=8000, reload=True)