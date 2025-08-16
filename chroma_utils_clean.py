# Document processing without langchain dependencies
import os
import pickle

from typing import List
from sentence_transformers import SentenceTransformer

# Simple document class
class Document:
    def __init__(self, page_content, metadata=None):
        self.page_content = page_content
        self.metadata = metadata or {}

# Simple document loaders
class PyPDFLoader:
    def __init__(self, file_path): 
        self.file_path = file_path
    
    def load(self):
        try:
            import pypdf
            text = ""
            with open(self.file_path, 'rb') as file:
                reader = pypdf.PdfReader(file)
                for page in reader.pages:
                    text += page.extract_text() + "\n"
            return [Document(page_content=text, metadata={"source": self.file_path})]
        except Exception as e:
            print(f"Error loading PDF: {e}")
            return []

class Docx2txtLoader:
    def __init__(self, file_path): 
        self.file_path = file_path  
    
    def load(self):
        try:
            import docx2txt
            text = docx2txt.process(self.file_path)
            return [Document(page_content=text, metadata={"source": self.file_path})]
        except Exception as e:
            print(f"Error loading DOCX: {e}")
            return []

# Simple text splitter
class RecursiveCharacterTextSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200, length_function=len, separators=None):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.length_function = length_function
        self.separators = separators or ["\n\n", "\n", " ", ""]
    
    def split_documents(self, documents):
        splits = []
        for doc in documents:
            text = doc.page_content
            chunks = self._split_text(text)
            for chunk in chunks:
                splits.append(Document(
                    page_content=chunk,
                    metadata=doc.metadata.copy()
                ))
        return splits
    
    def _split_text(self, text):
        chunks = []
        for separator in self.separators:
            if separator in text:
                parts = text.split(separator)
                current_chunk = ""
                for part in parts:
                    if len(current_chunk) + len(part) + len(separator) <= self.chunk_size:
                        current_chunk += part + separator
                    else:
                        if current_chunk:
                            chunks.append(current_chunk.strip())
                        current_chunk = part + separator
                if current_chunk:
                    chunks.append(current_chunk.strip())
                return [chunk for chunk in chunks if chunk.strip()]
        
        # If no separators found, split by chunk_size
        for i in range(0, len(text), self.chunk_size - self.chunk_overlap):
            chunk = text[i:i + self.chunk_size]
            if chunk.strip():
                chunks.append(chunk.strip())
        return chunks

# Simple embedding function
class HuggingFaceEmbeddings:
    def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
    
    def embed_documents(self, texts):
        return self.model.encode(texts).tolist()
    
    def embed_query(self, text):
        return self.model.encode([text])[0].tolist()

# Simple vector store
class SimpleVectorStore:
    def __init__(self, persist_directory="./chroma_db", embedding_function=None, collection_name="rag_documents"):
        self.persist_directory = persist_directory
        self.embedding_function = embedding_function
        self.collection_name = collection_name
        self.documents = {}
        self.embeddings = {}
        os.makedirs(persist_directory, exist_ok=True)
        self._load_data()
    
    def _load_data(self):
        data_file = os.path.join(self.persist_directory, "data.pkl")
        if os.path.exists(data_file):
            try:
                with open(data_file, 'rb') as f:
                    data = pickle.load(f)
                    self.documents = data.get('documents', {})
                    self.embeddings = data.get('embeddings', {})
            except:
                pass
    
    def _save_data(self):
        data_file = os.path.join(self.persist_directory, "data.pkl")
        try:
            with open(data_file, 'wb') as f:
                pickle.dump({
                    'documents': self.documents,
                    'embeddings': self.embeddings
                }, f)
        except Exception as e:
            print(f"Error saving data: {e}")
    
    def add_documents(self, documents):
        for doc in documents:
            doc_id = len(self.documents)
            self.documents[doc_id] = {
                'content': doc.page_content,
                'metadata': doc.metadata
            }
            # Simple embedding (just store text for now)
            self.embeddings[doc_id] = doc.page_content.lower()
        self._save_data()
    
    def similarity_search(self, query, k=3):
        query_lower = query.lower()
        scores = []
        
        for doc_id, embedding in self.embeddings.items():
            # Simple word overlap scoring
            query_words = set(query_lower.split())
            doc_words = set(embedding.split())
            overlap = len(query_words.intersection(doc_words))
            if overlap > 0:
                scores.append((overlap, doc_id))
        
        scores.sort(reverse=True)
        results = []
        
        for score, doc_id in scores[:k]:
            doc_data = self.documents[doc_id]
            results.append(Document(
                page_content=doc_data['content'],
                metadata=doc_data['metadata']
            ))
        
        return results
    
    def as_retriever(self, search_type="similarity", search_kwargs=None):
        search_kwargs = search_kwargs or {"k": 3}
        
        class SimpleRetriever:
            def __init__(self, vectorstore, k=3):
                self.vectorstore = vectorstore
                self.k = k
            
            def similarity_search(self, query, k=None):
                return self.vectorstore.similarity_search(query, k or self.k)
        
        return SimpleRetriever(self, search_kwargs.get("k", 3))
    
    @property
    def _collection(self):
        class MockCollection:
            def delete(self, where):
                # Simple delete based on metadata
                if 'file_id' in where:
                    file_id = where['file_id']
                    to_remove = []
                    for doc_id, doc_data in self.parent.documents.items():
                        if doc_data['metadata'].get('file_id') == file_id:
                            to_remove.append(doc_id)
                    
                    for doc_id in to_remove:
                        self.parent.documents.pop(doc_id, None)
                        self.parent.embeddings.pop(doc_id, None)
                    
                    self.parent._save_data()
            
            def count(self):
                return len(self.parent.documents)
        
        mock = MockCollection()
        mock.parent = self
        return mock

# Initialize components
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000, 
    chunk_overlap=200, 
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

embedding_function = HuggingFaceEmbeddings()

# Initialize vector store
vectorstore = SimpleVectorStore(
    persist_directory="./chroma_db", 
    embedding_function=embedding_function,
    collection_name="rag_documents"
)

def load_and_split_document(file_path: str) -> List[Document]:
    """Load and split a document into chunks."""
    try:
        if file_path.endswith('.txt'):
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": file_path})]
        elif file_path.endswith('.pdf'):
            loader = PyPDFLoader(file_path)
            documents = loader.load()
        elif file_path.endswith('.docx'):
            loader = Docx2txtLoader(file_path)
            documents = loader.load()
        else:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            documents = [Document(page_content=content, metadata={"source": file_path})]
        
        splits = text_splitter.split_documents(documents)
        return splits
    except Exception as e:
        print(f"Error loading and splitting document {file_path}: {e}")
        return []

def index_document_to_chroma(file_path: str, file_id: int) -> bool:
    """Index a document to vector store."""
    try:
        splits = load_and_split_document(file_path)
        
        if not splits:
            print(f"No content found in document: {file_path}")
            return False

        # Add metadata to each split
        for split in splits:
            split.metadata['file_id'] = file_id
            split.metadata['source'] = os.path.basename(file_path)

        # Add documents to vectorstore
        vectorstore.add_documents(splits)
        
        print(f"Successfully indexed {len(splits)} chunks from {file_path}")
        return True
    except Exception as e:
        print(f"Error indexing document {file_path}: {e}")
        return False

def delete_doc_from_chroma(file_id: int) -> bool:
    """Delete all document chunks with a specific file_id."""
    try:
        vectorstore._collection.delete(where={"file_id": file_id})
        print(f"Successfully deleted all documents with file_id {file_id}")
        return True
    except Exception as e:
        print(f"Error deleting document with file_id {file_id}: {str(e)}")
        return False

def search_documents(query: str, k: int = 3):
    """Search for relevant documents."""
    try:
        docs = vectorstore.similarity_search(query, k=k)
        return docs
    except Exception as e:
        print(f"Error searching documents: {e}")
        return []

def get_vectorstore_stats():
    """Get statistics about the vector store."""
    try:
        count = vectorstore._collection.count()
        return {"document_count": count}
    except Exception as e:
        print(f"Error getting vectorstore stats: {e}")
        return {"document_count": 0}

# Initialize vectorstore on import
try:
    stats = get_vectorstore_stats()
    print(f"Vectorstore initialized with {stats['document_count']} documents")
except Exception as e:
    print(f"Error initializing vectorstore: {e}")