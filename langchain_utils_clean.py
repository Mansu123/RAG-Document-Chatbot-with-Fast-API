# Google Gemini AI integration without langchain dependencies
import google.generativeai as genai
import os
from chroma_utils_clean import vectorstore
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Google API key from environment or use the provided key
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak")

# Set the API key for Google Generative AI
genai.configure(api_key=GOOGLE_API_KEY)

# Create retriever from vectorstore
if vectorstore:
    retriever = vectorstore.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}
    )
else:
    # Dummy retriever if vectorstore is not available
    class DummyRetriever:
        def similarity_search(self, query, k=3):
            return []
    retriever = DummyRetriever()

# Simple Google Gemini AI wrapper
class ChatGoogleGenerativeAI:
    def __init__(self, model="gemini-pro", temperature=0.1, api_key=None, **kwargs):
        self.model = model
        self.temperature = temperature
        if api_key:
            genai.configure(api_key=api_key)
        self.genai_model = genai.GenerativeModel(model)
    
    def invoke(self, message):
        try:
            if isinstance(message, str):
                response = self.genai_model.generate_content(message)
                return response.text
            else:
                # Handle message format
                content = str(message)
                response = self.genai_model.generate_content(content)
                return response.text
        except Exception as e:
            return f"Error generating response: {str(e)}"

# Simple RAG chain implementation
class SimpleRAGChain:
    def __init__(self, retriever, llm):
        self.retriever = retriever
        self.llm = llm
    
    def invoke(self, inputs):
        question = inputs.get('input', '')
        chat_history = inputs.get('chat_history', [])
        
        try:
            # Get relevant documents
            docs = self.retriever.similarity_search(question, k=3)
            context = "\n".join([doc.page_content for doc in docs])
            
            # Format chat history
            history_text = ""
            for msg in chat_history:
                if isinstance(msg, tuple):
                    role, content = msg
                    history_text += f"{role}: {content}\n"
                elif isinstance(msg, dict):
                    role = msg.get('role', 'user')
                    content = msg.get('content', '')
                    history_text += f"{role}: {content}\n"
            
            # Create prompt
            if context:
                prompt = f"""Based on the following context and chat history, answer the question. If the answer is not in the context, say so clearly.

Chat History:
{history_text}

Context:
{context}

Question: {question}

Answer:"""
            else:
                prompt = f"""Based on the chat history, answer the question.

Chat History:
{history_text}

Question: {question}

Answer: I don't have any relevant documents to answer this question. Please upload some documents first."""
            
            # Generate response
            response = self.llm.invoke(prompt)
            
            return {'answer': response}
            
        except Exception as e:
            return {'answer': f"Error processing query: {str(e)}"}

def get_rag_chain(model="gemini-pro", temperature=0.1):
    """
    Create and return a RAG chain using Google Gemini AI.
    
    Args:
        model (str): The Gemini model to use
        temperature (float): Temperature for response generation
        
    Returns:
        RAG chain for question answering
    """
    try:
        # Initialize the Gemini LLM
        llm = ChatGoogleGenerativeAI(
            model=model,
            temperature=temperature,
            api_key=GOOGLE_API_KEY
        )
        
        # Create the RAG chain
        rag_chain = SimpleRAGChain(retriever, llm)
        
        return rag_chain
    except Exception as e:
        print(f"Error creating RAG chain: {e}")
        # Return a simple fallback chain
        class FallbackChain:
            def invoke(self, inputs):
                question = inputs.get('input', '')
                return {
                    'answer': f"I received your question: '{question}'. However, there seems to be an issue with the RAG system configuration. Please check if all dependencies are properly installed."
                }
        return FallbackChain()

def test_gemini_connection():
    """Test the connection to Google Gemini AI."""
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content("Hello! Can you confirm you're working?")
        return True, response.text
    except Exception as e:
        return False, str(e)

def format_chat_history(chat_history):
    """Format chat history for use with the chains."""
    formatted_history = []
    for message in chat_history:
        if message["role"] == "human":
            formatted_history.append(("human", message["content"]))
        elif message["role"] == "ai":
            formatted_history.append(("ai", message["content"]))
    return formatted_history

# Test connection on import
if __name__ == "__main__":
    success, message = test_gemini_connection()
    if success:
        print("✅ Google Gemini AI connection successful!")
        print(f"Response: {message}")
    else:
        print("❌ Google Gemini AI connection failed!")
        print(f"Error: {message}")
else:
    try:
        success, message = test_gemini_connection()
        if success:
            print("✅ Google Gemini AI initialized successfully!")
        else:
            print(f"⚠️ Warning: Google Gemini AI test failed: {message}")
    except:
        print("⚠️ Warning: Could not test Google Gemini AI connection")