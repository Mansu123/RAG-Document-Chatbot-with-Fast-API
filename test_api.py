#!/usr/bin/env python3
"""
Test script for the RAG Chatbot API
"""

import requests
import json
import time
import os

# Configuration
API_BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

def test_health_check():
    """Test the health check endpoint."""
    print("🏥 Testing health check...")
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            print("✅ Health check passed")
            print(f"   Status: {response.json()}")
            return True
        else:
            print(f"❌ Health check failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Health check error: {e}")
        return False

def test_stats():
    """Test the stats endpoint."""
    print("\n📊 Testing statistics...")
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=5)
        if response.status_code == 200:
            stats = response.json()
            print("✅ Stats retrieved successfully")
            print(f"   Documents in vectorstore: {stats.get('vectorstore_document_count', 0)}")
            print(f"   Documents in database: {stats.get('database_document_count', 0)}")
            return True
        else:
            print(f"❌ Stats failed: {response.status_code}")
            return False
    except Exception as e:
        print(f"❌ Stats error: {e}")
        return False

def test_list_documents():
    """Test listing documents."""
    print("\n📄 Testing document listing...")
    try:
        response = requests.get(f"{API_BASE_URL}/list-docs", timeout=5)
        if response.status_code == 200:
            docs = response.json()
            print(f"✅ Found {len(docs)} documents")
            for doc in docs[:3]:  # Show first 3 documents
                print(f"   - {doc['filename']} (ID: {doc['id']})")
            return True, docs
        else:
            print(f"❌ List documents failed: {response.status_code}")
            return False, []
    except Exception as e:
        print(f"❌ List documents error: {e}")
        return False, []

def test_upload_document():
    """Test document upload with a sample text file."""
    print("\n📤 Testing document upload...")
    
    # Create a sample text file
    sample_content = """
    This is a sample document for testing the RAG chatbot system.
    
    The RAG (Retrieval-Augmented Generation) approach combines:
    1. Document retrieval from a vector database
    2. Language model generation based on retrieved context
    3. Real-time question answering capabilities
    
    Key benefits include:
    - Accurate responses based on specific documents
    - Ability to cite sources
    - Dynamic knowledge base updates
    - Contextual understanding
    
    This system uses Google Gemini AI for natural language processing
    and Chroma for vector storage and retrieval.
    """
    
    # Write sample file
    sample_file = "test_document.txt"
    try:
        with open(sample_file, "w", encoding="utf-8") as f:
            f.write(sample_content)
        
        # Upload the file
        with open(sample_file, "rb") as f:
            files = {"file": (sample_file, f, "text/plain")}
            response = requests.post(f"{API_BASE_URL}/upload-doc", files=files, timeout=30)
        
        # Clean up
        os.remove(sample_file)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document uploaded successfully")
            print(f"   File ID: {result.get('file_id')}")
            print(f"   Filename: {result.get('filename')}")
            return True, result.get('file_id')
        else:
            print(f"❌ Upload failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None
            
    except Exception as e:
        # Clean up on error
        if os.path.exists(sample_file):
            os.remove(sample_file)
        print(f"❌ Upload error: {e}")
        return False, None

def test_chat(question="What is RAG and what are its benefits?"):
    """Test the chat endpoint."""
    print(f"\n💬 Testing chat with question: '{question}'")
    try:
        payload = {
            "question": question,
            "model": "gemini-pro"
        }
        response = requests.post(f"{API_BASE_URL}/chat", json=payload, timeout=TEST_TIMEOUT)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Chat response received")
            print(f"   Session ID: {result.get('session_id', 'N/A')[:8]}...")
            print(f"   Model: {result.get('model', 'N/A')}")
            print(f"   Answer: {result.get('answer', 'No answer')[:200]}...")
            return True, result
        else:
            print(f"❌ Chat failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None
            
    except Exception as e:
        print(f"❌ Chat error: {e}")
        return False, None

def test_delete_document(file_id):
    """Test document deletion."""
    if not file_id:
        print("\n🗑️  Skipping delete test (no file ID)")
        return True
        
    print(f"\n🗑️  Testing document deletion (ID: {file_id})...")
    try:
        payload = {"file_id": file_id}
        response = requests.post(f"{API_BASE_URL}/delete-doc", json=payload, timeout=10)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ Document deleted successfully")
            print(f"   Message: {result.get('message', 'No message')}")
            return True
        else:
            print(f"❌ Delete failed: {response.status_code}")
            print(f"   Error: {response.text}")
            return False
            
    except Exception as e:
        print(f"❌ Delete error: {e}")
        return False

def run_full_test():
    """Run all tests in sequence."""
    print("🚀 Starting RAG Chatbot API Tests")
    print("=" * 50)
    
    # Test results
    results = {}
    
    # 1. Health check
    results['health'] = test_health_check()
    
    # 2. Stats
    results['stats'] = test_stats()
    
    # 3. List documents (initial)
    results['list_docs'], initial_docs = test_list_documents()
    
    # 4. Upload document
    results['upload'], uploaded_file_id = test_upload_document()
    
    # 5. List documents (after upload)
    if results['upload']:
        print("\n📄 Re-checking document list after upload...")
        success, updated_docs = test_list_documents()
        if success and len(updated_docs) > len(initial_docs):
            print("✅ Document count increased after upload")
        
    # 6. Chat test
    results['chat'], chat_result = test_chat()
    
    # 7. Chat with specific question about uploaded document
    if results['upload'] and results['chat']:
        results['chat_specific'], _ = test_chat("What are the key benefits of RAG systems?")
    
    # 8. Delete uploaded document
    if uploaded_file_id:
        results['delete'] = test_delete_document(uploaded_file_id)
    
    # Summary
    print("\n" + "=" * 50)
    print("📋 Test Summary")
    print("=" * 50)
    
    passed = sum(1 for result in results.values() if result)
    total = len(results)
    
    for test_name, result in results.items():
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{test_name.replace('_', ' ').title()}: {status}")
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! Your RAG Chatbot API is working correctly.")
    else:
        print("⚠️  Some tests failed. Check the output above for details.")
    
    return passed == total

if __name__ == "__main__":
    print("RAG Chatbot API Test Suite")
    print("Make sure the API is running on http://localhost:8000")
    print()
    
    # Check if API is accessible
    try:
        response = requests.get(f"{API_BASE_URL}/", timeout=5)
        if response.status_code == 200:
            print("✅ API is accessible")
            run_full_test()
        else:
            print(f"❌ API returned status code: {response.status_code}")
    except requests.exceptions.ConnectionError:
        print("❌ Cannot connect to API. Make sure it's running on http://localhost:8000")
        print("   Start the API with: uvicorn main:app --host 0.0.0.0 --port 8000")
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")