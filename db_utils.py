import sqlite3
from datetime import datetime
import os

DB_NAME = "rag_app.db"

def get_db_connection():
    """Create and return a database connection."""
    conn = sqlite3.connect(DB_NAME)
    conn.row_factory = sqlite3.Row
    return conn

def create_application_logs():
    """Create the application_logs table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS application_logs
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     session_id TEXT,
                     user_query TEXT,
                     gpt_response TEXT,
                     model TEXT,
                     created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def create_document_store():
    """Create the document_store table if it doesn't exist."""
    conn = get_db_connection()
    conn.execute('''CREATE TABLE IF NOT EXISTS document_store
                    (id INTEGER PRIMARY KEY AUTOINCREMENT,
                     filename TEXT,
                     upload_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    conn.close()

def insert_application_logs(session_id, user_query, gpt_response, model):
    """Insert a new chat log entry."""
    try:
        conn = get_db_connection()
        conn.execute('INSERT INTO application_logs (session_id, user_query, gpt_response, model) VALUES (?, ?, ?, ?)',
                     (session_id, user_query, gpt_response, model))
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"Error inserting application logs: {e}")
        return False

def get_chat_history(session_id):
    """Retrieve chat history for a given session ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT user_query, gpt_response FROM application_logs WHERE session_id = ? ORDER BY created_at', (session_id,))
        messages = []
        for row in cursor.fetchall():
            messages.extend([
                {"role": "human", "content": row['user_query']},
                {"role": "ai", "content": row['gpt_response']}
            ])
        conn.close()
        return messages
    except Exception as e:
        print(f"Error retrieving chat history: {e}")
        return []

def insert_document_record(filename):
    """Insert a new document record and return its ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('INSERT INTO document_store (filename) VALUES (?)', (filename,))
        file_id = cursor.lastrowid
        conn.commit()
        conn.close()
        return file_id
    except Exception as e:
        print(f"Error inserting document record: {e}")
        return None

def delete_document_record(file_id):
    """Delete a document record by file ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('DELETE FROM document_store WHERE id = ?', (file_id,))
        rows_affected = cursor.rowcount
        conn.commit()
        conn.close()
        return rows_affected > 0
    except Exception as e:
        print(f"Error deleting document record: {e}")
        return False

def get_all_documents():
    """Retrieve all document records."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, upload_timestamp FROM document_store ORDER BY upload_timestamp DESC')
        documents = cursor.fetchall()
        conn.close()
        return [dict(doc) for doc in documents]
    except Exception as e:
        print(f"Error retrieving documents: {e}")
        return []

def get_document_by_id(file_id):
    """Retrieve a specific document by ID."""
    try:
        conn = get_db_connection()
        cursor = conn.cursor()
        cursor.execute('SELECT id, filename, upload_timestamp FROM document_store WHERE id = ?', (file_id,))
        document = cursor.fetchone()
        conn.close()
        return dict(document) if document else None
    except Exception as e:
        print(f"Error retrieving document by ID: {e}")
        return None

# Initialize the database tables when this module is imported
if __name__ == "__main__":
    create_application_logs()
    create_document_store()
    print("Database tables initialized successfully!")
else:
    create_application_logs()
    create_document_store()