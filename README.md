# 🤖 RAG Chatbot with Google Gemini AI

A production-ready Retrieval-Augmented Generation (RAG) chatbot built with FastAPI, LangChain, and Google Gemini AI. This system allows you to upload documents and have intelligent conversations about their content.

## ✨ Features

- 🧠 **Google Gemini AI Integration**: Powered by Google's advanced language models
- 📚 **Document RAG**: Upload PDF, DOCX, HTML, and TXT files for intelligent Q&A
- 💬 **Session Management**: Maintains conversation context across interactions
- 🔄 **Real-time Document Management**: Upload, list, and delete documents dynamically
- 🎨 **Beautiful Web Interface**: User-friendly Streamlit frontend
- 🐳 **Docker Support**: Easy deployment with Docker and Docker Compose
- 📊 **System Monitoring**: Health checks and statistics endpoints
- 🔍 **Vector Search**: Efficient document retrieval using Chroma vector store

## 🏗️ Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Streamlit     │    │     FastAPI     │    │   Google        │
│   Frontend      │◄──►│   Backend API   │◄──►│   Gemini AI     │
│                 │    │                 │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
                              │
                              ▼
                    ┌─────────────────┐    ┌─────────────────┐
                    │   SQLite DB     │    │   Chroma        │
                    │   (Metadata)    │    │   Vector Store  │
                    └─────────────────┘    └─────────────────┘
```

## 📁 Project Structure

```
rag-fastapi-project/
├── main.py                 # FastAPI application
├── streamlit_app.py        # Streamlit frontend
├── langchain_utils.py      # LangChain RAG implementation
├── chroma_utils.py         # Vector store utilities
├── db_utils.py            # Database operations
├── pydantic_models.py     # Data models
├── requirements.txt       # Python dependencies
├── Dockerfile            # Docker configuration
├── docker-compose.yml    # Docker Compose setup
├── .env.example         # Environment variables template
├── README.md           # This file
└── chroma_db/         # Vector store persistence (created automatically)
```

## 🚀 Quick Start

### Option 1: Docker (Recommended)

1. **Clone and navigate to the project:**
   ```bash
   git clone <repository-url>
   cd rag-fastapi-project
   ```

2. **Start with Docker Compose:**
   ```bash
   docker-compose up --build
   ```

3. **Access the applications:**
   - **Streamlit Frontend**: http://localhost:8501
   - **FastAPI Backend**: http://localhost:8000
   - **API Documentation**: http://localhost:8000/docs

### Option 2: Local Development

1. **Clone the repository:**
   ```bash
   git clone <repository-url>
   cd rag-fastapi-project
   ```

2. **Create virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your configuration
   ```

5. **Start the FastAPI server:**
   ```bash
   uvicorn main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Start the Streamlit app (in another terminal):**
   ```bash
   streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0


   <img width="1366" height="768" alt="image" src="https://github.com/user-attachments/assets/ba670a8e-c2ea-4e1d-b143-163a2caedffb" />

   ```

## 🔧 Configuration

### Environment Variables

Create a `.env` file based on `.env.example`:

```bash
# Required
GOOGLE_API_KEY=your_google_api_key_here

# Optional (with defaults)
API_HOST=0.0.0.0
API_PORT=8000
STREAMLIT_PORT=8501
DB_NAME=rag_app.db
LOG_LEVEL=INFO
```

### Google Gemini AI Setup

1. Get your API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Set the `GOOGLE_API_KEY` environment variable
3. The system will automatically test the connection on startup

## 📚 Usage

### Web Interface (Streamlit)

1. **Upload Documents**: Use the sidebar to upload PDF, DOCX, HTML, or TXT files
2. **Ask Questions**: Type questions about your uploaded documents in the chat interface
3. **Manage Documents**: View, upload, and delete documents from the sidebar
4. **Session Management**: Start new sessions or clear chat history
5. **Monitor System**: Check API status and view system statistics

### API Endpoints

The FastAPI backend provides the following endpoints:

- `POST /chat` - Send a chat message
- `POST /upload-doc` - Upload and index a document
- `GET /list-docs` - List all indexed documents
- `POST /delete-doc` - Delete a document
- `GET /health` - Health check
- `GET /stats` - System statistics
- `GET /docs` - Interactive API documentation

### Example API Usage

```python
import requests

# Chat with the bot
response = requests.post("http://localhost:8000/chat", json={
    "question": "What is the main topic of the uploaded documents?",
    "model": "gemini-pro"
})

# Upload a document
with open("document.pdf", "rb") as f:
    response = requests.post(
        "http://localhost:8000/upload-doc",
        files={"file": f}
    )
```

## 🛠️ Development

### Adding New Document Types

To support new document types, update `chroma_utils.py`:

```python
def load_and_split_document(file_path: str) -> List[Document]:
    if file_path.endswith('.your_extension'):
        loader = YourCustomLoader(file_path)
    # ... existing code
```

### Customizing the RAG Pipeline

Modify `langchain_utils.py` to:
- Change embedding models
- Adjust chunk sizes
- Modify retrieval parameters
- Update prompts

### Adding New API Endpoints

Add new endpoints in `main.py` following the existing patterns.

## 🔍 Monitoring and Logging

- **Logs**: Application logs are saved to `app.log`
- **Health Check**: GET `/health` endpoint for monitoring
- **Statistics**: GET `/stats` endpoint for system metrics
- **Docker Health**: Built-in Docker health checks

## 🐛 Troubleshooting

### Common Issues

1. **Google API Key Issues**:
   ```bash
   # Test your API key
   curl -H "Authorization: Bearer YOUR_API_KEY" \
        "https://generativelanguage.googleapis.com/v1/models"
   ```

2. **Port Conflicts**:
   ```bash
   # Check what's using the ports
   lsof -i :8000
   lsof -i :8501
   ```

3. **Docker Issues**:
   ```bash
   # Rebuild containers
   docker-compose down
   docker-compose up --build
   
   # Check logs
   docker-compose logs rag-chatbot
   ```

4. **Database Issues**:
   ```bash
   # Reset database
   rm rag_app.db
   rm -rf chroma_db/
   ```

### Logs and Debugging

- Application logs: `app.log`
- Docker logs: `docker-compose logs`
- Streamlit logs: Check the terminal running Streamlit
- FastAPI logs: Check the terminal running uvicorn

## 🚀 Deployment

### Production Deployment

1. **Environment Variables**: Set secure environment variables
2. **Reverse Proxy**: Use nginx for SSL and load balancing
3. **Database**: Consider PostgreSQL for production
4. **Monitoring**: Add APM tools like Prometheus/Grafana
5. **Scaling**: Use Docker Swarm or Kubernetes

### Docker Production

```bash
# Production mode with nginx
docker-compose --profile production up -d
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🆘 Support

If you encounter any issues or have questions:

1. Check the troubleshooting section above
2. Review the logs for error messages
3. Ensure your Google API key is valid
4. Verify all dependencies are installed correctly

## 🙏 Acknowledgments

- [LangChain](https://langchain.com/) for the RAG framework
- [FastAPI](https://fastapi.tiangolo.com/) for the web framework
- [Streamlit](https://streamlit.io/) for the frontend
- [Google Gemini AI](https://ai.google.dev/) for the language model
- [Chroma](https://www.trychroma.com/) for vector storage

---


**Happy Chatting!** 🤖✨
