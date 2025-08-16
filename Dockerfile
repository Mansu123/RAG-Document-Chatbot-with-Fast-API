# Use Python 3.11 slim image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV GOOGLE_API_KEY=AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create necessary directories
RUN mkdir -p chroma_db logs

# Create a non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Expose ports
EXPOSE 8000 8501

# Create startup script
RUN echo '#!/bin/bash\n\
echo "Starting RAG Chatbot Application..."\n\
echo "Starting FastAPI server..."\n\
uvicorn main:app --host 0.0.0.0 --port 8000 &\n\
echo "FastAPI server started on port 8000"\n\
sleep 5\n\
echo "Starting Streamlit app..."\n\
streamlit run streamlit_app.py --server.port 8501 --server.address 0.0.0.0 &\n\
echo "Streamlit app started on port 8501"\n\
echo "Application ready!"\n\
echo "FastAPI docs: http://localhost:8000/docs"\n\
echo "Streamlit app: http://localhost:8501"\n\
wait' > start.sh && chmod +x start.sh

# Default command
CMD ["./start.sh"]