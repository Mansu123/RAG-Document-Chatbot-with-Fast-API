#!/bin/bash

# RAG Chatbot Startup Script
# This script starts both the FastAPI backend and Streamlit frontend

set -e  # Exit on any error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration
FASTAPI_HOST="0.0.0.0"
FASTAPI_PORT="8000"
STREAMLIT_HOST="0.0.0.0"
STREAMLIT_PORT="8501"
LOG_DIR="logs"

# Create logs directory if it doesn't exist
mkdir -p $LOG_DIR

echo -e "${BLUE}🚀 RAG Chatbot Startup Script${NC}"
echo -e "${BLUE}=================================${NC}"

# Function to check if a port is in use
check_port() {
    local port=$1
    if lsof -Pi :$port -sTCP:LISTEN -t >/dev/null ; then
        return 0  # Port is in use
    else
        return 1  # Port is free
    fi
}

# Function to wait for a service to be ready
wait_for_service() {
    local url=$1
    local service_name=$2
    local max_attempts=30
    local attempt=1

    echo -e "${YELLOW}⏳ Waiting for $service_name to be ready...${NC}"
    
    while [ $attempt -le $max_attempts ]; do
        if curl -s -f "$url" > /dev/null 2>&1; then
            echo -e "${GREEN}✅ $service_name is ready!${NC}"
            return 0
        fi
        
        echo -e "${YELLOW}   Attempt $attempt/$max_attempts - $service_name not ready yet...${NC}"
        sleep 2
        attempt=$((attempt + 1))
    done
    
    echo -e "${RED}❌ $service_name failed to start within expected time${NC}"
    return 1
}

# Function to cleanup background processes
cleanup() {
    echo -e "\n${YELLOW}🛑 Stopping services...${NC}"
    jobs -p | xargs -r kill 2>/dev/null || true
    echo -e "${GREEN}✅ Cleanup completed${NC}"
    exit 0
}

# Set up signal handling
trap cleanup SIGINT SIGTERM

echo -e "${BLUE}🔍 Checking system requirements...${NC}"

# Check if Python is installed
if ! command -v python3 &> /dev/null; then
    echo -e "${RED}❌ Python3 is required but not installed${NC}"
    exit 1
fi

# Check if required files exist
required_files=("main.py" "streamlit_app.py" "requirements.txt")
for file in "${required_files[@]}"; do
    if [ ! -f "$file" ]; then
        echo -e "${RED}❌ Required file $file not found${NC}"
        exit 1
    fi
done

echo -e "${GREEN}✅ System requirements check passed${NC}"

# Check if virtual environment should be used
if [ -d "venv" ]; then
    echo -e "${YELLOW}📦 Virtual environment detected, activating...${NC}"
    source venv/bin/activate
fi

# Check if dependencies are installed
echo -e "${BLUE}📦 Checking Python dependencies...${NC}"
if ! python3 -c "import fastapi, streamlit, langchain" 2>/dev/null; then
    echo -e "${YELLOW}⚠️  Some dependencies missing, installing...${NC}"
    pip install -r requirements.txt
fi

# Check ports
echo -e "${BLUE}🔌 Checking ports...${NC}"
if check_port $FASTAPI_PORT; then
    echo -e "${RED}❌ Port $FASTAPI_PORT is already in use${NC}"
    echo -e "${YELLOW}   Please stop the service using this port or change FASTAPI_PORT${NC}"
    exit 1
fi

if check_port $STREAMLIT_PORT; then
    echo -e "${RED}❌ Port $STREAMLIT_PORT is already in use${NC}"
    echo -e "${YELLOW}   Please stop the service using this port or change STREAMLIT_PORT${NC}"
    exit 1
fi

echo -e "${GREEN}✅ Ports $FASTAPI_PORT and $STREAMLIT_PORT are available${NC}"

# Create necessary directories
echo -e "${BLUE}📁 Creating necessary directories...${NC}"
mkdir -p chroma_db
mkdir -p $LOG_DIR

# Set environment variables
export GOOGLE_API_KEY="${GOOGLE_API_KEY:-AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak}"

# Start FastAPI server
echo -e "${BLUE}🔧 Starting FastAPI server...${NC}"
uvicorn main:app \
    --host $FASTAPI_HOST \
    --port $FASTAPI_PORT \
    --log-level info \
    > $LOG_DIR/fastapi.log 2>&1 &

FASTAPI_PID=$!
echo -e "${GREEN}✅ FastAPI server started (PID: $FASTAPI_PID)${NC}"

# Wait for FastAPI to be ready
if ! wait_for_service "http://localhost:$FASTAPI_PORT/health" "FastAPI"; then
    echo -e "${RED}❌ FastAPI failed to start${NC}"
    kill $FASTAPI_PID 2>/dev/null || true
    exit 1
fi

# Start Streamlit app
echo -e "${BLUE}🎨 Starting Streamlit app...${NC}"
streamlit run streamlit_app.py \
    --server.port $STREAMLIT_PORT \
    --server.address $STREAMLIT_HOST \
    --server.headless true \
    --browser.gatherUsageStats false \
    > $LOG_DIR/streamlit.log 2>&1 &

STREAMLIT_PID=$!
echo -e "${GREEN}✅ Streamlit app started (PID: $STREAMLIT_PID)${NC}"

# Wait for Streamlit to be ready
if ! wait_for_service "http://localhost:$STREAMLIT_PORT" "Streamlit"; then
    echo -e "${RED}❌ Streamlit failed to start${NC}"
    kill $FASTAPI_PID $STREAMLIT_PID 2>/dev/null || true
    exit 1
fi

# Display success message
echo -e "\n${GREEN}🎉 RAG Chatbot is now running!${NC}"
echo -e "${GREEN}=================================${NC}"
echo -e "${BLUE}📱 Web Interface:${NC}     http://localhost:$STREAMLIT_PORT"
echo -e "${BLUE}🔧 API Backend:${NC}      http://localhost:$FASTAPI_PORT"
echo -e "${BLUE}📚 API Documentation:${NC} http://localhost:$FASTAPI_PORT/docs"
echo -e "${BLUE}❤️  Health Check:${NC}     http://localhost:$FASTAPI_PORT/health"
echo -e "\n${YELLOW}📝 Logs are available in the '$LOG_DIR' directory${NC}"
echo -e "${YELLOW}🛑 Press Ctrl+C to stop all services${NC}"

# Test the system
echo -e "\n${BLUE}🧪 Running quick system test...${NC}"
if python3 test_api.py > /dev/null 2>&1; then
    echo -e "${GREEN}✅ System test passed${NC}"
else
    echo -e "${YELLOW}⚠️  System test had some issues (check test_api.py for details)${NC}"
fi

# Keep the script running and wait for interruption
echo -e "\n${GREEN}✨ All services are running. The application is ready to use!${NC}"
wait