# RAG Chatbot Startup Script for Windows PowerShell
# This script starts both the FastAPI backend and Streamlit frontend

param(
    [int]$FastAPIPort = 8000,
    [int]$StreamlitPort = 8501,
    [string]$LogDir = "logs"
)

# Function to write colored output
function Write-ColorOutput {
    param($ForegroundColor, $Message)
    Write-Host $Message -ForegroundColor $ForegroundColor
}

function Test-Port {
    param([int]$Port)
    try {
        $listener = [System.Net.NetworkInformation.IPGlobalProperties]::GetIPGlobalProperties().GetActiveTcpListeners()
        $portInUse = $listener | Where-Object { $_.Port -eq $Port }
        return $null -ne $portInUse
    }
    catch {
        return $false
    }
}

function Wait-ForService {
    param([string]$Url, [string]$ServiceName, [int]$MaxAttempts = 30)
    
    Write-ColorOutput Yellow "⏳ Waiting for $ServiceName to be ready..."
    
    for ($i = 1; $i -le $MaxAttempts; $i++) {
        try {
            $response = Invoke-WebRequest -Uri $Url -Method Get -TimeoutSec 5 -UseBasicParsing
            if ($response.StatusCode -eq 200) {
                Write-ColorOutput Green "✅ $ServiceName is ready!"
                return $true
            }
        }
        catch {
            # Service not ready yet
        }
        
        Write-ColorOutput Yellow "   Attempt $i/$MaxAttempts - $ServiceName not ready yet..."
        Start-Sleep -Seconds 2
    }
    
    Write-ColorOutput Red "❌ $ServiceName failed to start within expected time"
    return $false
}

# Main script
Clear-Host
Write-ColorOutput Blue "🚀 RAG Chatbot Startup Script"
Write-ColorOutput Blue "================================="
Write-Output ""

# Check system requirements
Write-ColorOutput Blue "🔍 Checking system requirements..."

# Check Python
try {
    $pythonVersion = python --version 2>&1
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Python found: $pythonVersion"
    } else {
        throw "Python not found"
    }
}
catch {
    Write-ColorOutput Red "❌ Python3 is required but not found"
    Write-ColorOutput Yellow "   Please install Python 3.8+ and add it to PATH"
    Read-Host "Press Enter to exit"
    exit 1
}

# Check if virtual environment exists and activate it
if (Test-Path ".venv\Scripts\Activate.ps1") {
    Write-ColorOutput Yellow "📦 Virtual environment found, activating..."
    try {
        & ".\.venv\Scripts\Activate.ps1"
        Write-ColorOutput Green "✅ Virtual environment activated"
    }
    catch {
        Write-ColorOutput Red "❌ Failed to activate virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
} else {
    Write-ColorOutput Yellow "📦 Creating virtual environment..."
    try {
        python -m venv .venv
        & ".\.venv\Scripts\Activate.ps1"
        Write-ColorOutput Green "✅ Virtual environment created and activated"
    }
    catch {
        Write-ColorOutput Red "❌ Failed to create virtual environment"
        Read-Host "Press Enter to exit"
        exit 1
    }
}

# Check dependencies
Write-ColorOutput Blue "📦 Checking Python dependencies..."
try {
    python -c "import fastapi, streamlit, langchain" 2>$null
    if ($LASTEXITCODE -eq 0) {
        Write-ColorOutput Green "✅ Dependencies are installed"
    } else {
        throw "Dependencies missing"
    }
}
catch {
    Write-ColorOutput Yellow "⚠️ Installing dependencies..."
    pip install -r requirements.txt
    if ($LASTEXITCODE -ne 0) {
        Write-ColorOutput Red "❌ Failed to install dependencies"
        Read-Host "Press Enter to exit"
        exit 1
    }
    Write-ColorOutput Green "✅ Dependencies installed successfully"
}

# Check ports
Write-ColorOutput Blue "🔌 Checking ports..."
if (Test-Port $FastAPIPort) {
    Write-ColorOutput Red "❌ Port $FastAPIPort is already in use"
    Write-ColorOutput Yellow "   Please stop the service using this port or change the port"
    Read-Host "Press Enter to exit"
    exit 1
}

if (Test-Port $StreamlitPort) {
    Write-ColorOutput Red "❌ Port $StreamlitPort is already in use"
    Write-ColorOutput Yellow "   Please stop the service using this port or change the port"
    Read-Host "Press Enter to exit"
    exit 1
}

Write-ColorOutput Green "✅ Ports $FastAPIPort and $StreamlitPort are available"

# Create necessary directories
Write-ColorOutput Blue "📁 Creating necessary directories..."
if (!(Test-Path "chroma_db")) { 
    New-Item -ItemType Directory -Path "chroma_db" | Out-Null
    Write-ColorOutput Green "✅ Created chroma_db directory"
}
if (!(Test-Path $LogDir)) { 
    New-Item -ItemType Directory -Path $LogDir | Out-Null
    Write-ColorOutput Green "✅ Created $LogDir directory"
}

# Set environment variables
if (!$env:GOOGLE_API_KEY) {
    $env:GOOGLE_API_KEY = "AIzaSyBciwV608FktD2u2Si_Lu0-aknjAH985ak"
    Write-ColorOutput Green "✅ Google API key set"
}

# Start FastAPI server
Write-ColorOutput Blue "🔧 Starting FastAPI server..."
$fastApiProcess = Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; uvicorn main:app --host 0.0.0.0 --port $FastAPIPort --reload" -PassThru
Write-ColorOutput Green "✅ FastAPI server started (Process ID: $($fastApiProcess.Id))"

# Wait for FastAPI to be ready
Start-Sleep -Seconds 5
if (!(Wait-ForService "http://localhost:$FastAPIPort/health" "FastAPI")) {
    Write-ColorOutput Red "❌ FastAPI failed to start"
    if (!$fastApiProcess.HasExited) {
        $fastApiProcess.Kill()
    }
    Read-Host "Press Enter to exit"
    exit 1
}

# Start Streamlit app
Write-ColorOutput Blue "🎨 Starting Streamlit app..."
$streamlitProcess = Start-Process powershell -ArgumentList "-Command", "cd '$PWD'; .\.venv\Scripts\Activate.ps1; streamlit run streamlit_app.py --server.port $StreamlitPort --server.address 0.0.0.0 --server.headless true --browser.gatherUsageStats false" -PassThru
Write-ColorOutput Green "✅ Streamlit app started (Process ID: $($streamlitProcess.Id))"

# Wait for Streamlit to be ready
Start-Sleep -Seconds 8
if (!(Wait-ForService "http://localhost:$StreamlitPort" "Streamlit")) {
    Write-ColorOutput Yellow "⚠️ Streamlit might still be starting up..."
}

# Display success message
Write-Output ""
Write-ColorOutput Green "🎉 RAG Chatbot is now running!"
Write-ColorOutput Green "================================="
Write-ColorOutput Blue "📱 Web Interface:     http://localhost:$StreamlitPort"
Write-ColorOutput Blue "🔧 API Backend:      http://localhost:$FastAPIPort"
Write-ColorOutput Blue "📚 API Documentation: http://localhost:$FastAPIPort/docs"
Write-ColorOutput Blue "❤️ Health Check:     http://localhost:$FastAPIPort/health"
Write-Output ""
Write-ColorOutput Yellow "📝 Both services are running in separate windows"
Write-ColorOutput Yellow "🌐 Opening web interface in your browser..."
Write-Output ""

# Open web interface
Start-Process "http://localhost:$StreamlitPort"

# Display final instructions
Write-ColorOutput Green "✨ All services are running. The application is ready to use!"
Write-Output ""
Write-ColorOutput Yellow "💡 To stop the services:"
Write-ColorOutput Yellow "   - Close the PowerShell windows that opened"
Write-ColorOutput Yellow "   - Or run: Get-Process | Where-Object {`$_.ProcessName -eq 'python'} | Stop-Process"
Write-Output ""
Write-ColorOutput Blue "🎯 Next steps:"
Write-ColorOutput Blue "   1. The web interface should open automatically"
Write-ColorOutput Blue "   2. Upload some documents (PDF, DOCX, TXT, HTML)"
Write-ColorOutput Blue "   3. Start asking questions about your documents"
Write-Output ""

Read-Host "Press Enter to close this window (services will continue running)"