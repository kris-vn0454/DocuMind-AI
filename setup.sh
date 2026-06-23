#!/bin/bash
set -e

echo "🧠 DocuMind AI — Setup Script"
echo "=================================="

# Check Python version
python_version=$(python3 --version 2>&1 | awk '{print $2}')
echo "✅ Python version: $python_version"

# Create virtual environment
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv venv
fi

# Activate virtual environment
source venv/bin/activate
echo "✅ Virtual environment activated"

# Install dependencies
echo "Installing dependencies (this may take a few minutes)..."
pip install --upgrade pip -q
pip install -r requirements.txt -q
echo "✅ Dependencies installed"

# Copy .env file
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "✅ Created .env file — add your ANTHROPIC_API_KEY"
fi

# Create directories
mkdir -p data/sample_docs data/vector_store mlruns logs

# Generate sample documents and ingest them
echo ""
echo "Generating sample documents and ingesting..."
python main.py --ingest-only

echo ""
echo "=================================="
echo "✅ Setup complete!"
echo ""
echo "Start the application:"
echo "  Streamlit UI:   streamlit run app/streamlit_app.py"
echo "  REST API:       uvicorn src.api.main:app --reload --port 8000"
echo "  MLflow UI:      mlflow ui --port 5000"
echo ""
echo "Edit .env to add your ANTHROPIC_API_KEY for LLM features."
echo "=================================="
