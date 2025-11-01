#!/bin/bash

# setup.sh - Script to install Ollama and download a Llama model

set -e  # Exit on error

echo "==================================="
echo "Llama Model Setup Script"
echo "==================================="
echo ""

# Check if ollama is already installed
if command -v ollama &> /dev/null; then
    echo "✓ Ollama is already installed"
    ollama --version
else
    echo "Installing Ollama..."

    # Install Ollama (official installation method)
    curl -fsSL https://ollama.com/install.sh | sh

    if [ $? -eq 0 ]; then
        echo "✓ Ollama installed successfully"
    else
        echo "✗ Failed to install Ollama"
        exit 1
    fi
fi

echo ""
echo "==================================="
echo "Starting Ollama service..."
echo "==================================="
echo ""

# Start ollama service in the background if not already running
if ! pgrep -x "ollama" > /dev/null; then
    ollama serve &
    OLLAMA_PID=$!
    echo "Ollama service started (PID: $OLLAMA_PID)"
    # Give it a moment to start up
    sleep 3
else
    echo "✓ Ollama service is already running"
fi

echo ""
echo "==================================="
echo "Downloading Llama model..."
echo "==================================="
echo ""

# Pull llama model (using llama3.2 as it's a good balance of size and performance)
# You can change this to other models like:
# - llama3.2:1b (smallest, fastest)
# - llama3.2:3b (medium size)
# - llama3.1:8b (larger, more capable)
# - llama2:7b (older but reliable)

MODEL="llama3.2"

echo "Pulling $MODEL model (this may take a while depending on your connection)..."
ollama pull $MODEL

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ Model $MODEL downloaded successfully!"
else
    echo ""
    echo "✗ Failed to download model $MODEL"
    exit 1
fi

echo ""
echo "==================================="
echo "Setup Complete!"
echo "==================================="
echo ""
echo "You can now use the model with:"
echo "  ollama run $MODEL"
echo ""
echo "Or in your Python code:"
echo "  # Install: pip install ollama"
echo "  import ollama"
echo "  response = ollama.chat(model='$MODEL', messages=[...])"
echo ""
echo "To list all models: ollama list"
echo "To remove a model: ollama rm $MODEL"
echo ""

