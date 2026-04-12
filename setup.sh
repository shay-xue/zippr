#!/bin/bash
# ──────────────────────────────────────────────────────────
# Science NeuroTech — One-command setup
#
# Run this script to get your development environment ready:
#   chmod +x setup.sh && ./setup.sh
# ──────────────────────────────────────────────────────────

set -e

echo "=========================================="
echo "  Science NeuroTech — Environment Setup"
echo "=========================================="

# Check Python version
PYTHON_CMD=""
if command -v python3 &> /dev/null; then
    PYTHON_CMD="python3"
elif command -v python &> /dev/null; then
    PYTHON_CMD="python"
else
    echo "ERROR: Python 3.10+ is required but not found."
    echo "Install Python from https://python.org"
    exit 1
fi

PY_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
PY_MAJOR=$(echo $PY_VERSION | cut -d. -f1)
PY_MINOR=$(echo $PY_VERSION | cut -d. -f2)

if [ "$PY_MAJOR" -lt 3 ] || ([ "$PY_MAJOR" -eq 3 ] && [ "$PY_MINOR" -lt 10 ]); then
    echo "ERROR: Python 3.10+ required, found $PY_VERSION"
    exit 1
fi
echo "[✓] Python $PY_VERSION"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo "[*] Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "[✓] Virtual environment created"
else
    echo "[✓] Virtual environment already exists"
fi

# Activate venv
source venv/bin/activate
echo "[✓] Virtual environment activated"

# Upgrade pip
pip install --upgrade pip --quiet

# Install dependencies
echo "[*] Installing Python dependencies..."
pip install --pre science-synapse --quiet
pip install -r requirements.txt --quiet
echo "[✓] All dependencies installed"

# Verify critical imports
echo "[*] Verifying installations..."
$PYTHON_CMD -c "
import synapse
import numpy
import torch
import onnx
import onnxruntime
import h5py
print('[✓] All packages verified')
" 2>&1

# Check synapsectl
if command -v synapsectl &> /dev/null || [ -f "venv/bin/synapsectl" ]; then
    SYNCTL_VERSION=$(venv/bin/synapsectl --version 2>&1 || synapsectl --version 2>&1)
    echo "[✓] synapsectl $SYNCTL_VERSION"
else
    echo "[!] synapsectl not found in PATH — run 'source venv/bin/activate' first"
fi

# Check Docker (needed for Synapse App builds)
if command -v docker &> /dev/null; then
    DOCKER_VERSION=$(docker --version 2>&1)
    echo "[✓] $DOCKER_VERSION"
else
    echo "[!] Docker not found — needed for Synapse App cross-compilation"
    echo "    Install from https://docker.com/get-started"
fi

# Create data directories
mkdir -p data/raw data/recordings models logs
echo "[✓] Data directories created"

echo ""
echo "=========================================="
echo "  Setup complete!"
echo "=========================================="
echo ""
echo "Activate the environment with:"
echo "  source venv/bin/activate"
echo ""
echo "SciFi device commands:"
echo "  synapsectl discover              # Find devices on network"
echo "  synapsectl -u <IP> info          # Device info"
echo ""
echo "Data collection:"
echo "  python scripts/device_check.py -d <IP>    # Verify device"
echo "  python scripts/collect_data.py -d <IP>     # Record training data"
echo ""
