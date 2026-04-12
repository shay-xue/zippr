#!/usr/bin/env bash
# =============================================================================
# run_pipeline.sh — Full BCI decoding pipeline CLI entrypoint
#
# Runs the complete pipeline in sequence:
#   1. Preprocess  — bin neural data, sweep k, save .npz files
#   2. Train       — fit MLP, XGBoost, 1D CNN, SVM for each k
#   3. Report      — generate plots + Markdown reports
#   (4. Git commit — optional, with --commit flag)
#
# Usage
# -----
#   bash run_pipeline.sh                    # full pipeline
#   bash run_pipeline.sh --skip-preprocess  # skip if .npz already exist
#   bash run_pipeline.sh --model mlp        # single model
#   bash run_pipeline.sh --k 3.0 3.5        # specific k values
#   bash run_pipeline.sh --commit           # also commit & push to medha
#
# Prerequisites
# -------------
#   conda activate <env>  OR  source venv/bin/activate
#   pip install -r requirements.txt
# =============================================================================

set -euo pipefail

# ── Resolve repo root ─────────────────────────────────────────────────────────
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ── Defaults ──────────────────────────────────────────────────────────────────
CONFIG="config.yaml"
SKIP_PREPROCESS=false
MODEL_ARGS=""
K_ARGS=""
DO_COMMIT=false

# Auto-detect Python: prefer conda (has sklearn/xgboost), fall back to python3
if command -v conda &>/dev/null && conda run python -c "import sklearn" &>/dev/null 2>&1; then
    PYTHON="conda run python"
elif [ -f "$HOME/miniconda3/python.exe" ]; then
    PYTHON="$HOME/miniconda3/python.exe"
elif [ -f "$HOME/miniconda3/bin/python" ]; then
    PYTHON="$HOME/miniconda3/bin/python"
elif [ -f "$HOME/anaconda3/bin/python" ]; then
    PYTHON="$HOME/anaconda3/bin/python"
else
    PYTHON="${PYTHON:-python3}"
fi
echo "Using Python: $PYTHON"

# ── Parse args ────────────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case $1 in
        --skip-preprocess) SKIP_PREPROCESS=true; shift ;;
        --model)           MODEL_ARGS="--model $2"; shift 2 ;;
        --k)               shift; K_LIST=""; while [[ $# -gt 0 && "$1" != --* ]]; do K_LIST="$K_LIST $1"; shift; done; K_ARGS="--k $K_LIST" ;;
        --commit)          DO_COMMIT=true; shift ;;
        --config)          CONFIG="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

echo "╔══════════════════════════════════════════════════════════════╗"
echo "║         BCI Decoding Pipeline — Science NeuroHack 2026       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
echo "Config: $CONFIG"
echo ""

# ── Step 1: Preprocessing ─────────────────────────────────────────────────────
if [ "$SKIP_PREPROCESS" = false ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo "STEP 1: Preprocessing (k sweep)"
    echo "──────────────────────────────────────────────────────────────"
    $PYTHON src/preprocess.py --config "$CONFIG" $K_ARGS
    echo ""
else
    echo "STEP 1: Skipped (--skip-preprocess)"
fi

# ── Step 2: Training ──────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
echo "STEP 2: Training models"
echo "──────────────────────────────────────────────────────────────"
$PYTHON src/pipeline.py --config "$CONFIG" $MODEL_ARGS $K_ARGS
echo ""

# ── Step 3: Reports ───────────────────────────────────────────────────────────
echo "──────────────────────────────────────────────────────────────"
echo "STEP 3: Reports generated"
echo "──────────────────────────────────────────────────────────────"
echo "  → reports/*.md"
echo "  → reports/figures/*.png"
echo ""

# ── Optional: Git commit ──────────────────────────────────────────────────────
if [ "$DO_COMMIT" = true ]; then
    echo "──────────────────────────────────────────────────────────────"
    echo "STEP 4: Git commit + push → medha"
    echo "──────────────────────────────────────────────────────────────"

    git checkout medha 2>/dev/null || git checkout -b medha

    git add data/preprocessed/ reports/ models/trained/ src/ config.yaml run_pipeline.sh
    git commit -m "Add preprocessed neural data with spike-threshold features (k sweep)"
    git push origin medha
    echo "  → Pushed to origin/medha"
fi

echo ""
echo "╔══════════════════════════════════════════════════════════════╗"
echo "║                     Pipeline complete!                       ║"
echo "╚══════════════════════════════════════════════════════════════╝"
