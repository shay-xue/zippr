#!/usr/bin/env bash
# deploy_synapse_app.sh
#
# Builds and deploys a Synapse App with an ONNX decoder model to a SciFi device.
# Supports both easy mode (32 channels) and hard mode (64 channels).
#
# Usage:
#   ./deploy_synapse_app.sh --model <path/to/model.onnx> --device <ip_or_name> [OPTIONS]
#
# Required:
#   --model   <path>      Path to the ONNX model file
#   --device  <ip|name>   SciFi device IP address or hostname
#
# Options:
#   --app-dir   <path>    Path to the Synapse App directory (default: ./synapse_app)
#   --mode      <mode>    Encoding mode: easy (32ch) or hard (64ch) (default: hard)
#   --model-name <name>   Model name to deploy as (default: decoder)
#   --sample-rate <hz>    Broadband source sample rate in Hz (default: 32000)
#   --config-out <path>   Where to write the generated start config (default: /tmp/synapse_start_config.json)
#   --skip-build          Skip the app build step (use existing .deb)
#   --skip-model          Skip the model deploy step
#   --skip-app            Skip the app deploy step
#   --dry-run             Print commands without executing them
#   --help                Show this message

set -euo pipefail

# ─── Colors ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
CYAN='\033[0;36m'
BOLD='\033[1m'
NC='\033[0m'

log()    { echo -e "${CYAN}[deploy]${NC} $*"; }
ok()     { echo -e "${GREEN}[  ok  ]${NC} $*"; }
warn()   { echo -e "${YELLOW}[ warn ]${NC} $*"; }
die()    { echo -e "${RED}[ fail ]${NC} $*" >&2; exit 1; }
header() { echo -e "\n${BOLD}${CYAN}══ $* ══${NC}"; }

# ─── Defaults ────────────────────────────────────────────────────────────────
MODEL_PATH=""
DEVICE=""
APP_DIR="./synapse_app"
MODE="hard"
MODEL_NAME="decoder"
SAMPLE_RATE=32000
CONFIG_OUT="/tmp/synapse_start_config.json"
SKIP_BUILD=false
SKIP_MODEL=false
SKIP_APP=false
DRY_RUN=false

# ─── Argument parsing ─────────────────────────────────────────────────────────
usage() {
  sed -n '/^# Usage:/,/^[^#]/{ /^#/{ s/^# \?//; p } }' "$0"
  exit 0
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)       MODEL_PATH="$2";   shift 2 ;;
    --device)      DEVICE="$2";       shift 2 ;;
    --app-dir)     APP_DIR="$2";      shift 2 ;;
    --mode)        MODE="$2";         shift 2 ;;
    --model-name)  MODEL_NAME="$2";   shift 2 ;;
    --sample-rate) SAMPLE_RATE="$2";  shift 2 ;;
    --config-out)  CONFIG_OUT="$2";   shift 2 ;;
    --skip-build)  SKIP_BUILD=true;   shift   ;;
    --skip-model)  SKIP_MODEL=true;   shift   ;;
    --skip-app)    SKIP_APP=true;     shift   ;;
    --dry-run)     DRY_RUN=true;      shift   ;;
    --help|-h)     usage ;;
    *) die "Unknown argument: $1. Run with --help for usage." ;;
  esac
done

# ─── Validation ───────────────────────────────────────────────────────────────
[[ -z "$MODEL_PATH" ]] && die "--model is required"
[[ -z "$DEVICE" ]]     && die "--device is required"

[[ "$MODE" == "easy" || "$MODE" == "hard" ]] || \
  die "--mode must be 'easy' or 'hard', got: $MODE"

[[ -f "$MODEL_PATH" ]] || die "Model file not found: $MODEL_PATH"
[[ "${MODEL_PATH##*.}" == "onnx" ]] || \
  die "Expected an .onnx file, got: $MODEL_PATH"

if [[ "$SKIP_BUILD" == false && "$SKIP_APP" == false ]]; then
  [[ -d "$APP_DIR" ]] || die "App directory not found: $APP_DIR"
  [[ -f "$APP_DIR/manifest.json" ]] || \
    die "manifest.json not found in $APP_DIR — is this a Synapse App?"
fi

# Check prerequisites (skipped for dry-run so you can preview on any machine)
if [[ "$DRY_RUN" == false ]]; then
  command -v synapsectl &>/dev/null || \
    die "'synapsectl' not found. Run: pip install --pre science-synapse"
fi

if [[ "$SKIP_BUILD" == false && "$DRY_RUN" == false ]]; then
  command -v docker &>/dev/null || die "'docker' not found — required for app build"
fi

# ─── Helpers ─────────────────────────────────────────────────────────────────
run() {
  if [[ "$DRY_RUN" == true ]]; then
    echo -e "${YELLOW}[dry-run]${NC} $*"
  else
    log "$ $*"
    "$@"
  fi
}

# ─── Channel count by mode ────────────────────────────────────────────────────
# Easy mode (testing): 32 encoded channels
# Hard mode (testing): 64 encoded channels
#   Hard mode training has 12 raw controller channels among the 64.
#   Training channels layout: channels 52-63 = raw (2 left joystick axes,
#   2 right joystick axes, A, B, X, Y, left trigger, right trigger,
#   left bumper, right bumper). Testing mode: all 64 are encoded neural data.

if [[ "$MODE" == "easy" ]]; then
  NUM_CHANNELS=32
else
  NUM_CHANNELS=64
fi

# ─── Generate electrode channel array ────────────────────────────────────────
# These electrode_id / reference_id values follow the pattern from the
# synapse-example-app simulator_32ch.json for channels 0-31, then extend
# with plausible sequential IDs for channels 32-63 (hard mode).
# Adjust to match your actual SciFi electrode map if needed.

generate_channels_json() {
  local count=$1

  # Electrode IDs from the reference 32-channel config (channels 0-31)
  local electrode_ids_32=(
    122 126 116 120 110 114 104 108
     98  66  92  60  86  54  80  48
     74  38  68  36  62   0  56   4
     50  12  44  14  42  20  32  26
  )
  local reference_ids_32=(
    513 513 513 513 513 513 513 513
    513 513 513 513 513 513 513 513
    513 513 513 513 513 512 513 512
    513 513 513 513 513 513 513 513
  )

  # Extended electrode IDs for channels 32-63 (hard mode).
  # These follow the ascending pattern of unused electrode addresses on the
  # SciFi's second bank. Replace with actual mapping from your device.
  local electrode_ids_ext=(
    128 132 136 140 144 148 152 156
    160 164 168 172 176 180 184 188
    192 196 200 204 208 212 216 220
    224 228 232 236 240 244 248 252
  )

  local channels=""
  local sep=""
  for ((i=0; i<count; i++)); do
    local eid ref
    if ((i < 32)); then
      eid="${electrode_ids_32[$i]}"
      ref="${reference_ids_32[$i]}"
    else
      j=$((i - 32))
      eid="${electrode_ids_ext[$j]}"
      ref=513
    fi
    channels+="${sep}{ \"id\": ${i}, \"electrode_id\": ${eid}, \"reference_id\": ${ref} }"
    sep=$',\n              '
  done
  echo "$channels"
}

# ─── Generate start config JSON ───────────────────────────────────────────────
# This is the runtime config passed to `synapsectl start`. It overrides the
# manifest.json defaults for this deployment.

header "Generating signal chain config (${MODE} mode, ${NUM_CHANNELS}ch)"

APP_NAME=$(python3 -c "import json; d=open('${APP_DIR}/manifest.json'); \
  print(json.load(d)['name'])" 2>/dev/null || echo "synapse-example-app")

CHANNELS_JSON=$(generate_channels_json "$NUM_CHANNELS")

# cursor_channels: channels that carry the primary decoded outputs.
# Hard mode decodes 12 controller signals; we pick 4 representative channels
# for the cursor interface. Adjust to match your model's output mapping.
if [[ "$MODE" == "easy" ]]; then
  CURSOR_CHANNELS='[0, 7, 16, 30]'
else
  CURSOR_CHANNELS='[0, 7, 16, 30, 40, 50, 52, 60]'
fi

cat > "$CONFIG_OUT" <<EOF
{
  "nodes": [
    {
      "type": "kBroadbandSource",
      "id": 1,
      "broadbandSource": {
        "peripheral_id": 100,
        "sample_rate_hz": ${SAMPLE_RATE},
        "bit_width": 12,
        "signal": {
          "electrode": {
            "channels": [
              ${CHANNELS_JSON}
            ],
            "low_cutoff_hz": 57,
            "high_cutoff_hz": 13489
          }
        }
      }
    },
    {
      "type": "kApplication",
      "id": 2,
      "application": {
        "name": "${APP_NAME}",
        "parameters": {
          "low_cutoff_hz": 200.0,
          "high_cutoff_hz": 5000.0,
          "spike_threshold_uv": 50.0,
          "waveform_size": 50,
          "refractory_period_us": 1000,
          "window_size": 5,
          "max_expected_rate": 10.0,
          "cursor_channels": ${CURSOR_CHANNELS},
          "enable_function_profiling": false,
          "enable_inference": true,
          "model_name": "${MODEL_NAME}",
          "num_channels": ${NUM_CHANNELS},
          "mode": "${MODE}"
        }
      }
    }
  ],
  "connections": [
    { "src_node_id": 1, "dst_node_id": 2 }
  ]
}
EOF

ok "Config written to: $CONFIG_OUT"
if [[ "$DRY_RUN" == false ]]; then
  log "Preview:"
  python3 -c "import json,sys; json.dump(json.load(open('${CONFIG_OUT}')), sys.stdout, indent=2)" \
    2>/dev/null || cat "$CONFIG_OUT"
fi

# ─── Step 1: Build ────────────────────────────────────────────────────────────
if [[ "$SKIP_BUILD" == false ]]; then
  header "Building Synapse App"
  log "App directory: $APP_DIR"
  run synapsectl apps build "$APP_DIR"
  ok "Build complete"
else
  warn "Skipping build (--skip-build)"
fi

# ─── Step 2: Deploy ONNX model ───────────────────────────────────────────────
if [[ "$SKIP_MODEL" == false ]]; then
  header "Deploying ONNX model"
  log "Model : $MODEL_PATH"
  log "Name  : $MODEL_NAME"
  log "Device: $DEVICE"
  run synapsectl deploy-model "$MODEL_PATH" \
    --name "$MODEL_NAME" \
    -u "$DEVICE"
  ok "Model deployed to /opt/scifi/data/models/${MODEL_NAME}.onnx"
else
  warn "Skipping model deploy (--skip-model)"
fi

# ─── Step 3: Deploy app ───────────────────────────────────────────────────────
if [[ "$SKIP_APP" == false ]]; then
  header "Deploying Synapse App"
  log "App   : $APP_NAME ($APP_DIR)"
  log "Device: $DEVICE"
  run synapsectl -u "$DEVICE" apps deploy "$APP_DIR"
  ok "App deployed"
else
  warn "Skipping app deploy (--skip-app)"
fi

# ─── Step 4: Start ───────────────────────────────────────────────────────────
header "Starting signal chain"
log "Config: $CONFIG_OUT"
log "Device: $DEVICE"
run synapsectl -u "$DEVICE" start "$CONFIG_OUT"
ok "Signal chain started"

# ─── Summary ─────────────────────────────────────────────────────────────────
echo ""
echo -e "${BOLD}${GREEN}╔══════════════════════════════════════════════════╗${NC}"
echo -e "${BOLD}${GREEN}║           Deployment Complete                    ║${NC}"
echo -e "${BOLD}${GREEN}╠══════════════════════════════════════════════════╣${NC}"
echo -e "${BOLD}${GREEN}║${NC}  Device    : ${BOLD}${DEVICE}${NC}"
echo -e "${BOLD}${GREEN}║${NC}  App       : ${BOLD}${APP_NAME}${NC}"
echo -e "${BOLD}${GREEN}║${NC}  Model     : ${BOLD}${MODEL_NAME}${NC} (${MODEL_PATH})"
echo -e "${BOLD}${GREEN}║${NC}  Mode      : ${BOLD}${MODE}${NC} (${NUM_CHANNELS} channels)"
echo -e "${BOLD}${GREEN}║${NC}  Inference : ${BOLD}enabled${NC}"
echo -e "${BOLD}${GREEN}╚══════════════════════════════════════════════════╝${NC}"
echo ""
log "To stream decoder output: synapsectl -u ${DEVICE} taps stream"
log "To view device logs     : synapsectl -u ${DEVICE} logs"
log "To stop                 : synapsectl -u ${DEVICE} stop"
