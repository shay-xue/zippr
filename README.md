# NeuroChess — Brain-Controlled Robotic Arm for Chess

A closed-loop brain-computer interface (BCI) system that decodes neural signals from a [Science Corp SciFi](https://science.xyz) headstage into motor commands, driving a 6-DOF robotic arm to play chess on a physical board — scored by information transfer rate (bits/second).

Built for the **Global NeuroHack 2026** hackathon, Science Corp track.

---

## System Overview

```
┌─────────────┐    32 kHz     ┌──────────────┐   joystick_out   ┌───────────────┐
│   SciFi     │──────────────▸│  Synapse App  │────────────────▸│  Brain-to-Arm  │
│  Headstage  │  64-ch neural │  (on-device   │   7-ch decoded  │    Bridge      │
│             │   broadband   │   GRU decoder)│    controller   │  (Python)      │
└─────────────┘               └──────────────┘                  └───────┬───────┘
                                                                        │ delta cmds
                                                                        ▼
┌─────────────┐   arm pose    ┌──────────────┐   servo targets  ┌───────────────┐
│   ZIPPR     │◂─────────────│  Arm FastAPI  │────────────────▸│   SO-101      │
│  Dashboard  │  grid coords  │   Server     │   via IK solver  │  Robot Arm    │
│ (Streamlit) │               │  (port 8000) │                  │  (6-DOF)      │
└─────────────┘               └──────────────┘                  └───────────────┘
```

**Signal flow**: Neural activity → on-device spike extraction & GRU inference → decoded joystick vector → inverse kinematics → arm moves chess piece → dashboard tracks position and scores bit rate.

---

## Key Results

| Metric | Value |
|--------|-------|
| Decoder R² (joystick axes) | **0.847** |
| Trigger R² | 0.763 |
| Gate accuracy | 97.1% |
| End-to-end latency | ~16 ms (10 ms bin + <1 ms features + <5 ms inference) |
| Model size | 467K parameters (2-layer GRU) |
| Training data | 11.1 minutes across 60 recordings |

---

## Repository Structure

```
science-neurotech/
├── app/                        # ZIPPR Streamlit dashboard
│   ├── streamlit_app.py        #   Main UI: chess grid, bit rate, waveforms
│   ├── arm_interface.py        #   MockArm (grid sim) + RealArm (SO-101 projection)
│   ├── decoder_client.py       #   WebSocket client + Synapse tap reader
│   ├── chess_grid.py           #   8x8 board rendering and target logic
│   ├── bit_rate.py             #   B = log2(64) * max(Sc - Si, 0) / t
│   ├── waveform_viz.py         #   Live neural waveform display (8-ch scrollable)
│   ├── config.py               #   All tunable constants
│   ├── data_loader.py          #   HDF5 recording loader
│   └── session_logger.py       #   Session event logging
│
├── arm/                        # SO-101 robotic arm control
│   ├── api_server.py           #   FastAPI server (IK, delta commands, teleop UI)
│   ├── ik.py                   #   Inverse kinematics via ikpy + URDF
│   ├── api.py                  #   SO101ArmAPI with EMA smoothing
│   ├── controller.py           #   Low-level servo controller (Feetech STS3215)
│   ├── config.py               #   Joint limits, servo IDs, calibration
│   ├── onehot_controller.py    #   One-hot vector → arm command mapping
│   ├── keyboard_teleop.py      #   Keyboard-driven manual control
│   └── web_teleop_server.py    #   Browser-based teleoperation interface
│
├── scripts/                    # Training, analysis, and deployment
│   ├── train_decoder_v9.py     #   Final GRU decoder training (v9)
│   ├── validate_decoder_v9.py  #   Robustness tests: leakage, shuffle, baselines
│   ├── brain_to_arm.py         #   Synapse tap → arm delta command bridge
│   ├── train_decoder_v[1-8]*.py#   Decoder iteration history (MLP → GRU evolution)
│   ├── analyze_decoder_rnn.py  #   Post-training analysis and visualization
│   ├── bin_size_sweep*.py      #   Temporal resolution optimization
│   ├── eda.py                  #   Exploratory data analysis
│   ├── diagnose_*.py           #   Spectral and encoding diagnostics
│   └── collect_data*.py        #   Automated data collection scripts
│
├── src/                        # ML pipeline module (model comparison)
│   ├── pipeline.py             #   Orchestrator: preprocess → train → evaluate
│   ├── preprocess.py           #   HDF5 → binned spike counts → .npz
│   ├── train.py                #   MLP, XGBoost, SVM, CNN1D training
│   ├── evaluate.py             #   Per-channel R², visualizations, reports
│   └── utils.py                #   Config loading, seeding, logging
│
├── synapse_app/                # On-device C++ decoder (ARM64 cross-compiled)
│   └── synapse-example-app/
│       ├── src/
│       │   ├── fixed_weight_decoder.cpp   # Spike extraction + ONNX inference
│       │   └── fixed_weight_decoder.hpp   # Constants: channel stds, feat norms
│       ├── CMakeLists.txt                 # Build config (ONNX Runtime, Synapse SDK)
│       ├── manifest.json                  # Synapse app metadata
│       └── deploy_synapse_app.sh          # Docker cross-compile + deploy script
│
├── models/
│   ├── decoder.onnx            # Production v9 GRU model (ONNX, 1.8 MB)
│   ├── so101.urdf              # Robot arm kinematics definition
│   └── assets/                 # CAD files (STL) for SO-101 arm assembly
│
├── analysis/                   # Training outputs and diagnostics
│   └── decoder_rnn_shay/       #   v7/v8/v9 training curves, R² plots, predictions
│
├── data_collection/            # 7 collection runs with metadata and visualizations
│   ├── run_001_idle_baseline/
│   ├── run_002_active_joystick/
│   ├── run_003_hard_mode_all_inputs/
│   ├── run_004_structured_mapping/
│   ├── run_005_individual_mapping/
│   ├── run_006*/               # 32 training recordings
│   └── run_007_structured_training/  # 28 additional recordings
│
├── reports/                    # Model comparison reports (SVM, XGB, MLP, CNN1D)
├── config/                     # Device peripheral configs (easy/hard training)
├── tests/                      # Arm subsystem unit tests
├── CHANNEL_MAP.md              # Definitive 76-channel layout reference
├── setup.sh                    # Environment setup (venv, deps, verification)
├── requirements.txt            # Python dependencies
└── pyproject.toml              # Package config + CLI entry points
```

---

## Neural Decoder

### Architecture

The production decoder is a **2-layer GRU** (Gated Recurrent Unit) that processes sequences of spike-count features extracted from broadband neural recordings.

```
Input: 64-ch broadband @ 32 kHz
  │
  ▼  Bandpass filter (200–5000 Hz, 2nd-order Butterworth)
  │
  ▼  Bin to 100 ms windows (3200 samples/bin)
  │
  ▼  Spike detection at 3σ / 4σ / 5σ thresholds
  │   → 64 channels × 3 thresholds = 192 features per bin
  │
  ▼  Normalize (subtract training mean, divide by training std)
  │
  ▼  Buffer last 15 bins (1.5 s context)
  │
  ▼  GRU (input_proj: 192→192, 2 layers, hidden=192, dropout=0.3)
  │
  ▼  Task-specific heads:
      ├── Joystick (4 axes) → Tanh [-1, 1]
      ├── Triggers (2)      → Sigmoid [0, 1]
      └── Gate (1)          → Sigmoid [0, 1]
```

### Decoder Evolution

| Version | Architecture | Features | Bin Size | R² (joy) | Notes |
|---------|-------------|----------|----------|----------|-------|
| v1–v3 | MLP | Raw / spike counts | 10 ms | <0.10 | Baseline exploration |
| v4–v5 | MLP (deeper) | Track A (192) | 10 ms | 0.15–0.25 | Feature engineering |
| v6 | MLP (1.5M params) | Track A+B (2880) | 10 ms | ~0.33 | Overfitting, slow |
| v7 | GRU (467K) | Track A (192) | 10 ms | 0.72 | Temporal modeling works |
| v8 | GRU (467K) | Track A (192) | 100 ms | 0.81 | Anti-lag, coarser bins |
| **v9** | **GRU (467K)** | **Track A (192)** | **100 ms** | **0.847** | **Production: +run007 data** |

### Training

```bash
python scripts/train_decoder_v9.py
```

- **Data**: 60 HDF5 recordings (32 from run006 + 28 from run007), 11.1 minutes total
- **Split**: File-based stratified train/val (no temporal leakage)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4) with cosine annealing
- **Regularization**: GRU dropout 0.3, gradient clipping 1.0, input noise augmentation (σ=0.05), early stopping (patience=40)
- **Validation**: `validate_decoder_v9.py` runs 8 robustness checks (shuffle test, leakage detection, naive baselines, autocorrelation analysis)

### ONNX Export

The model exports as `(1, 192, 15)` input → `(1, 7)` output to match the Synapse C++ app's convention. An internal transpose converts to GRU's `(1, 15, 192)` format.

---

## Hardware

### SciFi Headstage

- **Device**: Science Corp SciFi neural recording headstage
- **Channels**: 64 neural (mock-encoded broadband) + 12 controller labels (training mode)
- **Sample rate**: 32,000 Hz
- **Interface**: Synapse SDK over WiFi
- **Peripherals**: Easy (34 electrodes), Medium, Hard (64 electrodes) — training/testing pairs

### SO-101 Robot Arm

- **DOF**: 6 (shoulder pan, shoulder lift, elbow flex, wrist flex, wrist roll, gripper)
- **Servos**: Feetech STS3215
- **Kinematics**: URDF model + ikpy inverse kinematics solver
- **Control**: FastAPI server at `http://127.0.0.1:8000` accepting delta end-effector commands
- **Workspace mapping**: Physical (x, y) → chess grid (col, row) via perpendicular projection

### Controller Mapping

The bridge script (`brain_to_arm.py`) maps decoded outputs to arm degrees of freedom:

| Decoder Output | Arm Action | Scale |
|---------------|------------|-------|
| Left stick X (joy_x) | Shoulder pan (left/right) | 0.40 rad/unit |
| Left stick Y (joy_y) | Height (elbow up/down) | 0.10 m/unit |
| Right stick X (rot) | Wrist roll | 25.0 deg/unit |
| Right stick Y (depth) | Reach (extend/curl) | 0.10 m/unit |
| Left trigger | Gripper open | 10.0 units |
| Right trigger | Gripper close | 10.0 units |
| Gate | Motion enable/disable (threshold: 0.5) | — |

---

## ZIPPR Dashboard

The Streamlit-based dashboard provides real-time visualization and scoring:

- **Chess Grid**: 8x8 board showing piece position (mapped from real arm pose) and randomly generated targets
- **Bit Rate**: Information transfer rate calculated as `B = log2(64) * max(correct - incorrect, 0) / elapsed_seconds`
- **Neural Waveforms**: Live 8-channel scrollable display of broadband neural signals
- **Session Control**: Start/stop sessions, configurable duration, mock/live mode toggle

```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

---

## Quick Start

### Prerequisites

- Python 3.10+
- Science Corp Synapse SDK (`science-synapse >= 2.2.7`)
- Docker (for Synapse app cross-compilation)
- `synapsectl` CLI tool

### Setup

```bash
git clone https://github.com/manrajmondair/science-neurotech.git
cd science-neurotech
bash setup.sh
source venv/bin/activate
```

### Run the Full System

**1. Deploy the decoder to the SciFi device:**
```bash
cd synapse_app
bash deploy_synapse_app.sh          # Cross-compiles and deploys .deb
synapsectl start                    # Start the Synapse app on-device
```

**2. Start the arm API server:**
```bash
python arm/api_server.py            # Serves on http://127.0.0.1:8000
```

**3. Start the brain-to-arm bridge:**
```bash
python scripts/brain_to_arm.py --device-ip <SCIFI_IP>
```

**4. Launch the dashboard:**
```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

### Mock Mode (No Hardware)

Set in `app/config.py`:
```python
USE_MOCK_DECODER = True
USE_REAL_ARM = False
```

Then launch only the dashboard — the mock decoder server and simulated arm are built in.

---

## On-Device Inference

The Synapse app runs a real-time C++ pipeline on the SciFi's ARM64 processor:

1. Receive 64-channel broadband at 32 kHz
2. Bandpass filter (200–5000 Hz, 2nd-order Butterworth)
3. Accumulate 3200 samples → 1 bin (100 ms)
4. Extract spike counts at 3σ / 4σ / 5σ → 192 features
5. Normalize with training-derived mean/std
6. Push into circular buffer (15 bins)
7. Run ONNX model → 7 outputs `[joy_x, joy_y, rot, depth, lt, rt, gate]`
8. Publish on `joystick_out` tap at 10 Hz

Latency: ~16 ms per inference cycle.

---

## Data Collection

Seven structured recording runs were collected to iteratively build the training dataset:

| Run | Purpose | Files | Duration |
|-----|---------|-------|----------|
| 001 | Idle baseline | 1 | — |
| 002 | Active joystick | 1 | — |
| 003 | Hard mode (all inputs) | 2 | — |
| 004 | Structured channel mapping | 12 | — |
| 005 | Individual channel isolation | 12 | — |
| 006 | Primary training set | 32 | 5.5 min |
| 007 | Extended training set | 28 | 5.6 min |

Channel layout documented in [`CHANNEL_MAP.md`](CHANNEL_MAP.md): 64 neural channels (0–63) encoding mock broadband data, 12 label channels (64–75) carrying Xbox controller inputs during training.

---

## Model Comparison

Beyond the GRU decoder, several architectures were evaluated on the same data:

| Model | Parameters | R² (avg) | Notes |
|-------|-----------|----------|-------|
| SVM | — | Low | Linear kernel, poor temporal modeling |
| XGBoost | ~100 trees | Moderate | Strong per-bin, no sequence context |
| MLP (2-layer) | ~1.5M | 0.33 | Flattened window, overfits quickly |
| CNN1D | ~500K | Moderate | Local temporal patterns only |
| **GRU v9** | **467K** | **0.847** | **Sequential context, best by wide margin** |

Full comparison reports with per-channel R², residual plots, and prediction traces are in [`reports/`](reports/).

---

## Project Structure Details

### Key Configuration Files

| File | Purpose |
|------|---------|
| `app/config.py` | Dashboard constants (grid size, thresholds, device IPs) |
| `config/hard_training.json` | SciFi peripheral config (64+12 channels, 32 kHz) |
| `CHANNEL_MAP.md` | Definitive 76-channel layout reference |
| `models/so101.urdf` | Robot arm kinematics definition |
| `synapse_app/manifest.json` | On-device app metadata |

### CLI Entry Points (via pyproject.toml)

```bash
arm-api-server          # Start the FastAPI arm control server
arm-keyboard-teleop     # Manual keyboard control
arm-web-teleop          # Browser-based teleoperation
arm-check-motor-ids     # Verify servo hardware
arm-set-zero            # Calibrate arm zero position
```

---

## Team

- **Manraj Mondair** — Decoder pipeline (v1–v9), Synapse app deployment, system integration
- **Shay** — RNN decoder analysis, PCA diagnostics, bin-size optimization, v8/v9 evaluation
- **Medha** — ML comparison pipeline (SVM/XGB/MLP/CNN1D), preprocessing, ZIPPR dashboard UI
- **Yoyo** — Exploratory data analysis, spectral diagnostics, sklearn baselines

---

## License

This project was developed for the Global NeuroHack 2026 hackathon.
