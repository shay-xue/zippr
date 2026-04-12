<p align="center">
  <img src="https://img.shields.io/badge/R%C2%B2-0.847-brightgreen?style=for-the-badge" alt="R²">
  <img src="https://img.shields.io/badge/Latency-16ms-blue?style=for-the-badge" alt="Latency">
  <img src="https://img.shields.io/badge/Params-467K-orange?style=for-the-badge" alt="Parameters">
  <img src="https://img.shields.io/badge/Python-3.10+-yellow?style=for-the-badge&logo=python&logoColor=white" alt="Python">
</p>

<h1 align="center">ZIPPR</h1>

<p align="center">
  <strong>Neural signals in. Robotic motion out. Zipped together.</strong>
</p>

<p align="center">
  A general-purpose brain-to-robotic-arm interface that decodes neural activity into continuous multi-axis motor commands in real time. Built on the <a href="https://science.xyz">Science Corp SciFi</a> platform for the <strong>Global NeuroHack 2026</strong> hackathon.
</p>

---

## Why This Matters

ZIPPR is a **general-purpose neural-to-motor bridge** — decoded brain signals drive a 6-DOF robotic arm through continuous 3D motion with grip control. The decoder learns to map arbitrary neural patterns to 7 simultaneous output channels, meaning the same architecture can control any downstream actuator: a robotic arm, a prosthetic hand, a wheelchair, or a cursor.

We validated ZIPPR on chess because it provides a structured, quantifiable benchmark (information transfer in bits/second), but chess is just one of an infinite number of possible applications. The same pipeline generalizes directly to:

- **Prosthetic limbs** — continuous decoded intent mapped to multi-joint articulation for amputees or individuals with paralysis
- **Assistive robotics** — brain-driven manipulation for daily tasks, restoring independence to people with severe motor impairments
- **Teleoperation** — neural command of remote systems in surgical, industrial, or hazardous environments
- **Rehabilitation** — closed-loop neurofeedback with real-time motor decoding to accelerate recovery

The core contribution is a complete, end-to-end, deployable closed-loop BCI: from raw broadband neural recordings on a wireless headstage, through on-device real-time inference at 16 ms latency, to inverse-kinematics-driven arm motion — all running simultaneously with no external compute.

---

## System Architecture

```
 ┌──────────────┐                 ┌────────────────┐                    ┌─────────────────┐
 │              │    64-channel   │                │     7-channel      │                 │
 │    SciFi     │───broadband───▸│  Synapse App   │───decoded intent──▸│  Brain-to-Arm   │
 │  Headstage   │   @ 32 kHz     │  (on-device    │    vector @ 10 Hz  │    Bridge        │
 │              │                │   GRU decoder) │                    │   (Python)       │
 └──────────────┘                └────────────────┘                    └────────┬────────┘
                                                                                │
                                                                         delta commands
                                                                                │
 ┌──────────────┐    arm pose    ┌────────────────┐    servo targets   ┌────────▼────────┐
 │              │◂──projection───│                │───via IK solver───▸│                 │
 │    ZIPPR     │                │  Arm FastAPI   │                    │    SO-101       │
 │  Dashboard   │                │   Server       │                    │   Robot Arm     │
 │ (Streamlit)  │                │  (port 8000)   │                    │    (6-DOF)      │
 └──────────────┘                └────────────────┘                    └─────────────────┘
```

**Signal flow**: Neural activity is recorded and decoded entirely on-device into a 7-channel motor intent vector. A bridge script maps this to end-effector delta commands. The arm's FastAPI server solves inverse kinematics and drives servos. The dashboard reads the arm's physical pose, projects it onto the task space, and scores performance.

---

## Performance

| Metric | Value |
|:-------|------:|
| Decoder R² (joystick axes) | **0.847** |
| Decoder R² (triggers) | **0.763** |
| Gate classification accuracy | **97.1%** |
| End-to-end latency | **~16 ms** |
| Model parameters | **467K** |
| ONNX model size | **1.8 MB** |
| Training data required | **11.1 minutes** |
| On-device inference rate | **10 Hz** |
| Decoded degrees of freedom | **7** (4 axes + 2 triggers + gate) |

---

## Neural Decoder

### Architecture

A **2-layer GRU** processes sequences of multi-threshold spike-count features extracted from 64-channel broadband neural recordings.

```
64-ch broadband @ 32 kHz
  │
  ▼  Bandpass filter (200–5000 Hz, 2nd-order Butterworth)
  │
  ▼  Bin into 100 ms windows (3,200 samples/bin)
  │
  ▼  Spike detection at 3σ / 4σ / 5σ thresholds
  │   → 64 channels x 3 thresholds = 192 features per bin
  │
  ▼  Z-score normalize (training-derived mean/std)
  │
  ▼  Sequence buffer: last 15 bins (1.5 seconds of context)
  │
  ▼  GRU (input_proj 192→192, 2 layers, hidden=192, dropout=0.3)
  │
  ▼  Task-specific output heads:
      ├── Joystick  (4 axes)  → Tanh   [-1, 1]
      ├── Triggers  (2 axes)  → Sigmoid  [0, 1]
      └── Gate      (1 axis)  → Sigmoid  [0, 1]
```

### Decoder Evolution

Nine decoder versions were developed iteratively, progressing from simple baselines to the production GRU:

| Version | Architecture | Bin Size | R² (joy) | Key Insight |
|:-------:|:------------|:--------:|:--------:|:------------|
| v1–v3 | MLP | 10 ms | < 0.10 | Established baseline, identified feature needs |
| v4–v5 | MLP (deeper) | 10 ms | 0.15–0.25 | Multi-threshold spike features improve signal |
| v6 | MLP (1.5M params) | 10 ms | ~0.33 | Flattened windows destroy temporal structure |
| v7 | **GRU** (467K) | 10 ms | 0.72 | Sequential modeling captures temporal dynamics |
| v8 | GRU (467K) | 100 ms | 0.81 | Coarser bins reduce noise, improve generalization |
| **v9** | **GRU (467K)** | **100 ms** | **0.847** | **Additional training data pushes final accuracy** |

**Why GRU over MLP**: The MLP flattens T x F features into one vector, destroying temporal ordering. The GRU processes bins sequentially, learning temporal patterns in neural activity. With only 11 minutes of data, the GRU (467K params) is actually smaller than the v6 MLP (1.5M+ params), reducing overfitting.

### Training

```bash
python scripts/train_decoder_v9.py
```

- **Data**: 60 HDF5 recordings (32 from run006 + 28 from run007), 11.1 minutes total
- **Split**: File-based stratified train/val (no temporal leakage across files)
- **Optimizer**: AdamW (lr=5e-4, weight_decay=1e-4) with cosine annealing warm restarts
- **Regularization**: GRU dropout 0.3, gradient clipping at 1.0, input noise augmentation (σ=0.05), early stopping (patience=40)
- **Validation**: `validate_decoder_v9.py` runs 8 robustness checks — shuffle test, data leakage detection, feature normalization leakage, naive baselines, autocorrelation analysis, label coverage, per-file R², and prediction variance

### Model Comparison

Five architectures were benchmarked on identical preprocessed data:

| Model | Parameters | R² (avg) | Notes |
|:------|:---------:|:--------:|:------|
| SVM | 41K support vectors | 0.08 | Linear kernel, no temporal modeling |
| XGBoost | 100 estimators | 0.19 | Strong per-bin, but no sequence context |
| MLP | 1.5M | 0.33 | Flattened window input, prone to overfitting |
| CNN1D | 510K | 0.29 | Captures local patterns, limited receptive field |
| **GRU v9** | **467K** | **0.847** | **Sequential context over 1.5 s, production model** |

Full per-channel R² breakdowns, residual plots, and prediction traces are in [`reports/`](reports/).

---

## On-Device Inference

The Synapse app runs a real-time C++ inference pipeline directly on the SciFi headstage's ARM64 processor:

1. Record 64-channel broadband neural data at 32 kHz
2. Bandpass filter each channel (200–5000 Hz, 2nd-order Butterworth)
3. Accumulate 3,200 samples into one 100 ms bin
4. Extract spike counts at 3σ / 4σ / 5σ thresholds → 192 features
5. Z-score normalize using embedded training-derived mean/std arrays
6. Push features into a circular buffer of 15 bins (1.5 s context)
7. Run ONNX Runtime inference → 7 outputs: `[joy_x, joy_y, rot, depth, lt, rt, gate]`
8. Publish decoded vector on the `joystick_out` Synapse tap at 10 Hz

**Latency breakdown**: 10 ms bin accumulation + < 1 ms feature extraction + < 5 ms ONNX inference = **~16 ms total**.

The ONNX model accepts input shape `(1, 192, 15)` — batch, features, sequence — and outputs `(1, 7)`. An internal transpose converts to the GRU's native `(1, 15, 192)` ordering, making the model plug-and-play with the Synapse C++ SDK.

---

## Robotic Arm

### SO-101 Specifications

| Property | Value |
|:---------|:------|
| Degrees of freedom | 6 (shoulder pan/lift, elbow flex, wrist flex/roll, gripper) |
| Servos | Feetech STS3215 |
| Kinematics | URDF model + ikpy inverse kinematics solver |
| Control interface | FastAPI server accepting end-effector delta commands |
| Smoothing | Exponential moving average (α=0.7) on joint targets |

### Brain-to-Arm Mapping

The bridge script reads the decoded intent vector from the Synapse `joystick_out` tap and maps it to physical arm motion:

| Decoded Channel | Physical Action | Scale Factor |
|:---------------|:---------------|-------------:|
| Left stick X | Shoulder pan (left / right) | 0.40 rad/unit |
| Left stick Y | Elbow height (up / down) | 0.10 m/unit |
| Right stick X | Wrist roll | 25.0 deg/unit |
| Right stick Y | Reach (extend / curl) | 0.10 m/unit |
| Left trigger | Gripper open | 10.0 deg/unit |
| Right trigger | Gripper close | 10.0 deg/unit |
| Gate < 0.5 | Suppress all joystick motion | — |

### Workspace Calibration

The arm's physical end-effector position is projected into any downstream task coordinate system. For the chess validation task, this is a perpendicular projection from 3D arm pose onto the 2D board:
- **Arm Y** (shoulder pan) → board **column** &ensp;|&ensp; **Arm X** (radial reach) → board **row**
- Calibrated range: Y ∈ [-0.16, +0.16] m, X ∈ [0.08, 0.40] m

This projection layer is modular — swapping it out adapts the arm to any spatial task without retraining the decoder.

---

## ZIPPR Dashboard

The Streamlit-based real-time interface provides:

- **Task Grid** — 8x8 board displaying the current position (projected from real arm pose) and randomly generated targets
- **Bit Rate Scoring** — Information transfer rate: `B = log2(N) x max(correct - incorrect, 0) / elapsed_seconds`
- **Neural Waveforms** — Live 8-channel scrollable display of broadband neural signals with configurable time windows
- **Session Control** — Start/stop timed sessions, real-time correct/incorrect tallies
- **Decoder Telemetry** — Direction decisions, confidence scores, and gate state visualization

---

## Quick Start

### Prerequisites

- Python 3.10+
- [Science Corp Synapse SDK](https://science.xyz) (`science-synapse >= 2.2.7`)
- Docker (for cross-compilation to ARM64)
- `synapsectl` CLI

### Installation

```bash
git clone https://github.com/manrajmondair/science-neurotech.git
cd science-neurotech
bash setup.sh
source venv/bin/activate
```

### Full System Launch

```bash
# 1. Deploy decoder to SciFi device
cd synapse_app && bash deploy_synapse_app.sh && synapsectl start

# 2. Start arm control server
python arm/api_server.py

# 3. Start brain-to-arm bridge (set env vars for live hardware)
USE_REAL_ARM=true USE_MOCK_DECODER=false \
python scripts/brain_to_arm.py --device-ip <SCIFI_IP>

# 4. Launch dashboard
USE_REAL_ARM=true USE_MOCK_DECODER=false \
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

### Demo Mode (No Hardware)

The dashboard runs fully standalone with a built-in mock decoder and simulated arm:

```bash
streamlit run app/streamlit_app.py --server.port 8501 --server.headless true
```

No environment variables needed — mock mode is the default.

### CLI Tools

```bash
arm-api-server          # FastAPI arm control server
arm-keyboard-teleop     # Manual keyboard control
arm-web-teleop          # Browser-based teleoperation UI
arm-check-motor-ids     # Verify servo hardware connections
arm-set-zero            # Calibrate arm to zero position
```

---

## Data Collection

Seven structured recording sessions were conducted to iteratively build and validate the training dataset:

| Run | Purpose | Recordings | Duration |
|:---:|:--------|:----------:|:--------:|
| 001 | Idle baseline — characterize noise floor | 1 | — |
| 002 | Active joystick — verify neural encoding | 1 | — |
| 003 | Hard mode — all 12 inputs simultaneously | 2 | — |
| 004 | Structured mapping — systematic channel validation | 12 | — |
| 005 | Individual isolation — one input at a time | 12 | — |
| 006 | Primary training set | 32 | 5.5 min |
| 007 | Extended training set | 28 | 5.6 min |

Hardware: 64 neural channels + 12 label channels (Xbox controller ground truth), 32 kHz, 12-bit ADC. Full channel layout in [`CHANNEL_MAP.md`](CHANNEL_MAP.md).

---

## Repository Structure

```
zippr/
├── app/                          Streamlit dashboard (ZIPPR UI)
│   ├── streamlit_app.py            Main application (task grid, bit rate, waveforms)
│   ├── arm_interface.py            MockArm + RealArm (SO-101 projection)
│   ├── decoder_client.py           WebSocket + Synapse tap decoder clients
│   ├── chess_grid.py               Board rendering and target logic
│   ├── bit_rate.py                 Information transfer rate calculator
│   ├── waveform_viz.py             Live neural waveform visualization
│   └── config.py                   Central configuration constants
│
├── arm/                          SO-101 robotic arm control stack
│   ├── api_server.py               FastAPI server (IK, deltas, teleop UI)
│   ├── ik.py                       Inverse kinematics (ikpy + URDF)
│   ├── controller.py               Low-level servo controller (STS3215)
│   ├── onehot_controller.py        One-hot vector to arm command mapping
│   ├── keyboard_teleop.py          Keyboard-driven manual control
│   └── web_teleop_server.py        Browser-based teleoperation
│
├── scripts/                      Training, analysis, and deployment
│   ├── train_decoder_v9.py         Production GRU decoder training
│   ├── validate_decoder_v9.py      8-test robustness validation suite
│   ├── brain_to_arm.py             Synapse tap → arm delta command bridge
│   ├── train_decoder_v[1-8]*.py    Decoder iteration history
│   ├── analyze_decoder_rnn.py      Post-training analysis
│   └── bin_size_sweep*.py          Temporal resolution optimization
│
├── src/                          ML comparison pipeline
│   ├── pipeline.py                 Preprocess → train → evaluate orchestrator
│   ├── train.py                    MLP, XGBoost, SVM, CNN1D training
│   └── evaluate.py                 Per-channel R², visualization, reports
│
├── synapse_app/                  On-device C++ decoder (ARM64)
│   └── synapse-example-app/
│       ├── src/fixed_weight_decoder.cpp    Spike extraction + ONNX inference
│       ├── src/fixed_weight_decoder.hpp    Embedded normalization constants
│       ├── CMakeLists.txt                  ONNX Runtime + Synapse SDK build
│       └── deploy_synapse_app.sh           Cross-compile and deploy script
│
├── models/
│   ├── decoder.onnx              Production v9 GRU model (ONNX)
│   ├── so101.urdf                Robot arm kinematics definition
│   └── assets/                   CAD files (STL) for arm assembly
│
├── analysis/                     Training curves, R² plots, diagnostics
├── data_collection/              7 recording runs with metadata
├── reports/                      Model comparison reports with figures
├── config/                       Device peripheral configurations
├── tests/                        Arm subsystem unit tests
├── CHANNEL_MAP.md                76-channel layout reference
├── setup.sh                      Environment setup script
├── requirements.txt              Python dependencies
└── pyproject.toml                Package configuration + CLI entry points
```

---

## Key Configuration

| File | Purpose |
|:-----|:--------|
| `app/config.py` | Dashboard constants — grid size, decoder thresholds, device IPs |
| `config/hard_training.json` | SciFi peripheral config — 64+12 channels, 32 kHz |
| `CHANNEL_MAP.md` | Definitive 76-channel hardware layout |
| `models/so101.urdf` | Robot arm kinematics (joint limits, link geometry) |
| `synapse_app/manifest.json` | On-device app metadata and tap declarations |

---

<p align="center">
  <em>Global NeuroHack 2026 — Science Corp Track</em>
</p>
