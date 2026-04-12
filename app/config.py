"""
Central configuration for the BCI Streamlit application.

All tunable constants live here. Grep for '# DIMENSION' to find
hardcoded spatial values you may need to adjust for your chessboard.
"""

from __future__ import annotations

import os

# ── Decoder websocket ────────────────────────────────────────────────────────
DECODER_WS_URL: str = "ws://localhost:8765"
USE_MOCK_DECODER: bool = False

# ── Arm ──────────────────────────────────────────────────────────────────────
USE_REAL_ARM: bool = True

# ── SciFi device (live mode) ─────────────────────────────────────────────────
SCIFI_DEVICE_IP: str = "192.168.8.123"
SYNAPSE_TAP_NAME: str = "joystick_out"
ARM_API_URL: str = "http://127.0.0.1:8000"

# ── Decoder output mapping ──────────────────────────────────────────────────
# v9 GRU decoder outputs (7 channels):
#   [0] joy_x  (left stick X)  → grid col (LEFT/RIGHT)
#   [1] joy_y  (left stick Y)  → grid row (UP/DOWN)
#   [2] rot    (right stick X) → wrist roll (not used for grid)
#   [3] depth  (right stick Y) → reach (not used for grid)
#   [4] lt     (left trigger)  → gripper open
#   [5] rt     (right trigger) → gripper close
#   [6] gate   → motion enable/disable
DECODER_DIRECTION_THRESHOLD: float = 0.15  # min magnitude to register a direction
DECODER_GATE_THRESHOLD: float = 0.5

# ── Session ──────────────────────────────────────────────────────────────────
SESSION_DURATION: int = 60          # seconds
RANDOM_SEED: int = 42

# ── Bit rate ─────────────────────────────────────────────────────────────────
N_SQUARES: int = 64                 # 8×8 grid — solution space for bit rate
# B = log2(N_SQUARES) × max(Sc − Si, 0) / t
# log2(64) = 6.0 bits per correct selection

# ── Grid dimensions ──────────────────────────────────────────────────────────
GRID_COLS: int = 8                  # A–H                      # DIMENSION
GRID_ROWS: int = 8                  # 1–8                      # DIMENSION
GRID_STEP: float = 1.0             # 1 action = 1 square       # DIMENSION

# ── Mock arm workspace (grid-coordinate space) ──────────────────────────────
# Piece positions are integer grid coords: col ∈ [0, GRID_COLS-1],
# row ∈ [0, GRID_ROWS-1].
WORKSPACE_X_MIN: float = 0.0       # leftmost column           # DIMENSION
WORKSPACE_X_MAX: float = 7.0       # rightmost column          # DIMENSION
WORKSPACE_Y_MIN: float = 0.0       # bottom row                # DIMENSION
WORKSPACE_Y_MAX: float = 7.0       # top row                   # DIMENSION
WORKSPACE_Z_MIN: float = 0.0       #                           # DIMENSION
WORKSPACE_Z_MAX: float = 0.0       # 2-D grid → z unused       # DIMENSION

# ── Real arm workspace (SO-101 metres) ───────────────────────────────────────
# Map these to the grid range above when using a real arm.
REAL_ARM_X_RANGE: tuple[float, float] = (0.10, 0.40)   # DIMENSION
REAL_ARM_Y_RANGE: tuple[float, float] = (-0.15, 0.15)  # DIMENSION

# ── H5 recording layout ─────────────────────────────────────────────────────
N_NEURAL_CHANNELS: int = 64
N_TARGET_CHANNELS: int = 12
N_TOTAL_CHANNELS: int = 76
SAMPLE_RATE_HZ: int = 32_000

# ── Waveform display ────────────────────────────────────────────────────────
BUFFER_SECONDS: int = 5             # rolling waveform window (default)
WAVEFORM_UPDATE_MS: int = 200       # refresh interval

# ── One-hot channel layout (mirrors arm/onehot_controller.py) ────────────────
NUM_ONEHOT_CHANNELS: int = 12
CH_LEFT_STICK_X:  int = 0          # pan left/right  → grid col ±
CH_LEFT_STICK_Y:  int = 1          # up/down (z)     — unused for 2-D grid
CH_RIGHT_STICK_X: int = 2          # rotate θ        — unused for 2-D grid
CH_RIGHT_STICK_Y: int = 3          # reach fwd/back  → grid row ±
# Channels 4-9: buttons/bumpers (no-op)
CH_TRIGGER_LEFT:  int = 10         # gripper open
CH_TRIGGER_RIGHT: int = 11         # gripper close

RAW_STICK_MAX: float = 32767.0

# ── Data paths ───────────────────────────────────────────────────────────────
RECORDINGS_DIR: str = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "recordings")
