"""
ZIPPR — Neural BCI Real-Time Interface

Zipper-themed BCI demo dashboard.

Layout
------
  [ZIPPR logo + animated zipper teeth]
  ─────────────────────────────────────
  [Chess Grid (left)]  |  [Bit Rate + Stats (right)]
  ─────────────────────────────────────
  [Neural Waveforms — 8 channels, scrollable]

Launch with:  streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import math
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go

sys.path.insert(0, os.path.dirname(__file__))

from arm_interface import MockArm, RealArm
from bit_rate import BitRateCalculator, LOG2_N
from chess_grid import (
    optimal_direction,
    pick_random_target,
    render_grid_html,
    square_label,
)
from config import (
    ARM_API_URL,
    BUFFER_SECONDS,
    DECODER_DIRECTION_THRESHOLD,
    DECODER_GATE_THRESHOLD,
    DECODER_WS_URL,
    N_SQUARES,
    RANDOM_SEED,
    SCIFI_DEVICE_IP,
    SESSION_DURATION,
    SYNAPSE_TAP_NAME,
    USE_MOCK_DECODER,
    USE_REAL_ARM,
)
from data_loader import H5DataLoader, MockDataLoader
from decoder_client import (
    DecoderClient,
    MockDecoderServer,
    SynapseDecoderClient,
    set_target_direction_hint,
)
from session_logger import SessionEvent, SessionLogger
from waveform_viz import DIRECTION_LABELS, build_waveform_figure, get_top_channels

# ── Page config ───────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ZIPPR",
    page_icon="⚡",
    layout="wide",
)

# ── CSS injection ─────────────────────────────────────────────────────────────

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=VT323&family=Share+Tech+Mono&display=swap');

/* ── Global ─────────────────────────────────────────────────── */
html, body, [class*="css"], .stMarkdown p, .stText {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 16px !important;
    color: #161510;
}

/* ── Headings ────────────────────────────────────────────────── */
h1, h2, h3, h4, h5 {
    font-family: 'VT323', monospace !important;
    color: #161510 !important;
    letter-spacing: 0.06em !important;
}
h2 { font-size: 2.2em !important; }
h3 { font-size: 1.9em !important; }
h4 { font-size: 1.6em !important; }

/* ── ZIPPR Logo ──────────────────────────────────────────────── */
.zippr-logo {
    font-family: 'VT323', monospace;
    font-size: 120px;
    line-height: 1;
    color: #161510;
    text-align: center;
    letter-spacing: 0.06em;
    text-shadow: 6px 6px 0 #878672;
    margin: 0;
    padding: 14px 0 0;
    user-select: none;
}

/* ── Zipper teeth animation ───────────────────────────────────── */
.zipper-row {
    display: flex;
    justify-content: center;
    align-items: flex-end;
    gap: 0;
    padding-bottom: 6px;
    user-select: none;
}

.z-half-left {
    display: flex;
    flex-direction: row-reverse;
    gap: 3px;
    animation: zip-left 3s ease-in-out infinite;
}
.z-half-right {
    display: flex;
    gap: 3px;
    animation: zip-right 3s ease-in-out infinite;
}

@keyframes zip-left {
    0%, 100% { transform: translateX(0);     gap: 3px; }
    45%       { transform: translateX(-20px); gap: 9px; }
}
@keyframes zip-right {
    0%, 100% { transform: translateX(0);    gap: 3px; }
    45%       { transform: translateX(20px); gap: 9px; }
}

.ztooth {
    display: inline-block;
    width: 15px;
    height: 24px;
    background: #545333;
    border-radius: 3px 3px 0 0;
}
.ztooth:nth-child(even) {
    height: 18px;
    margin-bottom: 6px;
    background: #878672;
}

.z-pull {
    font-size: 32px;
    line-height: 1;
    color: #161510;
    margin: 0 8px;
    padding-bottom: 2px;
    animation: pull-beat 3s ease-in-out infinite;
}
@keyframes pull-beat {
    0%, 100% { transform: scale(1); }
    45%       { transform: scale(1.25) translateY(-2px); }
}

/* ── Status pill ──────────────────────────────────────────────── */
.status-pill {
    display: inline-block;
    background: #545333;
    color: #FDFBD4;
    font-family: 'VT323', monospace;
    font-size: 22px;
    letter-spacing: 0.1em;
    padding: 5px 20px;
    border-radius: 3px;
}

/* ── Bit rate big display ─────────────────────────────────────── */
.bps-number {
    font-family: 'VT323', monospace;
    font-size: 118px;
    line-height: 1;
    text-align: center;
    color: #161510;
    letter-spacing: -0.02em;
}
.bps-unit {
    font-family: 'VT323', monospace;
    font-size: 34px;
    text-align: center;
    color: #878672;
    letter-spacing: 0.08em;
    margin-top: -10px;
}

/* ── Metric cards — remove default shadow ─────────────────────── */
[data-testid="metric-container"] {
    background: #D9D7B6 !important;
    border: 2px solid #878672 !important;
    border-radius: 3px !important;
    box-shadow: none !important;
    padding: 10px 14px !important;
}
[data-testid="metric-container"] label {
    font-size: 15px !important;
    color: #545333 !important;
}
[data-testid="metric-container"] [data-testid="stMetricValue"] {
    font-size: 26px !important;
    font-family: 'VT323', monospace !important;
    color: #161510 !important;
}

/* ── Buttons ──────────────────────────────────────────────────── */
.stButton > button {
    font-family: 'VT323', monospace !important;
    font-size: 28px !important;
    letter-spacing: 0.12em !important;
    background: #545333 !important;
    color: #FDFBD4 !important;
    border: 3px solid #545333 !important;
    border-radius: 3px !important;
    padding: 8px 28px !important;
    transition: background 0.15s, border-color 0.15s !important;
}
.stButton > button:hover {
    background: #161510 !important;
    border-color: #161510 !important;
}
.stButton > button:disabled {
    background: #D9D7B6 !important;
    border-color: #878672 !important;
    color: #878672 !important;
}

/* ── Progress bar ─────────────────────────────────────────────── */
[data-testid="stProgressBar"] > div > div {
    background: #545333 !important;
}

/* ── Sidebar ──────────────────────────────────────────────────── */
[data-testid="stSidebar"] {
    background: #D9D7B6 !important;
}
[data-testid="stSidebar"] * {
    font-family: 'Share Tech Mono', monospace !important;
}

/* ── Welcome screen ───────────────────────────────────────────── */
.welcome-sub {
    font-family: 'Share Tech Mono', monospace !important;
    font-size: 18px !important;
    color: #878672 !important;
    text-align: center;
    margin: 0 0 32px !important;
}

/* ── Session frozen banner ────────────────────────────────────── */
.frozen-banner {
    background: #161510;
    color: #FDFBD4;
    font-family: 'VT323', monospace;
    font-size: 38px;
    text-align: center;
    padding: 18px;
    letter-spacing: 0.22em;
    border: 4px solid #545333;
    border-radius: 4px;
    margin: 12px 0 20px;
}

/* ── Final BPS display ────────────────────────────────────────── */
.final-bps-label {
    font-family: 'VT323', monospace;
    font-size: 30px;
    color: #878672;
    text-align: center;
}
.final-bps-value {
    font-family: 'VT323', monospace;
    font-size: 148px;
    line-height: 1;
    text-align: center;
    color: #FDFBD4;
    background: #161510;
    border: 4px solid #545333;
    border-radius: 4px;
    padding: 6px 20px;
    display: inline-block;
    letter-spacing: -0.02em;
}

/* ── Channel nav label ────────────────────────────────────────── */
.ch-nav-label {
    font-family: 'VT323', monospace;
    font-size: 22px;
    color: #545333;
    padding-top: 8px;
}

/* ── Divider ──────────────────────────────────────────────────── */
hr { border-color: #878672 !important; border-width: 2px !important; }

/* ── Remove top padding ───────────────────────────────────────── */
.block-container { padding-top: 0.4rem !important; }

/* ── Info / warning boxes ─────────────────────────────────────── */
.stAlert { font-size: 15px !important; }
</style>
"""

st.markdown(_CSS, unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────

def _render_header() -> None:
    n = 14
    left  = "".join('<div class="ztooth"></div>' for _ in range(n))
    right = "".join('<div class="ztooth"></div>' for _ in range(n))
    st.markdown(
        f'<div style="text-align:center;">'
        f'  <div class="zippr-logo">ZIPPR</div>'
        f'  <div class="zipper-row">'
        f'    <div class="z-half-left">{left}</div>'
        f'    <div class="z-pull">◉</div>'
        f'    <div class="z-half-right">{right}</div>'
        f'  </div>'
        f'</div>',
        unsafe_allow_html=True,
    )


_render_header()


# ── Session state init ────────────────────────────────────────────────────────

def _init_state() -> None:
    defaults: dict = {
        "session_running": False,
        "session_start_time": 0.0,
        "session_ended": False,
        "sc": 0,
        "si": 0,
        "piece_col": 0,
        "piece_row": 0,
        "target_col": 4,
        "target_row": 4,
        "flash": None,
        "flash_time": 0.0,
        "decoder_client": None,
        "mock_server": None,   # kept alive across sessions to avoid port race
        "data_loader": None,
        "arm": None,
        "bit_rate": None,
        "logger": None,
        "rng": None,
        "action_overlays": [],
        "last_decoder_ts": 0.0,
        "waveform_page": 0,
        "cfg_ws_url": DECODER_WS_URL,
        "cfg_mock_decoder": USE_MOCK_DECODER,
        "cfg_real_arm": USE_REAL_ARM,
        "cfg_time_window": BUFFER_SECONDS,
        "final_bps": 0.0,
        "final_sc": 0,
        "final_si": 0,
        "final_duration": 0,
        # Demo Day mode state
        "game_mode": "Chess Grid",
        "demo_target_piece": None,
        "demo_trial_active": False,
        "demo_awaiting_confirm": False,
        "demo_trial_start": 0.0,
        "demo_trials": [],       # list of {"piece", "correct", "time"}
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


_init_state()


# ── Session lifecycle — defined BEFORE sidebar so buttons can call them ───────

def _start_session() -> None:
    ss = st.session_state
    is_demo = ss.get("game_mode") == "Demo Day"
    ss["rng"] = random.Random(RANDOM_SEED)

    # Arm + grid setup (Chess Grid only)
    if not is_demo:
        if ss["cfg_real_arm"]:
            try:
                arm = RealArm(arm_url=ARM_API_URL, start_col=0, start_row=0)
                arm.get_position()  # probe
                ss["arm"] = arm
            except Exception as exc:
                st.warning(f"Real Arm failed ({exc}) — falling back to Mock Arm.")
                ss["arm"] = MockArm(start_col=0, start_row=0)
        else:
            ss["arm"] = MockArm(start_col=0, start_row=0)
        ss["piece_col"] = 0
        ss["piece_row"] = 0
        t_col, t_row = pick_random_target(ss["rng"])
        ss["target_col"] = t_col
        ss["target_row"] = t_row
        ss["flash"] = None

    # Demo Day reset
    ss["demo_target_piece"] = None
    ss["demo_trial_active"] = False
    ss["demo_awaiting_confirm"] = False
    ss["demo_trial_start"] = 0.0
    ss["demo_trials"] = []

    ss["sc"] = 0
    ss["si"] = 0
    ss["action_overlays"] = []
    ss["last_decoder_ts"] = time.time()
    ss["waveform_page"] = 0
    ss["final_bps"] = 0.0
    ss["final_sc"] = 0
    ss["final_si"] = 0

    brc = BitRateCalculator()
    brc.start()
    ss["bit_rate"] = brc
    ss["logger"] = SessionLogger()

    # Mock server — start once and keep alive to avoid port 8765 race condition.
    # On subsequent sessions the existing server continues serving.
    if ss["cfg_mock_decoder"] and ss.get("mock_server") is None:
        try:
            srv = MockDecoderServer()
            srv.start()
            ss["mock_server"] = srv
            time.sleep(0.4)
        except Exception as exc:
            st.warning(f"Mock server: {exc}")

    # Decoder client — fresh connection each session
    if ss.get("decoder_client"):
        try:
            ss["decoder_client"].stop()
        except Exception:
            pass

    if ss["cfg_mock_decoder"]:
        # Mock mode: WebSocket client
        try:
            client = DecoderClient(url=ss["cfg_ws_url"])
            client.start()
            ss["decoder_client"] = client
        except Exception as exc:
            st.warning(f"Decoder: {exc}")
    else:
        # Live mode: Synapse tap from SciFi device
        try:
            client = SynapseDecoderClient(
                device_ip=SCIFI_DEVICE_IP,
                tap_name=SYNAPSE_TAP_NAME,
                direction_threshold=DECODER_DIRECTION_THRESHOLD,
                gate_threshold=DECODER_GATE_THRESHOLD,
            )
            client.start()
            ss["decoder_client"] = client
        except Exception as exc:
            st.warning(f"Synapse decoder: {exc}")

    # Data loader — fresh each session (for waveform display from recordings)
    if ss.get("data_loader"):
        try:
            ss["data_loader"].stop_playback()
        except Exception:
            pass
    try:
        loader = H5DataLoader(buffer_seconds=ss["cfg_time_window"])
        loader.load()
        if loader.neural is not None:
            loader.start_playback()
            ss["data_loader"] = loader
        else:
            raise FileNotFoundError("No .h5 recordings found")
    except Exception:
        # Fall back to synthetic waveforms so the UI always shows data
        mock_loader = MockDataLoader(buffer_seconds=ss["cfg_time_window"])
        mock_loader.load()
        mock_loader.start_playback()
        ss["data_loader"] = mock_loader

    ss["session_running"] = True
    ss["session_ended"] = False
    ss["session_start_time"] = time.time()


def _reset_session() -> None:
    """Stop per-session components. Mock server is kept alive to hold port 8765."""
    ss = st.session_state
    if ss.get("decoder_client"):
        try:
            ss["decoder_client"].stop()
        except Exception:
            pass
    if ss.get("data_loader"):
        try:
            ss["data_loader"].stop_playback()
        except Exception:
            pass
    # Intentionally NOT stopping mock_server — keeps port 8765 bound and avoids
    # the race condition where a new server tries to bind before the old one fully closes.
    ss["decoder_client"] = None
    ss["data_loader"] = None
    ss["arm"] = None
    ss["bit_rate"] = None
    ss["logger"] = None
    ss["rng"] = None
    ss["session_running"] = False
    ss["session_ended"] = False
    ss["sc"] = 0
    ss["si"] = 0
    ss["action_overlays"] = []
    ss["final_bps"] = 0.0
    ss["final_sc"] = 0
    ss["final_si"] = 0
    ss["final_duration"] = 0
    # Demo Day reset
    ss["demo_target_piece"] = None
    ss["demo_trial_active"] = False
    ss["demo_awaiting_confirm"] = False
    ss["demo_trial_start"] = 0.0
    ss["demo_trials"] = []


# ── Demo Day constants ──────────────────────────────────────────────────────

DEMO_PIECES = ["King", "Queen", "Bishop", "Rook"]
DEMO_PIECE_SYMBOLS = {"King": "♚", "Queen": "♛", "Bishop": "♝", "Rook": "♜"}
DEMO_N_CHOICES = 4
DEMO_LOG2_N = math.log2(DEMO_N_CHOICES)  # 2.0


def _render_demo_board() -> str:
    """Return self-contained HTML for a wooden board with the Science Corp logo."""
    return """
<html><body style="margin:0;padding:0;background:transparent;display:flex;justify-content:center;">
<div style="
  width:400px;height:400px;
  background:linear-gradient(145deg, #d4b896 0%, #c4a67a 30%, #b8956a 70%, #a8845a 100%);
  border-radius:12px;
  border:3px solid #8b7355;
  box-shadow:inset 0 0 30px rgba(0,0,0,0.15), 0 4px 12px rgba(0,0,0,0.2);
  display:flex;align-items:center;justify-content:center;
  position:relative;overflow:hidden;
">
  <div style="position:absolute;inset:0;
    background:repeating-linear-gradient(90deg, transparent, transparent 40px, rgba(139,115,85,0.08) 40px, rgba(139,115,85,0.08) 42px);
  "></div>
  <div style="z-index:1;text-align:center;">
    <svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 200 260"
      width="110" height="143" fill="none" stroke="rgba(80,60,35,0.5)"
      stroke-width="7" stroke-linecap="round" stroke-linejoin="round">
      <path d="M100 130 C60 130, 55 80, 70 55 C80 38, 95 30, 100 28
        C105 30, 120 38, 130 55 C145 80, 140 130, 100 130 Z"/>
      <path d="M100 130 C60 130, 55 180, 70 205 C80 222, 95 230, 100 232
        C105 230, 120 222, 130 205 C145 180, 140 130, 100 130 Z"/>
      <line x1="100" y1="80" x2="100" y2="180"/>
      <line x1="55" y1="105" x2="145" y2="155"/>
      <line x1="55" y1="155" x2="145" y2="105"/>
      <line x1="55" y1="130" x2="145" y2="130"/>
    </svg>
  </div>
</div>
</body></html>
"""


def _demo_pick_piece(rng: random.Random) -> str:
    """Select a random piece for the next Demo Day trial."""
    return rng.choice(DEMO_PIECES)


# ── Sidebar ───────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### ⚙ ZIPPR Config")
    st.markdown("---")
    game_mode = st.radio("Game Mode", ["Chess Grid", "Demo Day"], horizontal=True, key="sb_game_mode")
    st.session_state["game_mode"] = game_mode
    mode = st.radio("Mode", ["Playback", "Live"], horizontal=True)
    time_window = st.slider("Time window (s)", 1, 10, BUFFER_SECONDS, key="sb_tw")
    st.session_state["cfg_time_window"] = time_window
    ws_url = st.text_input("Decoder WS URL", value=DECODER_WS_URL, key="sb_ws")
    st.session_state["cfg_ws_url"] = ws_url
    mock_dec = st.toggle("Mock Decoder (65% accuracy)", value=USE_MOCK_DECODER, key="sb_mock")
    st.session_state["cfg_mock_decoder"] = mock_dec
    real_arm = st.toggle("Real Arm (SO-101)", value=USE_REAL_ARM, key="sb_arm")
    st.session_state["cfg_real_arm"] = real_arm
    st.markdown("---")
    if st.button("↺ Reset", key="sb_reset", use_container_width=True):
        _reset_session()
        st.rerun()


# ── Event processing ──────────────────────────────────────────────────────────

def _process_decoder_events() -> None:
    """Read new decoder frames, update game state, and log events."""
    ss = st.session_state
    client = ss.get("decoder_client")
    arm = ss.get("arm")
    brc: Optional[BitRateCalculator] = ss.get("bit_rate")
    logger_inst: Optional[SessionLogger] = ss.get("logger")
    rng = ss.get("rng")

    if not all([client, arm, brc, logger_inst, rng]):
        return

    frames = client.get_all_since(ss["last_decoder_ts"])
    if not frames:
        return

    ss["last_decoder_ts"] = frames[-1].timestamp

    for frame in frames:
        if not frame.is_movement:
            continue

        direction = frame.direction
        target_dir = optimal_direction(
            ss["piece_col"], ss["piece_row"],
            ss["target_col"], ss["target_row"],
        )

        # Update mock server hint — biases 65% of future emissions toward correct direction
        set_target_direction_hint(target_dir)

        pos_before = arm.get_position()
        arm.execute_action(direction)
        pos_after = arm.get_position()
        ss["piece_col"] = int(pos_after.x)
        ss["piece_row"] = int(pos_after.y)

        dx = pos_after.x - pos_before.x
        dy = pos_after.y - pos_before.y
        dz = pos_after.z - pos_before.z

        correct = (direction == target_dir) if target_dir is not None else False

        if correct:
            brc.record_correct()
            ss["sc"] += 1
            ss["flash"] = "green"
        else:
            brc.record_incorrect()
            ss["si"] += 1
            ss["flash"] = "red"
        ss["flash_time"] = time.time()

        event = SessionEvent(
            timestamp=frame.timestamp,
            target_square=square_label(ss["target_col"], ss["target_row"]),
            target_direction=target_dir,
            decoded_action=direction,
            confidence=frame.confidence,
            arm_x_initial=pos_before.x,
            arm_y_initial=pos_before.y,
            arm_z_initial=pos_before.z,
            arm_x_final=pos_after.x,
            arm_y_final=pos_after.y,
            arm_z_final=pos_after.z,
            delta_x=dx,
            delta_y=dy,
            delta_z=dz,
            correct=correct,
        )
        logger_inst.log(event)

        overlay_ts = frame.timestamp - ss["session_start_time"]
        label = DIRECTION_LABELS.get(direction, direction)
        ss["action_overlays"].append({
            "timestamp": overlay_ts,
            "direction": direction,
            "label": label,
        })
        ss["action_overlays"] = ss["action_overlays"][-20:]

        # Pick new target when piece reaches it
        if ss["piece_col"] == ss["target_col"] and ss["piece_row"] == ss["target_row"]:
            t_col, t_row = pick_random_target(rng)
            ss["target_col"] = t_col
            ss["target_row"] = t_row

    # Clear flash after 500 ms
    if ss.get("flash") and time.time() - ss.get("flash_time", 0) > 0.5:
        ss["flash"] = None


def _check_session_end() -> bool:
    """Freeze state and return True when the 60-second session expires.

    Demo Day mode has no auto-timeout — the user ends manually.
    """
    ss = st.session_state
    if not ss["session_running"]:
        return False
    # Demo Day: no auto-timeout
    if ss.get("game_mode") == "Demo Day":
        return False
    elapsed = time.time() - ss["session_start_time"]
    if elapsed >= SESSION_DURATION:
        _end_session()
        return True
    return False


def _end_session() -> None:
    """Freeze session state and stop components."""
    ss = st.session_state
    elapsed = time.time() - ss["session_start_time"]
    ss["session_running"] = False
    ss["session_ended"] = True
    ss["final_duration"] = elapsed

    is_demo = ss.get("game_mode") == "Demo Day"
    if is_demo:
        t = elapsed if elapsed > 0 else 1.0
        ss["final_bps"] = DEMO_LOG2_N * max(ss["sc"] - ss["si"], 0) / t
    else:
        brc: Optional[BitRateCalculator] = ss.get("bit_rate")
        ss["final_bps"] = brc.get_current_bps() if brc else 0.0

    ss["final_sc"] = ss["sc"]
    ss["final_si"] = ss["si"]
    if ss.get("decoder_client"):
        try:
            ss["decoder_client"].stop()
        except Exception:
            pass
    if ss.get("data_loader"):
        try:
            ss["data_loader"].stop_playback()
        except Exception:
            pass


# ── Fragment: game panel (chess + bitrate) ────────────────────────────────────

@st.fragment(run_every=0.5)
def _game_panel() -> None:
    """Auto-refreshes every 0.5 s — processes events, draws chess + bitrate."""
    ss = st.session_state
    if not ss.get("session_running"):
        return

    _process_decoder_events()
    if _check_session_end():
        st.rerun(scope="app")
        return

    elapsed = time.time() - ss["session_start_time"]
    remaining = max(0.0, SESSION_DURATION - elapsed)
    brc: Optional[BitRateCalculator] = ss.get("bit_rate")
    logger_inst: Optional[SessionLogger] = ss.get("logger")

    # ── Timer / progress bar ──────────────────────────────────────────────────
    m_e, s_e = divmod(int(elapsed), 60)
    m_r, s_r = divmod(int(remaining), 60)
    st.markdown(
        f'<div style="text-align:center;margin-bottom:6px;">'
        f'<span class="status-pill">'
        f'⏱ {m_e:02d}:{s_e:02d} elapsed &nbsp;|&nbsp; ⏳ {m_r:02d}:{s_r:02d} left'
        f'</span></div>',
        unsafe_allow_html=True,
    )
    st.progress(min(elapsed / SESSION_DURATION, 1.0))

    col_chess, col_bps = st.columns([1, 1], gap="large")

    # ── LEFT: Chess grid ──────────────────────────────────────────────────────
    with col_chess:
        st.markdown("#### Chess Grid")
        grid_html = render_grid_html(
            piece_col=ss["piece_col"],
            piece_row=ss["piece_row"],
            target_col=ss["target_col"],
            target_row=ss["target_row"],
            flash=ss["flash"],
        )
        # st.html() renders tables faithfully; st.markdown would mangle the CSS
        st.html(grid_html)
        st.markdown(
            f'<div style="text-align:center;font-size:17px;color:#545333;margin-top:6px;">'
            f'♟ {square_label(ss["piece_col"], ss["piece_row"])}'
            f'&nbsp;→&nbsp;'
            f'⚑ {square_label(ss["target_col"], ss["target_row"])}'
            f'</div>',
            unsafe_allow_html=True,
        )

    # ── RIGHT: Bit rate + stats + timeline ────────────────────────────────────
    with col_bps:
        st.markdown("#### Live Bit Rate")

        bps = brc.get_current_bps() if brc else 0.0
        if bps > 2.0:
            bps_color = "#545333"
        elif bps >= 0.5:
            bps_color = "#878672"
        else:
            bps_color = "#030302"

        st.markdown(
            f'<div class="bps-number" style="color:{bps_color};">{bps:.2f}</div>'
            f'<div class="bps-unit">bits / sec</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="margin:8px 0;">', unsafe_allow_html=True)

        total = ss["sc"] + ss["si"]
        acc = (ss["sc"] / total * 100) if total > 0 else 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("✓ Correct", ss["sc"])
        with c2:
            st.metric("✗ Wrong", ss["si"])
        with c3:
            st.metric("Accuracy", f"{acc:.0f}%")

        # Timeline sparkline
        if logger_inst and logger_inst.events:
            events = logger_inst.events
            times  = [e.timestamp - ss["session_start_time"] for e in events]
            colors = ["#545333" if e.correct else "#030302" for e in events]
            labels = [
                f"{'✓' if e.correct else '✗'} {DIRECTION_LABELS.get(e.decoded_action, e.decoded_action)}"
                for e in events
            ]
            fig_tl = go.Figure()
            fig_tl.add_trace(go.Scatter(
                x=times, y=[0] * len(times),
                mode="markers",
                marker=dict(color=colors, size=14, symbol="line-ns-open",
                            line=dict(width=3)),
                showlegend=False,
                hovertext=labels,
                hoverinfo="text+x",
            ))
            fig_tl.update_layout(
                paper_bgcolor="rgba(253,251,212,0)",
                plot_bgcolor="rgba(217,215,182,0.3)",
                height=90,
                margin=dict(l=30, r=10, t=5, b=28),
                xaxis=dict(
                    title=dict(text="Time (s)", font=dict(size=13, color="#545333")),
                    range=[0, SESSION_DURATION],
                    color="#545333",
                    tickfont=dict(size=12),
                    gridcolor="rgba(135,134,114,0.25)",
                ),
                yaxis=dict(visible=False),
            )
            st.plotly_chart(fig_tl, width="stretch", key="tl_live")


# ── Fragment: Demo Day panel ─────────────────────────────────────────────────

@st.fragment(run_every=1.0)
def _demo_day_panel() -> None:
    """Demo Day game mode — pick the prompted piece and place it on the board."""
    ss = st.session_state
    if not ss.get("session_running"):
        return

    elapsed = time.time() - ss["session_start_time"]
    trials = ss.get("demo_trials", [])

    col_board, col_stats = st.columns([1, 1], gap="large")

    # ── LEFT: wooden board + piece prompt ─────────────────────────────────
    with col_board:
        # Show which piece to pick up
        if ss.get("demo_target_piece") and ss.get("demo_trial_active"):
            piece = ss["demo_target_piece"]
            symbol = DEMO_PIECE_SYMBOLS[piece]
            st.markdown(
                f'<div style="text-align:center;margin-bottom:12px;">'
                f'<span style="font-family:VT323,monospace;font-size:28px;color:#545333;">'
                f'Pick up the &nbsp;</span>'
                f'<span style="font-size:48px;">{symbol}</span>'
                f'<span style="font-family:VT323,monospace;font-size:28px;color:#545333;">'
                f'&nbsp; {piece}</span>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif ss.get("demo_awaiting_confirm"):
            st.markdown(
                '<div style="text-align:center;margin-bottom:12px;">'
                '<span style="font-family:VT323,monospace;font-size:24px;color:#878672;">'
                'Did you place it correctly?</span>'
                '</div>',
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                '<div style="text-align:center;margin-bottom:12px;">'
                '<span style="font-family:VT323,monospace;font-size:24px;color:#878672;">'
                'Press NEXT PIECE to begin a trial</span>'
                '</div>',
                unsafe_allow_html=True,
            )

        components.html(_render_demo_board(), height=420)

        # Action buttons below the board
        if ss.get("demo_awaiting_confirm"):
            b1, b2 = st.columns(2)
            with b1:
                if st.button("✓  YES", key="demo_yes", use_container_width=True):
                    ss["demo_trials"].append({
                        "piece": ss["demo_target_piece"],
                        "correct": True,
                        "time": time.time() - ss["demo_trial_start"],
                    })
                    ss["sc"] += 1
                    brc = ss.get("bit_rate")
                    if brc:
                        brc.record_correct()
                    ss["demo_awaiting_confirm"] = False
                    ss["demo_trial_active"] = False
                    ss["demo_target_piece"] = None
                    st.rerun()
            with b2:
                if st.button("✗  NO", key="demo_no", use_container_width=True):
                    ss["demo_trials"].append({
                        "piece": ss["demo_target_piece"],
                        "correct": False,
                        "time": time.time() - ss["demo_trial_start"],
                    })
                    ss["si"] += 1
                    brc = ss.get("bit_rate")
                    if brc:
                        brc.record_incorrect()
                    ss["demo_awaiting_confirm"] = False
                    ss["demo_trial_active"] = False
                    ss["demo_target_piece"] = None
                    st.rerun()
        elif ss.get("demo_trial_active"):
            _, cb, _ = st.columns([1, 2, 1])
            with cb:
                if st.button("✓  PLACED IT", key="demo_placed", use_container_width=True):
                    ss["demo_trial_active"] = False
                    ss["demo_awaiting_confirm"] = True
                    st.rerun()
        else:
            _, cb, _ = st.columns([1, 2, 1])
            with cb:
                if st.button("▶  NEXT PIECE", key="demo_next", use_container_width=True):
                    rng = ss.get("rng") or random.Random()
                    ss["demo_target_piece"] = _demo_pick_piece(rng)
                    ss["demo_trial_active"] = True
                    ss["demo_awaiting_confirm"] = False
                    ss["demo_trial_start"] = time.time()
                    st.rerun()

    # ── RIGHT: bit rate + trial stats ─────────────────────────────────────
    with col_stats:
        st.markdown("#### Live Bit Rate")

        brc = ss.get("bit_rate")
        bps = brc.get_current_bps() if brc else 0.0
        # Override with Demo Day formula: log2(4) instead of log2(64)
        t = elapsed if elapsed > 0 else 1.0
        demo_bps = DEMO_LOG2_N * max(ss["sc"] - ss["si"], 0) / t

        if demo_bps > 1.0:
            bps_color = "#545333"
        elif demo_bps >= 0.3:
            bps_color = "#878672"
        else:
            bps_color = "#030302"

        st.markdown(
            f'<div class="bps-number" style="color:{bps_color};">{demo_bps:.2f}</div>'
            f'<div class="bps-unit">bits / sec</div>',
            unsafe_allow_html=True,
        )
        st.markdown('<hr style="margin:8px 0;">', unsafe_allow_html=True)

        total = ss["sc"] + ss["si"]
        acc = (ss["sc"] / total * 100) if total > 0 else 0.0
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("✓ Correct", ss["sc"])
        with c2:
            st.metric("✗ Wrong", ss["si"])
        with c3:
            st.metric("Accuracy", f"{acc:.0f}%")

        st.markdown(
            f'<div style="font-size:15px;color:#545333;line-height:2;margin-top:8px;">'
            f'N = {DEMO_N_CHOICES} pieces &nbsp;·&nbsp; log₂(N) = {DEMO_LOG2_N:.1f}<br>'
            f'Trials: {len(trials)} &nbsp;·&nbsp; Elapsed: {int(elapsed)}s<br>'
            f'B = log₂(N) × max(Sc−Si, 0) / t'
            f'</div>',
            unsafe_allow_html=True,
        )

        # Trial history
        if trials:
            st.markdown("##### Trial History")
            for i, trial in enumerate(trials):
                icon = "✓" if trial["correct"] else "✗"
                sym = DEMO_PIECE_SYMBOLS[trial["piece"]]
                st.markdown(
                    f'<div style="font-size:15px;color:#545333;">'
                    f'{icon} &nbsp;{sym} {trial["piece"]} — {trial["time"]:.1f}s'
                    f'</div>',
                    unsafe_allow_html=True,
                )

        # End session button (Demo Day has no timer — user ends manually)
        st.markdown('<hr style="margin:12px 0;">', unsafe_allow_html=True)
        if st.button("■  END SESSION", key="demo_end", use_container_width=True):
            _end_session()
            st.rerun(scope="app")


# ── Fragment: waveform panel ──────────────────────────────────────────────────

@st.fragment(run_every=0.5)
def _waveform_panel() -> None:
    """Auto-refreshes every 0.5 s — draws 8-channel waveform with scroll."""
    ss = st.session_state
    # Guard: fragment may still fire briefly after session ends
    if not ss.get("session_running") and not ss.get("data_loader"):
        return

    loader: Optional[H5DataLoader] = ss.get("data_loader")
    if not loader:
        st.info("Data loader not initialised.")
        return

    buf = loader.get_buffer()
    if buf is None:
        st.info("Waiting for neural data…")
        return

    n_total  = buf.shape[0]
    per_page = 8
    n_pages  = max(1, (n_total + per_page - 1) // per_page)
    page     = int(ss.get("waveform_page", 0))
    page     = max(0, min(page, n_pages - 1))
    ss["waveform_page"] = page

    ch_start = page * per_page
    ch_end   = min(ch_start + per_page, n_total)
    current_chs = list(range(ch_start, ch_end))

    # Highlight channels with highest variance + command correlation
    top_chs   = get_top_channels(buf, ss.get("action_overlays", []), n_top=8)
    highlighted = [c for c in current_chs if c in top_chs]

    # Navigation row
    nav_l, nav_prev, nav_mid, nav_next = st.columns([3, 1, 2, 1])
    with nav_l:
        hint = "  ★ = high activity" if highlighted else ""
        st.markdown(
            f'<div class="ch-nav-label">Ch {ch_start+1}–{ch_end} / {n_total}{hint}</div>',
            unsafe_allow_html=True,
        )
    with nav_prev:
        if st.button("◀", key="ch_prev", disabled=(page == 0)):
            ss["waveform_page"] = page - 1
    with nav_mid:
        st.markdown(
            f'<div style="text-align:center;font-size:15px;color:#878672;padding-top:10px;">'
            f'page {page+1} / {n_pages}</div>',
            unsafe_allow_html=True,
        )
    with nav_next:
        if st.button("▶", key="ch_next", disabled=(page >= n_pages - 1)):
            ss["waveform_page"] = page + 1

    fig = build_waveform_figure(
        data=buf,
        channels=current_chs,
        time_window=ss["cfg_time_window"],
        actions=ss.get("action_overlays", []),
        highlighted_channels=highlighted,
    )
    st.plotly_chart(fig, width="stretch", key=f"wf_{page}")


# ── Welcome screen ────────────────────────────────────────────────────────────

ss = st.session_state

if not ss["session_running"] and not ss["session_ended"]:
    is_demo = ss.get("game_mode") == "Demo Day"
    st.markdown(
        '<div style="text-align:center;margin-top:10px;">'
        '<p style="font-family:VT323,monospace;font-size:32px;color:#545333;margin:0;">'
        f'{"DEMO DAY — PIECE PICK" if is_demo else "NEURAL BCI DEMO"}</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    subtitle = (
        'pick the prompted piece &nbsp;·&nbsp; place it on the board &nbsp;·&nbsp; confirm each trial'
        if is_demo else
        'configure in the sidebar &nbsp;·&nbsp; press START to begin a 60-second session'
    )
    st.markdown(
        f'<p class="welcome-sub">{subtitle}</p>',
        unsafe_allow_html=True,
    )
    _, col_btn, _ = st.columns([2, 1, 2])
    with col_btn:
        if st.button("▶  START", key="main_start", use_container_width=True):
            _start_session()
            st.rerun()
    st.stop()


# ── Session frozen / completed view ──────────────────────────────────────────

if ss["session_ended"]:
    final_bps = ss.get("final_bps", 0.0)
    final_sc  = ss.get("final_sc", ss["sc"])
    final_si  = ss.get("final_si", ss["si"])
    is_demo   = ss.get("game_mode") == "Demo Day"

    st.markdown(
        '<div class="frozen-banner">✦ SESSION COMPLETE — RESULTS FROZEN ✦</div>',
        unsafe_allow_html=True,
    )

    col_result, col_info = st.columns([1, 1], gap="large")

    with col_result:
        total = final_sc + final_si
        acc   = (final_sc / total * 100) if total > 0 else 0.0
        st.markdown(
            '<div class="final-bps-label">FINAL BIT RATE</div>'
            f'<div style="text-align:center;margin:6px 0;">'
            f'<span class="final-bps-value">{final_bps:.2f}</span>'
            f'</div>'
            '<div class="final-bps-label">bits / sec</div>',
            unsafe_allow_html=True,
        )
        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("✓ Correct", final_sc)
        with c2:
            st.metric("✗ Wrong", final_si)
        with c3:
            st.metric("Accuracy", f"{acc:.0f}%")

    with col_info:
        if is_demo:
            duration = ss.get("final_duration", 0)
            trials = ss.get("demo_trials", [])
            st.markdown(
                f'<div style="font-size:17px;color:#545333;line-height:2;">'
                f'N = {DEMO_N_CHOICES} pieces<br>'
                f'log₂(N) = {DEMO_LOG2_N:.1f} bits/correct<br>'
                f'Formula: B = log₂(N)×max(Sc−Si,0)/t<br>'
                f'Trials: {len(trials)} &nbsp;·&nbsp; Duration: {int(duration)}s'
                f'</div>',
                unsafe_allow_html=True,
            )
            # Show trial history
            if trials:
                st.markdown("##### Trial History")
                for i, trial in enumerate(trials):
                    icon = "✓" if trial["correct"] else "✗"
                    sym = DEMO_PIECE_SYMBOLS[trial["piece"]]
                    st.markdown(
                        f'<div style="font-size:15px;color:#545333;">'
                        f'{icon} &nbsp;{sym} {trial["piece"]} — {trial["time"]:.1f}s'
                        f'</div>',
                        unsafe_allow_html=True,
                    )
        else:
            logger_inst: Optional[SessionLogger] = ss.get("logger")
            st.markdown(
                f'<div style="font-size:17px;color:#545333;line-height:2;">'
                f'N = {N_SQUARES} squares<br>'
                f'log₂(N) = {LOG2_N:.1f} bits/correct<br>'
                f'Formula: B = log₂(N)×max(Sc−Si,0)/t<br>'
                f'Duration: {SESSION_DURATION}s'
                f'</div>',
                unsafe_allow_html=True,
            )
            if logger_inst:
                csv_data = logger_inst.to_csv_string()
                st.download_button(
                    "📥 Export CSV",
                    data=csv_data,
                    file_name="zippr_session_log.csv",
                    mime="text/csv",
                )

    st.markdown('<br>', unsafe_allow_html=True)
    _, col_replay, _ = st.columns([2, 1, 2])
    with col_replay:
        if st.button("↺  REPLAY", key="replay_btn", use_container_width=True):
            _reset_session()
            st.rerun()
    st.stop()


# ── Running session ───────────────────────────────────────────────────────────

st.markdown('<hr>', unsafe_allow_html=True)
if ss.get("game_mode") == "Demo Day":
    _demo_day_panel()
else:
    _game_panel()
st.markdown('<hr>', unsafe_allow_html=True)
st.markdown("#### Neural Waveforms")
_waveform_panel()
