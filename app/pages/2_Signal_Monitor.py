"""
Signal Dashboard — Live decoded BCI signal monitor.

Shows the 7-channel GRU decoder output in real-time directly from the Synapse tap.

Launch (live device):
    USE_MOCK_DECODER=false uv run streamlit run app/signal_dashboard.py

Launch (mock):
    uv run streamlit run app/signal_dashboard.py
"""

from __future__ import annotations

import math
import os
import sys
import time
import threading
from collections import deque
from typing import Deque, List, Optional

import numpy as np
import plotly.graph_objects as go
import streamlit as st

# ── config ────────────────────────────────────────────────────────────────────

DEVICE_IP   = os.getenv("SCIFI_DEVICE_IP", "192.168.0.25")
TAP_NAME    = "joystick_out"
USE_MOCK    = os.getenv("USE_MOCK_DECODER", "true").lower() in ("1", "true", "yes")
DEADZONE    = 0.05   # visualised as a threshold line
GATE_THRESH = 0.5
REFRESH_HZ  = 10
HISTORY_S   = 10

CH_NAMES  = ["joy_x", "joy_y", "rot", "depth", "lt", "rt", "gate"]
CH_LABELS = ["L-Stick X (L/R)", "L-Stick Y (U/D)", "Wrist Roll",
             "Reach Depth", "Gripper Open", "Gripper Close", "Gate"]
CH_RANGE  = [(-1,1),(-1,1),(-1,1),(-1,1),(0,1),(0,1),(0,1)]
CH_COLOR  = ["#4C9BE8","#4C9BE8","#A569BD","#A569BD","#E8834C","#E84C4C","#4CE87A"]

st.set_page_config(page_title="BCI Signal Monitor", page_icon="🧠", layout="wide")

# ── tap reader thread (singleton per session) ─────────────────────────────────

def _start_tap_reader():
    buf: Deque[List[float]] = deque(maxlen=HISTORY_S * REFRESH_HZ * 4)
    lock = threading.Lock()
    status = {"connected": False, "error": ""}

    def _run():
        try:
            from synapse.client.taps import Tap
            from synapse.api.datatype_pb2 import Tensor
        except ImportError as e:
            status["error"] = f"synapse SDK not found: {e}"
            return

        tap = Tap(DEVICE_IP)
        ok = tap.connect(TAP_NAME)
        if not ok:
            status["error"] = f"tap.connect() failed for {TAP_NAME} @ {DEVICE_IP}"
            return

        status["connected"] = True
        try:
            while not st.session_state.get("_tap_stop"):
                raw = tap.read()
                if raw is None:
                    time.sleep(0.005)
                    continue
                t = Tensor()
                t.ParseFromString(raw)
                v = np.frombuffer(t.data, dtype=np.float32)
                if v.size == 7:
                    with lock:
                        buf.append([float(x) for x in v])
        finally:
            tap.disconnect()
            status["connected"] = False

    thread = threading.Thread(target=_run, daemon=True)
    thread.start()
    return buf, lock, status, thread


def _mock_vector(phase: float) -> List[float]:
    return [
        0.6  * math.sin(phase * 0.7),
        0.5  * math.sin(phase * 0.5 + 1.0),
        0.4  * math.sin(phase * 1.1 + 2.0),
        0.35 * math.sin(phase * 0.9 + 0.5),
        max(0.0, 0.5 + 0.5 * math.sin(phase * 0.3 + 3.0)),
        max(0.0, 0.5 + 0.5 * math.sin(phase * 0.4 + 1.5)),
        0.5  + 0.45 * math.sin(phase * 0.2),
    ]

# ── session init ──────────────────────────────────────────────────────────────

if "history" not in st.session_state:
    st.session_state.history: Deque[List[float]] = deque(maxlen=HISTORY_S * REFRESH_HZ * 4)
    st.session_state.mock_phase = 0.0
    st.session_state.frame_count = 0
    st.session_state._tap_stop = False

if not USE_MOCK and "tap_buf" not in st.session_state:
    buf, lock, status, thread = _start_tap_reader()
    st.session_state.tap_buf    = buf
    st.session_state.tap_lock   = lock
    st.session_state.tap_status = status
    st.session_state.tap_thread = thread

# ── get latest vector ─────────────────────────────────────────────────────────

vector: Optional[List[float]] = None

if USE_MOCK:
    st.session_state.mock_phase += 1.0 / REFRESH_HZ
    vector = _mock_vector(st.session_state.mock_phase)
else:
    buf   = st.session_state.get("tap_buf")
    lock  = st.session_state.get("tap_lock")
    if buf is not None and lock is not None:
        with lock:
            vector = list(buf[-1]) if buf else None

if vector is not None:
    st.session_state.history.append(vector)
    st.session_state.frame_count += 1

# ── direction helper ──────────────────────────────────────────────────────────

def direction(v: List[float]) -> str:
    if v[6] < GATE_THRESH:
        return "GATED"
    jx, jy = v[0], v[1]
    ax = jx if abs(jx) >= DEADZONE else 0.0
    ay = jy if abs(jy) >= DEADZONE else 0.0
    if ax == 0.0 and ay == 0.0:
        return "REST"
    return ("RIGHT" if ax > 0 else "LEFT") if abs(ax) >= abs(ay) else ("UP" if ay > 0 else "DOWN")

DIR_COLOR = {"LEFT":"#4C9BE8","RIGHT":"#4C9BE8","UP":"#4CE87A",
             "DOWN":"#E84C4C","REST":"#888","GATED":"#E8834C"}

# ── layout ────────────────────────────────────────────────────────────────────

st.markdown("## 🧠 BCI Signal Monitor")

# status row
status_obj = st.session_state.get("tap_status", {})
connected  = status_obj.get("connected", False)
tap_error  = status_obj.get("error", "")

if USE_MOCK:
    src = "🟡 Mock"
elif connected:
    src = f"🟢 Live — {DEVICE_IP}/{TAP_NAME}"
elif tap_error:
    src = f"🔴 {tap_error}"
else:
    src = f"🟠 Connecting to {DEVICE_IP}…"

c1, c2, c3, c4 = st.columns([3,1,1,1])
c1.caption(f"**Source:** {src}")
c2.metric("Frames", st.session_state.frame_count)

if vector:
    d = direction(vector)
    c3.markdown(
        f"**Direction**<br>"
        f"<span style='font-size:1.5rem;font-weight:bold;color:{DIR_COLOR.get(d,'#888')}'>{d}</span>",
        unsafe_allow_html=True,
    )
    gate_val = vector[6]
    gate_ok  = gate_val >= GATE_THRESH
    c4.markdown(
        f"**Gate**<br>"
        f"<span style='font-size:1.1rem;font-weight:bold;color={'#4CE87A' if gate_ok else '#E8834C'}'>"
        f"{'ACTIVE' if gate_ok else 'GATED'}  {gate_val:.2f}</span>",
        unsafe_allow_html=True,
    )

st.divider()

# ── gauges ────────────────────────────────────────────────────────────────────

st.markdown("#### Current values")

if vector:
    cols = st.columns(7)
    for i, col in enumerate(cols):
        lo, hi = CH_RANGE[i]
        val  = float(vector[i])
        pct  = int((val - lo) / (hi - lo) * 100)
        col.markdown(
            f"<div style='text-align:center'>"
            f"<div style='font-size:.75rem;color:#aaa'>{CH_NAMES[i]}</div>"
            f"<div style='font-size:1.4rem;font-weight:bold;color:{CH_COLOR[i]}'>{val:+.3f}</div>"
            f"<div style='background:#1e2130;border-radius:6px;height:8px;margin:4px 0'>"
            f"<div style='background:{CH_COLOR[i]};width:{pct}%;height:100%;border-radius:6px'></div></div>"
            f"<div style='font-size:.68rem;color:#666'>{CH_LABELS[i]}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
else:
    st.info("Waiting for first frame…")

st.divider()

# ── time-series ───────────────────────────────────────────────────────────────

st.markdown("#### Signal history (last 10 s)")

hist = list(st.session_state.history)
if len(hist) >= 2:
    arr = np.array(hist)
    t   = np.linspace(0, len(hist) / REFRESH_HZ, len(hist))

    fig = go.Figure()
    for i in range(7):
        fig.add_trace(go.Scatter(
            x=t, y=arr[:, i],
            mode="lines", name=CH_NAMES[i],
            line=dict(color=CH_COLOR[i], width=1.8),
        ))
    # deadzone lines
    fig.add_hline(y= DEADZONE, line=dict(color="#555", dash="dot", width=1))
    fig.add_hline(y=-DEADZONE, line=dict(color="#555", dash="dot", width=1))

    fig.update_layout(
        height=300, margin=dict(l=0,r=0,t=10,b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(14,17,23,1)",
        font=dict(color="#ccc"),
        xaxis=dict(title="s", color="#666", gridcolor="#1e2130"),
        yaxis=dict(title="value", range=[-1.1,1.1], color="#666",
                   gridcolor="#1e2130", zeroline=True, zerolinecolor="#333"),
        legend=dict(orientation="h", x=0, y=1.08, font=dict(size=11)),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"Dotted lines = deadzone ±{DEADZONE}")
else:
    st.caption("Collecting samples…")

# ── refresh ───────────────────────────────────────────────────────────────────

time.sleep(1.0 / REFRESH_HZ)
st.rerun()
