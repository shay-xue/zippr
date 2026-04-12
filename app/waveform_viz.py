"""
Plotly-based neural waveform visualiser with action overlays.

Improvements over baseline:
- Smoothed traces (moving-average pre-filter before downsampling) to reduce noise
- Highlighted channels based on variance + command correlation
- Expanded direction labels ("LJ →" instead of just "RIGHT")
- ZIPPR color palette
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import plotly.graph_objects as go

from config import SAMPLE_RATE_HZ

# ── Direction labels (expanded, abbreviated) ──────────────────────────────────
# LEFT/RIGHT come from Left Joystick X; UP/DOWN from Right Joystick Y.
DIRECTION_LABELS: dict[str, str] = {
    "RIGHT": "LJ →",
    "LEFT":  "LJ ←",
    "UP":    "RJ ↑",
    "DOWN":  "RJ ↓",
}

# ── Overlay colours (ZIPPR palette) ──────────────────────────────────────────
DIRECTION_COLORS: dict[str, str] = {
    "UP":    "rgba(84,83,51,0.22)",      # dark olive
    "DOWN":  "rgba(135,134,114,0.28)",   # muted
    "LEFT":  "rgba(22,21,16,0.18)",      # near-black
    "RIGHT": "rgba(84,83,51,0.30)",      # olive
}

DIRECTION_LINE_COLORS: dict[str, str] = {
    "UP":    "#545333",
    "DOWN":  "#878672",
    "LEFT":  "#161510",
    "RIGHT": "#030302",
}

# ── Channel trace palette (8 distinct ZIPPR-adjacent hues) ───────────────────
_CH_COLORS = [
    "#545333",  # dark olive
    "#878672",  # muted grey-green
    "#030302",  # near-black
    "#161510",  # warm dark
    "#6b6845",  # mid olive
    "#a39e84",  # light warm
    "#3d3c27",  # deep olive
    "#c2bfa8",  # cream tan
]

MAX_DISPLAY_SAMPLES = 2000   # max samples per channel sent to Plotly
SMOOTH_WINDOW = 80           # moving-average window before downsampling (~2.5ms @32kHz)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _smooth(arr: np.ndarray, window: int = SMOOTH_WINDOW) -> np.ndarray:
    """Apply a causal moving-average to reduce high-frequency noise."""
    if window <= 1 or len(arr) < window:
        return arr
    kernel = np.ones(window, dtype=np.float32) / window
    return np.convolve(arr.astype(np.float32), kernel, mode="same")


def _downsample(arr: np.ndarray, target: int) -> np.ndarray:
    """Stride-downsample a 1-D array to *target* points."""
    if len(arr) <= target:
        return arr
    step = len(arr) // target
    return arr[::step][:target]


def _smooth_and_downsample(arr: np.ndarray, target: int = MAX_DISPLAY_SAMPLES) -> np.ndarray:
    """Smooth then downsample — removes noise before striding."""
    return _downsample(_smooth(arr), target)


# ── Top-channel selection ─────────────────────────────────────────────────────

def get_top_channels(
    data: np.ndarray,
    actions: list[dict],
    n_top: int = 8,
    sample_rate: int = SAMPLE_RATE_HZ,
) -> list[int]:
    """Return indices of the *n_top* most informative channels.

    Scoring: 0.6 × normalised_variance + 0.4 × |command_correlation|

    Command correlation measures how much each channel co-varies with the
    binary action signal (1 in a ±100ms window around each decoded action).
    """
    n_ch, n_samp = data.shape

    # Per-channel variance (recent window only — last 2 s worth)
    recent_samples = min(n_samp, int(2.0 * sample_rate))
    recent = data[:, -recent_samples:]
    variance = np.var(recent, axis=1)  # shape (n_ch,)

    # Command signal: binary spike at each action timestamp
    correlation = np.zeros(n_ch)
    if actions:
        cmd = np.zeros(n_samp, dtype=np.float32)
        window_samp = int(0.10 * sample_rate)  # ±100ms window
        t_total = n_samp / sample_rate
        for act in actions:
            ts = act.get("timestamp", 0.0)
            if ts < 0 or ts > t_total:
                continue
            idx = int(ts * sample_rate)
            lo = max(0, idx - window_samp)
            hi = min(n_samp, idx + window_samp)
            cmd[lo:hi] = 1.0

        if cmd.sum() > 0:
            cmd_norm = (cmd - cmd.mean()) / (cmd.std() + 1e-8)
            for ch in range(n_ch):
                ch_norm = (data[ch] - data[ch].mean()) / (data[ch].std() + 1e-8)
                correlation[ch] = float(np.dot(ch_norm, cmd_norm) / n_samp)

    # Normalise scores to [0, 1]
    var_max = variance.max() or 1.0
    corr_max = np.abs(correlation).max() or 1.0
    score = 0.6 * (variance / var_max) + 0.4 * (np.abs(correlation) / corr_max)

    top_indices = list(np.argsort(score)[::-1][:n_top])
    return top_indices


# ── Main figure builder ───────────────────────────────────────────────────────

def build_waveform_figure(
    data: np.ndarray,
    channels: list[int],
    time_window: float,
    sample_rate: int = SAMPLE_RATE_HZ,
    scale_factor: float = 2.5,
    actions: Optional[list[dict]] = None,
    highlighted_channels: Optional[list[int]] = None,
) -> go.Figure:
    """Build a stacked waveform Plotly figure.

    Parameters
    ----------
    data:
        Shape ``(n_channels, n_samples)`` — z-scored neural data.
    channels:
        Which channel indices to display (0-based).
    time_window:
        Display window in seconds.
    scale_factor:
        Vertical offset multiplier between channels.
    actions:
        List of overlay dicts: ``{timestamp, direction, label}``.
        Timestamps are seconds relative to session start.
    highlighted_channels:
        Subset of *channels* to draw in a brighter colour/thicker line.
    """
    if highlighted_channels is None:
        highlighted_channels = []

    n_samples = data.shape[1]
    t = np.linspace(0, n_samples / sample_rate, n_samples)

    # Trim to time_window from the right
    if len(t) > 1 and t[-1] > time_window:
        start_idx = int(np.searchsorted(t, t[-1] - time_window))
        t = t[start_idx:] - t[start_idx]
        data = data[:, start_idx:]

    fig = go.Figure()

    for i, ch in enumerate(channels):
        if ch >= data.shape[0]:
            continue

        offset = i * scale_factor
        is_highlighted = ch in highlighted_channels

        # Smooth + downsample for display
        t_ds = _downsample(t, MAX_DISPLAY_SAMPLES)
        trace_ds = _smooth_and_downsample(data[ch]) + offset

        color_idx = i % len(_CH_COLORS)
        color = _CH_COLORS[color_idx]
        line_width = 1.8 if is_highlighted else 1.0
        opacity = 1.0 if is_highlighted else 0.75

        label = f"Ch {ch + 1}" + (" ★" if is_highlighted else "")

        fig.add_trace(go.Scatter(
            x=t_ds,
            y=trace_ds,
            mode="lines",
            name=label,
            line=dict(color=color, width=line_width),
            opacity=opacity,
            hoverinfo="skip",
        ))

    # ── Action overlays ───────────────────────────────────────────────────────
    if actions:
        t_end = float(t[-1]) if len(t) > 0 else 0.0
        y_top = len(channels) * scale_factor + 0.4

        for act in actions[-20:]:
            ts = act.get("timestamp", 0.0)
            direction = act.get("direction", "UP")
            label = act.get("label", DIRECTION_LABELS.get(direction, direction))

            if ts < 0 or ts > t_end:
                continue

            line_color = DIRECTION_LINE_COLORS.get(direction, "#545333")
            shade_color = DIRECTION_COLORS.get(direction, "rgba(84,83,51,0.2)")

            fig.add_vline(
                x=ts,
                line=dict(color=line_color, width=1.5, dash="dash"),
                opacity=0.85,
            )
            fig.add_vrect(
                x0=ts - 0.08,
                x1=ts + 0.08,
                fillcolor=shade_color,
                line_width=0,
            )
            fig.add_annotation(
                x=ts,
                y=y_top,
                text=f"<b>{label}</b>",
                showarrow=False,
                font=dict(color="#161510", size=13, family="Share Tech Mono"),
                bgcolor="rgba(253,251,212,0.85)",
                bordercolor="#878672",
                borderwidth=1,
            )

    # ── Layout ────────────────────────────────────────────────────────────────
    fig.update_layout(
        template="none",
        paper_bgcolor="rgba(253,251,212,0.0)",
        plot_bgcolor="rgba(217,215,182,0.25)",
        height=min(60 + len(channels) * 38, 380),
        margin=dict(l=55, r=20, t=25, b=35),
        xaxis=dict(
            title=dict(text="Time (s)", font=dict(size=14, color="#545333")),
            showgrid=True,
            gridcolor="rgba(135,134,114,0.2)",
            color="#545333",
            tickfont=dict(size=13),
            zeroline=False,
        ),
        yaxis=dict(
            showticklabels=False,
            showgrid=False,
            zeroline=False,
        ),
        showlegend=True,
        legend=dict(
            orientation="v",
            x=1.01,
            y=1,
            xanchor="left",
            font=dict(size=12, color="#161510", family="Share Tech Mono"),
            bgcolor="rgba(253,251,212,0.8)",
            bordercolor="#878672",
            borderwidth=1,
        ),
    )

    return fig
