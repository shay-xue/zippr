"""
Exploratory data analysis for broadband HDF5 recordings.

Usage:
    uv run python scripts/eda.py data_collection/broadband_data_20260410_142940.h5
    uv run python scripts/eda.py data_collection/*.h5          # first file in glob
"""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── label helpers (mirrors train_decoder_sklearn.py) ──────────────────────────

JOYSTICK_X_CH = 0
JOYSTICK_Y_CH = 1
A_BUTTON_CH   = 2
AXIS_MAX      = 32767.0
DEAD_ZONE     = 0.3

CLASS_NAMES = [
    "center",
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
    "A_center", "A_upper", "A_lower",
]


def make_class_labels(labels):
    x = labels[JOYSTICK_X_CH] / AXIS_MAX
    y = labels[JOYSTICK_Y_CH] / AXIS_MAX
    a = labels[A_BUTTON_CH] > 0

    magnitude = np.sqrt(x**2 + y**2)
    active    = magnitude > DEAD_ZONE
    angle     = np.arctan2(y, x)

    pi = np.pi
    boundaries = np.linspace(-pi, pi, 9)
    sector_to_class = [7, 6, 5, 4, 3, 2, 1, 8]

    classes = np.zeros(len(x), dtype=np.int64)
    for sec_idx in range(8):
        lo, hi = boundaries[sec_idx], boundaries[sec_idx + 1]
        in_sector = active & (angle >= lo) & (angle < hi)
        classes[in_sector] = sector_to_class[sec_idx]

    classes[a & ~active]       = 9
    classes[a & active & (y > 0)]  = 10
    classes[a & active & (y <= 0)] = 11
    return classes


# ── H5 reader ─────────────────────────────────────────────────────────────────

def read_h5(path):
    with h5py.File(path, "r") as f:
        sr_candidates = ["sample_rate_hz", "sample_rate", "fs", "samplerate"]
        sample_rate = next(
            (int(f.attrs[k]) for k in sr_candidates if k in f.attrs), 32000
        )

        if "acquisition" in f and "ElectricalSeries" in f["acquisition"]:
            raw        = f["acquisition"]["ElectricalSeries"][()].astype(np.float32)
            n_channels = len(f["general"]["extracellular_ephys"]["electrodes"]["id"])
            n_samples  = len(f["acquisition"]["sequence_number"])
            timestamps = f["acquisition"]["timestamp_ns"][()].astype(np.float64)
            arr        = raw.reshape(n_samples, n_channels).T  # (C, N)
            neural     = arr[:-3]
            labels     = arr[-3:]
            return neural, labels, timestamps, sample_rate

        raise KeyError(f"Unrecognised H5 layout in {path}. Keys: {list(f.keys())}")


# ── plots ─────────────────────────────────────────────────────────────────────

def plot(path, out_png=None):
    neural, labels, timestamps, sr = read_h5(path)
    n_ch, n_samples = neural.shape
    t = (timestamps - timestamps[0]) * 1e-9   # ns → s (relative)

    classes = make_class_labels(labels)
    window_ms, stride_ms = 50, 25
    w = int(window_ms * sr / 1000)
    s = int(stride_ms * sr / 1000)
    n_win = (n_samples - w) // s + 1

    # Feature matrix X: mean + std per channel per window
    X = np.empty((n_win, n_ch * 2), dtype=np.float32)
    for i in range(n_win):
        chunk = neural[:, i*s : i*s + w]
        X[i, :n_ch] = chunk.mean(axis=1)
        X[i, n_ch:] = chunk.std(axis=1)

    # Window labels y (majority vote)
    y = np.array([
        np.bincount(classes[i*s : i*s + w], minlength=12).argmax()
        for i in range(n_win)
    ], dtype=np.int64)

    t_win = np.array([t[min(i*s + w//2, n_samples-1)] for i in range(n_win)])

    # ── figure ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(path, fontsize=10, y=0.98)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Raw neural traces (first 8 channels, 2-second snippet)
    ax1 = fig.add_subplot(gs[0, :])
    snip = min(2 * sr, n_samples)
    offset_scale = 4 * neural[:8, :snip].std()
    for ch in range(min(8, n_ch)):
        ax1.plot(t[:snip], neural[ch, :snip] + ch * offset_scale,
                 lw=0.4, alpha=0.8)
    ax1.set_title("Raw neural traces — first 2 s (channels 0-7, offset)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (ADC)")
    ax1.set_xlim(t[0], t[snip - 1])

    # 2. Label channels over time
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, labels[0] / AXIS_MAX, lw=0.6, label="joystick X (norm)")
    ax2.plot(t, labels[1] / AXIS_MAX, lw=0.6, label="joystick Y (norm)")
    ax2.plot(t, labels[2],            lw=0.8, label="A button",  alpha=0.7)
    ax2.set_title("Label channels over time")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(t[0], t[-1])

    # 3. Class label timeline (windowed y)
    ax3 = fig.add_subplot(gs[2, :])
    ax3.step(t_win, y, where="mid", lw=0.8)
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax3.set_title(f"Windowed class labels y  ({window_ms} ms window, {stride_ms} ms stride)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim(t_win[0], t_win[-1])
    ax3.grid(axis="y", ls="--", alpha=0.4)

    # 4. Class distribution bar chart
    ax4 = fig.add_subplot(gs[3, 0])
    unique, counts = np.unique(y, return_counts=True)
    ax4.bar([CLASS_NAMES[c] for c in unique], counts)
    ax4.set_title("Class distribution (y)")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Windows")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)

    # 5. Feature heatmap (X) — mean features only, all windows
    ax5 = fig.add_subplot(gs[3, 1])
    im = ax5.imshow(X[:, :n_ch].T, aspect="auto", interpolation="none",
                    origin="upper", cmap="RdBu_r",
                    vmin=np.percentile(X[:, :n_ch], 2),
                    vmax=np.percentile(X[:, :n_ch], 98))
    ax5.set_title("Feature matrix X (channel means per window)")
    ax5.set_xlabel("Window index")
    ax5.set_ylabel("Channel")
    plt.colorbar(im, ax=ax5, fraction=0.046)

    out = out_png or path.replace(".h5", "_eda.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    # ── summary ───────────────────────────────────────────────────────────────
    print(f"\n{'='*50}")
    print(f"File        : {path}")
    print(f"Sample rate : {sr} Hz")
    print(f"Duration    : {n_samples/sr:.2f} s  ({n_samples} samples)")
    print(f"Neural ch   : {n_ch}")
    print(f"Windows     : {n_win}  (X shape {X.shape}, y shape {y.shape})")
    print(f"\nClass distribution:")
    for c, n in zip(unique, counts):
        print(f"  {CLASS_NAMES[c]:10s}  {n:5d} windows  ({100*n/n_win:.1f}%)")
    print(f"\nX stats  min={X.min():.3g}  max={X.max():.3g}  "
          f"mean={X.mean():.3g}  std={X.std():.3g}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/eda.py <file.h5> [file2.h5 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        plot(path)
