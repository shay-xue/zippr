"""
Exploratory data analysis for broadband HDF5 recordings.

Produces four PNG files per recording:
  *_eda.png         — timeseries, labels, class distribution, feature heatmap
  *_spectral.png    — amplitude histogram, mean PSD, autocorrelation, cross-channel corr
  *_ch_psd.png      — per-channel power spectra (grid)
  *_ch_bandpower.png — per-channel bandpower heatmap + bar summary
  *_ch_spectrogram.png — per-channel spectrograms (grid)

Usage:
    uv run python scripts/eda.py data_collection/broadband_data_20260410_142940.h5
    uv run python scripts/eda.py data_collection/*.h5
"""

import sys
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# ── label helpers ─────────────────────────────────────────────────────────────

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

    classes[a & ~active]            = 9
    classes[a & active & (y > 0)]   = 10
    classes[a & active & (y <= 0)]  = 11
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
            arr        = raw.reshape(n_samples, n_channels).T   # (C, N)
            neural     = arr[:-3]
            labels     = arr[-3:]
            return neural, labels, timestamps, sample_rate

        raise KeyError(f"Unrecognised H5 layout in {path}. Keys: {list(f.keys())}")


# ── plot 1: timeseries + labels ───────────────────────────────────────────────

def plot_overview(path, neural, labels, timestamps, sr):
    n_ch, n_samples = neural.shape
    t = (timestamps - timestamps[0]) * 1e-9

    classes = make_class_labels(labels)
    window_ms, stride_ms = 50, 25
    w = int(window_ms * sr / 1000)
    s = int(stride_ms * sr / 1000)
    n_win = (n_samples - w) // s + 1

    X = np.empty((n_win, n_ch * 2), dtype=np.float32)
    for i in range(n_win):
        chunk = neural[:, i*s : i*s + w]
        X[i, :n_ch] = chunk.mean(axis=1)
        X[i, n_ch:] = chunk.std(axis=1)

    y = np.array([
        np.bincount(classes[i*s : i*s + w], minlength=12).argmax()
        for i in range(n_win)
    ], dtype=np.int64)
    t_win = np.array([t[min(i*s + w//2, n_samples-1)] for i in range(n_win)])

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(path, fontsize=9, y=0.99)
    gs = gridspec.GridSpec(4, 2, figure=fig, hspace=0.45, wspace=0.35)

    # Raw traces
    ax1 = fig.add_subplot(gs[0, :])
    snip = min(2 * sr, n_samples)
    offset_scale = 4 * neural[:8, :snip].std()
    for ch in range(min(8, n_ch)):
        ax1.plot(t[:snip], neural[ch, :snip] + ch * offset_scale, lw=0.4, alpha=0.8)
    ax1.set_title("Raw neural traces — first 2 s (ch 0-7, offset)")
    ax1.set_xlabel("Time (s)")
    ax1.set_ylabel("Amplitude (ADC)")
    ax1.set_xlim(t[0], t[snip - 1])

    # Label channels
    ax2 = fig.add_subplot(gs[1, :])
    ax2.plot(t, labels[0] / AXIS_MAX, lw=0.6, label="joystick X (norm)")
    ax2.plot(t, labels[1] / AXIS_MAX, lw=0.6, label="joystick Y (norm)")
    ax2.plot(t, labels[2],            lw=0.8, label="A button", alpha=0.7)
    ax2.set_title("Label channels over time")
    ax2.set_xlabel("Time (s)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_xlim(t[0], t[-1])

    # Class timeline
    ax3 = fig.add_subplot(gs[2, :])
    ax3.step(t_win, y, where="mid", lw=0.8)
    ax3.set_yticks(range(12))
    ax3.set_yticklabels(CLASS_NAMES, fontsize=7)
    ax3.set_title(f"Windowed class labels y  ({window_ms} ms window, {stride_ms} ms stride)")
    ax3.set_xlabel("Time (s)")
    ax3.set_xlim(t_win[0], t_win[-1])
    ax3.grid(axis="y", ls="--", alpha=0.4)

    # Class distribution
    ax4 = fig.add_subplot(gs[3, 0])
    unique, counts = np.unique(y, return_counts=True)
    ax4.bar([CLASS_NAMES[c] for c in unique], counts)
    ax4.set_title("Class distribution (y)")
    ax4.set_xlabel("Class")
    ax4.set_ylabel("Windows")
    ax4.tick_params(axis="x", rotation=45, labelsize=7)

    # Feature heatmap
    ax5 = fig.add_subplot(gs[3, 1])
    im = ax5.imshow(X[:, :n_ch].T, aspect="auto", interpolation="none",
                    origin="upper", cmap="RdBu_r",
                    vmin=np.percentile(X[:, :n_ch], 2),
                    vmax=np.percentile(X[:, :n_ch], 98))
    ax5.set_title("Feature matrix X (channel means per window)")
    ax5.set_xlabel("Window index")
    ax5.set_ylabel("Channel")
    plt.colorbar(im, ax=ax5, fraction=0.046)

    out = path.replace(".h5", "_eda.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")

    unique, counts = np.unique(y, return_counts=True)
    print(f"\n{'='*50}")
    print(f"File        : {path}")
    print(f"Sample rate : {sr} Hz")
    print(f"Duration    : {n_samples/sr:.2f} s  ({n_samples} samples)")
    print(f"Neural ch   : {n_ch}")
    print(f"Windows     : {n_win}  (X shape {X.shape}, y shape {y.shape})")
    print(f"Class dist  :")
    for c, n in zip(unique, counts):
        print(f"  {CLASS_NAMES[c]:10s}  {n:5d} windows  ({100*n/n_win:.1f}%)")
    print(f"X stats     : min={X.min():.3g}  max={X.max():.3g}  "
          f"mean={X.mean():.3g}  std={X.std():.3g}")


# ── plot 2: spectral analysis ─────────────────────────────────────────────────

def plot_spectral(path, neural, sr):
    """Amplitude histogram, power spectrum, and autocorrelation."""
    n_ch, n_samples = neural.shape

    # Z-score each channel for fair comparison across channels
    z = (neural - neural.mean(axis=1, keepdims=True)) / (neural.std(axis=1, keepdims=True) + 1e-8)

    fig = plt.figure(figsize=(16, 14))
    fig.suptitle(f"{path} — spectral analysis", fontsize=9, y=0.99)
    gs = gridspec.GridSpec(3, 2, figure=fig, hspace=0.45, wspace=0.35)

    # ── 1. Amplitude histogram (all channels pooled + per-channel overlay) ────
    ax1 = fig.add_subplot(gs[0, 0])
    all_vals = neural.ravel()
    ax1.hist(all_vals, bins=200, color="steelblue", alpha=0.7, density=True,
             label="all channels")
    # overlay a few individual channels in lighter colours
    for ch in range(min(4, n_ch)):
        ax1.hist(neural[ch], bins=200, histtype="step", density=True,
                 lw=0.8, alpha=0.6, label=f"ch {ch}")
    ax1.set_title("Amplitude histogram (ADC counts)")
    ax1.set_xlabel("Amplitude")
    ax1.set_ylabel("Density")
    ax1.legend(fontsize=7)

    # ── 2. Per-channel amplitude stats (mean ± std) ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    ch_mean = neural.mean(axis=1)
    ch_std  = neural.std(axis=1)
    ch_idx  = np.arange(n_ch)
    ax2.fill_between(ch_idx, ch_mean - ch_std, ch_mean + ch_std,
                     alpha=0.3, color="steelblue", label="±1 std")
    ax2.plot(ch_idx, ch_mean, color="steelblue", lw=1.2, label="mean")
    ax2.set_title("Per-channel amplitude: mean ± std")
    ax2.set_xlabel("Channel")
    ax2.set_ylabel("ADC counts")
    ax2.legend(fontsize=8)

    # ── 3. Mean power spectrum (Welch-style average across channels) ──────────
    ax3 = fig.add_subplot(gs[1, :])
    # Use rfft on each channel, average power across channels
    fft_len = min(n_samples, 4 * sr)   # cap at 4 s for speed
    window  = np.hanning(fft_len)
    psd_sum = np.zeros(fft_len // 2 + 1)
    n_segs  = 0
    step    = fft_len // 2
    for start in range(0, n_samples - fft_len, step):
        seg = z[:, start : start + fft_len] * window   # (C, fft_len)
        F   = np.fft.rfft(seg, axis=1)                 # (C, fft_len//2+1)
        psd_sum += (np.abs(F) ** 2).mean(axis=0)
        n_segs  += 1
    psd = psd_sum / max(n_segs, 1)
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)

    ax3.semilogy(freqs, psd, lw=0.8, color="darkorange")
    ax3.set_title("Mean power spectrum (averaged across all channels & segments)")
    ax3.set_xlabel("Frequency (Hz)")
    ax3.set_ylabel("Power (log scale)")
    ax3.set_xlim(0, sr / 2)
    # annotate common neural bands
    for band, (lo, hi, col) in {
        "delta (1-4)":   (1,    4,    "purple"),
        "theta (4-8)":   (4,    8,    "blue"),
        "alpha (8-13)":  (8,    13,   "cyan"),
        "beta (13-30)":  (13,   30,   "green"),
        "gamma (30-80)": (30,   80,   "red"),
        "HFO (80-300)":  (80,   300,  "maroon"),
    }.items():
        ax3.axvspan(lo, hi, alpha=0.07, color=col, label=band)
    ax3.legend(fontsize=7, loc="upper right")

    # ── 4. Autocorrelation (average across channels, lags up to 200 ms) ──────
    ax4 = fig.add_subplot(gs[2, 0])
    max_lag_samples = int(0.2 * sr)   # 200 ms
    ac_sum = np.zeros(max_lag_samples)
    for ch in range(n_ch):
        sig = z[ch] - z[ch].mean()
        # normalised autocorrelation via FFT (fast)
        n_fft = 2 ** int(np.ceil(np.log2(2 * n_samples - 1)))
        F  = np.fft.rfft(sig, n=n_fft)
        ac = np.fft.irfft(F * np.conj(F))[:n_samples]
        ac = ac / (ac[0] + 1e-12)             # normalise to lag-0 = 1
        ac_sum += ac[:max_lag_samples]
    ac_mean = ac_sum / n_ch
    lags_ms = np.arange(max_lag_samples) * 1000.0 / sr

    ax4.plot(lags_ms, ac_mean, lw=0.9, color="teal")
    ax4.axhline(0, color="k", lw=0.5, ls="--")
    # 95% confidence bound for white noise
    ci = 1.96 / np.sqrt(n_samples)
    ax4.axhline( ci, color="grey", lw=0.7, ls=":", label="95% CI (white noise)")
    ax4.axhline(-ci, color="grey", lw=0.7, ls=":")
    ax4.set_title("Mean autocorrelation (averaged across channels, lags 0–200 ms)")
    ax4.set_xlabel("Lag (ms)")
    ax4.set_ylabel("Normalised AC")
    ax4.legend(fontsize=8)

    # ── 5. Cross-channel correlation matrix ───────────────────────────────────
    ax5 = fig.add_subplot(gs[2, 1])
    # Pearson correlation of z-scored channels over time
    corr = np.corrcoef(z)   # (n_ch, n_ch)
    im = ax5.imshow(corr, vmin=-1, vmax=1, cmap="RdBu_r", aspect="auto")
    ax5.set_title("Cross-channel Pearson correlation")
    ax5.set_xlabel("Channel")
    ax5.set_ylabel("Channel")
    plt.colorbar(im, ax=ax5, fraction=0.046)

    out = path.replace(".h5", "_spectral.png")
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── per-channel helpers ───────────────────────────────────────────────────────

BANDS = {
    "delta":  (1,   4),
    "theta":  (4,   8),
    "alpha":  (8,  13),
    "beta":   (13, 30),
    "gamma":  (30, 80),
    "hfo":    (80, 300),
}


def _channel_psd(sig, sr, fft_len=None):
    """Welch-averaged PSD for a single channel. Returns (freqs, psd)."""
    n = len(sig)
    fft_len = fft_len or min(n, 4 * sr)
    hop     = fft_len // 2
    win     = np.hanning(fft_len)
    psd     = np.zeros(fft_len // 2 + 1)
    count   = 0
    for start in range(0, n - fft_len, hop):
        frame = (sig[start : start + fft_len] - sig[start : start + fft_len].mean()) * win
        psd  += np.abs(np.fft.rfft(frame)) ** 2
        count += 1
    if count:
        psd /= count
    freqs = np.fft.rfftfreq(fft_len, d=1.0 / sr)
    return freqs, psd


def _channel_spectrogram(sig, sr, window_ms=50, hop_ms=10):
    """STFT spectrogram for a single channel. Returns (freqs, times, S_db)."""
    w   = int(window_ms * sr / 1000)
    hop = int(hop_ms    * sr / 1000)
    win = np.hanning(w)
    n   = len(sig)
    n_frames = (n - w) // hop + 1
    freqs    = np.fft.rfftfreq(w, d=1.0 / sr)
    S        = np.zeros((len(freqs), n_frames), dtype=np.float32)
    for i in range(n_frames):
        frame  = (sig[i*hop : i*hop + w] - sig[i*hop : i*hop + w].mean()) * win
        S[:, i] = np.abs(np.fft.rfft(frame)) ** 2
    times = np.arange(n_frames) * hop / sr
    S_db  = 10 * np.log10(S + 1e-12)
    return freqs, times, S_db


def _bandpower(freqs, psd, lo, hi):
    """Trapezoidal integration of PSD within [lo, hi] Hz."""
    mask = (freqs >= lo) & (freqs <= hi)
    if mask.sum() < 2:
        return 0.0
    return float(np.trapezoid(psd[mask], freqs[mask]))


# ── plot 3: per-channel PSD grid ──────────────────────────────────────────────

def plot_ch_psd(path, neural, sr):
    n_ch = neural.shape[0]
    ncols = 8
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5),
                             sharex=False, sharey=False)
    fig.suptitle(f"{path} — per-channel PSD", fontsize=9, y=1.01)
    axes = axes.ravel()

    for ch in range(n_ch):
        ax = axes[ch]
        freqs, psd = _channel_psd(neural[ch], sr)
        ax.semilogy(freqs, psd, lw=0.7, color="steelblue")
        for name, (lo, hi) in BANDS.items():
            ax.axvspan(lo, hi, alpha=0.08)
        ax.set_title(f"ch {ch}", fontsize=7, pad=2)
        ax.set_xlim(0, min(300, sr / 2))
        ax.tick_params(labelsize=6)
        ax.set_xlabel("Hz", fontsize=6)

    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.tight_layout()
    out = path.replace(".h5", "_ch_psd.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── plot 4: per-channel bandpower heatmap ────────────────────────────────────

def plot_ch_bandpower(path, neural, sr):
    n_ch      = neural.shape[0]
    band_names = list(BANDS.keys())
    bp         = np.zeros((n_ch, len(band_names)))

    for ch in range(n_ch):
        freqs, psd = _channel_psd(neural[ch], sr)
        for b, (name, (lo, hi)) in enumerate(BANDS.items()):
            bp[ch, b] = _bandpower(freqs, psd, lo, hi)

    # Log-scale for display
    bp_log = np.log10(bp + 1e-12)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, n_ch * 0.25 + 2)),
                                   gridspec_kw={"width_ratios": [2, 1]})
    fig.suptitle(f"{path} — per-channel bandpower", fontsize=9)

    # Heatmap
    im = ax1.imshow(bp_log, aspect="auto", cmap="inferno", origin="upper")
    ax1.set_xticks(range(len(band_names)))
    ax1.set_xticklabels(band_names, fontsize=8)
    ax1.set_yticks(range(n_ch))
    ax1.set_yticklabels([f"ch {c}" for c in range(n_ch)], fontsize=6)
    ax1.set_title("log₁₀ bandpower  (channels × bands)")
    plt.colorbar(im, ax=ax1, fraction=0.03, label="log₁₀ power")

    # Bar chart: mean across channels per band
    ax2.barh(band_names, bp.mean(axis=0), color="steelblue")
    ax2.set_title("Mean bandpower across channels")
    ax2.set_xlabel("Power (linear)")
    ax2.tick_params(labelsize=8)

    fig.tight_layout()
    out = path.replace(".h5", "_ch_bandpower.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── plot 5: per-channel spectrogram grid ─────────────────────────────────────

def plot_ch_spectrogram(path, neural, sr):
    n_ch  = neural.shape[0]
    ncols = 8
    nrows = int(np.ceil(n_ch / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 3, nrows * 2.5),
                             sharex=False, sharey=False)
    fig.suptitle(f"{path} — per-channel spectrogram (0–300 Hz)", fontsize=9, y=1.01)
    axes = axes.ravel()

    # Compute all spectrograms first to get a shared colour range
    spectrograms = []
    for ch in range(n_ch):
        freqs, times, S_db = _channel_spectrogram(neural[ch], sr)
        spectrograms.append((freqs, times, S_db))

    vmin = np.percentile([s for _, _, s in spectrograms], 5)
    vmax = np.percentile([s for _, _, s in spectrograms], 99)
    freq_mask = freqs <= 300

    for ch, (freqs, times, S_db) in enumerate(spectrograms):
        ax = axes[ch]
        ax.imshow(S_db[freq_mask], aspect="auto", origin="lower",
                  extent=[times[0], times[-1], freqs[freq_mask][0], freqs[freq_mask][-1]],
                  cmap="inferno", vmin=vmin, vmax=vmax, interpolation="nearest")
        ax.set_title(f"ch {ch}", fontsize=7, pad=2)
        ax.tick_params(labelsize=6)
        ax.set_xlabel("s", fontsize=6)
        ax.set_ylabel("Hz", fontsize=6)

    for ax in axes[n_ch:]:
        ax.set_visible(False)

    fig.tight_layout()
    out = path.replace(".h5", "_ch_spectrogram.png")
    fig.savefig(out, dpi=100, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved → {out}")


# ── main ──────────────────────────────────────────────────────────────────────

def run(path):
    neural, labels, timestamps, sr = read_h5(path)
    plot_overview(path, neural, labels, timestamps, sr)
    plot_spectral(path, neural, sr)
    plot_ch_psd(path, neural, sr)
    plot_ch_bandpower(path, neural, sr)
    plot_ch_spectrogram(path, neural, sr)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: uv run python scripts/eda.py <file.h5> [file2.h5 ...]")
        sys.exit(1)
    for path in sys.argv[1:]:
        run(path)
