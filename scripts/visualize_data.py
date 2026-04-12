"""
Visualize recorded training data from the SciFi headstage.

Plots neural channels alongside ground truth controller labels
to understand the encoding and verify data quality.

Usage:
    python scripts/visualize_data.py --file data/recordings/easy_20260410.h5
"""

import argparse
import numpy as np
import h5py
import matplotlib.pyplot as plt


def parse_args():
    parser = argparse.ArgumentParser(description="Visualize SciFi training data")
    parser.add_argument("--file", "-f", required=True, help="Path to HDF5 recording")
    parser.add_argument("--start", type=float, default=0, help="Start time in seconds")
    parser.add_argument("--window", type=float, default=5.0, help="Window duration in seconds")
    parser.add_argument("--channels", type=int, nargs="+", default=None,
                        help="Specific neural channels to plot (default: first 8)")
    return parser.parse_args()


def load_data(filepath, start_sec, window_sec):
    """Load a time window from the HDF5 file."""
    with h5py.File(filepath, "r") as f:
        sample_rate = f.attrs["sample_rate"]
        mode = f.attrs["mode"]
        n_neural = f.attrs["n_neural_channels"]
        n_labels = f.attrs["n_label_channels"]

        start_idx = int(start_sec * sample_rate)
        end_idx = int((start_sec + window_sec) * sample_rate)

        neural = f["neural"][:, start_idx:end_idx]
        labels = f["labels"][:, start_idx:end_idx]

    return neural, labels, sample_rate, mode, n_neural, n_labels


def plot_overview(neural, labels, sample_rate, mode, channels=None):
    """Plot neural channels and label channels side by side."""
    n_neural_ch = neural.shape[0]
    n_label_ch = labels.shape[0]
    n_samples = neural.shape[1]
    t = np.arange(n_samples) / sample_rate

    if channels is None:
        channels = list(range(min(8, n_neural_ch)))

    n_plot_neural = len(channels)
    n_rows = n_plot_neural + n_label_ch

    fig, axes = plt.subplots(n_rows, 1, figsize=(14, 2 * n_rows), sharex=True)
    fig.suptitle(f"SciFi Training Data — {mode} mode", fontsize=14, fontweight="bold")

    # Plot selected neural channels
    for i, ch in enumerate(channels):
        axes[i].plot(t, neural[ch], linewidth=0.5, color="steelblue")
        axes[i].set_ylabel(f"Neural {ch}")
        axes[i].set_xlim(t[0], t[-1])

    # Plot label channels
    label_names_easy = ["Joystick X", "Joystick Y"]
    label_names_hard = [
        "L-Stick X", "L-Stick Y", "R-Stick X", "R-Stick Y",
        "A", "B", "X", "Y",
        "L-Trigger", "R-Trigger", "L-Bumper", "R-Bumper"
    ]
    label_names = label_names_easy if mode == "easy" else label_names_hard

    for i in range(n_label_ch):
        ax_idx = n_plot_neural + i
        axes[ax_idx].plot(t, labels[i], linewidth=1.0, color="tomato")
        name = label_names[i] if i < len(label_names) else f"Label {i}"
        axes[ax_idx].set_ylabel(name)
        axes[ax_idx].set_xlim(t[0], t[-1])

    axes[-1].set_xlabel("Time (s)")
    plt.tight_layout()
    plt.show()


def plot_correlation_matrix(neural, labels, sample_rate):
    """Plot correlation between neural channels and label channels."""
    # Downsample for correlation analysis (100 Hz bins)
    bin_size = int(sample_rate / 100)
    n_bins = neural.shape[1] // bin_size

    neural_binned = neural[:, :n_bins * bin_size].reshape(neural.shape[0], n_bins, bin_size).mean(axis=2)
    labels_binned = labels[:, :n_bins * bin_size].reshape(labels.shape[0], n_bins, bin_size).mean(axis=2)

    # Compute correlation between each neural channel and each label
    all_data = np.vstack([neural_binned, labels_binned])
    corr = np.corrcoef(all_data)

    n_n = neural_binned.shape[0]
    n_l = labels_binned.shape[0]

    # Extract neural-label cross-correlation block
    cross_corr = corr[:n_n, n_n:]

    fig, ax = plt.subplots(figsize=(max(6, n_l * 2), max(8, n_n * 0.3)))
    im = ax.imshow(np.abs(cross_corr), aspect="auto", cmap="hot", vmin=0, vmax=1)
    ax.set_xlabel("Label Channel")
    ax.set_ylabel("Neural Channel")
    ax.set_title("Neural-Label Correlation (absolute)")
    plt.colorbar(im, ax=ax, label="|correlation|")
    plt.tight_layout()
    plt.show()

    # Print top correlated channels per label
    print("\nTop 5 correlated neural channels per label:")
    for l_idx in range(n_l):
        top_ch = np.argsort(np.abs(cross_corr[:, l_idx]))[::-1][:5]
        corrs = cross_corr[top_ch, l_idx]
        print(f"  Label {l_idx}: {list(zip(top_ch.tolist(), [f'{c:.3f}' for c in corrs]))}")


def plot_channel_spectra(neural, sample_rate, channels=None):
    """Plot power spectra of neural channels to understand frequency content."""
    if channels is None:
        channels = list(range(min(4, neural.shape[0])))

    fig, axes = plt.subplots(len(channels), 1, figsize=(12, 3 * len(channels)), sharex=True)
    if len(channels) == 1:
        axes = [axes]

    for i, ch in enumerate(channels):
        freqs = np.fft.rfftfreq(neural.shape[1], 1.0 / sample_rate)
        psd = np.abs(np.fft.rfft(neural[ch])) ** 2
        psd_db = 10 * np.log10(psd + 1e-12)

        axes[i].plot(freqs, psd_db, linewidth=0.5)
        axes[i].set_ylabel(f"Ch {ch} (dB)")
        axes[i].set_xlim(0, min(5000, sample_rate / 2))

    axes[-1].set_xlabel("Frequency (Hz)")
    fig.suptitle("Neural Channel Power Spectra", fontsize=13)
    plt.tight_layout()
    plt.show()


def main():
    args = parse_args()
    neural, labels, sr, mode, n_n, n_l = load_data(args.file, args.start, args.window)

    print(f"Loaded: mode={mode}, {n_n} neural ch, {n_l} label ch, "
          f"sr={sr} Hz, {neural.shape[1]} samples ({neural.shape[1]/sr:.1f}s)")

    plot_overview(neural, labels, sr, mode, args.channels)
    plot_correlation_matrix(neural, labels, sr)
    plot_channel_spectra(neural, sr, args.channels)


if __name__ == "__main__":
    main()
