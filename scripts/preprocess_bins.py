"""
Preprocessing — save canonical binned spike-count features to disk.

Run this ONCE before any training. All model scripts load from these files
so every experiment uses identical features.

For each bin size W:
  1. Load all 32 training HDF5s (channels 0-63 neural, 64-75 labels)
  2. Z-score each neural channel using global training statistics
  3. Count threshold crossings (|signal| > SPIKE_THRESH * std) per bin
  4. Average labels within each bin, normalise to [-1,1] / [0,1]
  5. Save to data/preprocessed/bins_{W}ms.npz

Saved arrays per file:
  X          (n_bins, 64)   float32  — spike counts per bin per channel
  Y          (n_bins, 12)   float32  — normalised labels
  ch_mean    (64,)          float32  — per-channel raw voltage mean
  ch_std     (64,)          float32  — per-channel raw voltage std
  feat_mean  (64,)          float32  — per-feature spike-count mean (for MLP normalisation)
  feat_std   (64,)          float32  — per-feature spike-count std

Metadata stored in the npz (scalars):
  bin_ms, spike_thresh, n_files, total_seconds, sample_rate

Usage:
  python scripts/preprocess_bins.py               # all bin sizes
  python scripts/preprocess_bins.py --bins 50 100 # specific sizes only
  python scripts/preprocess_bins.py --force       # overwrite existing files
"""

import argparse
import h5py
import numpy as np
import glob
import gc
import os
import time

SAMPLE_RATE  = 32_000
N_NEURAL     = 64
N_LABELS     = 12
RECORDINGS   = 'data/recordings'
OUT_DIR      = 'data/preprocessed'
SPIKE_THRESH = 2.0
ALL_BIN_SIZES_MS = [20, 30, 40, 50, 60, 70, 80, 100]


def log(msg):
    print(msg, flush=True)


def load_all_raw():
    files = sorted(glob.glob(f'{RECORDINGS}/train_*/broadband_data_*.h5'))
    if not files:
        raise FileNotFoundError(f"No training files found under {RECORDINGS}/train_*/")
    log(f"Found {len(files)} training files")

    neural_parts, label_parts = [], []
    for i, f in enumerate(files):
        with h5py.File(f, 'r') as hf:
            raw = hf['acquisition/ElectricalSeries'][:]
        n = len(raw) // 76
        d = raw[:n * 76].reshape(n, 76).astype(np.float32)
        neural_parts.append(d[:, :N_NEURAL])
        label_parts.append(d[:, N_NEURAL:N_NEURAL + N_LABELS])
        del raw, d
        if (i + 1) % 8 == 0 or (i + 1) == len(files):
            log(f"  Loaded {i+1}/{len(files)} files")

    neural = np.concatenate(neural_parts)
    labels = np.concatenate(label_parts)
    del neural_parts, label_parts
    gc.collect()

    total_sec = len(neural) / SAMPLE_RATE
    log(f"Total: {neural.shape[0]:,} samples  ({total_sec:.1f}s, {total_sec/60:.1f} min)")
    return neural, labels, len(files), total_sec


def compute_bins(neural, labels, bin_ms):
    bin_samples = int(SAMPLE_RATE * bin_ms / 1000)
    n_bins = len(neural) // bin_samples

    neural_tr = neural[:n_bins * bin_samples]
    labels_tr = labels[:n_bins * bin_samples]

    # Global channel stats (from all data — this is training-only data)
    ch_mean = neural.mean(0).astype(np.float32)
    ch_std  = (neural.std(0) + 1e-8).astype(np.float32)

    # Spike counts: threshold crossings in either direction
    above = neural_tr > (ch_mean + SPIKE_THRESH * ch_std)
    below = neural_tr < (ch_mean - SPIKE_THRESH * ch_std)
    spikes = (above | below).astype(np.float32)
    del above, below

    X = spikes.reshape(n_bins, bin_samples, N_NEURAL).sum(1).astype(np.float32)
    del spikes
    gc.collect()

    # Per-feature stats for downstream normalisation
    feat_mean = X.mean(0).astype(np.float32)
    feat_std  = (X.std(0) + 1e-8).astype(np.float32)

    # Labels: average within bin, normalise
    Y_raw = labels_tr.reshape(n_bins, bin_samples, N_LABELS).mean(1)
    Y = np.empty_like(Y_raw)
    Y[:, :4]  = np.clip(Y_raw[:, :4]  / 32767.0, -1.0, 1.0)   # joystick axes
    Y[:, 4:]  = np.clip(Y_raw[:, 4:]  / 32767.0,  0.0, 1.0)   # buttons / triggers
    Y = Y.astype(np.float32)

    return X, Y, ch_mean, ch_std, feat_mean, feat_std


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--bins', type=int, nargs='+', default=ALL_BIN_SIZES_MS,
                        help='Bin sizes in ms to preprocess')
    parser.add_argument('--force', action='store_true',
                        help='Overwrite existing preprocessed files')
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    t_total = time.time()

    log(f"Spike threshold : {SPIKE_THRESH}σ")
    log(f"Bin sizes       : {args.bins} ms")
    log(f"Output dir      : {OUT_DIR}/\n")

    # Check which sizes still need processing
    to_run = []
    for bin_ms in args.bins:
        out_path = os.path.join(OUT_DIR, f'bins_{bin_ms}ms.npz')
        if os.path.exists(out_path) and not args.force:
            log(f"  SKIP  bins_{bin_ms}ms.npz (exists — use --force to overwrite)")
        else:
            to_run.append(bin_ms)

    if not to_run:
        log("\nAll files already exist. Done.")
        return

    # Load raw data once, reuse for all bin sizes
    log(f"\nLoading raw data...")
    neural, labels, n_files, total_sec = load_all_raw()

    for bin_ms in to_run:
        t_bin = time.time()
        log(f"\nProcessing {bin_ms}ms bins...")

        X, Y, ch_mean, ch_std, feat_mean, feat_std = compute_bins(neural, labels, bin_ms)

        out_path = os.path.join(OUT_DIR, f'bins_{bin_ms}ms.npz')
        np.savez_compressed(
            out_path,
            X           = X,
            Y           = Y,
            ch_mean     = ch_mean,
            ch_std      = ch_std,
            feat_mean   = feat_mean,
            feat_std    = feat_std,
            # Metadata
            bin_ms      = np.float32(bin_ms),
            spike_thresh= np.float32(SPIKE_THRESH),
            n_files     = np.int32(n_files),
            total_seconds = np.float32(total_sec),
            sample_rate = np.int32(SAMPLE_RATE),
        )

        size_mb = os.path.getsize(out_path) / 1e6
        log(f"  Saved {out_path}  ({X.shape[0]:,} bins, {size_mb:.1f} MB, {time.time()-t_bin:.1f}s)")

        del X, Y, ch_mean, ch_std, feat_mean, feat_std
        gc.collect()

    del neural, labels
    gc.collect()

    log(f"\nAll done in {time.time()-t_total:.1f}s")
    log(f"Load in training scripts with:")
    log(f"  d = np.load('data/preprocessed/bins_50ms.npz')")
    log(f"  X, Y = d['X'], d['Y']")


if __name__ == '__main__':
    main()
