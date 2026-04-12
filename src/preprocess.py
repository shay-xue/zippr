"""
preprocess.py -- Neural recording preprocessing pipeline.

Pipeline (per file, per k value):
  1. Load HDF5 recording -> flat ElectricalSeries array, reshape to (76, T)
  2. Split: neural channels 0-63, target channels 64-75
  3. Z-score normalize each of the 64 neural channels independently
  4. Temporal binning: 20 ms windows (640 samples @ 32 kHz)
  5. Spike counting per bin: count samples where |z| > k
  6. Target aggregation per bin: mean of raw controller values
  7. Save X (num_bins, 64), Y (num_bins, 12) -> .npz

Target aggregation choice -- MEAN:
  Controller axes (joysticks) are continuous signals in [-32767, +32767].
  Buttons are step functions (0 or 32767).  Using the bin mean:
    - Captures sustained joystick positions accurately
    - Represents "fraction of bin pressed" for buttons (a useful soft label)
    - Is more robust than "last sample" to single-sample noise at bin edges
  Mean is strictly better than last-value for regression targets.

Usage
-----
    python src/preprocess.py                   # uses config.yaml
    python src/preprocess.py --config config.yaml --k 3.5
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import List, Optional, Tuple

import h5py
import numpy as np

# Ensure repo root is on path so 'src' imports work from the CLI
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    ensure_dir, get_logger, load_config, preprocessed_path,
    set_seed, zscore_per_channel,
)

log = get_logger("preprocess")

# -- Constants (overridden by config at runtime) --------------------------------
N_NEURAL     = 64
N_TARGETS    = 12
N_TOTAL      = 76
SAMPLE_HZ    = 32_000
BIN_MS       = 20.0
BIN_SAMPLES  = 640   # = BIN_MS * SAMPLE_HZ / 1000


# ==============================================================================
# Public API
# ==============================================================================

def run_preprocessing(cfg: dict) -> None:
    """
    Entry point: sweep over all recording files x all k values.

    Skips output files that already exist (skip_existing=True in config).
    Prints sanity-check stats for the very first processed file.

    Parameters
    ----------
    cfg : Loaded config.yaml dict.
    """
    rec_cfg   = cfg["recording"]
    pre_cfg   = cfg["preprocessing"]
    data_cfg  = cfg["data"]

    rec_dir   = Path(data_cfg["recordings_dir"])
    out_dir   = ensure_dir(Path(data_cfg["preprocessed_dir"]))
    k_values  = pre_cfg["k_values"]
    patterns  = data_cfg.get("file_patterns", ["broadband_data*.h5"])
    skip      = pre_cfg.get("skip_existing", True)
    do_sanity = pre_cfg.get("sanity_check", True)

    # Collect files
    files: List[Path] = []
    for pat in patterns:
        files.extend(sorted(rec_dir.glob(pat)))
    files = [f for f in files if f.stat().st_size > 1024]   # skip LFS pointers

    if not files:
        log.error(f"No valid .h5 files found in {rec_dir}")
        return

    log.info(f"Found {len(files)} recording(s): {[f.name for f in files]}")

    first_save = True
    for fpath in files:
        log.info(f"\n{'-'*60}")
        log.info(f"Loading: {fpath.name}  ({fpath.stat().st_size/1e6:.1f} MB)")

        try:
            neural, targets, sample_hz = load_h5_recording(
                fpath,
                n_neural=rec_cfg["n_neural"],
                n_targets=rec_cfg["n_targets"],
                n_total=rec_cfg["n_total"],
            )
        except Exception as exc:
            log.error(f"  Failed to load {fpath.name}: {exc}")
            continue

        log.info(f"  neural={neural.shape}  targets={targets.shape}  hz={sample_hz}")

        bin_samples = int(pre_cfg["bin_ms"] * sample_hz / 1000)

        for k in k_values:
            out_path = preprocessed_path(out_dir, fpath.name, k)

            if skip and out_path.exists():
                log.info(f"  [skip] {out_path.name} already exists")
                continue

            X, Y = preprocess_recording(
                neural, targets, bin_samples,
                k=k,
                aggregation=pre_cfg["target_aggregation"],
            )

            np.savez_compressed(out_path, X=X, Y=Y)
            log.info(f"  Saved -> {out_path.name}  X={X.shape}  Y={Y.shape}")

            if do_sanity and first_save:
                _print_sanity(X, Y, fpath.name, k)
                first_save = False

    log.info(f"\nPreprocessing complete. Output dir: {out_dir}")


def preprocess_recording(
    neural: np.ndarray,
    targets: np.ndarray,
    bin_samples: int,
    k: float,
    aggregation: str = "mean",
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform a single recording into binned feature + target arrays.

    Parameters
    ----------
    neural      : (n_neural, n_samples) float32 -- raw neural voltages
    targets     : (n_targets, n_samples) float32 -- raw controller signals
    bin_samples : number of samples per bin (e.g. 640 for 20 ms @ 32 kHz)
    k           : spike-detection threshold multiplier (|z| > k counts as spike)
    aggregation : "mean" or "last" -- how to aggregate targets per bin

    Returns
    -------
    X : (num_bins, n_neural)   float32 -- spike counts per bin per channel
    Y : (num_bins, n_targets)  float32 -- aggregated controller signals
    """
    n_neural, n_samples = neural.shape

    # -- Step 1: Z-score normalize each neural channel independently ---------
    z, _, _ = zscore_per_channel(neural)  # (n_neural, n_samples)

    # -- Step 2: Temporal binning -- truncate to whole bins -------------------
    num_bins = n_samples // bin_samples
    if num_bins == 0:
        raise ValueError(
            f"Recording too short ({n_samples} samples) for bin={bin_samples}"
        )
    usable = num_bins * bin_samples

    z_bins = z[:, :usable].reshape(n_neural, num_bins, bin_samples)
    # z_bins: (n_neural, num_bins, bin_samples)

    t_bins = targets[:, :usable].reshape(targets.shape[0], num_bins, bin_samples)

    # -- Step 3: Spike counting -- |z| > k ------------------------------------
    # Count per (channel, bin) -> transpose to (num_bins, n_neural)
    X = (np.abs(z_bins) > k).sum(axis=-1).T.astype(np.float32)
    # X: (num_bins, n_neural)

    # -- Step 4: Target aggregation -------------------------------------------
    if aggregation == "mean":
        Y = t_bins.mean(axis=-1).T.astype(np.float32)
    elif aggregation == "last":
        Y = t_bins[:, :, -1].T.astype(np.float32)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation!r}")
    # Y: (num_bins, n_targets)

    return X, Y


def load_h5_recording(
    fpath: Path,
    n_neural: int = N_NEURAL,
    n_targets: int = N_TARGETS,
    n_total: int = N_TOTAL,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Load a Science Corp SciFi HDF5 recording.

    The ElectricalSeries dataset is stored as a flat 1-D array of interleaved
    samples: [ch0_t0, ch1_t0, ..., ch75_t0, ch0_t1, ch1_t1, ..., ch75_tN].
    We reshape to (n_total, n_timepoints) and split into neural + target arrays.

    Parameters
    ----------
    fpath    : Path to the .h5 file.
    n_neural : Number of neural channels (default 64).
    n_targets: Number of target/label channels (default 12).
    n_total  : Total channels expected (default 76).

    Returns
    -------
    neural  : (n_neural, n_samples) float32
    targets : (n_targets, n_samples) float32
    hz      : int -- sample rate from file attributes
    """
    with h5py.File(fpath, "r") as f:
        hz = int(f.attrs.get("sample_rate_hz", SAMPLE_HZ))
        raw = f["acquisition/ElectricalSeries"][:]
        # Read the electrode count actually stored in the file
        actual_n_ch = int(f["general/extracellular_ephys/electrodes/id"].shape[0])

    # Guard: skip recordings with the wrong channel layout (e.g. early 35-ch runs)
    if actual_n_ch != n_total:
        raise ValueError(
            f"Expected {n_total} channels but file has {actual_n_ch}. "
            f"Skipping (not a hard-mode 76-channel recording)."
        )

    # Interleaved flat -> (n_total, n_timepoints)
    n_timepoints = raw.shape[0] // n_total
    data = raw[: n_timepoints * n_total].reshape(n_timepoints, n_total).T.astype(np.float32)

    neural  = data[:n_neural]                          # (64, T)
    targets = data[n_neural : n_neural + n_targets]    # (12, T)

    return neural, targets, hz


# ==============================================================================
# Internal helpers
# ==============================================================================

def _print_sanity(X: np.ndarray, Y: np.ndarray, fname: str, k: float) -> None:
    """Print sanity-check statistics for the first processed output."""
    print("\n" + "=" * 60)
    print(f"SANITY CHECK -- {fname}  k={k}")
    print("=" * 60)
    print(f"  X.shape  = {X.shape}  (num_bins x n_neural)")
    print(f"  Y.shape  = {Y.shape}  (num_bins x n_targets)")
    print(f"  X.min()  = {X.min():.4f}")
    print(f"  X.max()  = {X.max():.4f}")
    print(f"  X.mean() = {X.mean():.4f}  (expected: spike rate ~= 0-640 counts/bin)")
    print(f"  Y.min()  = {Y.min():.4f}")
    print(f"  Y.max()  = {Y.max():.4f}")
    print(f"  Non-zero X bins: {(X.sum(axis=1) > 0).mean()*100:.1f}%")
    print("=" * 60 + "\n")


# ==============================================================================
# CLI
# ==============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Preprocess neural HDF5 recordings.")
    p.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    p.add_argument(
        "--k", type=float, nargs="+", default=None,
        help="Override k values (e.g. --k 3.0 3.5). Default: from config."
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    if args.k:
        cfg["preprocessing"]["k_values"] = args.k
        log.info(f"k override: {args.k}")

    set_seed(cfg["split"]["random_seed"])
    run_preprocessing(cfg)
