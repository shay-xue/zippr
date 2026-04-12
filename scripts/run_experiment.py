#!/usr/bin/env python3
"""
run_experiment.py

End-to-end pipeline:
  1. Fetch HDF5 recordings from the `manraj` branch via git + LFS
  2. Auto-detect HDF5 layout and extract the (n_channels, n_samples) array
  3. Window continuous recordings into (n_windows, n_timesteps, 64) trials
  4. Label each window (argmax of absolute label-channel activity, channels 64-75)
  5. Build a stratified 80/20 train-test split and save to data/processed/
  6. Call compare_models.py on the prepared dataset

Usage
-----
    python scripts/run_experiment.py
    python scripts/run_experiment.py --window-ms 50 --stride-ms 25 --epochs 80 --out-dir "models/attempt 2"
    python scripts/run_experiment.py --skip-fetch   # if data already pulled
"""

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import h5py
import numpy as np
from sklearn.model_selection import train_test_split

# ─── Constants ────────────────────────────────────────────────────────────────
REPO_ROOT    = Path(__file__).resolve().parent.parent   # worktree root
DATA_DIR     = REPO_ROOT / "data" / "recordings"
MAPPING_DIR  = REPO_ROOT / "data" / "recordings" / "mappings"
PROC_DIR     = REPO_ROOT / "data" / "processed"
MAPPING_META = REPO_ROOT / "data_collection" / "run_004_structured_mapping" / "metadata.json"

N_NEURAL  = 64   # first 64 channels → model input
N_LABEL   = 12   # last 12 channels  → one-hot label
N_TOTAL   = 76   # full channel count for recordings we care about
SAMPLE_HZ = 32_000

# Only load recordings that have the full 76-channel layout
RELEVANT_FILES = [
    "broadband_data_20260410_144548.h5",   # Run 003 – hard mode, all inputs
    "broadband_data_20260410_151848.h5",   # Run 004 – structured mapping
]


# ─── Step 1: Fetch data from manraj branch ────────────────────────────────────

def fetch_from_manraj():
    """
    Checkout HDF5 recordings and mapping metadata from the manraj branch.
    Falls back to copying from the local LFS object store if git-lfs smudge
    doesn't materialise the files automatically.
    """
    print("─── Fetching data from manraj branch ─────────────────────────────")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    MAPPING_DIR.mkdir(parents=True, exist_ok=True)

    # Checkout H5 files from manraj
    h5_paths = [f"data/recordings/{f}" for f in RELEVANT_FILES]
    _git("checkout", "manraj", "--", *h5_paths)

    # Checkout mappings subfolder if it exists on the branch
    mapping_exists = _git_path_exists("manraj", "data/recordings/mappings")
    if mapping_exists:
        _git("checkout", "manraj", "--", "data/recordings/mappings")
    else:
        print("  (no data/recordings/mappings on manraj – using data_collection metadata)")

    # Checkout the channel-mapping metadata JSON
    meta_path = "data_collection/run_004_structured_mapping/metadata.json"
    if _git_path_exists("manraj", meta_path):
        MAPPING_META.parent.mkdir(parents=True, exist_ok=True)
        _git("checkout", "manraj", "--", meta_path)

    # Ensure LFS pointers are replaced with real file contents
    _ensure_lfs(h5_paths)


def _git(*args):
    cmd = ["git"] + list(args)
    result = subprocess.run(cmd, cwd=REPO_ROOT, capture_output=True, text=True)
    if result.returncode != 0:
        # Non-fatal – print warning and continue
        print(f"  [git warning] {' '.join(args[1:3])}: {result.stderr.strip()[:120]}")
    return result.returncode == 0


def _git_path_exists(branch, path):
    r = subprocess.run(
        ["git", "ls-tree", "--name-only", branch, path],
        cwd=REPO_ROOT, capture_output=True, text=True,
    )
    return bool(r.stdout.strip())


def _ensure_lfs(relative_paths):
    """
    If any of the checked-out files are still LFS pointer stubs (< 1 kB),
    copy the real content from the local LFS object store.
    """
    lfs_store = REPO_ROOT.parent / ".git" / "lfs" / "objects"
    # Also check the worktree's own git dir
    worktree_git = REPO_ROOT / ".git"
    if worktree_git.is_file():                   # worktree .git is a file, not dir
        gitdir_text = worktree_git.read_text().strip()
        if gitdir_text.startswith("gitdir:"):
            worktree_gitdir = Path(gitdir_text.split(":", 1)[1].strip())
            lfs_store2 = worktree_gitdir / "lfs" / "objects"
        else:
            lfs_store2 = None
    else:
        lfs_store2 = worktree_git / "lfs" / "objects"

    for rel in relative_paths:
        dest = REPO_ROOT / rel
        if not dest.exists() or dest.stat().st_size < 1024:
            # File is an LFS pointer – read its OID
            try:
                text = dest.read_text(errors="replace")
            except Exception:
                print(f"  [lfs] cannot read {rel}, skipping"); continue

            oid = None
            for line in text.splitlines():
                if line.startswith("oid sha256:"):
                    oid = line.split(":", 1)[1].strip()
            if not oid:
                print(f"  [lfs] no OID in {rel}"); continue

            # Find the object in one of the LFS stores
            obj = None
            for store in [lfs_store, lfs_store2]:
                if store is None:
                    continue
                candidate = store / oid[:2] / oid[2:4] / oid
                if candidate.exists():
                    obj = candidate; break

            if obj:
                print(f"  [lfs] copying {oid[:12]}… → {rel}")
                import shutil; shutil.copy2(obj, dest)
            else:
                # Last resort: git lfs pull
                print(f"  [lfs] running git lfs pull for {rel}")
                subprocess.run(
                    ["git", "lfs", "pull", "--include", rel],
                    cwd=REPO_ROOT, capture_output=True,
                )


# ─── Step 2: Inspect HDF5 layout ─────────────────────────────────────────────

def find_data_array(f: h5py.File):
    """
    Recursively search an HDF5 file for a 2-D dataset shaped
    (n_channels, n_samples) or (n_samples, n_channels) with n_channels == N_TOTAL.
    Returns the array as (n_channels, n_samples).
    """
    found = {}

    def _visit(name, obj):
        if isinstance(obj, h5py.Dataset) and obj.ndim == 2:
            s = obj.shape
            if N_TOTAL in s:
                found[name] = s

    f.visititems(_visit)

    if not found:
        raise ValueError(
            f"No 2-D dataset with {N_TOTAL} channels found.\n"
            f"Datasets: {_list_datasets(f)}"
        )

    # Pick the dataset with the most samples (largest non-N_TOTAL dimension)
    best = max(found, key=lambda n: max(found[n]))
    shape = found[best]
    print(f"  Using dataset '{best}' shape={shape}")

    arr = f[best][:]
    # Normalise to (n_channels, n_samples)
    if arr.shape[0] == N_TOTAL:
        return arr.astype(np.float32)
    else:
        return arr.T.astype(np.float32)


def _list_datasets(f):
    names = []
    f.visititems(lambda n, o: names.append(n) if isinstance(o, h5py.Dataset) else None)
    return names


# ─── Step 3: Window + label ───────────────────────────────────────────────────

def make_windows(neural, label_raw, sample_hz, window_ms, stride_ms):
    """
    Segment continuous data into overlapping windows.

    Parameters
    ----------
    neural     : (64, n_samples) float32  – neural channels
    label_raw  : (12, n_samples) float32  – raw label channels
    window_ms  : window duration (ms)
    stride_ms  : hop between windows (ms)

    Returns
    -------
    X : (n_windows, window_samples, 64)   – raw neural time-series, no preprocessing
    y : (n_windows,)                       – integer class label 0-11
    """
    win  = int(window_ms  * sample_hz / 1000)
    hop  = int(stride_ms  * sample_hz / 1000)
    n_total = neural.shape[1]
    n_win = max(0, (n_total - win) // hop + 1)

    if n_win == 0:
        raise ValueError(
            f"Recording too short ({n_total} samples) for window={win} samples"
        )

    # Normalise each label channel to [0, 1] across the whole recording
    # so axes (±32767) and buttons (0/32767) are comparable
    lab_abs = np.abs(label_raw)
    col_max = lab_abs.max(axis=1, keepdims=True).clip(1)   # avoid /0
    lab_norm = lab_abs / col_max                            # (12, n_samples) in [0,1]

    X = np.empty((n_win, win, 64), dtype=np.float32)
    y = np.empty(n_win, dtype=np.int64)

    for i in range(n_win):
        s, e = i * hop, i * hop + win
        X[i] = neural[:, s:e].T             # (win, 64)
        # Label = argmax of mean normalised activity over window
        window_label_mean = lab_norm[:, s:e].mean(axis=1)   # (12,)
        y[i] = int(window_label_mean.argmax())

    return X, y


# ─── Step 4: Load all relevant recordings ────────────────────────────────────

def load_recordings(window_ms, stride_ms):
    all_X, all_y = [], []

    for fname in RELEVANT_FILES:
        fpath = DATA_DIR / fname
        if not fpath.exists():
            print(f"  [skip] {fname} not found"); continue
        if fpath.stat().st_size < 1024:
            print(f"  [skip] {fname} is still an LFS pointer"); continue

        print(f"\nLoading {fname}  ({fpath.stat().st_size / 1e6:.1f} MB) …")
        try:
            with h5py.File(fpath, "r") as f:
                sample_hz = int(f.attrs.get("sample_rate_hz", SAMPLE_HZ))
                arr = find_data_array(f)   # (76, n_samples)
        except Exception as e:
            print(f"  [error] {e}"); continue

        neural    = arr[:N_NEURAL]                   # (64, n_samples)
        label_raw = arr[N_NEURAL:N_NEURAL + N_LABEL] # (12, n_samples)
        print(f"  neural={neural.shape}  label={label_raw.shape}  hz={sample_hz}")

        X, y = make_windows(neural, label_raw, sample_hz, window_ms, stride_ms)
        counts = {int(c): int(n) for c, n in zip(*np.unique(y, return_counts=True))}
        print(f"  → {len(y)} windows  class distribution: {counts}")
        all_X.append(X); all_y.append(y)

    if not all_X:
        raise RuntimeError("No valid 76-channel recordings found.")

    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nCombined dataset: X={X.shape}  y={y.shape}")
    return X, y


# ─── Step 5: Train-test split + save ─────────────────────────────────────────

def save_split(X, y, seed=42):
    """
    Stratified 80/20 split.  Saves to data/processed/train.npz + test.npz,
    and a combined dataset.npz for compare_models.py.
    """
    # Drop any classes with fewer than 2 samples (can't stratify)
    classes, counts = np.unique(y, return_counts=True)
    valid_classes = classes[counts >= 2]
    mask = np.isin(y, valid_classes)
    X, y = X[mask], y[mask]

    # Re-map labels to 0..N-1 (in case some classes are missing entirely)
    label_map = {old: new for new, old in enumerate(np.unique(y))}
    y = np.array([label_map[c] for c in y], dtype=np.int64)
    n_classes = len(label_map)
    print(f"Classes present after filtering: {n_classes}")

    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=0.2, random_state=seed, stratify=y
    )
    print(f"Train: {len(y_tr)}  Test: {len(y_te)}")

    PROC_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(PROC_DIR / "train.npz", X=X_tr, y=y_tr)
    np.savez_compressed(PROC_DIR / "test.npz",  X=X_te, y=y_te)
    np.savez_compressed(PROC_DIR / "dataset.npz", X=X, y=y)
    print(f"Saved to {PROC_DIR}/")

    return str(PROC_DIR / "dataset.npz"), n_classes


# ─── Step 6: Run compare_models.py ───────────────────────────────────────────

def run_comparison(dataset_path, out_dir, epochs, batch_size):
    print("\n─── Running compare_models.py ────────────────────────────────────")
    script = REPO_ROOT / "scripts" / "compare_models.py"
    cmd = [
        sys.executable, str(script),
        "--data",       dataset_path,
        "--out-dir",    out_dir,
        "--epochs",     str(epochs),
        "--batch-size", str(batch_size),
    ]
    print(" ".join(cmd))
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    return result.returncode


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Fetch → prepare → compare models")
    parser.add_argument("--skip-fetch",  action="store_true",
                        help="Skip git fetch (data already present)")
    parser.add_argument("--window-ms",   type=int, default=50,
                        help="Window size in ms (default 50 → 1600 samples @ 32kHz)")
    parser.add_argument("--stride-ms",   type=int, default=25,
                        help="Hop size in ms (default 25 → 50%% overlap)")
    parser.add_argument("--out-dir",     type=str, default="models/attempt 2",
                        help="Directory for plots/results")
    parser.add_argument("--epochs",      type=int, default=50)
    parser.add_argument("--batch-size",  type=int, default=64)
    parser.add_argument("--seed",        type=int, default=42)
    args = parser.parse_args()

    t_total = time.perf_counter()

    # 1. Fetch
    if not args.skip_fetch:
        fetch_from_manraj()
    else:
        print("─── Skipping fetch (--skip-fetch) ────────────────────────────────")

    # 2-3. Load + window
    print("\n─── Loading and windowing recordings ─────────────────────────────")
    X, y = load_recordings(args.window_ms, args.stride_ms)

    # 4. Split + save
    print("\n─── Saving train/test split ───────────────────────────────────────")
    dataset_path, n_classes = save_split(X, y, args.seed)
    print(f"n_classes in data: {n_classes}  (model always outputs {12} logits)")

    # 5. Compare models
    rc = run_comparison(dataset_path, args.out_dir, args.epochs, args.batch_size)

    elapsed = time.perf_counter() - t_total
    print(f"\nTotal time: {elapsed:.1f}s  exit_code={rc}")
    sys.exit(rc)


if __name__ == "__main__":
    main()
