"""
utils.py -- Shared utilities for the BCI decoding pipeline.

Provides:
  - Config loading (YAML -> dataclass)
  - Reproducibility seeding
  - Logging helpers
  - Path management
  - Numpy / Torch array helpers
"""

from __future__ import annotations

import logging
import os
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import yaml

# -- Logging --------------------------------------------------------------------

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Return a consistently formatted logger."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(
            logging.Formatter("%(asctime)s [%(name)s] %(levelname)s: %(message)s",
                              datefmt="%H:%M:%S")
        )
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


log = get_logger("utils")


# -- Seeding --------------------------------------------------------------------

def set_seed(seed: int = 42) -> None:
    """
    Set random seeds for Python, NumPy, and (if installed) PyTorch.

    This ensures reproducible results for:
      - numpy random ops (data splitting, shuffling)
      - torch model weight initialization and dropout
      - sklearn (uses numpy under the hood)
    """
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Deterministic cuDNN (may slow GPU training)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    except ImportError:
        pass


# -- Config ---------------------------------------------------------------------

def load_config(path: str | Path = "config.yaml") -> Dict[str, Any]:
    """
    Load YAML config file and return as a nested dict.

    Parameters
    ----------
    path : Path to config.yaml (default: 'config.yaml' relative to cwd).

    Returns
    -------
    dict with all pipeline hyperparameters.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(path, "r") as fh:
        cfg = yaml.safe_load(fh)
    log.info(f"Loaded config from {path}")
    return cfg


def get_path(cfg: Dict[str, Any], *keys: str) -> Path:
    """
    Navigate nested config dict and return a resolved Path.

    Example
    -------
    get_path(cfg, "data", "preprocessed_dir")  ->  Path("data/preprocessed")
    """
    val = cfg
    for k in keys:
        val = val[k]
    return Path(val)


# -- File helpers ---------------------------------------------------------------

def ensure_dir(path: Path | str) -> Path:
    """Create directory (and parents) if it doesn't exist. Returns path."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def preprocessed_path(
    out_dir: Path,
    original_name: str,
    k: float,
) -> Path:
    """
    Construct the deterministic output path for a preprocessed file.

    Pattern: {out_dir}/{stem}_k_{k:.1f}.npz
    Example: data/preprocessed/broadband_data_20260410_144548_k_3.5.npz
    """
    stem = Path(original_name).stem
    return out_dir / f"{stem}_k_{k:.1f}.npz"


# -- Array helpers --------------------------------------------------------------

def train_test_split_temporal(
    X: np.ndarray,
    Y: np.ndarray,
    test_fraction: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Chronological (no-shuffle) 80/20 train-test split.

    Splitting at a fixed index preserves temporal ordering, which is critical
    for neural time-series -- shuffling would allow future information to leak
    into the training set.

    Parameters
    ----------
    X             : (n_bins, n_features)
    Y             : (n_bins, n_targets)
    test_fraction : fraction of samples reserved for testing (default 0.20)

    Returns
    -------
    X_train, X_test, Y_train, Y_test  -- all np.ndarray
    """
    n = len(X)
    split = int(n * (1.0 - test_fraction))
    return X[:split], X[split:], Y[:split], Y[split:]


def zscore_per_channel(
    data: np.ndarray,
    eps: float = 1e-8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Z-score normalize each channel independently.

    Parameters
    ----------
    data : (n_channels, n_samples) float32

    Returns
    -------
    z    : (n_channels, n_samples) float32 -- normalized signal
    mean : (n_channels,) float32
    std  : (n_channels,) float32
    """
    mean = data.mean(axis=-1, keepdims=True)
    std  = data.std(axis=-1, keepdims=True).clip(min=eps)
    z = ((data - mean) / std).astype(np.float32)
    return z, mean.squeeze(-1), std.squeeze(-1)


# -- Results I/O ----------------------------------------------------------------

def save_results(results: list[dict], out_path: Path) -> None:
    """Save list-of-dicts results to a .npz file."""
    import json
    out_path = Path(out_path)
    ensure_dir(out_path.parent)
    with open(out_path.with_suffix(".json"), "w") as fh:
        json.dump(results, fh, indent=2)
    log.info(f"Saved results -> {out_path.with_suffix('.json')}")
