"""
pipeline.py -- Top-level orchestrator: preprocess -> train -> evaluate -> report.

This module ties together all pipeline stages. It is invoked by run_pipeline.sh
but can also be run directly:

    python src/pipeline.py
    python src/pipeline.py --skip-preprocess
    python src/pipeline.py --model mlp xgboost --k 3.0 3.5
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_dir, get_logger, load_config, save_results, set_seed
from src.preprocess import run_preprocessing
from src.train import run_training
from src.evaluate import run_evaluation

log = get_logger("pipeline")


# ==============================================================================
# NumPy-aware JSON serialiser
# ==============================================================================

class _NumpyEncoder(json.JSONEncoder):
    """Allow json.dump to handle numpy scalars and arrays."""
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return super().default(obj)


# ==============================================================================
# Main pipeline
# ==============================================================================

def run_pipeline(
    cfg: dict,
    skip_preprocess: bool = False,
    model_names: Optional[List[str]] = None,
) -> None:
    """
    Execute the full BCI decoding pipeline.

    Stages
    ------
    1. Preprocessing  -- load .h5 files, bin + threshold, save .npz
    2. Training       -- fit 4 models across all k values
    3. Evaluation     -- compute metrics, generate plots, write reports

    Parameters
    ----------
    cfg             : Loaded config dict.
    skip_preprocess : If True, assume .npz files already exist.
    model_names     : Optional subset of models to train.
    """
    seed = cfg["split"]["random_seed"]
    set_seed(seed)

    # -- Stage 1: Preprocessing ------------------------------------------------
    if not skip_preprocess:
        log.info("\n" + "=" * 60)
        log.info("STAGE 1 -- PREPROCESSING")
        log.info("=" * 60)
        run_preprocessing(cfg)
    else:
        log.info("STAGE 1 -- Skipped (skip_preprocess=True)")

    # -- Stage 2: Training -----------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("STAGE 2 -- TRAINING")
    log.info("=" * 60)
    results = run_training(cfg, model_names=model_names)

    # Persist results (strip non-serialisable arrays for JSON, keep for plots)
    results_path = Path(cfg["data"].get("reports_dir", "reports")) / "results.json"
    ensure_dir(results_path.parent)

    # Serialise: keep Y_test/Y_pred_test in memory for plots but save scalars
    serialisable = {}
    for mname, k_dict in results.items():
        serialisable[mname] = {}
        for k, res in k_dict.items():
            serialisable[mname][str(k)] = {
                "train_r2":             res["train_r2"],
                "test_r2":              res["test_r2"],
                "per_channel_r2_train": res["per_channel_r2_train"],
                "per_channel_r2_test":  res["per_channel_r2_test"],
            }

    with open(results_path, "w") as fh:
        json.dump(serialisable, fh, indent=2, cls=_NumpyEncoder)
    log.info(f"Results saved -> {results_path}")

    # -- Stage 3: Evaluation + Reports ----------------------------------------
    log.info("\n" + "=" * 60)
    log.info("STAGE 3 -- EVALUATION & REPORTS")
    log.info("=" * 60)
    run_evaluation(cfg, results)

    # -- Summary ---------------------------------------------------------------
    _print_summary(results)


def _print_summary(results: dict) -> None:
    """Print a clean final summary table."""
    print("\n" + "=" * 70)
    print("FINAL RESULTS SUMMARY")
    print("=" * 70)
    print(f"{'Model':<12} {'k':>5}  {'Train R2':>10}  {'Test R2':>10}  {'Best':>6}")
    print("-" * 70)

    best_overall = ("", 0.0, -1.0)
    for mname, k_results in results.items():
        if not k_results:
            continue
        best_k = max(k_results, key=lambda k: k_results[k]["test_r2"])
        for k in sorted(k_results.keys()):
            r    = k_results[k]
            star = "" if k == best_k else ""
            print(
                f"{mname:<12} {k:>5.1f}  "
                f"{r['train_r2']:>10.4f}  {r['test_r2']:>10.4f}  {star:>6}"
            )
            if r["test_r2"] > best_overall[2]:
                best_overall = (mname, k, r["test_r2"])

    print("=" * 70)
    print(
        f"\nBest overall: {best_overall[0].upper()}  k={best_overall[1]:.1f}"
        f"  Test R2={best_overall[2]:.4f}"
    )
    print("Reports: reports/")
    print("Figures: reports/figures/")
    print("Models:  models/trained/")


# ==============================================================================
# CLI
# ==============================================================================

def _parse_args():
    p = argparse.ArgumentParser(
        description="Run the full BCI decoding pipeline: preprocess -> train -> report."
    )
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--skip-preprocess", action="store_true",
                   help="Skip preprocessing (use existing .npz files).")
    p.add_argument("--model", nargs="+", default=None,
                   choices=["mlp", "xgboost", "cnn1d", "svm"])
    p.add_argument("--k", type=float, nargs="+", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    if args.k:
        cfg["preprocessing"]["k_values"] = args.k

    run_pipeline(
        cfg,
        skip_preprocess=args.skip_preprocess,
        model_names=args.model,
    )
