"""
evaluate.py -- Evaluation metrics, visualisation, and Markdown report generation.

For each model x k combination, this module:
  1. Computes per-channel R2 and mean R2
  2. Generates four plot types (saved to reports/figures/):
       - r2_vs_k line plot
       - per_channel R2 bar chart
       - pred_vs_actual scatter for channel 0
       - residual plot for channel 0
  3. Writes a Markdown report to reports/{model_name}.md

Usage
-----
    python src/evaluate.py                    # evaluate all results
    python src/evaluate.py --model mlp        # single model
    python src/evaluate.py --config config.yaml --results results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from textwrap import dedent
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")   # headless rendering
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import ensure_dir, get_logger, load_config

log = get_logger("evaluate")

MODEL_DESCRIPTIONS = {
    "mlp": (
        "Multi-Layer Perceptron (2 hidden layers, 256 units, ReLU, Adam optimizer)",
        "Chosen as a universal approximator for non-linear mapping from spike-count "
        "features to continuous controller outputs. Simple to train and easily "
        "exportable to ONNX for on-device Synapse App inference."
    ),
    "xgboost": (
        "XGBoost Gradient-Boosted Trees (MultiOutputRegressor, 100 estimators)",
        "Strong classical baseline for tabular spike-count data. Handles non-linear "
        "feature interactions without explicit engineering. One independent ensemble "
        "is trained per output channel."
    ),
    "cnn1d": (
        "1-D Convolutional Neural Network (2 Conv1d layers, 64 filters, kernel=3)",
        "Captures local spatial correlations between adjacent electrodes. The conv "
        "layers learn channel-interaction patterns that a pure MLP might miss, "
        "while remaining lightweight for real-time inference."
    ),
    "svm": (
        "Support Vector Regression with RBF kernel (MultiOutputRegressor(SVR))",
        "Classical kernel method providing a smooth non-linear baseline. Works well "
        "on bounded, normalized spike-count features. Included as a comparison point "
        "against gradient-based methods."
    ),
}


# ==============================================================================
# Main entry point
# ==============================================================================

def run_evaluation(cfg: dict, results: Dict) -> None:
    """
    Generate all plots and Markdown reports for every model in results.

    Parameters
    ----------
    cfg     : Loaded config dict.
    results : Output of train.run_training() -- nested dict
              { model_name: { k: { train_r2, test_r2, per_channel_r2_train,
                                   per_channel_r2_test, Y_test, Y_pred_test } } }
    """
    fig_dir  = ensure_dir(Path(cfg["data"]["figures_dir"]))
    rep_dir  = ensure_dir(Path(cfg["data"]["reports_dir"]))
    eval_cfg = cfg["evaluation"]
    sample_ch = eval_cfg.get("plot_sample_channel", 0)
    dpi       = eval_cfg.get("fig_dpi", 120)

    for model_name, k_results in results.items():
        if not k_results:
            continue
        log.info(f"\nGenerating report for {model_name.upper()}")

        plot_paths = {}

        # -- 1. R2 vs k line plot ---------------------------------------------
        p = _plot_r2_vs_k(model_name, k_results, fig_dir, dpi)
        plot_paths["r2_vs_k"] = p

        for k, res in k_results.items():
            # -- 2. Per-channel R2 bar chart ----------------------------------
            p = _plot_per_channel_r2(model_name, k, res, fig_dir, dpi)
            plot_paths[f"per_channel_k{k:.1f}"] = p

            # -- 3. Predicted vs actual ---------------------------------------
            if "Y_test" in res and "Y_pred_test" in res:
                p = _plot_pred_vs_actual(
                    model_name, k, res, sample_ch, fig_dir, dpi
                )
                plot_paths[f"pred_vs_actual_k{k:.1f}"] = p

                # -- 4. Residuals ---------------------------------------------
                p = _plot_residuals(
                    model_name, k, res, sample_ch, fig_dir, dpi
                )
                plot_paths[f"residuals_k{k:.1f}"] = p

        # -- Markdown report --------------------------------------------------
        _write_report(model_name, k_results, plot_paths, rep_dir, cfg)
        log.info(f"  Report -> reports/{model_name}.md")


# ==============================================================================
# Plots
# ==============================================================================

def _plot_r2_vs_k(
    model_name: str,
    k_results: Dict,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Line plot: mean Train R2 and Test R2 vs k value."""
    k_vals  = sorted(k_results.keys())
    tr_r2   = [k_results[k]["train_r2"] for k in k_vals]
    te_r2   = [k_results[k]["test_r2"]  for k in k_vals]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(k_vals, tr_r2, "o-", color="#2196F3", linewidth=2, label="Train R2")
    ax.plot(k_vals, te_r2, "s--", color="#F44336", linewidth=2, label="Test R2")
    ax.axhline(0, color="gray", linewidth=0.8, linestyle=":")
    ax.set_xlabel("Threshold k", fontsize=12)
    ax.set_ylabel("Mean R2 (12 channels)", fontsize=12)
    ax.set_title(f"{model_name.upper()} -- R2 vs Spike Threshold k", fontsize=13)
    ax.legend(fontsize=11)
    ax.set_xticks(k_vals)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = fig_dir / f"{model_name}_r2_vs_k.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _plot_per_channel_r2(
    model_name: str,
    k: float,
    res: Dict,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Horizontal bar chart: per-channel Test R2 for one k."""
    r2_vals  = np.array(res["per_channel_r2_test"])
    n_ch     = len(r2_vals)
    channels = [f"Ch {i}" for i in range(n_ch)]
    colors   = ["#4CAF50" if v >= 0 else "#F44336" for v in r2_vals]

    fig, ax = plt.subplots(figsize=(7, max(4, n_ch * 0.35)))
    y_pos = np.arange(n_ch)
    ax.barh(y_pos, r2_vals, color=colors, edgecolor="white", linewidth=0.5)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(channels, fontsize=8)
    ax.set_xlabel("Test R2", fontsize=11)
    ax.set_title(f"{model_name.upper()} -- Per-Channel R2  (k={k:.1f})", fontsize=12)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()

    out = fig_dir / f"{model_name}_per_channel_k{k:.1f}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _plot_pred_vs_actual(
    model_name: str,
    k: float,
    res: Dict,
    channel: int,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Scatter plot: predicted vs actual for one output channel."""
    Y_true = np.array(res["Y_test"])[:, channel]
    Y_pred = np.array(res["Y_pred_test"])[:, channel]

    # Subsample for legibility
    n_plot = min(2000, len(Y_true))
    idx    = np.linspace(0, len(Y_true) - 1, n_plot, dtype=int)
    yt, yp = Y_true[idx], Y_pred[idx]

    lim = max(abs(yt).max(), abs(yp).max()) * 1.05

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(yt, yp, s=4, alpha=0.4, color="#1565C0", rasterized=True)
    ax.plot([-lim, lim], [-lim, lim], "r--", linewidth=1.5, label="y = x")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_xlabel("Actual", fontsize=11)
    ax.set_ylabel("Predicted", fontsize=11)
    ax.set_title(
        f"{model_name.upper()} -- Pred vs Actual\n"
        f"Ch {channel}  (k={k:.1f})  R2={res['per_channel_r2_test'][channel]:.3f}",
        fontsize=11,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out = fig_dir / f"{model_name}_pred_vs_actual_k{k:.1f}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


def _plot_residuals(
    model_name: str,
    k: float,
    res: Dict,
    channel: int,
    fig_dir: Path,
    dpi: int,
) -> Path:
    """Residual histogram for one output channel."""
    Y_true = np.array(res["Y_test"])[:, channel]
    Y_pred = np.array(res["Y_pred_test"])[:, channel]
    resid  = Y_true - Y_pred

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))

    # Time-series residuals
    axes[0].plot(resid[:500], color="#7B1FA2", linewidth=0.8, alpha=0.9)
    axes[0].axhline(0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_xlabel("Bin index (first 500)", fontsize=10)
    axes[0].set_ylabel("Residual", fontsize=10)
    axes[0].set_title("Residuals over time", fontsize=11)
    axes[0].grid(True, alpha=0.3)

    # Residual histogram
    axes[1].hist(resid, bins=60, color="#7B1FA2", edgecolor="white", linewidth=0.4)
    axes[1].axvline(0, color="red", linewidth=1.5, linestyle="--")
    axes[1].set_xlabel("Residual", fontsize=10)
    axes[1].set_ylabel("Count", fontsize=10)
    axes[1].set_title(
        f"Residual distribution  u={resid.mean():.1f}  ={resid.std():.1f}",
        fontsize=11,
    )
    axes[1].grid(True, alpha=0.3)

    fig.suptitle(
        f"{model_name.upper()} Residuals -- Ch {channel}  k={k:.1f}", fontsize=12
    )
    fig.tight_layout()

    out = fig_dir / f"{model_name}_residuals_k{k:.1f}.png"
    fig.savefig(out, dpi=dpi)
    plt.close(fig)
    return out


# ==============================================================================
# Markdown report
# ==============================================================================

def _write_report(
    model_name: str,
    k_results: Dict,
    plot_paths: Dict[str, Path],
    rep_dir: Path,
    cfg: Dict,
) -> None:
    """Write a fully-structured Markdown report for one model."""
    k_vals  = sorted(k_results.keys())
    desc, rationale = MODEL_DESCRIPTIONS.get(
        model_name, (model_name, "No description available.")
    )

    best_k  = max(k_vals, key=lambda k: k_results[k]["test_r2"])
    best_r2 = k_results[best_k]["test_r2"]

    # Results table
    table_rows = []
    for k in k_vals:
        r = k_results[k]
        table_rows.append(
            f"| {k:.1f} | {r['train_r2']:.4f} | {r['test_r2']:.4f} |"
        )

    # Per-channel R2 for best k
    best_pc = k_results[best_k]["per_channel_r2_test"]
    ch_rows = "\n".join(
        f"| Ch {i} | {v:.4f} |" for i, v in enumerate(best_pc)
    )

    # Build plot embed section
    def _embed(key: str, caption: str, alt: str = "") -> str:
        p = plot_paths.get(key)
        if p is None:
            return f"*{caption} -- not generated*"
        rel = Path("figures") / p.name
        return f"![{alt or caption}]({rel})\n*{caption}*"

    # Overfitting analysis
    gap_by_k = {k: k_results[k]["train_r2"] - k_results[k]["test_r2"] for k in k_vals}
    worst_gap_k = max(gap_by_k, key=lambda k: gap_by_k[k])
    worst_gap   = gap_by_k[worst_gap_k]

    overfit_discussion = (
        f"The largest train-test gap is **{worst_gap:.4f}** at k={worst_gap_k:.1f}. "
    )
    if worst_gap > 0.15:
        overfit_discussion += (
            "This suggests meaningful overfitting -- the model has memorised "
            "training patterns that don't generalise. Consider adding dropout, "
            "reducing model capacity, or collecting more data."
        )
    elif worst_gap > 0.05:
        overfit_discussion += (
            "A moderate gap indicates mild overfitting. Regularisation or "
            "early stopping could close this gap."
        )
    else:
        overfit_discussion += (
            "The model generalises well; train/test performance is closely matched."
        )

    # Trend analysis
    trend_direction = "increases" if k_results[k_vals[-1]]["test_r2"] > k_results[k_vals[0]]["test_r2"] else "decreases"

    md = dedent(f"""\
    # Model: {model_name.upper()}

    ## Overview

    **Description:** {desc}

    **Why chosen:** {rationale}

    ---

    ## Data

    ### Preprocessing Summary

    | Parameter | Value |
    |-----------|-------|
    | Sampling rate | 32,000 Hz |
    | Neural channels | 64 (channels 0-63) |
    | Target channels | 12 (channels 64-75) |
    | Bin size | 20 ms = 640 samples |
    | Feature type | Spike count (\\|z\\| > k) |
    | Target aggregation | Mean per bin |
    | Train/test split | 80/20 chronological |

    ### k Values Tested

    k  {{ {', '.join(str(k) for k in k_vals)} }}

    ---

    ## Results

    ### R2 Summary Table

    | k | Train R2 | Test R2 |
    |---|---------|--------|
    {chr(10).join(table_rows)}

    **Best k:** `{best_k:.1f}` -> Test R2 = **{best_r2:.4f}**

    ### Per-Channel R2 Breakdown (k = {best_k:.1f})

    | Channel | Test R2 |
    |---------|---------|
    {ch_rows}

    ---

    ## Plots

    ### R2 vs Threshold k

    {_embed("r2_vs_k", f"Mean train and test R2 across k values for {model_name.upper()}", "R2 vs k")}

    ### Per-Channel R2 (Best k = {best_k:.1f})

    {_embed(f"per_channel_k{best_k:.1f}", f"Per-channel test R2 for k={best_k:.1f}", "Per-channel R2")}

    ### Predicted vs Actual -- Channel 0 (Best k = {best_k:.1f})

    {_embed(f"pred_vs_actual_k{best_k:.1f}", f"Predicted vs actual for channel 0, k={best_k:.1f}", "Pred vs Actual")}

    ### Residuals -- Channel 0 (Best k = {best_k:.1f})

    {_embed(f"residuals_k{best_k:.1f}", f"Residual distribution for channel 0, k={best_k:.1f}", "Residuals")}

    ---

    ## Analysis

    ### Performance Trend vs k

    Test R2 **{trend_direction}** as k increases from {k_vals[0]} to {k_vals[-1]}.
    At low k, the threshold is loose -- many sub-threshold noise fluctuations are
    counted as spikes, inflating feature values and adding noise.
    At high k, only strong deflections are counted -- fewer features, but higher
    SNR. The optimal k for this model is **{best_k:.1f}**.

    ### Overfitting / Underfitting

    {overfit_discussion}

    ### Strengths

    {_model_strengths(model_name)}

    ### Weaknesses

    {_model_weaknesses(model_name)}

    ---

    ## Conclusion

    {_model_conclusion(model_name, best_k, best_r2, k_results)}
    """)

    out = rep_dir / f"{model_name}.md"
    out.write_text(md, encoding="utf-8")


# -- Report text helpers --------------------------------------------------------

def _model_strengths(name: str) -> str:
    return {
        "mlp":      "Fast training and inference. Simple architecture makes ONNX export straightforward. Handles non-linear channel interactions.",
        "xgboost":  "Excellent on tabular spike-count data. Robust to outlier counts. No normalisation of inputs required. Interpretable via feature importance.",
        "cnn1d":    "Captures local spatial correlations between adjacent electrode channels. Weight sharing reduces overfitting risk on limited data.",
        "svm":      "Theoretically well-motivated with RBF kernel. No gradient issues. Works well with a small number of samples.",
    }.get(name, "N/A")


def _model_weaknesses(name: str) -> str:
    return {
        "mlp":      "Ignores temporal order across bins. Can overfit with limited data. Sensitive to learning rate.",
        "xgboost":  "Trains independent regressors per channel -- misses cross-channel correlations. Slow with large datasets.",
        "cnn1d":    "Assumes channel ordering carries spatial meaning -- not guaranteed for arbitrary electrode placement.",
        "svm":      "O(n2) training complexity; impractical on >10K samples without subsampling. RBF bandwidth requires careful tuning.",
    }.get(name, "N/A")


def _model_conclusion(
    name: str, best_k: float, best_r2: float, k_results: Dict
) -> str:
    perf_label = (
        "strong" if best_r2 > 0.5 else
        "moderate" if best_r2 > 0.2 else
        "weak"
    )
    return (
        f"**{name.upper()}** achieves {perf_label} decoding performance with a "
        f"best test R2 of **{best_r2:.4f}** at k={best_k:.1f}. "
        f"We recommend using **k={best_k:.1f}** for this model in the final BCI pipeline. "
        f"{'For deployment in the Synapse App, this model is a good candidate due to its speed and ONNX compatibility.' if name in ('mlp', 'cnn1d') else 'This model is best suited for offline analysis rather than real-time on-device inference.'}"
    )


# ==============================================================================
# CLI
# ==============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Evaluate trained models and generate reports.")
    p.add_argument("--config",  default="config.yaml")
    p.add_argument("--results", default="results.json",
                   help="JSON results file produced by train.py")
    p.add_argument("--model",   nargs="+", default=None)
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    results_path = Path(args.results)
    if not results_path.exists():
        log.error(f"Results file not found: {results_path}")
        sys.exit(1)

    with open(results_path) as fh:
        results = json.load(fh)

    if args.model:
        results = {k: v for k, v in results.items() if k in args.model}

    run_evaluation(cfg, results)
