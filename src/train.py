"""
train.py -- Multi-model training for BCI controller decoding.

Models
------
MLP       : 2-layer fully-connected network, PyTorch, 50 epochs, Adam
XGBoost   : MultiOutputRegressor wrapping XGBRegressor, 100 trees
1D CNN    : Conv1d feature extractor + FC head, PyTorch, 50 epochs, Adam
SVM       : MultiOutputRegressor wrapping SVR(rbf), optional subsampling

All models:
  - Input  X: (num_bins, 64)  -- spike counts per channel
  - Output Y: (num_bins, 12)  -- controller signals (regression)
  - Split  : chronological 80/20 (no temporal leakage)
  - Seed   : fixed for reproducibility

Usage
-----
    python src/train.py                          # train all models, all k
    python src/train.py --model mlp --k 3.5     # single model + k
    python src/train.py --config config.yaml
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.utils import (
    ensure_dir, get_logger, load_config, preprocessed_path,
    set_seed, train_test_split_temporal,
)

log = get_logger("train")

# ==============================================================================
# Public entry point
# ==============================================================================

def run_training(cfg: dict, model_names: Optional[List[str]] = None) -> Dict:
    """
    Train all enabled models across all k values.

    Parameters
    ----------
    cfg          : Loaded config dict.
    model_names  : Optional list to restrict which models are trained.

    Returns
    -------
    results : nested dict  { model_name: { k: { "train_r2", "test_r2",
                              "per_channel_r2_train", "per_channel_r2_test" } } }
    """
    data_cfg  = cfg["data"]
    split_cfg = cfg["split"]
    pre_cfg   = cfg["preprocessing"]
    mdl_cfg   = cfg["models"]

    preprocessed_dir = Path(data_cfg["preprocessed_dir"])
    models_dir = ensure_dir(Path(data_cfg.get("models_dir", "models/trained")))
    k_values   = pre_cfg["k_values"]
    seed       = split_cfg["random_seed"]
    test_frac  = split_cfg["test_fraction"]

    enabled = {
        name for name in ("mlp", "xgboost", "cnn1d", "svm")
        if mdl_cfg.get(name, {}).get("enabled", True)
    }
    if model_names:
        enabled &= set(model_names)

    results: Dict = {m: {} for m in enabled}

    for k in k_values:
        log.info(f"\n{'='*60}")
        log.info(f"Loading data for k={k}")
        X, Y = _load_all_preprocessed(preprocessed_dir, k)
        if X is None:
            log.warning(f"No preprocessed data found for k={k}, skipping.")
            continue

        log.info(f"  X={X.shape}  Y={Y.shape}")
        X_train, X_test, Y_train, Y_test = train_test_split_temporal(
            X, Y, test_fraction=test_frac
        )
        log.info(f"  Train: {X_train.shape}  Test: {X_test.shape}")

        set_seed(seed)

        for model_name in sorted(enabled):
            log.info(f"\n-- Training {model_name.upper()} (k={k}) --")
            m_cfg = mdl_cfg[model_name]

            try:
                model, preds = _dispatch_train(
                    model_name, m_cfg,
                    X_train, Y_train, X_test, Y_test,
                    seed=seed,
                )
            except Exception as exc:
                log.error(f"  {model_name} FAILED: {exc}")
                import traceback; traceback.print_exc()
                continue

            Y_pred_train, Y_pred_test = preds
            r2_train_pc = _per_channel_r2(Y_train, Y_pred_train)
            r2_test_pc  = _per_channel_r2(Y_test,  Y_pred_test)

            entry = {
                "train_r2":             float(np.mean(r2_train_pc)),
                "test_r2":              float(np.mean(r2_test_pc)),
                "per_channel_r2_train": r2_train_pc.tolist(),
                "per_channel_r2_test":  r2_test_pc.tolist(),
                "Y_test":               Y_test,
                "Y_pred_test":          Y_pred_test,
            }
            results[model_name][k] = entry

            log.info(
                f"  Train R2={entry['train_r2']:.4f}  "
                f"Test R2={entry['test_r2']:.4f}"
            )

            # Persist model weights
            _save_model(model, model_name, k, models_dir)

    return results


# ==============================================================================
# Model implementations
# ==============================================================================

# -- MLP -----------------------------------------------------------------------

class MLP:
    """
    2-layer fully-connected network for multi-output regression.

    Architecture: Input(64) -> Linear(256) -> ReLU -> Linear(256) -> ReLU -> Linear(12)

    Chosen because:
      - Simple baseline that handles non-linear channel interactions
      - Fast to train on spike-count features (no temporal modelling needed)
      - Easy to export to ONNX for on-device Synapse App inference
    """

    def __init__(self, cfg: dict, n_in: int = 64, n_out: int = 12):
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        h = cfg["hidden_size"]
        self.net = nn.Sequential(
            nn.Linear(n_in, h), nn.ReLU(),
            nn.Linear(h, h),   nn.ReLU(),
            nn.Linear(h, n_out),
        ).to(self.device)
        self.cfg = cfg

    def fit(self, X_tr: np.ndarray, Y_tr: np.ndarray) -> "MLP":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        epochs = cfg["epochs"]
        lr     = cfg["learning_rate"]
        bs     = cfg["batch_size"]

        Xt = torch.tensor(X_tr, dtype=torch.float32).to(self.device)
        Yt = torch.tensor(Y_tr, dtype=torch.float32).to(self.device)
        ds = TensorDataset(Xt, Yt)
        dl = DataLoader(ds, batch_size=bs, shuffle=False)  # preserve time order

        opt = torch.optim.Adam(self.net.parameters(), lr=lr)
        loss_fn = torch.nn.MSELoss()

        self.net.train()
        for epoch in range(epochs):
            total_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                pred = self.net(xb)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                log.info(f"    [MLP] epoch {epoch+1}/{epochs}  loss={total_loss/len(dl):.5f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.net.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).to(self.device)
            return self.net(xt).cpu().numpy()


# -- 1D CNN --------------------------------------------------------------------

class CNN1D:
    """
    1-D convolutional network that treats the 64 neural channels as a sequence.

    Architecture:
      Input: (batch, 1, 64)  -- 64 channels as 1-D signal
      Conv1d(1,  64, k=3, pad=1) -> ReLU -> Conv1d(64, 64, k=3, pad=1) -> ReLU
      -> AdaptiveAvgPool1d(1) -> Flatten -> Linear(64, 256) -> ReLU -> Linear(256, 12)

    Rationale:
      Nearby electrode channels often share spatial correlations.
      A small conv kernel captures local multi-channel patterns without
      assuming global channel ordering, while remaining lightweight.
    """

    def __init__(self, cfg: dict, n_in: int = 64, n_out: int = 12):
        import torch
        import torch.nn as nn

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        nf = cfg["n_filters"]
        ks = cfg["kernel_size"]
        pad = ks // 2

        layers = []
        in_ch = 1
        for i in range(cfg["n_conv_layers"]):
            out_ch = nf
            layers += [nn.Conv1d(in_ch, out_ch, kernel_size=ks, padding=pad), nn.ReLU()]
            if cfg.get("dropout", 0) > 0:
                layers.append(nn.Dropout(cfg["dropout"]))
            in_ch = out_ch

        layers += [nn.AdaptiveAvgPool1d(1)]
        self.conv = nn.Sequential(*layers).to(self.device)
        self.head = nn.Sequential(
            nn.Linear(nf, 256), nn.ReLU(),
            nn.Linear(256, n_out),
        ).to(self.device)
        self.cfg = cfg

    def fit(self, X_tr: np.ndarray, Y_tr: np.ndarray) -> "CNN1D":
        import torch
        from torch.utils.data import DataLoader, TensorDataset

        cfg = self.cfg
        Xt = torch.tensor(X_tr, dtype=torch.float32).unsqueeze(1).to(self.device)
        Yt = torch.tensor(Y_tr, dtype=torch.float32).to(self.device)
        dl = DataLoader(TensorDataset(Xt, Yt), batch_size=cfg["batch_size"], shuffle=False)

        opt = torch.optim.Adam(
            list(self.conv.parameters()) + list(self.head.parameters()),
            lr=cfg["learning_rate"],
        )
        loss_fn = torch.nn.MSELoss()

        for epoch in range(cfg["epochs"]):
            total_loss = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                feat = self.conv(xb).squeeze(-1)
                pred = self.head(feat)
                loss = loss_fn(pred, yb)
                loss.backward()
                opt.step()
                total_loss += loss.item()
            if (epoch + 1) % 10 == 0:
                log.info(f"    [CNN1D] epoch {epoch+1}/{cfg['epochs']}  loss={total_loss/len(dl):.5f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        import torch
        self.conv.eval(); self.head.eval()
        with torch.no_grad():
            xt = torch.tensor(X, dtype=torch.float32).unsqueeze(1).to(self.device)
            feat = self.conv(xt).squeeze(-1)
            return self.head(feat).cpu().numpy()


# -- XGBoost -------------------------------------------------------------------

def build_xgboost(cfg: dict, n_out: int = 12):
    """
    MultiOutputRegressor wrapping XGBRegressor.

    XGBoost is a strong non-linear baseline for tabular spike-count data:
      - Handles non-linearities and channel interactions without feature engineering
      - Robust to outlier spike counts
      - Fast inference, easy ONNX conversion

    MultiOutputRegressor trains one independent tree ensemble per target channel,
    which is appropriate here because the 12 controller signals are semi-independent.
    """
    from sklearn.multioutput import MultiOutputRegressor
    from xgboost import XGBRegressor

    base = XGBRegressor(
        n_estimators=cfg["n_estimators"],
        max_depth=cfg["max_depth"],
        learning_rate=cfg["learning_rate"],
        subsample=cfg["subsample"],
        colsample_bytree=cfg["colsample_bytree"],
        n_jobs=cfg["n_jobs"],
        random_state=cfg["random_seed"],
        verbosity=0,
        tree_method="hist",
    )
    return MultiOutputRegressor(base, n_jobs=1)


# -- SVM -----------------------------------------------------------------------

def build_svm(cfg: dict, n_out: int = 12):
    """
    MultiOutputRegressor wrapping SVR(rbf).

    SVM with RBF kernel provides a smooth nonlinear mapping.
    It works well on normalized, bounded features (spike counts).
    Included as a classical ML comparison point.

    Note: SVR is O(n2) in training time; max_train_samples caps this.
    """
    from sklearn.multioutput import MultiOutputRegressor
    from sklearn.svm import SVR

    base = SVR(kernel=cfg["kernel"], C=cfg["C"], epsilon=cfg["epsilon"])
    return MultiOutputRegressor(base, n_jobs=-1)


# ==============================================================================
# Dispatch & helpers
# ==============================================================================

def _dispatch_train(
    model_name: str,
    m_cfg: dict,
    X_train: np.ndarray, Y_train: np.ndarray,
    X_test:  np.ndarray, Y_test:  np.ndarray,
    seed: int = 42,
) -> Tuple[object, Tuple[np.ndarray, np.ndarray]]:
    """Train one model and return (fitted_model, (Y_pred_train, Y_pred_test))."""

    set_seed(seed)

    if model_name == "mlp":
        model = MLP(m_cfg, n_in=X_train.shape[1], n_out=Y_train.shape[1])
        model.fit(X_train, Y_train)
        preds = (model.predict(X_train), model.predict(X_test))

    elif model_name == "cnn1d":
        model = CNN1D(m_cfg, n_in=X_train.shape[1], n_out=Y_train.shape[1])
        model.fit(X_train, Y_train)
        preds = (model.predict(X_train), model.predict(X_test))

    elif model_name == "xgboost":
        model = build_xgboost(m_cfg, n_out=Y_train.shape[1])
        model.fit(X_train, Y_train)
        preds = (model.predict(X_train), model.predict(X_test))

    elif model_name == "svm":
        max_n = m_cfg.get("max_train_samples")
        Xtr = X_train[:max_n] if max_n else X_train
        Ytr = Y_train[:max_n] if max_n else Y_train
        model = build_svm(m_cfg, n_out=Y_train.shape[1])
        log.info(f"  SVM fitting on {len(Xtr)} samples (may take a minute)")
        model.fit(Xtr, Ytr)
        preds = (model.predict(X_train), model.predict(X_test))

    else:
        raise ValueError(f"Unknown model: {model_name!r}")

    return model, preds


def _per_channel_r2(Y_true: np.ndarray, Y_pred: np.ndarray) -> np.ndarray:
    """
    Compute R2 score per output channel.

    Uses the standard definition:  R2 = 1 - SS_res / SS_tot
    Channels where the target has zero variance get R2 = 0 (not NaN).

    Returns
    -------
    r2 : (n_outputs,) float64
    """
    ss_res = ((Y_true - Y_pred) ** 2).sum(axis=0)
    ss_tot = ((Y_true - Y_true.mean(axis=0)) ** 2).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        r2 = np.where(ss_tot > 0, 1 - ss_res / ss_tot, 0.0)
    return r2.astype(np.float64)


def _load_all_preprocessed(
    preprocessed_dir: Path,
    k: float,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Load and concatenate all .npz files matching k into (X, Y).

    Files are concatenated in sorted order to maintain temporal consistency
    across multi-session training.
    """
    pattern = f"*_k_{k:.1f}.npz"
    files = sorted(preprocessed_dir.glob(pattern))

    if not files:
        return None, None

    X_parts, Y_parts = [], []
    for fp in files:
        d = np.load(fp)
        X_parts.append(d["X"])
        Y_parts.append(d["Y"])
        log.info(f"  Loaded {fp.name}  X={d['X'].shape}")

    return np.concatenate(X_parts, axis=0), np.concatenate(Y_parts, axis=0)


def _save_model(model: object, model_name: str, k: float, models_dir: Path) -> None:
    """Persist a trained model to disk."""
    import pickle
    import torch

    out = models_dir / f"{model_name}_k{k:.1f}"

    if isinstance(model, (MLP, CNN1D)):
        # Save PyTorch state dict
        out = out.with_suffix(".pt")
        if isinstance(model, MLP):
            torch.save(model.net.state_dict(), out)
        else:
            torch.save({"conv": model.conv.state_dict(),
                        "head": model.head.state_dict()}, out)
    else:
        # Sklearn / XGBoost -> pickle
        out = out.with_suffix(".pkl")
        with open(out, "wb") as f:
            pickle.dump(model, f)

    log.info(f"  Saved model -> {out.name}")


# ==============================================================================
# CLI
# ==============================================================================

def _parse_args():
    p = argparse.ArgumentParser(description="Train BCI decoder models.")
    p.add_argument("--config", default="config.yaml")
    p.add_argument("--model",  nargs="+", default=None,
                   choices=["mlp", "xgboost", "cnn1d", "svm"],
                   help="Restrict to specific model(s).")
    p.add_argument("--k", type=float, nargs="+", default=None,
                   help="Override k values.")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    cfg  = load_config(args.config)

    if args.k:
        cfg["preprocessing"]["k_values"] = args.k

    set_seed(cfg["split"]["random_seed"])
    results = run_training(cfg, model_names=args.model)

    # Quick summary table
    print("\n" + "=" * 70)
    print(f"{'Model':<12} {'k':>5}  {'Train R2':>10}  {'Test R2':>10}")
    print("-" * 70)
    for mname, k_results in results.items():
        for k, res in sorted(k_results.items()):
            print(f"{mname:<12} {k:>5.1f}  {res['train_r2']:>10.4f}  {res['test_r2']:>10.4f}")
    print("=" * 70)
