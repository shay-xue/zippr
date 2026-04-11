"""
Sklearn MLP decoder for SciFi easy-mode recordings.

Loads an HDF5 file with 32 neural channels + 3 label channels
(joystick X, joystick Y, A button), extracts mean+std features
per sliding window, trains an MLPClassifier over 12 action classes,
and exports to ONNX via skl2onnx.

Usage:
    uv run python scripts/train_decoder_sklearn.py --data data_collection/broadband_data_20260410_142940.h5
Loading data_collection/broadband_data_20260410_142940.h5 
    python scripts/train_decoder_sklearn.py --data path/to/broadband.h5 --output models/decoder.onnx
"""

import argparse
import os
import numpy as np
import h5py
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType


# ── Label channels (easy mode) ────────────────────────────────────────────────

JOYSTICK_X_CH = 0   # index into labels array (label channel 32 → index 0)
JOYSTICK_Y_CH = 1   # label channel 33 → index 1
A_BUTTON_CH   = 2   # label channel 34 → index 2

AXIS_MAX      = 32767.0
DEAD_ZONE     = 0.3   # normalized threshold to consider stick active


# ── 12-class action scheme ────────────────────────────────────────────────────
#
# Without A button (9 classes): center + 8 octant directions
#   0  center
#   1  N   (up)
#   2  NE
#   3  E   (right)
#   4  SE
#   5  S   (down)
#   6  SW
#   7  W   (left)
#   8  NW
#
# With A button (3 classes):
#   9  A + center  (button press, stick idle)
#  10  A + upper   (button press + any upward stick)
#  11  A + lower   (button press + any downward stick)

CLASS_NAMES = [
    "center",
    "N", "NE", "E", "SE", "S", "SW", "W", "NW",
    "A_center", "A_upper", "A_lower",
]
N_CLASSES = len(CLASS_NAMES)  # 12


def make_class_labels(labels: np.ndarray) -> np.ndarray:
    """
    Convert raw label channels → integer class indices.

    Args:
        labels: (3, n_samples) — [joystick_x, joystick_y, a_button]

    Returns:
        classes: (n_samples,) int array in [0, 11]
    """
    x = labels[JOYSTICK_X_CH] / AXIS_MAX   # normalized to [-1, 1]
    y = labels[JOYSTICK_Y_CH] / AXIS_MAX
    a = labels[A_BUTTON_CH] > 0            # boolean

    magnitude = np.sqrt(x ** 2 + y ** 2)
    active = magnitude > DEAD_ZONE
    angle = np.arctan2(y, x)               # radians in (-π, π]

    # Sector boundaries for 8 octants, starting at E and going CCW
    # We remap so N = up = positive Y
    # arctan2(y, x): E=0, N=π/2, W=±π, S=-π/2
    # Octants: each is π/4 wide
    pi = np.pi
    boundaries = np.linspace(-pi, pi, 9)   # 8 sectors

    # Sector index 0–7 maps CCW from W: W(0), SW(1), S(2), SE(3), E(4), NE(5), N(6), NW(7)
    # Reorder to match class scheme: N=1, NE=2, E=3, SE=4, S=5, SW=6, W=7, NW=8
    sector_to_class = [7, 6, 5, 4, 3, 2, 1, 8]  # sector 0–7 → class 1–8

    classes = np.zeros(len(x), dtype=np.int64)  # default: center (0)

    for sec_idx in range(8):
        lo, hi = boundaries[sec_idx], boundaries[sec_idx + 1]
        in_sector = active & (angle >= lo) & (angle < hi)
        classes[in_sector] = sector_to_class[sec_idx]

    # A button overrides: 9 = A+center, 10 = A+upper, 11 = A+lower
    a_center = a & ~active
    a_upper  = a & active & (y > 0)
    a_lower  = a & active & (y <= 0)

    classes[a_center] = 9
    classes[a_upper]  = 10
    classes[a_lower]  = 11

    return classes


# ── Feature extraction ────────────────────────────────────────────────────────

def extract_features(neural: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """
    Sliding window over neural data, extracting mean + std per channel.

    Args:
        neural: (n_channels, n_samples) z-scored neural data
        window_samples: samples per window
        stride_samples: samples between window starts

    Returns:
        features: (n_windows, n_channels * 2)
    """
    n_channels, n_total = neural.shape
    n_windows = (n_total - window_samples) // stride_samples + 1

    features = np.empty((n_windows, n_channels * 2), dtype=np.float32)
    for i in range(n_windows):
        start = i * stride_samples
        end   = start + window_samples
        chunk = neural[:, start:end]           # (n_channels, window_samples)
        features[i, :n_channels] = chunk.mean(axis=1)
        features[i, n_channels:] = chunk.std(axis=1)

    return features


def window_labels(classes: np.ndarray, window_samples: int, stride_samples: int) -> np.ndarray:
    """Majority-vote label per window."""
    n_total = len(classes)
    n_windows = (n_total - window_samples) // stride_samples + 1
    wlabels = np.empty(n_windows, dtype=np.int64)
    for i in range(n_windows):
        start = i * stride_samples
        end   = start + window_samples
        wlabels[i] = np.bincount(classes[start:end], minlength=N_CLASSES).argmax()
    return wlabels


# ── Load + preprocess ─────────────────────────────────────────────────────────

def _read_h5(f: h5py.File):
    """
    Parse an HDF5 recording regardless of layout.

    Supported layouts:
      A) Separate datasets: f["neural"] (32×N) + f["labels"] (3×N)
      B) Single dataset:    f["data"] or f["broadband"] (35×N)
         — first 32 rows are neural, last 3 are labels
      C) Transposed single: shape (N, 35) — auto-transposed

    Returns:
        neural      : (32, N) float32
        labels      : (3,  N) float32
        sample_rate : int
    """
    sr_candidates = ["sample_rate", "sample_rate_hz", "fs", "samplerate", "Fs"]
    sample_rate = None
    for key in sr_candidates:
        if key in f.attrs:
            sample_rate = int(f.attrs[key])
            break
    if sample_rate is None:
        sample_rate = 32000
        print(f"  [warn] sample_rate not found in attrs, defaulting to {sample_rate} Hz")

    # NWB layout (science-synapse): acquisition/ElectricalSeries is a flat
    # 1-D array of interleaved samples: (n_samples * n_channels,)
    if "acquisition" in f and "ElectricalSeries" in f["acquisition"]:
        raw = f["acquisition"]["ElectricalSeries"][()].astype(np.float32)
        n_channels = len(f["general"]["extracellular_ephys"]["electrodes"]["id"])
        n_samples  = len(f["acquisition"]["sequence_number"])
        arr = raw.reshape(n_samples, n_channels).T   # (n_channels, n_samples)
        n_neural = n_channels - 3
        return arr[:n_neural], arr[n_neural:], sample_rate

    # Separate neural / labels datasets
    if "neural" in f and "labels" in f:
        neural = f["neural"][:].astype(np.float32)
        labels = f["labels"][:].astype(np.float32)
        if neural.ndim == 2 and neural.shape[0] > neural.shape[1]:
            neural = neural.T
        if labels.ndim == 2 and labels.shape[0] > labels.shape[1]:
            labels = labels.T
        return neural, labels, sample_rate

    # Single combined array
    for key in ("data", "broadband", "raw", "recording"):
        if key in f:
            arr = f[key][:].astype(np.float32)
            if arr.ndim == 2 and arr.shape[0] > arr.shape[1]:
                arr = arr.T
            n_neural = arr.shape[0] - 3
            return arr[:n_neural], arr[n_neural:], sample_rate

    raise KeyError(
        f"Could not find neural data in {f.filename}. "
        f"Top-level keys: {list(f.keys())}."
    )


def load_recording(h5_path: str, window_ms: int = 50, stride_ms: int = 25):
    """
    Load easy-mode HDF5 recording and return feature matrix + labels.

    Returns:
        X: (n_windows, 64)  — mean+std features
        y: (n_windows,)     — integer class labels 0–11
    """
    print(f"Loading {h5_path} ...")
    with h5py.File(h5_path, "r") as f:
        neural, labels, sample_rate = _read_h5(f)

    print(f"  Neural: {neural.shape}  Labels: {labels.shape}  SR: {sample_rate} Hz")

    # Z-score per channel
    mean = neural.mean(axis=1, keepdims=True)
    std  = neural.std(axis=1,  keepdims=True) + 1e-8
    neural = (neural - mean) / std

    window_samples = int(window_ms * sample_rate / 1000)
    stride_samples = int(stride_ms * sample_rate / 1000)

    classes = make_class_labels(labels)

    X = extract_features(neural, window_samples, stride_samples)
    y = window_labels(classes, window_samples, stride_samples)

    # Class distribution
    unique, counts = np.unique(y, return_counts=True)
    dist = {CLASS_NAMES[c]: int(n) for c, n in zip(unique, counts)}
    print(f"  Windows: {len(y)}  Class dist: {dist}")

    return X, y


# ── Train ─────────────────────────────────────────────────────────────────────

def train(X: np.ndarray, y: np.ndarray):
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"\nTrain: {len(y_train)}  Val: {len(y_val)}")

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("mlp", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            max_iter=300,
            random_state=42,
            verbose=False,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=20,
        )),
    ])

    print("Training MLP ...")
    model.fit(X_train, y_train)

    val_acc = model.score(X_val, y_val)
    print(f"Val accuracy: {val_acc:.4f}\n")

    y_pred = model.predict(X_val)
    present = sorted(set(y_val))
    print(classification_report(
        y_val, y_pred,
        labels=present,
        target_names=[CLASS_NAMES[c] for c in present],
    ))

    return model


# ── ONNX export ───────────────────────────────────────────────────────────────

def export_onnx(model: Pipeline, n_features: int, output_path: str):
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=13)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"ONNX model saved → {output_path}")

    # Quick verify
    import onnxruntime as ort
    sess = ort.InferenceSession(output_path)
    dummy = np.zeros((1, n_features), dtype=np.float32)
    out = sess.run(None, {"float_input": dummy})
    print(f"ONNX verify: output shape {out[0].shape}  classes {out[1]}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description="Train sklearn MLP decoder (easy mode)")
    parser.add_argument("--data",   "-d", required=True, nargs="+", help="HDF5 recording file(s)")
    parser.add_argument("--output", "-o", default="models/decoder.onnx", help="Output ONNX path")
    parser.add_argument("--window-ms", type=int, default=50,  help="Window size in ms")
    parser.add_argument("--stride-ms", type=int, default=25,  help="Stride in ms")
    return parser.parse_args()


def main():
    args = parse_args()
    all_X, all_y = [], []
    for path in args.data:
        X, y = load_recording(path, args.window_ms, args.stride_ms)
        all_X.append(X)
        all_y.append(y)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    print(f"\nTotal windows across all files: {len(y)}")
    model = train(X, y)
    export_onnx(model, X.shape[1], args.output)


if __name__ == "__main__":
    main()
