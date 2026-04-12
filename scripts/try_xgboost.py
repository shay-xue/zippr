"""Try XGBoost on different feature sets to find the R² ceiling."""
import h5py, numpy as np, glob, gc, time
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SAMPLE_RATE = 32000
label_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY', 'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

# ── Load data ──────────────────────────────────────────────────────────────
files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
bs = int(SAMPLE_RATE * 10 / 1000)  # 320 samples = 10ms bins

all_mean = []
all_fft = []
all_labels = []

for fi, f in enumerate(files):
    with h5py.File(f, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d

    nb = len(neural) // bs
    chunks = neural[:nb * bs].reshape(nb, bs, 64)
    label_bins = labels[:nb * bs].reshape(nb, bs, 12).mean(1)

    # Mean voltage per bin
    all_mean.append(chunks.mean(1))

    # FFT: full spectrum per channel (keep more detail)
    fft_data = np.fft.rfft(chunks, axis=1)  # (nb, 161, 64)
    power = np.abs(fft_data) ** 2
    # 10 frequency bands instead of 5
    band_edges = [(0,3), (3,6), (6,10), (10,16), (16,25), (25,40), (40,60), (60,80), (80,120), (120,161)]
    parts = [power[:, lo:hi, :].mean(1) for lo, hi in band_edges]
    all_fft.append(np.concatenate(parts, axis=1))  # 10*64=640

    all_labels.append(label_bins)
    del neural, labels, chunks, fft_data, power, parts
    if (fi+1) % 8 == 0:
        print(f"  Loaded {fi+1}/{len(files)}", flush=True)

print(f"Loaded {len(files)} files", flush=True)

Xmean = np.concatenate(all_mean); del all_mean
Xfft = np.concatenate(all_fft); del all_fft
Y = np.concatenate(all_labels); del all_labels
gc.collect()

print(f"Mean: {Xmean.shape}, FFT: {Xfft.shape}, Y: {Y.shape}", flush=True)

# ── Window features ────────────────────────────────────────────────────────
def window(X, w=15):
    Xw = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).reshape(-1, w * X.shape[1])
    return Xw

# Prepare windowed feature sets
w = 15
Xmean_w = window(Xmean, w)
Yw = Y[w-1:]

# Combined: mean + fft
Xcomb = np.concatenate([Xmean, Xfft], axis=1)
Xcomb_w = window(Xcomb, w)
del Xcomb
gc.collect()

print(f"Windowed mean: {Xmean_w.shape}, combined: {Xcomb_w.shape}", flush=True)

# ── Temporal split (no data leakage!) ──────────────────────────────────────
n = len(Yw)
split = int(0.8 * n)
print(f"Temporal split: train={split}, test={n-split}", flush=True)

# ── Try XGBoost ────────────────────────────────────────────────────────────
try:
    from xgboost import XGBRegressor
    HAS_XGB = True
    print("XGBoost available!", flush=True)
except ImportError:
    HAS_XGB = False
    print("XGBoost not installed, trying: pip install xgboost", flush=True)
    import subprocess
    subprocess.check_call(['pip', 'install', 'xgboost', '-q'])
    from xgboost import XGBRegressor
    HAS_XGB = True

# Also try gradient boosting from sklearn as backup
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor

def test_model(name, X, Y, model_fn):
    Xtr, Xte = X[:split], X[split:]
    Ytr, Yte = Y[:split], Y[split:]

    # Normalize
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - m) / s
    Xte = (Xte - m) / s

    print(f"\n=== {name} ===", flush=True)
    r2s = []
    for li in range(4):  # joystick only
        t0 = time.time()
        model = model_fn()
        model.fit(Xtr, Ytr[:, li])
        pred = model.predict(Xte)
        r2 = r2_score(Yte[:, li], pred)
        r2s.append(r2)
        print(f"  {label_names[li]:10s}: R²={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
    avg = np.mean(r2s)
    print(f"  Mean joystick R²: {avg:.4f}", flush=True)
    return avg

# XGBoost on mean voltage (fast)
test_model("XGBoost mean voltage w=15", Xmean_w, Yw,
           lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                n_jobs=-1, verbosity=0))

# XGBoost on combined features
test_model("XGBoost mean+fft w=15", Xcomb_w, Yw,
           lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                n_jobs=-1, verbosity=0))

# HistGradientBoosting (sklearn, very fast)
test_model("HistGBR mean+fft w=15", Xcomb_w, Yw,
           lambda: HistGradientBoostingRegressor(max_iter=500, max_depth=6,
                                                  learning_rate=0.1, max_leaf_nodes=63))

# XGBoost deeper
test_model("XGBoost deep mean+fft w=15", Xcomb_w, Yw,
           lambda: XGBRegressor(n_estimators=1000, max_depth=8, learning_rate=0.05,
                                subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                n_jobs=-1, verbosity=0))

# Try wider window
print("\n\n--- Wider window w=30 ---", flush=True)
Xcomb30 = np.concatenate([Xmean, Xfft], axis=1)
Xcomb30_w = window(Xcomb30, 30)
Yw30 = Y[29:]
split30 = int(0.8 * len(Yw30))

def test_model_30(name, X, Y, model_fn):
    Xtr, Xte = X[:split30], X[split30:]
    Ytr, Yte = Y[:split30], Y[split30:]
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - m) / s
    Xte = (Xte - m) / s
    print(f"\n=== {name} ===", flush=True)
    r2s = []
    for li in range(4):
        t0 = time.time()
        model = model_fn()
        model.fit(Xtr, Ytr[:, li])
        pred = model.predict(Xte)
        r2 = r2_score(Yte[:, li], pred)
        r2s.append(r2)
        print(f"  {label_names[li]:10s}: R²={r2:.4f} ({time.time()-t0:.1f}s)", flush=True)
    avg = np.mean(r2s)
    print(f"  Mean joystick R²: {avg:.4f}", flush=True)

test_model_30("XGBoost mean+fft w=30", Xcomb30_w, Yw30,
              lambda: XGBRegressor(n_estimators=500, max_depth=6, learning_rate=0.1,
                                   subsample=0.8, colsample_bytree=0.8, tree_method='hist',
                                   n_jobs=-1, verbosity=0))
