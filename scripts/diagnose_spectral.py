"""Diagnose: what frequency features does the mock encoder use?"""
import h5py, numpy as np, glob
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

SAMPLE_RATE = 32000

label_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY', 'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

# Load all data
files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
all_feat_mean = []
all_feat_fft = []
all_feat_multi = []
all_labels = []

bin_ms = 10
bs = int(SAMPLE_RATE * bin_ms / 1000)  # 320 samples

for fi, f in enumerate(files):
    with h5py.File(f, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d

    nb = len(neural) // bs
    neural = neural[:nb * bs]
    labels = labels[:nb * bs]

    chunks = neural.reshape(nb, bs, 64)
    label_bins = labels.reshape(nb, bs, 12).mean(1)

    # Feature set 1: just mean voltage (64 features)
    feat_mean = chunks.mean(1)

    # Feature set 2: FFT power in frequency bands per channel
    # For each 320-sample chunk, compute FFT and extract band powers
    fft_data = np.fft.rfft(chunks, axis=1)  # (nb, 161, 64)
    power = np.abs(fft_data) ** 2

    # Frequency resolution: 32000/320 = 100 Hz per bin
    # Bands: 0-500 Hz (bins 0-5), 500-2000 (5-20), 2000-5000 (20-50), 5000-16000 (50-160)
    band_edges = [(0, 5), (5, 20), (20, 50), (50, 100), (100, 161)]
    feat_fft_parts = []
    for lo, hi in band_edges:
        feat_fft_parts.append(power[:, lo:hi, :].mean(1))  # mean power in band
    feat_fft = np.concatenate(feat_fft_parts, axis=1)  # 5 bands * 64 ch = 320 features

    # Feature set 3: multi-feature (mean, std, fft bands)
    feat_multi = np.concatenate([
        chunks.mean(1),        # 64
        chunks.std(1),         # 64
        feat_fft,              # 320
    ], axis=1)                 # total: 448

    all_feat_mean.append(feat_mean)
    all_feat_fft.append(feat_fft)
    all_feat_multi.append(feat_multi)
    all_labels.append(label_bins)
    del neural, labels, chunks, fft_data, power

    if (fi + 1) % 8 == 0:
        print(f"  Loaded {fi+1}/{len(files)}")

print(f"Loaded {len(files)} files")

Xmean = np.concatenate(all_feat_mean)
Xfft = np.concatenate(all_feat_fft)
Xmulti = np.concatenate(all_feat_multi)
Y = np.concatenate(all_labels)
del all_feat_mean, all_feat_fft, all_feat_multi, all_labels

print(f"Shapes: mean={Xmean.shape}, fft={Xfft.shape}, multi={Xmulti.shape}, Y={Y.shape}")

# Test each feature set with windowed Ridge (15 bins = 150ms)
def test_features(name, X, Y, w=15):
    Xw = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).reshape(-1, w * X.shape[1])
    Yw = Y[w-1:]
    Xtr, Xte, Ytr, Yte = train_test_split(Xw, Yw, test_size=0.2, random_state=42)

    # Normalize
    m, s = Xtr.mean(0), Xtr.std(0) + 1e-8
    Xtr = (Xtr - m) / s
    Xte = (Xte - m) / s

    print(f"\n=== {name} (w={w}, {Xw.shape[1]} features) ===")
    r2s = []
    for li, lname in enumerate(label_names[:4]):  # just joysticks for speed
        model = Ridge(alpha=1.0)
        model.fit(Xtr, Ytr[:, li])
        pred = model.predict(Xte)
        r2 = r2_score(Yte[:, li], pred)
        r2s.append(r2)
        print(f"  {lname:10s}: R²={r2:.4f}")
    print(f"  Mean joystick R²: {np.mean(r2s):.4f}")
    return np.mean(r2s)

# Compare feature sets
test_features("Mean voltage only", Xmean, Y, w=15)
test_features("FFT bands only", Xfft, Y, w=15)
test_features("Mean+Std+FFT", Xmulti, Y, w=15)

# Also try wider windows
test_features("Mean voltage w=30 (300ms)", Xmean, Y, w=30)
test_features("FFT bands w=30", Xfft, Y, w=30)
test_features("Mean+Std+FFT w=30", Xmulti, Y, w=30)

# Try even wider
test_features("Mean voltage w=50 (500ms)", Xmean, Y, w=50)
test_features("FFT bands w=50", Xfft, Y, w=50)

# Try smaller bins - 2ms instead of 10ms
print("\n\n========= TRYING 2ms BINS =========")
bs2 = int(SAMPLE_RATE * 2 / 1000)  # 64 samples per bin
all_feat2 = []
all_labels2 = []
for fi, f in enumerate(files):
    with h5py.File(f, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    neural = d[:, :64].astype(np.float32)
    labels = d[:, 64:76].astype(np.float32)
    del raw, d
    nb = len(neural) // bs2
    neural = neural[:nb * bs2]
    labels = labels[:nb * bs2]
    chunks = neural.reshape(nb, bs2, 64)
    all_feat2.append(chunks.mean(1))
    all_labels2.append(labels.reshape(nb, bs2, 12).mean(1))
    del neural, labels, chunks

X2 = np.concatenate(all_feat2); del all_feat2
Y2 = np.concatenate(all_labels2); del all_labels2
print(f"2ms bins: X={X2.shape}, Y={Y2.shape}")

# 75 bins of 2ms = 150ms (same temporal context as 15 bins of 10ms)
test_features("Mean 2ms bins w=75 (150ms)", X2, Y2, w=75)
# 150 bins of 2ms = 300ms
test_features("Mean 2ms bins w=150 (300ms)", X2, Y2, w=150)
