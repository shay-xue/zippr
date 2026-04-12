"""Quick diagnostic: what is the actual relationship between neural channels and controller inputs?"""
import h5py, numpy as np, glob
from scipy.stats import pearsonr

SAMPLE_RATE = 32000

# Load one file with varied movement
f = sorted(glob.glob('data/recordings/train_009_freestyle1.h5/broadband_data_*.h5'))[0]
print(f"Loading {f}")
with h5py.File(f, 'r') as hf:
    raw = hf['acquisition/ElectricalSeries'][:]

n = len(raw) // 76
d = raw[:n * 76].reshape(n, 76)
neural = d[:, :64].astype(np.float32)
labels = d[:, 64:76].astype(np.float32)
del raw, d

# Bin to 10ms
bs = int(SAMPLE_RATE * 10 / 1000)  # 320 samples per bin
nb = len(neural) // bs
neural_binned = neural[:nb * bs].reshape(nb, bs, 64).mean(1)  # just mean voltage
labels_binned = labels[:nb * bs].reshape(nb, bs, 12).mean(1)

label_names = ['LStickX', 'LStickY', 'RStickX', 'RStickY', 'A', 'B', 'X', 'Y', 'LB', 'RB', 'LT', 'RT']

print(f"\nNeural: {neural_binned.shape}, Labels: {labels_binned.shape}")
print(f"Neural value range: [{neural.min():.0f}, {neural.max():.0f}]")
print(f"Label ranges:")
for i, name in enumerate(label_names):
    col = labels_binned[:, i]
    print(f"  {name}: [{col.min():.1f}, {col.max():.1f}], std={col.std():.1f}")

# Correlation: each neural channel vs each label
print("\n=== Pearson correlations: top 5 neural channels per controller input ===")
for li, lname in enumerate(label_names):
    corrs = []
    for ch in range(64):
        r, p = pearsonr(neural_binned[:, ch], labels_binned[:, li])
        corrs.append((abs(r), r, ch))
    corrs.sort(reverse=True)
    top = corrs[:5]
    print(f"\n{lname}:")
    for absR, r, ch in top:
        print(f"  ch{ch:02d}: r={r:+.4f}")

# Try a simple linear regression on raw binned means
print("\n\n=== Linear regression (raw binned mean voltage → each label) ===")
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

# Load ALL freestyle files for better coverage
files = sorted(glob.glob('data/recordings/train_*.h5/broadband_data_*.h5'))
all_neural, all_labels = [], []
for f in files:
    with h5py.File(f, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76)
    nb = n // bs
    all_neural.append(d[:nb * bs, :64].astype(np.float32).reshape(nb, bs, 64).mean(1))
    all_labels.append(d[:nb * bs, 64:76].astype(np.float32).reshape(nb, bs, 12).mean(1))
    del raw, d

X = np.concatenate(all_neural)
Y = np.concatenate(all_labels)
del all_neural, all_labels

print(f"\nAll data: X={X.shape}, Y={Y.shape}")

Xtr, Xte, Ytr, Yte = train_test_split(X, Y, test_size=0.2, random_state=42)

# Simple Ridge regression per output
for li, lname in enumerate(label_names):
    model = Ridge(alpha=1.0)
    model.fit(Xtr, Ytr[:, li])
    pred = model.predict(Xte)
    r2 = r2_score(Yte[:, li], pred)
    print(f"  {lname:10s}: R²={r2:.4f}")

# Also try with a few history bins (lag features)
print("\n=== Ridge with 5-bin history (50ms context) ===")
w = 5
Xw = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).reshape(-1, w * 64)
Yw = Y[w-1:]
Xwtr, Xwte, Ywtr, Ywte = train_test_split(Xw, Yw, test_size=0.2, random_state=42)
for li, lname in enumerate(label_names):
    model = Ridge(alpha=1.0)
    model.fit(Xwtr, Ywtr[:, li])
    pred = model.predict(Xwte)
    r2 = r2_score(Ywte[:, li], pred)
    print(f"  {lname:10s}: R²={r2:.4f}")

print("\n=== Ridge with 15-bin history (150ms context) ===")
w = 15
Xw = np.lib.stride_tricks.sliding_window_view(X, w, axis=0).reshape(-1, w * 64)
Yw = Y[w-1:]
Xwtr, Xwte, Ywtr, Ywte = train_test_split(Xw, Yw, test_size=0.2, random_state=42)
for li, lname in enumerate(label_names):
    model = Ridge(alpha=1.0)
    model.fit(Xwtr, Ywtr[:, li])
    pred = model.predict(Xwte)
    r2 = r2_score(Ywte[:, li], pred)
    print(f"  {lname:10s}: R²={r2:.4f}")
