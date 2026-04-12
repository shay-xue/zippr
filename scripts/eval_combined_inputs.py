"""
Evaluate v9 decoder specifically on combined-input recordings:
  - Both sticks simultaneously
  - Sticks + RT/LT
  - Natural play / freestyle
"""

import h5py
import numpy as np
import glob
import os
import gc
import torch
from scipy.signal import butter, sosfilt
from sklearn.metrics import r2_score

SAMPLE_RATE = 32_000
N_NEURAL    = 64
N_LABELS    = 12
BIN_MS      = 100
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)
N_FEATURES  = 192
SEQ_LEN     = 15
RECORDINGS  = 'data/recordings'
V9_DIR      = 'analysis/decoder_rnn_shay/v9'
JOY_NAMES   = ['LStX', 'LStY', 'RStX', 'RStY']
TRIG_NAMES  = ['LT',   'RT']
DEVICE      = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')

def log(msg): print(msg, flush=True)

def bandpass_filter(neural):
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)

def extract_features(neural_filtered):
    ch_std = neural_filtered.std(0) + 1e-8
    n_bins = len(neural_filtered) // BIN_SAMPLES
    nt = neural_filtered[:n_bins * BIN_SAMPLES]
    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        spikes = ((nt > thresh) | (nt < -thresh)).astype(np.float32)
        counts = spikes.reshape(n_bins, BIN_SAMPLES, N_NEURAL).sum(1)
        parts.append(counts)
        del spikes
    return np.concatenate(parts, axis=1).astype(np.float32)

def load_file(path):
    with h5py.File(path, 'r') as hf:
        raw = hf['acquisition/ElectricalSeries'][:]
    n = len(raw) // 76
    d = raw[:n * 76].reshape(n, 76).astype(np.float32)
    neural, labels = d[:, :N_NEURAL], d[:, N_NEURAL:N_NEURAL + N_LABELS]
    del raw, d
    feat = extract_features(bandpass_filter(neural))
    del neural
    n_bins = len(feat)
    lbl = labels[:n_bins * BIN_SAMPLES].reshape(n_bins, BIN_SAMPLES, N_LABELS).mean(1)
    gc.collect()
    joy  = np.clip(lbl[:, :4]    / 32767.0, -1.0, 1.0).astype(np.float32)
    trig = np.clip(lbl[:, 10:12] / 32767.0,  0.0, 1.0).astype(np.float32)
    return feat, joy, trig

def load_model():
    import sys; sys.path.insert(0, 'scripts')
    from train_decoder_v9 import GRUDecoder
    norm = np.load(os.path.join(V9_DIR, 'feat_norm.npz'))
    feat_mean = norm['feat_mean'].reshape(1, -1)
    feat_std  = norm['feat_std'].reshape(1, -1)
    model = GRUDecoder().to(DEVICE)
    model.load_state_dict(torch.load(os.path.join(V9_DIR, 'best_model.pt'),
                                     map_location=DEVICE))
    model.eval()
    return model, feat_mean, feat_std

def predict(model, feat, feat_mean, feat_std):
    feat_n = (feat - feat_mean) / feat_std
    preds_joy, preds_trig = [], []
    with torch.no_grad():
        for start in range(0, len(feat_n) - SEQ_LEN + 1, SEQ_LEN):
            x = feat_n[start:start + SEQ_LEN]
            if len(x) < SEQ_LEN: break
            xb = torch.from_numpy(x[None]).to(DEVICE)
            joy_p, trig_p, _ = model(xb)
            preds_joy.append(joy_p.cpu().numpy()[0])
            preds_trig.append(trig_p.cpu().numpy()[0])
    return (np.array(preds_joy)  if preds_joy  else np.zeros((0, 4)),
            np.array(preds_trig) if preds_trig else np.zeros((0, 2)))

def eval_file(path, model, feat_mean, feat_std):
    feat, joy, trig = load_file(path)
    jp, tp = predict(model, feat, feat_mean, feat_std)
    n = len(jp)
    if n == 0:
        return None

    # Align ground truth to prediction windows
    jt = joy [SEQ_LEN - 1::SEQ_LEN][:n]
    tt = trig[SEQ_LEN - 1::SEQ_LEN][:n]
    n = min(len(jt), len(jp))
    jt, jp = jt[:n], jp[:n]
    tt, tp = tt[:n], tp[:n]

    joy_r2  = [r2_score(jt[:, i], jp[:, i]) if jt[:, i].std() > 0.02 else float('nan')
               for i in range(4)]
    trig_r2 = [r2_score(tt[:, i], tp[:, i]) if tt[:, i].std() > 0.02 else float('nan')
               for i in range(2)]

    # How much were each axis actually active in this recording?
    joy_std  = [jt[:, i].std() for i in range(4)]
    trig_std = [tt[:, i].std() for i in range(2)]

    return {
        'joy_r2':  joy_r2,
        'trig_r2': trig_r2,
        'joy_std': joy_std,
        'trig_std': trig_std,
        'n_pred':  n,
        'duration_s': n * SEQ_LEN * BIN_MS / 1000,
    }

def print_result(label, result):
    if result is None:
        log(f"  {label}: (no data)")
        return
    log(f"\n  {label}  ({result['duration_s']:.0f}s, {result['n_pred']} windows)")
    log(f"  {'Axis':<8}  {'R²':>7}  {'label std':>10}  {'active?':>8}")
    for i, name in enumerate(JOY_NAMES):
        r2  = result['joy_r2'][i]
        std = result['joy_std'][i]
        active = 'yes' if std > 0.1 else 'idle'
        r2_str = f"{r2:+.3f}" if not np.isnan(r2) else "  n/a"
        log(f"  {name:<8}  {r2_str:>7}  {std:>10.3f}  {active:>8}")
    for i, name in enumerate(TRIG_NAMES):
        r2  = result['trig_r2'][i]
        std = result['trig_std'][i]
        active = 'yes' if std > 0.05 else 'idle'
        r2_str = f"{r2:+.3f}" if not np.isnan(r2) else "  n/a"
        log(f"  {name:<8}  {r2_str:>7}  {std:>10.3f}  {active:>8}")

def main():
    model, feat_mean, feat_std = load_model()

    COMBINED_FILES = [
        ("Both sticks circles (run007)",
         f"{RECORDINGS}/run007_both_sticks_circles_20260411_214155.h5/broadband_data_20260411_214155.h5"),
        ("Sticks + RT/LT take 1 (run007)",
         f"{RECORDINGS}/run007_sticks_and_buttons_20260411_214535.h5/broadband_data_20260411_214535.h5"),
        ("Sticks + RT/LT take 2 (run007)",
         f"{RECORDINGS}/run007_sticks_and_triggers_20260411_214754.h5/broadband_data_20260411_214754.h5"),
        ("Natural play (run007)",
         f"{RECORDINGS}/run007_natural_play_20260411_215119.h5/broadband_data_20260411_215119.h5"),
        ("Both sticks (run006)",
         f"{RECORDINGS}/train_004_bothsticks.h5/broadband_data_20260410_164619.h5"),
        ("Freestyle 1 (run006)",
         f"{RECORDINGS}/train_009_freestyle1.h5/broadband_data_20260410_164857.h5"),
        ("Freestyle 2 (run006)",
         f"{RECORDINGS}/train_010_freestyle2.h5/broadband_data_20260410_165002.h5"),
        ("Freestyle 3 (run006)",
         f"{RECORDINGS}/train_022_freestyle3.h5/broadband_data_20260410_194610.h5"),
    ]

    log("="*60)
    log("  V9 DECODER — COMBINED INPUT EVALUATION")
    log("="*60)

    for label, path in COMBINED_FILES:
        if not os.path.exists(path):
            # Try to find the broadband file inside the directory
            parent = path.rsplit('/broadband', 1)[0]
            matches = glob.glob(f"{parent}/broadband_data_*.h5")
            if matches:
                path = matches[0]
            else:
                log(f"\n  {label}: file not found, skipping")
                continue
        result = eval_file(path, model, feat_mean, feat_std)
        print_result(label, result)

    log("\n" + "="*60)
    log("  R² interpretation: >0.7 good | 0.4–0.7 fair | <0.4 poor")
    log("  n/a = axis was idle in that recording (std < threshold)")
    log("="*60)

if __name__ == '__main__':
    main()
