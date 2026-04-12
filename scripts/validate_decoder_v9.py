"""
Deep validation of decoder v9 R² scores.

Stress tests:
  1. Data leakage — any file in both train and val?
  2. Feature normalisation leakage — were stats computed on val data?
  3. Naive baselines — predict-zero, predict-mean, predict-last-bin
  4. Label autocorrelation — how much does label[t] predict label[t+1]?
  5. Val set label coverage — is val dominated by rest/zero signal?
  6. Per-val-file R² — does every file contribute, or is one file driving results?
  7. Prediction variance — is the model actually moving or predicting a constant?
  8. Shuffle test — does R² collapse when labels are shuffled?
"""

import h5py
import numpy as np
import glob
import os
import gc
import json
import torch
from scipy.signal import butter, sosfilt
from sklearn.metrics import r2_score

# ── Mirror v9 constants exactly ───────────────────────────────────────────────
SAMPLE_RATE = 32_000
N_NEURAL    = 64
N_LABELS    = 12
BIN_MS      = 100
BIN_SAMPLES = int(SAMPLE_RATE * BIN_MS / 1000)   # 3200
N_FEATURES  = 192
SEQ_LEN     = 15
RECORDINGS  = 'data/recordings'
V9_DIR      = 'analysis/decoder_rnn_shay/v9'
JOY_NAMES   = ['LStX', 'LStY', 'RStX', 'RStY']
TRIG_NAMES  = ['LT', 'RT']

DEVICE = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')


def log(msg): print(msg, flush=True)
def sep(title): log(f"\n{'─'*60}\n  {title}\n{'─'*60}")


# ── Reuse v9 preprocessing exactly ───────────────────────────────────────────

def bandpass_filter(neural):
    sos = butter(2, [200, 5000], btype='bandpass', fs=SAMPLE_RATE, output='sos')
    return sosfilt(sos, neural, axis=0).astype(np.float32)

def extract_features(neural_filtered):
    ch_std = neural_filtered.std(0) + 1e-8
    n_bins = len(neural_filtered) // BIN_SAMPLES
    neural_tr = neural_filtered[:n_bins * BIN_SAMPLES]
    parts = []
    for sigma in [3.0, 4.0, 5.0]:
        thresh = sigma * ch_std
        spikes = ((neural_tr > thresh) | (neural_tr < -thresh)).astype(np.float32)
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

def get_category(path):
    dirname = os.path.basename(os.path.dirname(path))
    rules = [
        ('rest',       ['rest_start', 'rest_end', '_rest_']),
        ('lstick',     ['lsty', 'lstx', 'lst_circle', 'lstick']),
        ('rstick',     ['rsty', 'rstx', 'rst_circle', 'rstick']),
        ('triggers',   ['lt_sweep', 'lt_pump', 'rt_sweep', 'rt_pump',
                        'lt_spam', 'rt_spam', 'lt_analog', 'rt_analog']),
        ('bothsticks', ['both_sticks', 'sticks_and', 'bothsticks']),
        ('freestyle',  ['natural_play', 'freestyle', 'slow_all', 'robot_sim']),
        ('buttons',    ['btn_', 'buttons', 'ab_alt', 'xy_alt']),
        ('bumpers',    ['bumpers', 'lb_spam', 'rb_spam']),
    ]
    for cat, keywords in rules:
        if any(kw in dirname for kw in keywords):
            return cat
    return 'other'

def get_split(files):
    by_cat = {}
    for f in files:
        by_cat.setdefault(get_category(f), []).append(f)
    val_files, train_files = [], []
    for cat, flist in sorted(by_cat.items()):
        val_idx = len(flist) // 2
        val_files.append(flist[val_idx])
        train_files.extend(f for i, f in enumerate(flist) if i != val_idx)
    return train_files, val_files


# ── Load model + norm stats ───────────────────────────────────────────────────

def load_model_and_stats():
    import sys
    sys.path.insert(0, 'scripts')
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
    """Run model on a single file's features → joy predictions (n_bins - SEQ_LEN + 1, 4)."""
    feat_n = (feat - feat_mean) / feat_std
    preds = []
    with torch.no_grad():
        for start in range(0, len(feat_n) - SEQ_LEN + 1, SEQ_LEN):
            x = feat_n[start:start + SEQ_LEN]
            if len(x) < SEQ_LEN:
                break
            xb = torch.from_numpy(x[None]).to(DEVICE)
            joy_p, _, _ = model(xb)
            preds.append(joy_p.cpu().numpy()[0])
    return np.array(preds) if preds else np.zeros((0, 4))


# ── Main validation ───────────────────────────────────────────────────────────

def main():
    log("Loading file list...")
    run006 = sorted(glob.glob(f'{RECORDINGS}/train_*/broadband_data_*.h5'))
    run007 = sorted(glob.glob(f'{RECORDINGS}/run007_*/broadband_data_*.h5'))
    files  = run006 + run007
    train_files, val_files = get_split(files)

    # ── 1. Data leakage check ─────────────────────────────────────────────────
    sep("1. DATA LEAKAGE CHECK")
    train_stems = {os.path.basename(os.path.dirname(f)) for f in train_files}
    val_stems   = {os.path.basename(os.path.dirname(f)) for f in val_files}
    overlap = train_stems & val_stems
    if overlap:
        log(f"  !! LEAKAGE DETECTED — {len(overlap)} files in both sets: {overlap}")
    else:
        log(f"  OK — no file appears in both train and val")
    log(f"\n  Val files ({len(val_files)}):")
    for f in val_files:
        log(f"    {get_category(f):12s}  {os.path.basename(os.path.dirname(f))}")

    # ── 2. Feature normalisation leakage check ────────────────────────────────
    sep("2. FEATURE NORMALISATION LEAKAGE CHECK")
    log("  Norm stats are computed inside train_rnn() from feats_tr only.")
    log("  Checking: do val file features differ from train distribution?")
    log("  Loading norm stats from v9...")
    norm = np.load(os.path.join(V9_DIR, 'feat_norm.npz'))
    feat_mean = norm['feat_mean'].reshape(1, -1)
    feat_std  = norm['feat_std'].reshape(1, -1)

    log("\n  Loading one val file to check normalised feature distribution...")
    val_feat, _, _ = load_file(val_files[0])
    val_feat_n = (val_feat - feat_mean) / feat_std
    log(f"  Val feat normalised mean: {val_feat_n.mean():.3f}  (expect ~0 if dist matches train)")
    log(f"  Val feat normalised std:  {val_feat_n.std():.3f}   (expect ~1 if dist matches train)")
    del val_feat, val_feat_n; gc.collect()

    # ── 3. Load val data ──────────────────────────────────────────────────────
    sep("3. LOADING VAL DATA")
    val_feats, val_joys, val_trigs = [], [], []
    for f in val_files:
        feat, joy, trig = load_file(f)
        val_feats.append(feat)
        val_joys.append(joy)
        val_trigs.append(trig)
        log(f"  {os.path.basename(os.path.dirname(f)):50s}  {len(feat)} bins")

    # ── 4. Label coverage in val set ─────────────────────────────────────────
    sep("4. VAL SET LABEL COVERAGE")
    all_joy  = np.concatenate(val_joys,  axis=0)
    all_trig = np.concatenate(val_trigs, axis=0)
    log(f"\n  {'Label':<8}  {'mean':>8}  {'std':>8}  {'min':>8}  {'max':>8}  {'% near-zero':>12}")
    for i, name in enumerate(JOY_NAMES):
        col = all_joy[:, i]
        pct_zero = (np.abs(col) < 0.05).mean() * 100
        log(f"  {name:<8}  {col.mean():>8.3f}  {col.std():>8.3f}  {col.min():>8.3f}  {col.max():>8.3f}  {pct_zero:>11.1f}%")
    for i, name in enumerate(TRIG_NAMES):
        col = all_trig[:, i]
        pct_zero = (col < 0.05).mean() * 100
        log(f"  {name:<8}  {col.mean():>8.3f}  {col.std():>8.3f}  {col.min():>8.3f}  {col.max():>8.3f}  {pct_zero:>11.1f}%")

    # ── 5. Label autocorrelation ──────────────────────────────────────────────
    sep("5. LABEL AUTOCORRELATION (lag-1, i.e. how well does t predict t+1?)")
    log("  If label[t] ≈ label[t+1], a model that 'cheats' by copying context")
    log("  could score high R² without using neural signal at all.\n")
    for i, name in enumerate(JOY_NAMES):
        col = all_joy[:, i]
        if len(col) > 1:
            r2_lag1 = r2_score(col[1:], col[:-1])
            log(f"  {name:<8}  lag-1 R²={r2_lag1:+.4f}  "
                f"({'HIGH — temporal cheating possible' if r2_lag1 > 0.8 else 'OK'})")

    # ── 6. Naive baselines ────────────────────────────────────────────────────
    sep("6. NAIVE BASELINES")

    # predict-zero
    zero_r2 = [r2_score(all_joy[:, i], np.zeros(len(all_joy))) for i in range(4)]
    log(f"\n  Predict-zero R²:  {[f'{r:+.3f}' for r in zero_r2]}  avg={np.mean(zero_r2):+.3f}")

    # predict-mean (train mean = 0 after normalisation, so use val mean)
    mean_r2 = [r2_score(all_joy[:, i], np.full(len(all_joy), all_joy[:, i].mean())) for i in range(4)]
    log(f"  Predict-mean R²:  {[f'{r:+.3f}' for r in mean_r2]}  avg={np.mean(mean_r2):+.3f}")

    # predict-last (copy label from SEQ_LEN bins ago)
    lag_r2s = []
    for i, name in enumerate(JOY_NAMES):
        col = all_joy[:, i]
        if len(col) > SEQ_LEN:
            r2_lag = r2_score(col[SEQ_LEN:], col[:-SEQ_LEN])
            lag_r2s.append(r2_lag)
        else:
            lag_r2s.append(float('nan'))
    log(f"  Predict-lag-{SEQ_LEN} R²: {[f'{r:+.3f}' for r in lag_r2s]}  avg={np.nanmean(lag_r2s):+.3f}")
    log(f"\n  (Model must beat these baselines to prove it uses neural signal)")

    # ── 7. Per-val-file R² ────────────────────────────────────────────────────
    sep("7. PER-VAL-FILE R² (model predictions)")
    model, feat_mean_m, feat_std_m = load_model_and_stats()

    all_jp, all_jt = [], []
    log(f"\n  {'File':50s}  {'LStX':>7}  {'LStY':>7}  {'RStX':>7}  {'RStY':>7}  {'avg':>7}")
    for f, feat, joy in zip(val_files, val_feats, val_joys):
        jp = predict(model, feat, feat_mean_m, feat_std_m)
        if len(jp) == 0:
            log(f"  {os.path.basename(os.path.dirname(f)):50s}  (too short)")
            continue
        jt = joy[SEQ_LEN - 1: SEQ_LEN - 1 + len(jp):SEQ_LEN][:len(jp)]
        if len(jt) != len(jp):
            min_len = min(len(jt), len(jp))
            jt, jp = jt[:min_len], jp[:min_len]
        r2s = [r2_score(jt[:, i], jp[:, i]) if jt[:, i].std() > 0.01 else float('nan')
               for i in range(4)]
        avg = np.nanmean(r2s)
        log(f"  {os.path.basename(os.path.dirname(f)):50s}  "
            f"  {r2s[0]:>+.3f}  {r2s[1]:>+.3f}  {r2s[2]:>+.3f}  {r2s[3]:>+.3f}  {avg:>+.3f}")
        all_jp.append(jp)
        all_jt.append(jt)

    # ── 8. Prediction variance ────────────────────────────────────────────────
    sep("8. PREDICTION VARIANCE")
    log("  A model predicting near-constant output can get high R² if labels are")
    log("  also near-constant. Check that predictions actually vary.\n")
    if all_jp:
        jp_cat = np.concatenate(all_jp, axis=0)
        jt_cat = np.concatenate(all_jt, axis=0)
        log(f"  {'Label':<8}  {'true std':>10}  {'pred std':>10}  {'std ratio':>10}")
        for i, name in enumerate(JOY_NAMES):
            ts = jt_cat[:, i].std()
            ps = jp_cat[:, i].std()
            log(f"  {name:<8}  {ts:>10.4f}  {ps:>10.4f}  {ps/ts if ts > 0 else float('nan'):>10.3f}")

    # ── 9. Shuffle test ───────────────────────────────────────────────────────
    sep("9. SHUFFLE TEST")
    log("  Shuffle val labels randomly. R² should collapse to near 0 or negative.")
    log("  If it stays high, the score is driven by distribution, not signal.\n")
    if all_jp:
        jp_cat = np.concatenate(all_jp, axis=0)
        jt_cat = np.concatenate(all_jt, axis=0)
        rng = np.random.default_rng(42)
        shuffled_idx = rng.permutation(len(jt_cat))
        jt_shuffled = jt_cat[shuffled_idx]
        r2s_shuf = [r2_score(jt_shuffled[:, i], jp_cat[:, i]) for i in range(4)]
        log(f"  Shuffled label R²: {[f'{r:+.3f}' for r in r2s_shuf]}  avg={np.mean(r2s_shuf):+.3f}")
        log(f"  (Should be strongly negative — model predicts wrong pattern on shuffled labels)")

    sep("SUMMARY")
    with open(os.path.join(V9_DIR, 'model_meta.json')) as f:
        meta = json.load(f)
    log(f"  Reported val R² (from training):  joy avg={meta['joy_r2_avg']:+.4f}")
    if all_jp:
        jp_cat = np.concatenate(all_jp, axis=0)
        jt_cat = np.concatenate(all_jt, axis=0)
        recomputed = [r2_score(jt_cat[:, i], jp_cat[:, i]) for i in range(4)]
        log(f"  Recomputed R² (this script):      joy avg={np.mean(recomputed):+.4f}")
        log(f"  Per-axis: {[f'{r:+.3f}' for r in recomputed]}")
        log(f"\n  Predict-lag-{SEQ_LEN} avg R²:       {np.nanmean(lag_r2s):+.4f}  ← model must beat this")
        log(f"  Model margin over lag baseline:   {np.mean(recomputed) - np.nanmean(lag_r2s):+.4f}")


if __name__ == '__main__':
    main()
