# Decoder Pipeline — v7 GRU

Model trained to decode joystick axes and trigger pressure from SciFi neural mock data.

---

## Results (Run 006 training data, 100ms bins)

| Output | Val R² |
|--------|--------|
| Left Stick X | ~0.65 |
| Left Stick Y | ~0.45 (data gap — fixed by Run 007) |
| Right Stick X | ~0.65 |
| Right Stick Y | ~0.65 |
| LT | ~0.60 |
| RT | ~0.55 |
| Button gate (no-press) | ~0.85 accuracy |

---

## Preprocessing

### 1. Load HDF5
- File layout: `acquisition/ElectricalSeries` — flattened int16 array
- Reshape to `(n_samples, 76)`: channels 0–63 = neural, channels 64–75 = labels
- Labels: `[LStX, LStY, RStX, RStY, A, B, X, Y, LB, RB, LT, RT]`

### 2. Bandpass filter (neural channels only)
- 2nd-order Butterworth bandpass: **200–5000 Hz**
- Applied per file along the time axis

### 3. Bin into 10ms windows
- Bin size: **10ms = 320 samples** at 32,000 Hz
- Each file produces `n_samples // 320` bins

### 4. Spike counting — Track A (192 features/bin)
- Compute per-channel std from the filtered signal
- Count threshold crossings at **3σ, 4σ, 5σ** — both positive and negative
- Each threshold produces 64 counts/bin → concatenated: **192 features/bin**

### 5. Label binning
- Labels averaged over each 10ms window → `(n_bins, 12)`

### 6. Label normalisation
- Joystick axes (ch 64–67): divide by 32767 → **[-1, +1]**
- Triggers LT/RT (ch 74–75): divide by 32767 → **[0, +1]**
- Buttons (ch 68–73): threshold at 16000 → binary 0/1
- Gate: 1 if no buttons pressed, 0 otherwise

### 7. Feature normalisation
- Global mean/std computed from training bins only
- Applied to both train and val sets: `(x - mean) / std`

---

## Architecture — GRUDecoder

```
Input: (batch, seq_len, 192)
  → Linear(192 → 192)           # input projection
  → GRU(hidden=192, layers=2, dropout=0.3)
  → last hidden state (batch, 192)
  → joy_head:  Linear→GELU→Linear→Tanh     → (batch, 4)   joystick axes
  → trig_head: Linear→GELU→Linear→Sigmoid  → (batch, 2)   LT/RT
  → gate_head: Linear→GELU→Linear→Sigmoid  → (batch, 1)   button gate
```

Total parameters: ~330k

---

## Sequence construction

- **Sequence length**: 30 bins = 300ms context window
- **Training stride**: 1 (every possible window)
- **Val stride**: 30 (non-overlapping, no leakage)
- Sequences never cross file boundaries

---

## Training

| Hyperparameter | Value |
|----------------|-------|
| Optimizer | AdamW |
| Learning rate | 5e-4 |
| Weight decay | 1e-4 |
| Scheduler | CosineAnnealingWarmRestarts (T₀=30, T_mult=2) |
| Batch size | 128 |
| Max epochs | 300 |
| Early stopping patience | 40 epochs |
| Input noise augmentation | Gaussian, σ=0.05 |
| Gradient clipping | max norm 1.0 |

**Loss**: `MSE(joy) + MSE(trig) + 0.5 × BCE(gate)`  
**Best model selected by**: val joystick R²

---

## Train/val split

File-based stratified split — one file per recording category held out for val:

| Category | Keywords matched |
|----------|-----------------|
| rest | rest |
| lstick | lstick, lstickx, lsticky, lstick_diag |
| rstick | rstick, rstickx, rsticky, rstick_diag |
| buttons | buttons, ab_alt, xy_alt |
| bumpers | bumpers, lb_spam, rb_spam |
| triggers | lt_spam, rt_spam, lt_analog, rt_analog |
| freestyle | freestyle, slow_all, robot_sim |
| bothsticks | bothsticks |

Val file chosen as the middle file of each category to avoid boundary effects.

---

## ONNX export

Wrapped in `GRUDecoderForExport` for Synapse App compatibility:
- **Input**: `(1, 192, seq_len)` — channels-first (transposed from training convention)
- **Output**: `(1, 7)` — `[LStX, LStY, RStX, RStY, LT, RT, gate]`

Export path: `models/decoder_v7.onnx`  
Normalisation stats saved alongside: `models/decoder_v7_stats.npz`

---

## Scripts

| Script | Purpose |
|--------|---------|
| `train_decoder_v7.py` | Full training pipeline — loads data, trains, exports ONNX |
| `bin_size_sweep_v7.py` | Sweep over bin sizes (20–100ms) to find optimal bin width |
| `analyze_decoder_rnn.py` | Post-training analysis — R² per label, label distribution plots |
| `collect_data_run007.py` | Guided data collection — 32 structured sessions on SciFi device |

---

## Known limitations / next steps

- LStY underrepresented in Run 006 val set (std ratio 2.25x train vs val) — addressed by Run 007
- RT nearly absent in Run 006 val set — addressed by Run 007
- Button outputs (A/B/X/Y/LB/RB) not used in current game — gate head only
- Retrain with Run 006 + Run 007 combined after collection completes
