# EDA Report — SciFi Broadband Recordings
**Device:** funky-frisky-gerbil (SFI100112) · **Date:** 2026-04-10 · **Sample rate:** 32,000 Hz

---

## 1. Recording Summary

| File | Time | Duration | Channels (neural) | Mode | Notes |
|------|------|----------|-------------------|------|-------|
| `broadband_data_20260410_142940.h5` | 14:29 | 6.7 s | 32 | Easy | Run 001 — idle baseline |
| `broadband_data_20260410_143032.h5` | 14:30 | 20.7 s | 32 | Easy | Run 002 — active joystick + A button |
| `broadband_data_20260410_144229.h5` | 14:42 | 5.1 s | 73 | Hard | Run 003 — all inputs (unstructured) |
| `broadband_data_20260410_144548.h5` | 14:45 | 3.7 s | 73 | Hard | Run 004 — structured mapping |
| `broadband_data_20260410_151848.h5` | 15:18 | 27.2 s | 73 | Hard | Run 005 — individual isolated inputs |

Easy-mode files store 35 channels total (32 neural + 3 labels). Hard-mode files store 76 channels (73 neural + 3 labels decoded in easy-mode script; full hard-mode has 64 neural + 12 labels — decoder needs updating for hard mode).

---

## 2. Amplitude

### Easy mode (runs 001–002)
- Amplitudes are tightly bounded: **min ≈ −354, max ≈ +127 ADC counts**
- Per-channel mean ≈ −0.2 to −1.4, std ≈ 16–25 ADC counts
- Distribution is approximately Gaussian and zero-centred — consistent with broadband neural noise

### Hard mode (runs 003–005)
- Amplitudes rail at **±32,767** — the full 16-bit signed range
- Per-channel std blows out to 265–1,566 ADC counts
- This almost certainly reflects **ADC saturation or clipping** on the hard-mode peripheral (peripheral ID 108), or a gain/scaling mismatch in the H5 writer
- **Action required:** verify headstage gain settings for peripheral 108 before using hard-mode data for training

---

## 3. Class Distribution

### Run 001 — Idle baseline (`142940`)
| Class | Windows | % |
|-------|---------|---|
| center | 267 | 100% |

No controller input, as expected. Useful as a pure resting-state neural baseline.

### Run 002 — Active joystick (`143032`)
| Class | Windows | % |
|-------|---------|---|
| center | 253 | 30.6% |
| S | 115 | 13.9% |
| W | 108 | 13.1% |
| SE | 88 | 10.7% |
| N | 74 | 9.0% |
| NE | 68 | 8.2% |
| SW | 49 | 5.9% |
| E | 41 | 5.0% |
| NW | 18 | 2.2% |
| A_center | 12 | 1.5% |

Good directional coverage across 9 of 12 classes. A_upper and A_lower are absent — the A button was only pressed while the stick was idle. **Class imbalance is moderate**; center is overrepresented at 30%. Consider augmentation or class-weighted loss.

### Run 004 — Structured mapping (`144548`)
| Class | Windows | % |
|-------|---------|---|
| center | 68 | 46.9% |
| E | 55 | 37.9% |
| A_center | 9 | 6.2% |
| A_upper | 9 | 6.2% |
| A_lower | 2 | 1.4% |
| N | 2 | 1.4% |

Only two non-center directions captured. Short duration (3.7 s) limits usefulness as standalone training data.

### Run 005 — Isolated inputs (`151848`)
| Class | Windows | % |
|-------|---------|---|
| center | 985 | 90.5% |
| E | 57 | 5.2% |
| N | 26 | 2.4% |
| A_center | 20 | 1.8% |

Heavily center-dominant. Individual inputs are isolated but brief — the inter-stimulus rest dominates. This recording is better suited for **mapping/validation** than for balanced classifier training.

---

## 4. Power Spectrum

### Easy mode
- PSD follows a **1/f (pink noise) shape** with power decreasing from low to high frequencies
- **Dominant band: HFO (80–300 Hz)** by integrated power — at 32 kHz, there is substantial energy in the upper spectrum
- Relative bandpower ordering: HFO >> gamma >> beta >> delta/theta/alpha
- Welch-average PSDs are smooth and consistent across channels — no obvious dead electrodes

| Band | Freq range | Mean power (run 001) | Mean power (run 002) |
|------|-----------|----------------------|----------------------|
| delta | 1–4 Hz | 7.4×10⁷ | 1.2×10⁹ |
| theta | 4–8 Hz | 1.0×10⁸ | 5.0×10⁸ |
| alpha | 8–13 Hz | 1.4×10⁸ | 4.8×10⁸ |
| beta | 13–30 Hz | 4.2×10⁸ | 1.4×10⁹ |
| gamma | 30–80 Hz | 1.3×10⁹ | 4.1×10⁹ |
| HFO | 80–300 Hz | 5.8×10⁹ | 2.2×10¹⁰ |

Run 002 shows **~4× higher broadband power** than run 001 across all bands, consistent with movement-related neural activity increasing overall signal amplitude.

### Hard mode
- Absolute bandpower is orders of magnitude higher (~10¹³–10¹⁵) due to amplitude saturation/clipping
- **Delta dominates** in hard-mode runs — characteristic of a clipped/DC-offset signal, not meaningful neural content
- Hard-mode spectral data should not be used until amplitude scaling is resolved

---

## 5. Autocorrelation

- Easy-mode signals show **rapid autocorrelation decay within ~5 ms** (consistent with broadband neural noise)
- Some residual correlation out to ~50 ms, suggesting low-frequency fluctuations (movement or common-mode)
- Autocorrelation in hard-mode data is not interpretable due to saturation

---

## 6. Cross-Channel Correlation

- Easy-mode channels show **low pairwise correlation** (most |r| < 0.2), indicating the 32 channels carry largely independent information
- Small clusters of moderately correlated channels (~0.3–0.5) are present — likely electrodes in close physical proximity sharing some common drive
- No channels are fully correlated with each other — no obvious short circuits or bridged electrodes

---

## 7. Per-Channel Notes

### Easy mode (32 channels)
- All 32 channels are active with std ≈ 16–25 ADC counts
- No dead channels detected
- Channel amplitude distributions are Gaussian and zero-centred
- Spectrograms show continuous broadband activity — no burst artefacts or line noise spikes visible in run 001; more dynamic activity in run 002 consistent with motor-related modulation

### Hard mode (73 neural channels)
- Most channels rail at ±32,767 — cannot assess individual channel quality
- Before re-recording: **reduce headstage gain or verify digital scaling for peripheral 108**

---

## 8. Recommendations

| Priority | Action |
|----------|--------|
| High | Fix hard-mode amplitude saturation — adjust gain or LSB scaling for peripheral 108 |
| High | Collect more A_upper / A_lower examples in easy mode (currently 0 windows in any file) |
| Medium | Balance class distribution: center is 30–90% in all files; use stratified sampling or class weights |
| Medium | Collect NW direction data (only 18 windows across all easy-mode recordings) |
| Low | Consider re-running hard mode once saturation is resolved — 64 channels + 12 labels is the real target |

---

## 9. Files Generated

Each `.h5` file produced five PNGs alongside it in `data_collection/`:

| Suffix | Contents |
|--------|----------|
| `_eda.png` | Raw traces, label channels, class timeline, class distribution, feature heatmap |
| `_spectral.png` | Amplitude histogram, per-channel mean/std, mean PSD with band annotations, autocorrelation, cross-channel correlation matrix |
| `_ch_psd.png` | Per-channel Welch PSD grid (log scale) |
| `_ch_bandpower.png` | Per-channel bandpower heatmap (channels × bands) + mean bar chart |
| `_ch_spectrogram.png` | Per-channel STFT spectrogram grid (0–300 Hz, shared colour scale) |

Generated by `scripts/eda.py` — re-run at any time:
```bash
uv run python scripts/eda.py data_collection/*.h5
```
