# Run 006 — Structured Training Data Collection

**Date:** 2026-04-10 16:45–17:30  
**Mode:** Hard Training (Peripheral 108)  
**Purpose:** Collect diverse, labeled training data for decoder model development.

## Method
32 sequential recordings covering isolated inputs, combined inputs, sweeps, and freestyle.
Each trial recorded separately to avoid packet-loss-induced alignment issues.

## Recordings

| # | Input | Duration |
|---|-------|----------|
| 001 | Rest (baseline) | 13.7s |
| 002 | Left stick circles | 15.0s |
| 003 | Right stick circles | 14.3s |
| 004 | Both sticks | 7.2s |
| 005 | A/B/X/Y buttons | 13.9s |
| 006 | LB/RB/LT/RT bumpers | 15.7s |
| 007 | Left stick + buttons | 12.4s |
| 008 | Right stick + bumpers | 15.2s |
| 009 | Freestyle (all) #1 | 15.5s |
| 010 | Freestyle (all) #2 | 14.3s |
| 011 | LB spam | 17.2s |
| 012 | RB spam | 20.0s |
| 013 | LT spam | 16.6s |
| 014 | RT spam | 17.7s |
| 015 | A/B alternating | 21.0s |
| 016 | X/Y alternating | 17.1s |
| 017 | Right stick X only | 10.6s |
| 018 | Right stick Y only | 20.1s |
| 019 | Left stick + ABXY | 26.1s |
| 020 | Right stick + bumpers/triggers | 24.9s |
| 021 | Slow all inputs | 29.8s |
| 022 | Freestyle #3 | 27.7s |
| 023 | Left stick X sweep | 30.5s |
| 024 | Left stick Y sweep | 25.1s |
| 025 | Right stick X sweep | 29.6s |
| 026 | Right stick Y sweep | 29.1s |
| 027 | Left stick diagonals | 30.5s |
| 028 | Right stick diagonals | 29.2s |
| 029 | LT analog range | 29.7s |
| 030 | RT analog range | 22.4s |
| 031 | Left stick + triggers | 20.6s |
| 032 | Robot sim (full arm) | 33.4s |

**Total usable data: 666.3 seconds (11.1 minutes)**

## Channel Map Reference
- Channels 0–63: Neural data (64 channels)
- Channel 64: Left Stick X | 65: Left Stick Y
- Channel 66: Right Stick X | 67: Right Stick Y
- Channel 68: A | 69: B | 70: X | 71: Y
- Channel 72: LB | 73: RB | 74: LT | 75: RT

## Data Locations
Raw HDF5 files: `data/recordings/train_001_rest.h5/` through `train_032_robot_sim.h5/`

## Decoder Training Results (v4 — spike rate + voltage features)
- Architecture: 3 separate models (joystick, trigger, button gate)
- Features: spike rates at 3/4/5x thresholds + binned voltage stats (448 features/bin)
- Window: 15 bins of 10ms = 150ms context
- Best joystick R² (random split): 0.61 — but inflated by data leakage from overlapping windows
- Honest evaluation (file-based split, linear model): R² ~0.33
- Key finding: per-channel correlations with controller inputs are extremely low (max r=0.03),
  suggesting the mock encoder uses a distributed, nonlinear encoding across all 64 channels
