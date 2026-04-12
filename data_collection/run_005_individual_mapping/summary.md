# Run 005 — Definitive Channel Mapping

**Date:** 2026-04-10 15:32–15:34  
**Mode:** Hard Training (Peripheral 108)  
**Purpose:** Map each of the 12 label channels to its controller input using individual isolated recordings.

## Method
12 separate 5-second recordings, each with exactly one controller input active.

For joystick axes: the primary axis hits **full range** [-32767, +32767] while crosstalk on the adjacent axis stays well below full range — this cleanly distinguishes X from Y on each stick.

For buttons/bumpers/triggers: exactly one channel shows [0, 32767] activity while all others remain at zero.

## Confirmed Channel Map (100% confidence, all 12)

| Channel | Controller Input | Type | Proof |
|---------|-----------------|------|-------|
| **Ch 64** | Left Stick X | Analog | Only channel with full range in L-stick-X recording |
| **Ch 65** | Left Stick Y | Analog | Only channel with full range in L-stick-Y recording |
| **Ch 66** | Right Stick X | Analog | Only channel with full range in R-stick-X recording |
| **Ch 67** | Right Stick Y | Analog | Only channel with full range in R-stick-Y recording |
| **Ch 68** | A Button | Binary | Only channel active, all others zero |
| **Ch 69** | B Button | Binary | Only channel active, all others zero |
| **Ch 70** | X Button | Binary | Only channel active, all others zero |
| **Ch 71** | Y Button | Binary | Only channel active, all others zero |
| **Ch 72** | Left Bumper (LB) | Binary | Only channel active, all others zero |
| **Ch 73** | Right Bumper (RB) | Binary | Only channel active, all others zero |
| **Ch 74** | Left Trigger (LT) | Binary | ON 53.7% in LT recording, 0% in RT recording |
| **Ch 75** | Right Trigger (RT) | Binary | ON 35.7% in RT recording, 0% in LT recording |

## Joystick Crosstalk Explained

Analog sticks naturally wobble on both axes. When pushing Left Stick X:
- Ch 64 (L-stick X): range = 65534 (full) ← **this is the signal**
- Ch 65 (L-stick Y): range = 35465 (partial) ← this is just stick wobble, not a mapping

The full-range test cleanly separates primary axis from crosstalk in every case.

## Channel Mapping Heatmap
![Mapping Heatmap](channel_mapping_heatmap.png)

## Mapping Diagram
![Mapping Diagram](channel_mapping_diagram.png)

## Recording Details

| # | Input | File | Duration | Packet Loss |
|---|-------|------|----------|-------------|
| 1 | Left Stick X | broadband_data_20260410_153221.h5 | 4.4s | 5.5% |
| 2 | Left Stick Y | broadband_data_20260410_153232.h5 | 3.1s | 31.4% |
| 3 | Right Stick X | broadband_data_20260410_153243.h5 | 4.9s | 0.0% |
| 4 | Right Stick Y | broadband_data_20260410_153254.h5 | 4.9s | 0.0% |
| 5 | A Button | broadband_data_20260410_153305.h5 | 5.2s | 0.0% |
| 6 | B Button | broadband_data_20260410_153316.h5 | 2.6s | 34.8% |
| 7 | X Button | broadband_data_20260410_153328.h5 | 2.9s | 25.7% |
| 8 | Y Button | broadband_data_20260410_153339.h5 | 2.1s | 45.2% |
| 9 | Left Trigger | broadband_data_20260410_153353.h5 | 4.7s | 0.0% |
| 10 | Right Trigger | broadband_data_20260410_153403.h5 | 4.5s | 2.2% |
| 11 | Left Bumper | broadband_data_20260410_153413.h5 | 4.0s | 12.4% |
| 12 | Right Bumper | broadband_data_20260410_153428.h5 | 4.8s | 0.0% |
