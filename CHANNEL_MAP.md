# SciFi Hard Mode — Confirmed Channel Map

**Device:** funky-frisky-gerbil (SFI100112)  
**Peripheral:** 108 (Hard Training)  
**Sample Rate:** 32,000 Hz  
**Total Channels:** 76 (64 neural + 12 labels)

---

## Neural Data

| Channels | Content |
|----------|---------|
| 0–63 | Encoded mock neural data (only available in training + testing) |

## Controller Labels (training mode only)

| Channel | Input | Type | Range | Confidence |
|---------|-------|------|-------|------------|
| 64 | Left Stick X | Analog axis | [-32767, +32767] | 100% |
| 65 | Left Stick Y | Analog axis | [-32767, +32767] | 100% |
| 66 | Right Stick X | Analog axis | [-32767, +32767] | 100% |
| 67 | Right Stick Y | Analog axis | [-32767, +32767] | 100% |
| 68 | A Button | Binary | 0 / 32767 | 100% |
| 69 | B Button | Binary | 0 / 32767 | 100% |
| 70 | X Button | Binary | 0 / 32767 | 100% |
| 71 | Y Button | Binary | 0 / 32767 | 100% |
| 72 | Left Bumper (LB) | Binary | 0 / 32767 | 100% |
| 73 | Right Bumper (RB) | Binary | 0 / 32767 | 100% |
| 74 | Left Trigger (LT) | Binary | 0 / 32767 | 100% |
| 75 | Right Trigger (RT) | Binary | 0 / 32767 | 100% |

## Verification Method

Each mapping was confirmed by recording 5 seconds of isolated input (one input at a time). For joystick axes, the primary axis hits full range [-32767, +32767] while crosstalk on the other axis stays well below full range. For buttons/bumpers/triggers, exactly one channel activates while all others remain zero.

Full verification data: `data_collection/run_005_individual_mapping/`

## Peripheral IDs

| Mode | Training | Testing |
|------|----------|---------|
| Easy | 104 | 103 |
| Medium | 106 | 105 |
| Hard | 108 | 107 |

## Notes

- Label channels (64-75) are **only present in training mode**. In testing mode, only channels 0-63 are available.
- The decoder model must learn to predict controller state from channels 0-63 alone.
- Joystick resting positions: Ch 64 = -128, Ch 65 = +128, Ch 66 = -128, Ch 67 = +128 (slight offsets from zero).
