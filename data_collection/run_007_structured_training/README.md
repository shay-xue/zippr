# Run 007 — Structured Training v2

**Device:** axiomatic-perch-of-revolution (SFI100118)  
**Mode:** Hard Training (Peripheral 108)  
**Sample Rate:** 32,000 Hz | 76 channels (64 neural + 12 labels)

## Why this run exists

Model analysis after Run 006 revealed two critical data gaps:

| Issue | Evidence | Fix |
|-------|----------|-----|
| LStY severely underrepresented in val set | val std = 0.152 vs train std = 0.430 (2.25× gap) | Dedicated up / down / sweep / circle sessions |
| RT nearly absent in val set | val std = 0.111 vs train std = 0.326 | Dedicated RT sweep + pump sessions |
| Buttons not individually isolated | No clean single-button recordings in Run 006 | 90 s per button with explicit rest between presses |

---

## Session Plan

Sessions are ordered by priority — LStY gap is addressed first.

| # | Slug | What to do | Duration |
|---|------|-----------|----------|
| 01 | `rest_start` | Controller on table, don't touch | 2 min |
| 02 | `lsty_up` | Left stick full UP only, rhythm: up 2s → center 1s | 2 min |
| 03 | `lsty_down` | Left stick full DOWN only, same rhythm | 2 min |
| 04 | `lsty_sweep` | Left stick slow continuous up↔down sweep | 2 min |
| 05 | `lstx_right` | Left stick full RIGHT only | 2 min |
| 06 | `lstx_left` | Left stick full LEFT only | 2 min |
| 07 | `lstx_sweep` | Left stick slow continuous left↔right sweep | 2 min |
| 08 | `lst_circle_cw` | Left stick full clockwise circles | 2 min |
| 09 | `lst_circle_ccw` | Left stick full counter-clockwise circles | 2 min |
| 10 | `rsty_up` | Right stick full UP only | 2 min |
| 11 | `rsty_down` | Right stick full DOWN only | 2 min |
| 12 | `rsty_sweep` | Right stick slow continuous up↔down sweep | 2 min |
| 13 | `rstx_right` | Right stick full RIGHT only | 2 min |
| 14 | `rstx_left` | Right stick full LEFT only | 2 min |
| 15 | `rstx_sweep` | Right stick slow continuous left↔right sweep | 2 min |
| 16 | `rst_circle_cw` | Right stick full clockwise circles | 2 min |
| 17 | `rst_circle_ccw` | Right stick full counter-clockwise circles | 2 min |
| 18 | `lt_sweep` | LT slow analog squeeze: 0 → full → 0, 4s cycle | 2 min |
| 19 | `rt_sweep` | RT slow analog squeeze: 0 → full → 0, 4s cycle | 2 min |
| 20 | `lt_pump` | LT fast pumps ~1/sec | 1 min |
| 21 | `rt_pump` | RT fast pumps ~1/sec | 1 min |
| 22 | `btn_a` | A button: hold 1.5s → release 1s → repeat | 90 s |
| 23 | `btn_b` | B button: same | 90 s |
| 24 | `btn_x` | X button: same | 90 s |
| 25 | `btn_y` | Y button: same | 90 s |
| 26 | `btn_lb` | LB: same | 90 s |
| 27 | `btn_rb` | RB: same | 90 s |
| 28 | `both_sticks_circles` | Both sticks simultaneously in circles | 2 min |
| 29 | `sticks_and_buttons` | Both sticks + random face button taps | 2 min |
| 30 | `sticks_and_triggers` | Both sticks + LT/RT varies | 2 min |
| 31 | `natural_play` | Full freestyle — play naturally | 5 min |
| 32 | `rest_end` | Controller down, don't touch | 1 min |

**Total: ~32 sessions, ~57 minutes**

---

## Key recording rules

1. **One thing at a time** in isolated sessions — if doing LStY up, don't accidentally press buttons
2. **Full range** — push all the way to the edge, not halfway
3. **Slow and deliberate** for sweeps — ~4 second cycles
4. **Consistent rhythm** for button sessions — don't vary the hold/release timing too much
5. The controller is always recording labels automatically — no manual annotation needed

---

## Running the collection script

```bash
# From the repo root:
source .venv/bin/activate
python scripts/collect_data_run007.py
```

The script will:
- Tell you exactly what to do before each session
- Count down 3-2-1 before recording starts
- Save each file as `data/recordings/run007_<slug>_<timestamp>.h5/`
- Let you pause between sessions
- Write `session_log.json` after each recording so you can resume if interrupted

**Resume after interruption:**
```bash
python scripts/collect_data_run007.py --start 5   # resume from session 6
```

**Preview the plan without recording:**
```bash
python scripts/collect_data_run007.py --dry-run
```

---

## Output files

- `data/recordings/run007_<slug>_<timestamp>.h5/` — raw HDF5 per session (via Git LFS)
- `data_collection/run_007_structured_training/session_log.json` — progress log (auto-updated)
- `data_collection/run_007_structured_training/summary.md` — generated when all sessions complete

---

## Channel map reference

| Channel | Input | Type |
|---------|-------|------|
| 0–63 | Neural data | 64 ch |
| 64 | Left Stick X | Analog [-32767, +32767] |
| 65 | Left Stick Y | Analog [-32767, +32767] |
| 66 | Right Stick X | Analog [-32767, +32767] |
| 67 | Right Stick Y | Analog [-32767, +32767] |
| 68 | A Button | Binary 0 / 32767 |
| 69 | B Button | Binary 0 / 32767 |
| 70 | X Button | Binary 0 / 32767 |
| 71 | Y Button | Binary 0 / 32767 |
| 72 | LB | Binary 0 / 32767 |
| 73 | RB | Binary 0 / 32767 |
| 74 | LT | Binary 0 / 32767 |
| 75 | RT | Binary 0 / 32767 |
