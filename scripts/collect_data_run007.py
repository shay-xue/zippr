"""
Guided data collection — Run 007 (Structured Training v2)

Walks you through each recording session with real-time movement cues.
The script calls out exactly what to do and when during every recording.

Usage:
    python scripts/collect_data_run007.py

    # Resume from a specific session number (1-based):
    python scripts/collect_data_run007.py --start 5

    # Dry run (print plan only, no recording):
    python scripts/collect_data_run007.py --dry-run
"""

import argparse
import json
import os
import subprocess
import sys
import threading
import time
from datetime import datetime

DEVICE     = "192.168.16.227"
MANIFEST   = "configs/manifest_hard_training.json"
RECORDINGS = "data/recordings"
RUN_DIR    = "data_collection/run_007_structured_training"
LOG_FILE   = os.path.join(RUN_DIR, "session_log.json")

# ── Cue sequences ─────────────────────────────────────────────────────────────
# Each cue is (hold_seconds, message_to_print)
# The cue player loops through these until the recording duration is up.

def cue_rest():
    return [(5, "Controller down — do not touch")]

def cue_hold_release(direction, hold=2, rest=1):
    return [(hold, f">>> {direction} <<<"), (rest, "--- CENTER ---")]

def cue_sweep(axis, hold=4):
    parts = axis.split("/")
    a, b = parts[0], parts[1]
    return [(hold, f"Slowly slide to {a}..."), (hold, f"Slowly slide to {b}...")]

def cue_circle(direction, secs_per_rotation=3):
    return [(secs_per_rotation, f"Rotating {direction}... keep going...")]

def cue_trigger_sweep(side, hold=4):
    return [(hold, f"Slowly SQUEEZE {side} trigger all the way..."),
            (hold, f"Slowly RELEASE {side} trigger all the way...")]

def cue_trigger_pump(side):
    return [(0.5, f"PRESS {side}"), (0.5, f"RELEASE {side}")]

def cue_button(name, hold=1.5, rest=1.0):
    return [(hold, f"HOLD {name}"), (rest, "RELEASE — rest")]

def cue_both_circles():
    return [(3, "L stick CW + R stick CCW — keep going...")]

def cue_combined(msg):
    return [(4, msg)]


# ── Session plan ──────────────────────────────────────────────────────────────
# Each entry: (slug, label, duration_s, intro_text, cue_fn)

SESSIONS = [
    # ── Rest ──────────────────────────────────────────────────────────────────
    ("rest_start",
     "Rest — Baseline START",
     120,
     "PUT THE CONTROLLER DOWN completely.\n"
     "Do not touch it at all for 2 minutes.\n"
     "This teaches the model what silence looks like.",
     cue_rest()),

    # ── Left Stick Y (LStY — biggest gap in existing data) ───────────────────
    ("lsty_up",
     "Left Stick Y — Full UP",
     120,
     "LEFT STICK — vertical axis only.\n"
     "Push fully UP, hold, then return to CENTER.\n"
     "Do NOT move left or right at all.",
     cue_hold_release("LEFT STICK FULL UP", hold=2, rest=1)),

    ("lsty_down",
     "Left Stick Y — Full DOWN",
     120,
     "LEFT STICK — vertical axis only.\n"
     "Push fully DOWN, hold, then return to CENTER.\n"
     "Do NOT move left or right at all.",
     cue_hold_release("LEFT STICK FULL DOWN", hold=2, rest=1)),

    ("lsty_sweep",
     "Left Stick Y — Slow vertical sweep",
     120,
     "LEFT STICK — vertical axis only.\n"
     "Slide slowly and continuously UP then DOWN, full range.\n"
     "~4 seconds each direction. No left/right movement.",
     cue_sweep("UP/DOWN", hold=4)),

    # ── Left Stick X ──────────────────────────────────────────────────────────
    ("lstx_right",
     "Left Stick X — Full RIGHT",
     120,
     "LEFT STICK — horizontal axis only.\n"
     "Push fully RIGHT, hold, then return to CENTER.\n"
     "Do NOT move up or down.",
     cue_hold_release("LEFT STICK FULL RIGHT", hold=2, rest=1)),

    ("lstx_left",
     "Left Stick X — Full LEFT",
     120,
     "LEFT STICK — horizontal axis only.\n"
     "Push fully LEFT, hold, then return to CENTER.\n"
     "Do NOT move up or down.",
     cue_hold_release("LEFT STICK FULL LEFT", hold=2, rest=1)),

    ("lstx_sweep",
     "Left Stick X — Slow horizontal sweep",
     120,
     "LEFT STICK — horizontal axis only.\n"
     "Slide slowly and continuously LEFT then RIGHT, full range.\n"
     "~4 seconds each direction. No up/down movement.",
     cue_sweep("LEFT/RIGHT", hold=4)),

    # ── Left Stick circles ────────────────────────────────────────────────────
    ("lst_circle_cw",
     "Left Stick — Full circle CLOCKWISE",
     120,
     "LEFT STICK — rotate in large clockwise circles.\n"
     "Stay at the outer edge the whole time.\n"
     "One full rotation every ~3 seconds.",
     cue_circle("CLOCKWISE", secs_per_rotation=3)),

    ("lst_circle_ccw",
     "Left Stick — Full circle COUNTER-CLOCKWISE",
     120,
     "LEFT STICK — rotate in large counter-clockwise circles.\n"
     "Stay at the outer edge the whole time.\n"
     "One full rotation every ~3 seconds.",
     cue_circle("COUNTER-CLOCKWISE", secs_per_rotation=3)),

    # ── Right Stick Y ─────────────────────────────────────────────────────────
    ("rsty_up",
     "Right Stick Y — Full UP",
     120,
     "RIGHT STICK — vertical axis only.\n"
     "Push fully UP, hold, then return to CENTER.",
     cue_hold_release("RIGHT STICK FULL UP", hold=2, rest=1)),

    ("rsty_down",
     "Right Stick Y — Full DOWN",
     120,
     "RIGHT STICK — vertical axis only.\n"
     "Push fully DOWN, hold, then return to CENTER.",
     cue_hold_release("RIGHT STICK FULL DOWN", hold=2, rest=1)),

    ("rsty_sweep",
     "Right Stick Y — Slow vertical sweep",
     120,
     "RIGHT STICK — vertical axis only.\n"
     "Slide slowly and continuously UP then DOWN, full range.\n"
     "~4 seconds each direction.",
     cue_sweep("UP/DOWN", hold=4)),

    # ── Right Stick X ─────────────────────────────────────────────────────────
    ("rstx_right",
     "Right Stick X — Full RIGHT",
     120,
     "RIGHT STICK — horizontal axis only.\n"
     "Push fully RIGHT, hold, then return to CENTER.",
     cue_hold_release("RIGHT STICK FULL RIGHT", hold=2, rest=1)),

    ("rstx_left",
     "Right Stick X — Full LEFT",
     120,
     "RIGHT STICK — horizontal axis only.\n"
     "Push fully LEFT, hold, then return to CENTER.",
     cue_hold_release("RIGHT STICK FULL LEFT", hold=2, rest=1)),

    ("rstx_sweep",
     "Right Stick X — Slow horizontal sweep",
     120,
     "RIGHT STICK — horizontal axis only.\n"
     "Slide slowly and continuously LEFT then RIGHT, full range.\n"
     "~4 seconds each direction.",
     cue_sweep("LEFT/RIGHT", hold=4)),

    # ── Right Stick circles ───────────────────────────────────────────────────
    ("rst_circle_cw",
     "Right Stick — Full circle CLOCKWISE",
     120,
     "RIGHT STICK — rotate in large clockwise circles.\n"
     "Stay at the outer edge. One full rotation every ~3 seconds.",
     cue_circle("CLOCKWISE", secs_per_rotation=3)),

    ("rst_circle_ccw",
     "Right Stick — Full circle COUNTER-CLOCKWISE",
     120,
     "RIGHT STICK — rotate in large counter-clockwise circles.\n"
     "One full rotation every ~3 seconds.",
     cue_circle("COUNTER-CLOCKWISE", secs_per_rotation=3)),

    # ── Triggers ──────────────────────────────────────────────────────────────
    ("lt_sweep",
     "Left Trigger — Slow analog sweep",
     120,
     "LT ONLY — do not touch anything else.\n"
     "Slowly squeeze from 0 to full, then slowly release back to 0.\n"
     "~4 seconds to fully press, ~4 seconds to fully release.",
     cue_trigger_sweep("LEFT (LT)", hold=4)),

    ("rt_sweep",
     "Right Trigger — Slow analog sweep",
     120,
     "RT ONLY — do not touch anything else.\n"
     "Slowly squeeze from 0 to full, then slowly release back to 0.\n"
     "~4 seconds to fully press, ~4 seconds to fully release.",
     cue_trigger_sweep("RIGHT (RT)", hold=4)),

    ("lt_pump",
     "Left Trigger — Fast pumps",
     60,
     "LT ONLY — quick full presses at ~1 per second.\n"
     "Full press all the way down, then full release. Keep the rhythm.",
     cue_trigger_pump("LT")),

    ("rt_pump",
     "Right Trigger — Fast pumps",
     60,
     "RT ONLY — quick full presses at ~1 per second.\n"
     "Full press all the way down, then full release. Keep the rhythm.",
     cue_trigger_pump("RT")),

    # ── Buttons ───────────────────────────────────────────────────────────────
    ("btn_a",
     "A Button — Isolated",
     90,
     "A BUTTON ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.\n"
     "Do not press any other button.",
     cue_button("A", hold=1.5, rest=1.0)),

    ("btn_b",
     "B Button — Isolated",
     90,
     "B BUTTON ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.",
     cue_button("B", hold=1.5, rest=1.0)),

    ("btn_x",
     "X Button — Isolated",
     90,
     "X BUTTON ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.",
     cue_button("X", hold=1.5, rest=1.0)),

    ("btn_y",
     "Y Button — Isolated",
     90,
     "Y BUTTON ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.",
     cue_button("Y", hold=1.5, rest=1.0)),

    ("btn_lb",
     "Left Bumper (LB) — Isolated",
     90,
     "LB ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.",
     cue_button("LB", hold=1.5, rest=1.0)),

    ("btn_rb",
     "Right Bumper (RB) — Isolated",
     90,
     "RB ONLY.\n"
     "Press and hold for ~1.5 seconds, then fully release for ~1 second.",
     cue_button("RB", hold=1.5, rest=1.0)),

    # ── Combined ──────────────────────────────────────────────────────────────
    ("both_sticks_circles",
     "Both Sticks — Simultaneous circles",
     120,
     "BOTH STICKS at the same time.\n"
     "Left stick clockwise circles, right stick counter-clockwise.\n"
     "Stay at the outer edge on both. One rotation every ~3 seconds.",
     cue_both_circles()),

    ("sticks_and_buttons",
     "Both Sticks + Random RT/LT (Take 1)",
     120,
     "Move BOTH STICKS freely while randomly squeezing RT and LT.\n"
     "Vary timing and pressure naturally. No strict pattern.",
     cue_combined("Move both sticks + random RT/LT squeezes")),

    ("sticks_and_triggers",
     "Both Sticks + Random RT/LT (Take 2)",
     120,
     "Move BOTH STICKS freely while randomly squeezing RT and LT.\n"
     "Vary timing and pressure naturally. No strict pattern.",
     cue_combined("Move both sticks + random RT/LT squeezes")),

    ("natural_play",
     "Natural Play — Freestyle",
     300,
     "USE THE CONTROLLER HOWEVER FEELS NATURAL.\n"
     "Move sticks, press buttons, use triggers — as if playing a real game.\n"
     "5 minutes. No rules. Most realistic training data.",
     cue_combined("Play naturally — sticks, buttons, triggers — anything goes")),

    # ── Rest end ──────────────────────────────────────────────────────────────
    ("rest_end",
     "Rest — Baseline END",
     60,
     "PUT THE CONTROLLER DOWN completely.\n"
     "Do not touch it for 1 minute.",
     cue_rest()),
]

TOTAL_SESSIONS = len(SESSIONS)
TOTAL_SECONDS  = sum(s[2] for s in SESSIONS)


# ── Helpers ───────────────────────────────────────────────────────────────────

def hms(seconds):
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    if h:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s" if m else f"{s}s"


def banner(text, width=66, char="="):
    print("\n" + char * width)
    for line in text.strip().split("\n"):
        print(f"  {line}")
    print(char * width)


def countdown(n=3):
    for i in range(n, 0, -1):
        print(f"  {i}...", end="  ", flush=True)
        time.sleep(1)
    print("GO!\n")


def run_cues(cues, total_seconds, stop_event):
    """Background thread: cycle through cues, printing each with a timer."""
    elapsed = 0.0
    tick    = 0.25   # check stop_event every 0.25s
    while not stop_event.is_set() and elapsed < total_seconds:
        for hold, msg in cues:
            if stop_event.is_set() or elapsed >= total_seconds:
                break
            remaining    = total_seconds - elapsed
            actual_hold  = min(hold, remaining)
            deadline     = elapsed + actual_hold
            while elapsed < deadline and not stop_event.is_set():
                print(f"\r  [{hms(elapsed):>6} / {hms(total_seconds)}]  {msg:<50}",
                      end="", flush=True)
                time.sleep(tick)
                elapsed += tick
    if not stop_event.is_set():
        print(f"\r  [{hms(total_seconds):>6} / {hms(total_seconds)}]  Recording complete.{' '*40}")


def load_log():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE) as f:
            return json.load(f)
    return {"completed": [], "files": []}


def save_log(log):
    with open(LOG_FILE, "w") as f:
        json.dump(log, f, indent=2)


def start_device():
    """Configure and start the device once. Returns the synapse Device object."""
    import synapse
    sig = synapse.SignalConfig()
    for i in range(64):
        ch = sig.electrode.channels.add()
        ch.id = i
    src = synapse.BroadbandSource(
        peripheral_id=108, bit_width=16,
        sample_rate_hz=32000, gain=1.0, signal=sig)
    cfg = synapse.Config()
    cfg.add_node(src)
    d = synapse.Device(DEVICE)
    d.configure(cfg)
    d.start()
    time.sleep(2.0)   # wait for tap to appear
    return d


def record_python(slug, duration, cues, tap_name):
    """
    Record directly via ZMQ tap — no synapsectl, device stays running.
    Saves an HDF5 file to RECORDINGS/run007_<slug>_<ts>.h5/broadband_data_<ts>.h5
    matching the layout synapsectl produces.
    """
    from synapse.client.taps import Tap
    from synapse.api.datatype_pb2 import BroadbandFrame
    import numpy as np
    import h5py
    import concurrent.futures

    ts      = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(RECORDINGS, f"run007_{slug}_{ts}.h5")
    os.makedirs(out_dir, exist_ok=True)
    h5_path = os.path.join(out_dir, f"broadband_data_{ts}.h5")

    # Start cue display BEFORE connecting tap so the user sees instructions
    # even if the tap connection takes a moment.
    stop_event = threading.Event()
    cue_thread = threading.Thread(target=run_cues,
                                  args=(cues, duration, stop_event),
                                  daemon=True)
    cue_thread.start()

    tap = Tap(DEVICE)

    # Connect with a timeout — if it hangs, fail fast instead of freezing.
    connect_ok = threading.Event()
    connect_err = [None]

    def _connect():
        try:
            tap.connect(tap_name)
            connect_ok.set()
        except Exception as e:
            connect_err[0] = e

    conn_thread = threading.Thread(target=_connect, daemon=True)
    conn_thread.start()
    conn_thread.join(timeout=10)

    if not connect_ok.is_set():
        stop_event.set()
        cue_thread.join(timeout=2)
        err = connect_err[0] or "tap.connect timed out after 10s"
        raise RuntimeError(f"Could not connect to tap '{tap_name}': {err}")

    samples = []
    t_start = time.time()
    try:
        for raw in tap.stream(timeout_ms=200):
            if time.time() - t_start >= duration:
                break
            if stop_event.is_set():
                break
            frame = BroadbandFrame()
            frame.ParseFromString(raw)
            if frame.frame_data:
                arr = np.array(list(frame.frame_data), dtype=np.int16)
                n_ch = 76
                n_samp = len(arr) // n_ch
                if n_samp > 0:
                    samples.append(arr[:n_samp * n_ch].reshape(n_samp, n_ch))
    finally:
        stop_event.set()
        cue_thread.join(timeout=2)
        try:
            tap.disconnect()
        except Exception:
            pass

    elapsed = time.time() - t_start

    if samples:
        data = np.concatenate(samples, axis=0)
        with h5py.File(h5_path, 'w') as hf:
            grp = hf.require_group('acquisition')
            ds  = grp.create_dataset('ElectricalSeries', data=data.flatten(),
                                     dtype=np.int16)
            ds.attrs['n_channels'] = 76
            ds.attrs['sample_rate'] = 32000
        n_sec = len(data) / 32000
        print(f"\r  Recorded {len(data):,} samples ({n_sec:.1f}s) → {h5_path}{' '*20}")
        success = True
    else:
        print(f"\r  WARNING: no data received for {slug}{' '*40}")
        success = False

    return out_dir, success, elapsed


def get_tap_name():
    """Query the device for the current tap name, restarting only if needed."""
    from synapse.client.taps import Tap
    tap = Tap(DEVICE)
    taps = tap.list_taps()
    if taps:
        return taps[0].name
    # Device not running — start it
    print("\n  Device not running, restarting...", end="", flush=True)
    start_device()
    for attempt in range(5):
        taps = tap.list_taps()
        if taps:
            print(f" ready ({taps[0].name}).")
            return taps[0].name
        time.sleep(1)
    raise RuntimeError("Could not find tap after starting device")


def record(slug, duration, cues, dry_run=False):
    if dry_run:
        print(f"  [DRY RUN] {slug} — {hms(duration)}")
        time.sleep(0.5)
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        return os.path.join(RECORDINGS, f"run007_{slug}_{ts}.h5"), True, duration

    tap_name = get_tap_name()
    out_dir, success, elapsed = record_python(slug, duration, cues, tap_name)
    return out_dir, success, elapsed


def write_summary(log):
    lines = [
        "# Run 007 — Structured Training v2\n",
        f"**Date:** {datetime.now().strftime('%Y-%m-%d')}  ",
        "**Device:** axiomatic-perch-of-revolution (SFI100118)  ",
        "**Mode:** Hard Training (Peripheral 108)\n",
        "## Sessions\n",
        "| # | Label | Duration | Status |",
        "|---|-------|----------|--------|",
    ]
    for i, (slug, label, dur, _, _) in enumerate(SESSIONS):
        status = "done" if slug in log["completed"] else "skipped"
        lines.append(f"| {i+1:02d} | {label} | {hms(dur)} | {status} |")

    total_done = sum(s[2] for s in SESSIONS if s[0] in log["completed"])
    lines += [
        "",
        f"**Total recorded: {hms(total_done)} across {len(log['completed'])} sessions**",
        "",
        "Raw HDF5 files: `data/recordings/run007_<slug>_<timestamp>.h5/`",
    ]
    summary_path = os.path.join(RUN_DIR, "summary.md")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"\n  Saved: {summary_path}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--start",   type=int, default=1,
                        help="Resume from session number (1-based, default=1)")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    os.makedirs(RECORDINGS, exist_ok=True)
    log = load_log()

    banner(
        f"RUN 007 — STRUCTURED TRAINING\n"
        f"Device  : {DEVICE}\n"
        f"Sessions: {TOTAL_SESSIONS}  |  Total time: ~{hms(TOTAL_SECONDS)}\n"
        f"\nThe script will call out exactly what to do during each recording.\n"
        f"Press ENTER to start each session. Take breaks whenever you need."
    )

    # Start device if not already running
    if not args.dry_run:
        print("\n  Starting device...", end="", flush=True)
        start_device()
        print(" ready.")

    for i, (slug, label, duration, intro, cues) in enumerate(SESSIONS):
        session_num = i + 1
        if session_num < args.start:
            continue
        if slug in log["completed"]:
            print(f"\n  [{session_num:02d}/{TOTAL_SESSIONS}] {label} — already done, skipping")
            continue

        remaining_time = sum(s[2] for s in SESSIONS[i:])

        banner(
            f"SESSION {session_num:02d} of {TOTAL_SESSIONS}  —  {label}\n"
            f"Duration: {hms(duration)}   |   Remaining after this: {hms(remaining_time - duration)}\n"
            f"\n{intro}",
            char="-"
        )

        if not args.dry_run:
            input("\n  Press ENTER when you are in position and ready to start...")
            print()
            countdown(3)

        t_start  = time.time()
        out_path, success, elapsed = record(slug, duration, cues,
                                            dry_run=args.dry_run)
        elapsed  = time.time() - t_start

        if success and not args.dry_run:
            log["completed"].append(slug)
            log["files"].append({
                "slug":      slug,
                "label":     label,
                "path":      out_path,
                "duration":  duration,
                "elapsed":   round(elapsed, 1),
                "timestamp": datetime.now().isoformat(),
            })
            save_log(log)
            print(f"\n  Saved to: {out_path}")
        else:
            print(f"\n  Recording failed. Re-run with --start {session_num} to retry.")

        if session_num < TOTAL_SESSIONS and not args.dry_run:
            input("\n  Press ENTER to continue to the next session (or Ctrl+C to stop).\n")

    banner(
        f"ALL SESSIONS COMPLETE\n"
        f"{len(log['completed'])}/{TOTAL_SESSIONS} sessions recorded"
    )
    write_summary(log)
    print("\n  Next step: python scripts/train_decoder_v7.py\n")


if __name__ == "__main__":
    main()
