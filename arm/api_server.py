"""FastAPI server for high-level SO-101 arm control."""

from __future__ import annotations

import argparse
import logging
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field

from .api import EndEffectorDeltaResult, SO101ArmAPI
from .commands import EndEffectorDeltaCommand
from .config import ArmSettings, load_settings
from .controller import SO101ArmController
from .ik import ArmPose, SO101IKTranslator

logger = logging.getLogger(__name__)

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000
DEFAULT_URDF_PATH = Path("models/so101.urdf")
KEYBOARD_UI_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 FastAPI Control</title>
  <style>
    :root {
      --bg: #f3efe7;
      --panel: rgba(255, 251, 244, 0.9);
      --ink: #1d2528;
      --muted: #617177;
      --line: #d4cab8;
      --accent: #005f73;
      --accent-2: #bb3e03;
      --ok: #1b7f4d;
    }
    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-height: 100vh;
      color: var(--ink);
      font-family: "IBM Plex Mono", "SF Mono", monospace;
      background:
        radial-gradient(circle at top left, rgba(0, 95, 115, 0.12), transparent 28z),
        radial-gradient(circle at bottom right, rgba(187, 62, 3, 0.14), transparent 34%),
        linear-gradient(135deg, #f8f4eb, #ece7dc 55%, #f4efe6);
      padding: 24px;
    }
    main {
      max-width: 980px;
      margin: 0 auto;
      display: grid;
      gap: 18px;
    }
    .panel {
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 50px rgba(29, 37, 40, 0.08);
      padding: 20px;
      backdrop-filter: blur(8px);
    }
    h1, h2 { margin: 0 0 10px; }
    p { margin: 0; color: var(--muted); line-height: 1.55; }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(210px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    .card {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.72);
    }
    .card strong {
      display: block;
      margin-bottom: 8px;
      color: var(--accent);
    }
    .row {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(160px, 1fr));
      gap: 12px;
      margin-top: 16px;
    }
    label {
      display: grid;
      gap: 6px;
      color: var(--muted);
      font-size: 13px;
    }
    input {
      width: 100%;
      border: 1px solid var(--line);
      border-radius: 10px;
      padding: 10px 12px;
      background: white;
      color: var(--ink);
      font: inherit;
    }
    code, pre {
      font-family: inherit;
    }
    #status {
      margin-top: 14px;
      padding: 12px 14px;
      border-radius: 12px;
      background: rgba(0, 95, 115, 0.08);
      color: var(--accent);
      border: 1px solid rgba(0, 95, 115, 0.16);
    }
    #status.error {
      background: rgba(187, 62, 3, 0.08);
      color: var(--accent-2);
      border-color: rgba(187, 62, 3, 0.16);
    }
    #status.ok {
      background: rgba(27, 127, 77, 0.08);
      color: var(--ok);
      border-color: rgba(27, 127, 77, 0.16);
    }
    .pillbar {
      display: flex;
      flex-wrap: wrap;
      gap: 8px;
      margin-top: 12px;
      min-height: 34px;
    }
    .pill {
      border-radius: 999px;
      border: 1px solid var(--line);
      padding: 6px 10px;
      background: white;
      color: var(--ink);
      font-size: 13px;
    }
    .debug {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
      gap: 12px;
      margin-bottom: 14px;
    }
    .debug-box {
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 12px;
      background: rgba(255, 255, 255, 0.72);
    }
    .debug-box strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
      font-size: 13px;
    }
    .debug-box span {
      font-size: 22px;
      line-height: 1.1;
    }
    .target-grid {
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px 14px;
    }
    .target-pair {
      display: grid;
      gap: 2px;
    }
    .target-pair label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .target-pair code {
      font-size: 14px;
    }
    pre {
      margin: 0;
      padding: 14px;
      border-radius: 14px;
      background: #192226;
      color: #e8f0ef;
      overflow: auto;
      min-height: 180px;
    }
    .callout {
      color: var(--accent-2);
      margin-top: 10px;
    }
    .viz-row {
      display: grid;
      grid-template-columns: auto auto 1fr;
      gap: 16px;
      align-items: start;
      margin-top: 14px;
    }
    .arm-viz {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: rgba(255,255,255,0.88);
      display: block;
    }
    .viz-label {
      font-size: 11px;
      color: var(--muted);
      text-transform: uppercase;
      letter-spacing: 0.04em;
      margin-bottom: 6px;
    }
    .pose-coord-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
      margin-bottom: 10px;
    }
    .pose-coord-table th {
      text-align: left;
      color: var(--muted);
      font-weight: normal;
      padding: 2px 10px 2px 0;
      font-size: 11px;
      text-transform: uppercase;
      letter-spacing: 0.04em;
    }
    .pose-coord-table td {
      padding: 3px 10px 3px 0;
      font-variant-numeric: tabular-nums;
    }
    .viz-legend {
      display: flex;
      gap: 14px;
      font-size: 11px;
      flex-wrap: wrap;
      margin-top: 10px;
      color: var(--muted);
    }
    .legend-item { display: flex; align-items: center; gap: 5px; }
    .legend-dot { width: 9px; height: 9px; border-radius: 50%; flex-shrink: 0; }
  </style>
</head>
<body>
  <main>
    <section class="panel">
      <h1>SO-101 Keyboard Command UI</h1>
      <p>Click this page, keep it focused, and use the keyboard to send mixed reach, shoulder-pan, and end-effector delta commands to <code>/api/end-effector-delta</code>.</p>
      <div class="grid">
        <div class="card"><strong>Move X</strong><code>w</code> positive, <code>s</code> negative forward reach</div>
        <div class="card"><strong>Move Y</strong><code>a</code> negative, <code>d</code> positive shoulder pan</div>
        <div class="card"><strong>Move Z</strong><code>i</code> positive, <code>k</code> negative</div>
        <div class="card"><strong>Tool Roll</strong><code>j</code> negative, <code>l</code> positive</div>
        <div class="card"><strong>Gripper</strong><code>u</code> open, <code>o</code> close</div>
        <div class="card"><strong>Send</strong>Commands stream while keys are held</div>
      </div>
      <div class="row">
        <label>Reach/Z step (meters)
          <input id="linearStep" type="number" step="0.001" value="__LINEAR_STEP__">
        </label>
        <label>Shoulder pan step (radians)
          <input id="panStep" type="number" step="0.01" value="__PAN_STEP__">
        </label>
        <label>Roll step (degrees)
          <input id="rollStep" type="number" step="0.1" value="__ROLL_STEP__">
        </label>
        <label>Jaw step
          <input id="jawStep" type="number" step="0.1" value="__JAW_STEP__">
        </label>
        <label>Repeat interval (ms)
          <input id="repeatMs" type="number" step="10" min="20" value="__REPEAT_MS__">
        </label>
      </div>
      <div id="status">Idle. Press and hold control keys to send commands.</div>
      <div class="pillbar" id="keys"></div>
      <p class="callout">This UI binds only to the same FastAPI server process. Use <code>--dry-run</code> first before sending commands to real hardware.</p>
    </section>
    <section class="panel">
      <h2>Command + Targets</h2>
      <div class="debug">
        <div class="debug-box">
          <strong>Browser Command</strong>
          <pre id="commandPayload">{}</pre>
        </div>
        <div class="debug-box">
          <strong>LeRobot Send / Merged State</strong>
          <div class="target-grid" id="targets">
            <div class="target-pair"><label>Shoulder Pan</label><code id="sent-shoulder_pan">send: --</code><code id="merged-shoulder_pan">merged: --</code></div>
            <div class="target-pair"><label>Shoulder Lift</label><code id="sent-shoulder_lift">send: --</code><code id="merged-shoulder_lift">merged: --</code></div>
            <div class="target-pair"><label>Elbow Flex</label><code id="sent-elbow_flex">send: --</code><code id="merged-elbow_flex">merged: --</code></div>
            <div class="target-pair"><label>Wrist Flex</label><code id="sent-wrist_flex">send: --</code><code id="merged-wrist_flex">merged: --</code></div>
            <div class="target-pair"><label>Wrist Roll</label><code id="sent-wrist_roll">send: --</code><code id="merged-wrist_roll">merged: --</code></div>
            <div class="target-pair"><label>Gripper</label><code id="sent-gripper">send: --</code><code id="merged-gripper">merged: --</code></div>
          </div>
        </div>
      </div>
      <h2>Last Response</h2>
      <pre id="response">{}</pre>
    </section>
    <section class="panel">
      <h2>Pose Visualizer</h2>
      <p>Forward kinematics of the last command targets.
         <span style="color:#005f73">Merged</span> = what we commanded &mdash;
         <span style="color:#bb3e03">Sent</span> = what LeRobot accepted &mdash;
         <span style="color:#1b7f4d">Live</span> = current hardware FK (polls <code>/api/pose</code>).
      </p>
      <div class="viz-row">
        <div>
          <div class="viz-label">Side view (X&ndash;Z, mm)</div>
          <svg id="viz-xz" class="arm-viz" viewBox="0 0 280 260" width="280" height="260"></svg>
        </div>
        <div>
          <div class="viz-label">Top view (X&ndash;Y, mm)</div>
          <svg id="viz-xy" class="arm-viz" viewBox="0 0 260 260" width="260" height="260"></svg>
        </div>
        <div>
          <div class="viz-label">Coordinates (m)</div>
          <table class="pose-coord-table">
            <thead>
              <tr><th></th><th>x</th><th>y</th><th>z</th><th>roll°</th></tr>
            </thead>
            <tbody>
              <tr>
                <td style="color:#1b7f4d;padding-right:10px">live</td>
                <td id="c-live-x">—</td><td id="c-live-y">—</td>
                <td id="c-live-z">—</td><td id="c-live-r">—</td>
              </tr>
              <tr>
                <td style="color:#005f73;padding-right:10px">merged</td>
                <td id="c-merged-x">—</td><td id="c-merged-y">—</td>
                <td id="c-merged-z">—</td><td id="c-merged-r">—</td>
              </tr>
              <tr>
                <td style="color:#bb3e03;padding-right:10px">sent</td>
                <td id="c-sent-x">—</td><td id="c-sent-y">—</td>
                <td id="c-sent-z">—</td><td id="c-sent-r">—</td>
              </tr>
            </tbody>
          </table>
          <div class="viz-legend">
            <div class="legend-item"><div class="legend-dot" style="background:#1b7f4d"></div>Live (hardware)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#005f73"></div>Merged (commanded)</div>
            <div class="legend-item"><div class="legend-dot" style="background:#bb3e03"></div>Sent (returned)</div>
          </div>
        </div>
      </div>
    </section>
  </main>
  <script>
    const activeKeys = new Set();
    const keyPills = document.getElementById("keys");
    const status = document.getElementById("status");
    const commandPayload = document.getElementById("commandPayload");
    const response = document.getElementById("response");
    const targetFields = {
      shoulder_pan: {
        sent: document.getElementById("sent-shoulder_pan"),
        merged: document.getElementById("merged-shoulder_pan"),
      },
      shoulder_lift: {
        sent: document.getElementById("sent-shoulder_lift"),
        merged: document.getElementById("merged-shoulder_lift"),
      },
      elbow_flex: {
        sent: document.getElementById("sent-elbow_flex"),
        merged: document.getElementById("merged-elbow_flex"),
      },
      wrist_flex: {
        sent: document.getElementById("sent-wrist_flex"),
        merged: document.getElementById("merged-wrist_flex"),
      },
      wrist_roll: {
        sent: document.getElementById("sent-wrist_roll"),
        merged: document.getElementById("merged-wrist_roll"),
      },
      gripper: {
        sent: document.getElementById("sent-gripper"),
        merged: document.getElementById("merged-gripper"),
      },
    };
    let timerId = null;

    const mappings = {
      w: ["dx", 1],
      s: ["dx", -1],
      a: ["dy", -1],
      d: ["dy", 1],
      i: ["dz", 1],
      k: ["dz", -1],
      j: ["d_rot", 1],
      l: ["d_rot", -1],
      u: ["d_jaw", 1],
      o: ["d_jaw", -1],
    };

    function readFloat(id) {
      const value = Number(document.getElementById(id).value);
      return Number.isFinite(value) ? value : 0;
    }

    function renderKeys() {
      const keys = Array.from(activeKeys).sort();
      keyPills.innerHTML = "";
      if (keys.length === 0) {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = "No active keys";
        keyPills.appendChild(pill);
        return;
      }
      for (const key of keys) {
        const pill = document.createElement("div");
        pill.className = "pill";
        pill.textContent = key;
        keyPills.appendChild(pill);
      }
    }

    function formatTargetValue(value) {
      if (!Number.isFinite(value)) return "--";
      return value.toFixed(3);
    }

    function renderTargets(sentTargets, mergedTargets) {
      for (const [joint, fields] of Object.entries(targetFields)) {
        fields.sent.textContent = `send: ${formatTargetValue(sentTargets?.[joint])}`;
        fields.merged.textContent = `merged: ${formatTargetValue(mergedTargets?.[joint])}`;
      }
    }

    function buildCommand() {
      const linearStep = readFloat("linearStep");
      const panStep = readFloat("panStep");
      const rollStep = readFloat("rollStep");
      const jawStep = readFloat("jawStep");
      const command = {dx: 0, dy: 0, dz: 0, d_rot: 0, d_jaw: 0};

      for (const key of activeKeys) {
        const mapping = mappings[key];
        if (!mapping) continue;
        const [axis, direction] = mapping;
        const step =
          axis === "dy" ? panStep : axis === "d_rot" ? rollStep : axis === "d_jaw" ? jawStep : linearStep;
        command[axis] += direction * step;
      }
      return command;
    }

    async function sendCommand() {
      if (activeKeys.size === 0) return;
      const command = buildCommand();
      try {
        const res = await fetch("/api/end-effector-delta", {
          method: "POST",
          headers: {"Content-Type": "application/json"},
          body: JSON.stringify(command),
        });
        const payload = await res.json();
        commandPayload.textContent = JSON.stringify(payload.command ?? command, null, 2);
        renderTargets(payload.sent_targets, payload.merged_targets);
        response.textContent = JSON.stringify(payload, null, 2);
        status.className = "ok";
        status.textContent = `Sent ${JSON.stringify(command)}`;
        if (payload.merged_pose) { vizState.merged = payload.merged_pose; setPoseCoords("merged", payload.merged_pose); }
        if (payload.sent_pose)   { vizState.sent   = payload.sent_pose;   setPoseCoords("sent",   payload.sent_pose);   }
        refreshVizDots();
      } catch (error) {
        status.className = "error";
        status.textContent = `Request failed: ${error}`;
      }
    }

    function startLoop() {
      if (timerId !== null) return;
      sendCommand();
      const interval = Math.max(20, readFloat("repeatMs"));
      timerId = window.setInterval(sendCommand, interval);
    }

    function stopLoop() {
      if (timerId !== null) {
        window.clearInterval(timerId);
        timerId = null;
      }
    }

    function normalizeKey(event) {
      if (event.key.length === 1) return event.key.toLowerCase();
      return null;
    }

    window.addEventListener("keydown", (event) => {
      const key = normalizeKey(event);
      if (!(key in mappings)) return;
      event.preventDefault();
      activeKeys.add(key);
      renderKeys();
      startLoop();
    });

    window.addEventListener("keyup", (event) => {
      const key = normalizeKey(event);
      if (!(key in mappings)) return;
      event.preventDefault();
      activeKeys.delete(key);
      renderKeys();
      if (activeKeys.size === 0) {
        stopLoop();
        status.className = "";
        status.textContent = "Idle. Press and hold control keys to send commands.";
      }
    });

    window.addEventListener("blur", () => {
      activeKeys.clear();
      renderKeys();
      stopLoop();
      status.className = "";
      status.textContent = "Focus lost. Active keys cleared.";
    });

    renderKeys();

    // ── Pose Visualizer ──────────────────────────────────────────────────────
    const vizState = { live: null, merged: null, sent: null };
    const SVG_NS = "http://www.w3.org/2000/svg";

    // Physical bounds and SVG canvas sizes (all in meters)
    const XZ = { id: "viz-xz", W: 280, H: 260, xMin: -0.05, xMax: 0.46, yMin: -0.08, yMax: 0.50 };
    const XY = { id: "viz-xy", W: 260, H: 260, xMin: -0.05, xMax: 0.46, yMin: -0.26, yMax: 0.26 };

    function toSvg(b, u, v) {
      return {
        x: (u - b.xMin) / (b.xMax - b.xMin) * b.W,
        y: (1 - (v - b.yMin) / (b.yMax - b.yMin)) * b.H,
      };
    }

    function mkEl(tag, attrs) {
      const el = document.createElementNS(SVG_NS, tag);
      for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, String(v));
      return el;
    }
    function addEl(parent, tag, attrs) { const el = mkEl(tag, attrs); parent.appendChild(el); return el; }

    function buildVizGrid(b, hAxis, vAxis) {
      const svg = document.getElementById(b.id);
      if (!svg) return;
      const {W, H, xMin, xMax, yMin, yMax} = b;
      const STEP_MM = 100;

      // Grid lines
      for (let mm = Math.ceil(xMin * 1000 / STEP_MM) * STEP_MM; mm <= xMax * 1000 + 0.5; mm += STEP_MM) {
        const {x} = toSvg(b, mm / 1000, 0);
        addEl(svg, "line", {x1: x, y1: 0, x2: x, y2: H, stroke: "#e2d8cc", "stroke-width": 0.8});
      }
      for (let mm = Math.ceil(yMin * 1000 / STEP_MM) * STEP_MM; mm <= yMax * 1000 + 0.5; mm += STEP_MM) {
        const {y} = toSvg(b, 0, mm / 1000);
        addEl(svg, "line", {x1: 0, y1: y, x2: W, y2: y, stroke: "#e2d8cc", "stroke-width": 0.8});
      }

      // Max-reach guide (~420mm)
      const R = 0.42;
      if (b.id === "viz-xz") {
        let d = "";
        for (let i = 0; i <= 40; i++) {
          const a = (i / 40) * Math.PI / 2;
          const {x, y} = toSvg(b, R * Math.cos(a), R * Math.sin(a));
          d += (i === 0 ? "M" : "L") + `${x.toFixed(1)},${y.toFixed(1)} `;
        }
        addEl(svg, "path", {d, stroke: "#c4d9d0", "stroke-width": 1.5, fill: "none", "stroke-dasharray": "5 3"});
      } else {
        const c = toSvg(b, 0, 0);
        const rPx = (R / (xMax - xMin)) * W;
        addEl(svg, "circle", {cx: c.x, cy: c.y, r: rPx, stroke: "#c4d9d0", "stroke-width": 1.5, fill: "none", "stroke-dasharray": "5 3"});
      }

      // Axes
      const o = toSvg(b, 0, 0);
      addEl(svg, "line", {x1: 0, y1: o.y, x2: W, y2: o.y, stroke: "#a89880", "stroke-width": 0.8});
      addEl(svg, "line", {x1: o.x, y1: 0, x2: o.x, y2: H, stroke: "#a89880", "stroke-width": 0.8});

      // Base marker
      addEl(svg, "circle", {cx: o.x, cy: o.y, r: 4, fill: "#1f2a2e"});

      // Tick labels on x-axis
      const ts = "font-size:9;fill:#9a8e80;font-family:monospace";
      for (let mm = STEP_MM; mm <= xMax * 1000; mm += STEP_MM) {
        const {x} = toSvg(b, mm / 1000, 0);
        const ty = Math.min(o.y + 12, H - 2);
        const lbl = mkEl("text", {"text-anchor": "middle", style: ts, x: x.toFixed(1), y: ty.toFixed(1)});
        lbl.textContent = mm;
        svg.appendChild(lbl);
      }

      // Axis labels
      const as = "font-size:10;fill:#617177;font-family:monospace";
      const xLab = mkEl("text", {"text-anchor": "end", style: as, x: W - 3, y: (o.y - 4).toFixed(1)});
      xLab.textContent = hAxis;
      svg.appendChild(xLab);
      const yLab = mkEl("text", {"text-anchor": "start", style: as, x: (o.x + 3).toFixed(1), y: 11});
      yLab.textContent = vAxis;
      svg.appendChild(yLab);

      // Dots (on top of everything else)
      for (const [key, color] of [["live","#1b7f4d"],["merged","#005f73"],["sent","#bb3e03"]]) {
        addEl(svg, "circle", {
          id: `${b.id}-dot-${key}`, r: key === "live" ? 5 : 7,
          fill: color, stroke: "white", "stroke-width": 1.5,
          opacity: 0.9, visibility: "hidden", cx: -999, cy: -999,
        });
      }
    }

    function moveDot(b, key, pose, vExtractor) {
      const dot = document.getElementById(`${b.id}-dot-${key}`);
      if (!dot) return;
      if (!pose) { dot.setAttribute("visibility", "hidden"); return; }
      dot.setAttribute("visibility", "visible");
      const {x, y} = toSvg(b, pose.x, vExtractor(pose));
      dot.setAttribute("cx", x.toFixed(1));
      dot.setAttribute("cy", y.toFixed(1));
    }

    function refreshVizDots() {
      for (const key of ["live", "merged", "sent"]) {
        moveDot(XZ, key, vizState[key], p => p.z);
        moveDot(XY, key, vizState[key], p => p.y);
      }
    }

    function setPoseCoords(key, pose) {
      const fmt = v => (typeof v === "number") ? v.toFixed(3) : "—";
      document.getElementById(`c-${key}-x`).textContent = fmt(pose?.x);
      document.getElementById(`c-${key}-y`).textContent = fmt(pose?.y);
      document.getElementById(`c-${key}-z`).textContent = fmt(pose?.z);
      document.getElementById(`c-${key}-r`).textContent = fmt(pose?.tool_roll);
    }

    async function pollLivePose() {
      try {
        const res = await fetch("/api/pose");
        if (!res.ok) return;
        const pose = await res.json();
        vizState.live = pose;
        setPoseCoords("live", pose);
        refreshVizDots();
      } catch (_) { /* server may not be ready yet */ }
    }

    buildVizGrid(XZ, "x →", "z ↑");
    buildVizGrid(XY, "x →", "y ↑");
    setInterval(pollLivePose, 500);
    pollLivePose();
  </script>
</body>
</html>
"""


def render_keyboard_ui_html(settings: ArmSettings) -> str:
    step_sizes = settings.api_ui_step_sizes
    return (
        KEYBOARD_UI_HTML_TEMPLATE.replace("__LINEAR_STEP__", str(step_sizes["linear"]))
        .replace("__PAN_STEP__", str(step_sizes["pan"]))
        .replace("__ROLL_STEP__", str(step_sizes["roll"]))
        .replace("__JAW_STEP__", str(step_sizes["jaw"]))
        .replace("__REPEAT_MS__", str(max(20, settings.api_ui_repeat_ms)))
    )


class EndEffectorDeltaRequest(BaseModel):
    dx: float = Field(default=0.0)
    dy: float = Field(default=0.0)
    dz: float = Field(default=0.0)
    d_rot: float = Field(default=0.0)
    d_jaw: float = Field(default=0.0)


class JointPositionsResponse(BaseModel):
    shoulder_pan: float
    shoulder_lift: float
    elbow_flex: float
    wrist_flex: float
    wrist_roll: float
    gripper: float


class ArmPoseResponse(BaseModel):
    x: float
    y: float
    z: float
    tool_roll: float


class EndEffectorDeltaResponse(BaseModel):
    command: EndEffectorDeltaRequest
    sent_targets: dict[str, float]
    merged_targets: JointPositionsResponse
    sent_pose: ArmPoseResponse | None = None
    merged_pose: ArmPoseResponse | None = None


@dataclass
class ArmServerRuntime:
    arm_api: SO101ArmAPI


def build_arm_api(settings: ArmSettings, urdf_path: str | None = None) -> SO101ArmAPI:
    resolved_urdf_path = urdf_path
    if resolved_urdf_path is None and DEFAULT_URDF_PATH.exists():
        resolved_urdf_path = str(DEFAULT_URDF_PATH)

    controller = SO101ArmController(settings=settings)
    ik_translator = SO101IKTranslator(
        urdf_path=resolved_urdf_path,
        use_degrees=settings.use_degrees,
        joint_names=settings.ik_joint_names,
        end_effector_link=settings.ik_end_effector_link,
    )
    return SO101ArmAPI(
        controller=controller,
        ik_translator=ik_translator,
        smoothing_alpha=settings.smoothing_alpha,
    )


def create_app(arm_api: SO101ArmAPI, settings: ArmSettings | None = None) -> FastAPI:
    settings = settings or ArmSettings()
    runtime = ArmServerRuntime(arm_api=arm_api)
    keyboard_ui_html = render_keyboard_ui_html(settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI):
        runtime.arm_api.connect()
        try:
            yield
        finally:
            runtime.arm_api.disconnect()

    app = FastAPI(title="SO-101 Arm API", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    def index() -> str:
        return keyboard_ui_html

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/api/joints", response_model=JointPositionsResponse)
    def get_joint_positions() -> JointPositionsResponse:
        return JointPositionsResponse(**runtime.arm_api.get_joint_positions())

    @app.get("/api/pose", response_model=ArmPoseResponse)
    def get_end_effector_pose() -> ArmPoseResponse:
        pose = runtime.arm_api.get_end_effector_pose()
        return _pose_response(pose)

    @app.post("/api/end-effector-delta", response_model=EndEffectorDeltaResponse)
    def send_end_effector_delta(
        command: EndEffectorDeltaRequest,
    ) -> EndEffectorDeltaResponse:
        result = runtime.arm_api.send_end_effector_delta(
            EndEffectorDeltaCommand(
                dx=command.dx,
                dy=command.dy,
                dz=command.dz,
                d_rot=command.d_rot,
                d_jaw=command.d_jaw,
            )
        )
        logger.info(
            "API /api/end-effector-delta sent_targets=%s merged_targets=%s",
            result.sent_targets,
            result.merged_targets,
        )
        sent_pose = _targets_to_pose(runtime.arm_api.ik_translator, result.sent_targets)
        merged_pose = _targets_to_pose(
            runtime.arm_api.ik_translator, result.merged_targets
        )
        return EndEffectorDeltaResponse(
            command=command,
            sent_targets=result.sent_targets,
            merged_targets=JointPositionsResponse(**result.merged_targets),
            sent_pose=sent_pose,
            merged_pose=merged_pose,
        )

    return app


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run a FastAPI server for high-level SO-101 arm control."
    )
    parser.add_argument("--host", default=DEFAULT_HOST, help="Host interface to bind.")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to bind.")
    parser.add_argument(
        "--urdf-path",
        help="Optional local path to so101.urdf. Defaults to models/so101.urdf if present.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run without hardware by using the controller dry-run mode.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    import uvicorn

    logging.basicConfig(level=logging.INFO)

    parser = build_parser()
    args = parser.parse_args(argv)

    settings = load_settings()
    if args.dry_run:
        settings = settings.with_overrides({"dry_run": True})

    app = create_app(
        build_arm_api(settings=settings, urdf_path=args.urdf_path), settings=settings
    )
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def _pose_response(pose: ArmPose) -> ArmPoseResponse:
    return ArmPoseResponse(
        x=pose.x,
        y=pose.y,
        z=pose.z,
        tool_roll=pose.tool_roll,
    )


def _targets_to_pose(
    ik_translator: SO101IKTranslator, targets: dict[str, float]
) -> ArmPoseResponse | None:
    """Run forward kinematics on a joint target dict and return the end-effector pose."""
    try:
        obs = {f"{k}.pos": v for k, v in targets.items()}
        return _pose_response(ik_translator.forward(obs))
    except Exception:
        return None


if __name__ == "__main__":
    raise SystemExit(main())
