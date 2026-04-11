"""Local web server for browser-based SO-101 keyboard teleoperation."""

from __future__ import annotations

import json
import threading
import time
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from socket import timeout as SocketTimeout
from typing import Any

from .config import ArmSettings, load_settings, validate_settings
from .controller import SO101ArmController
from .keyboard_teleop import JointTargetTracker, build_joint_delta_command

HOST = "127.0.0.1"
PORT = 8765

HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>SO-101 Keyboard Teleop</title>
  <style>
    :root {
      --bg: #f2efe8;
      --card: #fffaf0;
      --ink: #1f2a2e;
      --accent: #0b6e4f;
      --muted: #5b676b;
      --line: #d7d0c3;
    }
    body {
      margin: 0;
      font-family: "SF Mono", "IBM Plex Mono", monospace;
      background: linear-gradient(135deg, #f7f2e7, #e7efe9);
      color: var(--ink);
      min-height: 100vh;
      display: grid;
      place-items: center;
    }
    main {
      width: min(820px, calc(100vw - 32px));
      background: var(--card);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: 0 18px 60px rgba(31, 42, 46, 0.14);
      padding: 28px;
    }
    h1 {
      margin-top: 0;
      margin-bottom: 8px;
      font-size: 28px;
    }
    p {
      color: var(--muted);
      line-height: 1.5;
    }
    .grid {
      display: grid;
      grid-template-columns: repeat(auto-fit, minmax(170px, 1fr));
      gap: 12px;
      margin: 20px 0;
    }
    .key {
      border: 1px solid var(--line);
      border-radius: 12px;
      padding: 12px;
      background: white;
    }
    .key strong {
      display: block;
      margin-bottom: 6px;
      color: var(--accent);
    }
    code {
      background: #eef4f1;
      padding: 2px 6px;
      border-radius: 6px;
    }
    #status {
      margin-top: 20px;
      padding: 12px 14px;
      border-radius: 12px;
      background: #eef4f1;
      border: 1px solid #c8ddd4;
    }
    .action-tables {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 12px;
      margin-top: 16px;
    }
    .action-table-wrap {
      border: 1px solid var(--line);
      border-radius: 12px;
      background: white;
      padding: 10px 14px;
    }
    .action-table-wrap h3 {
      margin: 0 0 8px;
      font-size: 13px;
      color: var(--accent);
      text-transform: uppercase;
      letter-spacing: 0.05em;
    }
    .action-table {
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }
    .action-table td {
      padding: 2px 6px;
    }
    .action-table td:first-child {
      color: var(--muted);
    }
    .action-table td:last-child {
      text-align: right;
      font-variant-numeric: tabular-nums;
    }
    .warn {
      color: #8b3a2b;
    }
  </style>
</head>
<body>
  <main>
    <h1>SO-101 Keyboard Teleop Server</h1>
    <p>Click anywhere on this page, keep it focused, and use the keyboard to drive the arm.</p>
    <div class="grid">
      <div class="key"><strong>q / a</strong>shoulder pan</div>
      <div class="key"><strong>w / s</strong>shoulder lift</div>
      <div class="key"><strong>e / d</strong>elbow flex</div>
      <div class="key"><strong>r / f</strong>wrist flex</div>
      <div class="key"><strong>t / g</strong>wrist roll</div>
      <div class="key"><strong>y / h</strong>gripper</div>
      <div class="key"><strong>space</strong>freeze target</div>
      <div class="key"><strong>esc</strong>stop server</div>
    </div>
    <p>Current URL: <code id="url"></code></p>
    <div id="status">Connecting to arm...</div>
    <div class="action-tables">
      <div class="action-table-wrap">
        <h3>Merged (sent to send_action)</h3>
        <table class="action-table" id="merged-table"><tbody></tbody></table>
      </div>
      <div class="action-table-wrap">
        <h3>Sent (returned from send_action)</h3>
        <table class="action-table" id="sent-table"><tbody></tbody></table>
      </div>
    </div>
    <p class="warn">Run this only on a trusted local machine. The server binds to <code>127.0.0.1</code>.</p>
  </main>
  <script>
    const activeKeys = new Set();
    const status = document.getElementById("status");
    document.getElementById("url").textContent = window.location.href;

    function normalizeKey(event) {
      if (event.key === " ") return "space";
      if (event.key === "Escape") return "esc";
      if (event.key.length === 1) return event.key.toLowerCase();
      return null;
    }

    function renderActionTable(tableId, data) {
      const tbody = document.querySelector(`#${tableId} tbody`);
      const keys = Object.keys(data).sort();
      if (keys.length === 0) {
        tbody.innerHTML = "<tr><td colspan='2' style='color:var(--muted)'>—</td></tr>";
        return;
      }
      tbody.innerHTML = keys.map(k =>
        `<tr><td>${k}</td><td>${data[k].toFixed(4)}</td></tr>`
      ).join("");
    }

    async function refreshStatus() {
      const response = await fetch("/api/status");
      const payload = await response.json();
      const prefix = payload.connected
        ? "Arm connected"
        : payload.last_error
          ? "Arm error"
          : "Arm starting";
      const details = payload.last_error ? `: ${payload.last_error}` : "";
      const keySummary = payload.active_keys.length ? ` | keys: ${payload.active_keys.join(", ")}` : "";
      status.textContent = `${prefix}${details}${keySummary}`;
      renderActionTable("merged-table", payload.last_merged || {});
      renderActionTable("sent-table", payload.last_sent || {});
    }

    async function pushKeys() {
      const keys = Array.from(activeKeys).sort();
      const response = await fetch("/api/keys", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ keys }),
      });
      const payload = await response.json();
      status.textContent = payload.message;
    }

    window.addEventListener("keydown", async (event) => {
      const key = normalizeKey(event);
      if (!key) return;
      event.preventDefault();
      activeKeys.add(key);
      if (key === "esc") {
        status.textContent = "Stopping server...";
      }
      await pushKeys();
    });

    window.addEventListener("keyup", async (event) => {
      const key = normalizeKey(event);
      if (!key) return;
      event.preventDefault();
      activeKeys.delete(key);
      await pushKeys();
    });

    window.addEventListener("blur", async () => {
      activeKeys.clear();
      await pushKeys();
      status.textContent = "Focus lost. Keys cleared.";
    });

    refreshStatus();
    setInterval(refreshStatus, 500);
  </script>
</body>
</html>
"""


class BrowserKeyState:
    """Thread-safe state container for browser key presses."""

    def __init__(self) -> None:
        self._keys: set[str] = set()
        self._stop_requested = False
        self._lock = threading.Lock()

    def set_keys(self, keys: set[str]) -> None:
        with self._lock:
            self._keys = set(keys)
            if "esc" in keys:
                self._stop_requested = True

    def snapshot(self) -> set[str]:
        with self._lock:
            return set(self._keys)

    @property
    def should_exit(self) -> bool:
        with self._lock:
            return self._stop_requested


class RunnerStatus:
    """Shared runtime state for the control thread and browser."""

    def __init__(self) -> None:
        self._connected = False
        self._last_error: str | None = None
        self._started = False
        self._last_merged: dict[str, float] = {}
        self._last_sent: dict[str, float] = {}
        self._lock = threading.Lock()

    def mark_starting(self) -> None:
        with self._lock:
            self._started = True
            self._connected = False
            self._last_error = None

    def mark_connected(self) -> None:
        with self._lock:
            self._started = True
            self._connected = True
            self._last_error = None

    def mark_error(self, exc: Exception) -> None:
        with self._lock:
            self._started = True
            self._connected = False
            self._last_error = str(exc)

    def mark_disconnected(self) -> None:
        with self._lock:
            self._connected = False

    def update_action_values(self, merged: dict[str, float], sent: dict[str, float]) -> None:
        with self._lock:
            self._last_merged = dict(merged)
            self._last_sent = dict(sent)

    def snapshot(self, active_keys: set[str]) -> dict[str, Any]:
        with self._lock:
            return {
                "started": self._started,
                "connected": self._connected,
                "last_error": self._last_error,
                "active_keys": sorted(active_keys),
                "last_merged": dict(self._last_merged),
                "last_sent": dict(self._last_sent),
            }


class WebTeleopRunner:
    """Background control loop driven by browser-reported keys."""

    def __init__(
        self,
        controller: SO101ArmController,
        settings: ArmSettings,
        key_state: BrowserKeyState,
        status: RunnerStatus,
        sleeper: Any = time.sleep,
    ) -> None:
        self.controller = controller
        self.settings = settings
        self.key_state = key_state
        self.status = status
        self.sleeper = sleeper
        self.tracker = JointTargetTracker(settings)
        self._thread: threading.Thread | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run_loop, name="web-teleop-loop", daemon=True)
        self._thread.start()

    def join(self) -> None:
        if self._thread is not None:
            self._thread.join()
        if self._error is not None:
            raise self._error

    def _run_loop(self) -> None:
        try:
            self.status.mark_starting()
            validate_settings(self.settings)
            self.controller.connect()
            self.status.mark_connected()
            initial_observation = self.controller.get_observation()
            self.tracker.seed(initial_observation)
            loop_dt = 1.0 / self.settings.loop_hz

            while not self.key_state.should_exit:
                observation = self.controller.get_observation()
                pressed = self.key_state.snapshot()
                command = build_joint_delta_command(pressed, self.settings)
                targets = self.tracker.next_targets(observation, command)
                sent = self.controller.send_joint_targets(targets)
                self.status.update_action_values(dict(targets), sent)
                self.sleeper(loop_dt)
        except Exception as exc:
            self._error = exc
            self.status.mark_error(exc)
            self.key_state.set_keys({"esc"})
        finally:
            self.controller.disconnect()
            self.status.mark_disconnected()

    def wait_until_ready(self, timeout_s: float = 2.0) -> None:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            if self._error is not None:
                raise self._error
            if self.status.snapshot(self.key_state.snapshot())["connected"]:
                return
            time.sleep(0.05)
        if self._error is not None:
            raise self._error
        raise TimeoutError("Timed out waiting for the arm control loop to connect.")


def make_handler(key_state: BrowserKeyState, status: RunnerStatus) -> type[BaseHTTPRequestHandler]:
    class WebTeleopHandler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/api/status":
                payload = json.dumps(status.snapshot(key_state.snapshot())).encode("utf-8")
                self.send_response(HTTPStatus.OK)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if self.path != "/":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return
            payload = HTML.encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)

        def do_POST(self) -> None:
            if self.path != "/api/keys":
                self.send_error(HTTPStatus.NOT_FOUND, "Not found")
                return

            length = int(self.headers.get("Content-Length", "0"))
            raw = self.rfile.read(length)
            data = json.loads(raw.decode("utf-8") or "{}")
            keys = {str(key).lower() for key in data.get("keys", [])}
            key_state.set_keys(keys)

            status_snapshot = status.snapshot(keys)
            if status_snapshot["last_error"]:
                message = f"Arm error: {status_snapshot['last_error']}"
            elif status_snapshot["connected"]:
                message = "Arm connected | Active keys: " + (", ".join(sorted(keys)) if keys else "(none)")
            else:
                message = "Arm starting | Active keys: " + (", ".join(sorted(keys)) if keys else "(none)")
            response = json.dumps({"message": message, **status_snapshot}).encode("utf-8")
            self.send_response(HTTPStatus.OK)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(response)))
            self.end_headers()
            self.wfile.write(response)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return WebTeleopHandler


def serve(host: str = HOST, port: int = PORT) -> int:
    settings = load_settings()
    key_state = BrowserKeyState()
    status = RunnerStatus()
    controller = SO101ArmController(settings=settings)
    runner = WebTeleopRunner(controller=controller, settings=settings, key_state=key_state, status=status)

    server = ThreadingHTTPServer((host, port), make_handler(key_state, status))
    server.timeout = 0.5
    runner.start()
    runner.wait_until_ready()
    print(f"SO-101 web teleop server running at http://{host}:{port}")
    print("Open that URL in a browser, focus the page, and use the keyboard controls.")
    print("Press Esc in the browser page to stop.")

    try:
        while not key_state.should_exit:
            try:
                server.handle_request()
            except SocketTimeout:
                continue
    finally:
        server.server_close()
        key_state.set_keys({"esc"})
        runner.join()
    return 0


def main() -> int:
    return serve()


if __name__ == "__main__":
    raise SystemExit(main())
