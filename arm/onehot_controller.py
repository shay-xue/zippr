"""
One-hot vector → SO-101 arm controller.

Input: a vector of length 12 where exactly one element is non-zero.
The POSITION of the active element selects the controller input.
The VALUE of the active element carries direction and intensity:
  - Analog sticks: signed value in [-32767, +32767] → direction + magnitude
  - Buttons / bumpers / triggers: 0 or 32767 → binary on/off

Channel layout (matching the BCI decoder output):

  Analog axes
  ────────────
  LEFT_STICK_X    Left stick horizontal   [-32767, +32767]  → x   (pan left/right)
  LEFT_STICK_Y    Left stick vertical     [-32767, +32767]  → z   (up/down)
  RIGHT_STICK_X   Right stick horizontal  [-32767, +32767]  → θ   (rotate CCW/CW)
  RIGHT_STICK_Y   Right stick vertical    [-32767, +32767]  → y   (reach fwd/back)

  Buttons (unused — mapped to no-op)
  ───────
  BUTTON_A / BUTTON_B / BUTTON_X / BUTTON_Y

  Bumpers (unused — mapped to no-op)
  ───────
  BUMPER_LEFT / BUMPER_RIGHT

  Triggers
  ────────
  TRIGGER_LEFT    Left trigger            0 / 32767         → gripper open
  TRIGGER_RIGHT   Right trigger           0 / 32767         → gripper close
"""

from __future__ import annotations

import enum
import logging
import time
from dataclasses import dataclass
from typing import Iterator

import numpy as np

from .api import SO101ArmAPI
from .commands import EndEffectorDeltaCommand
from .config import ArmSettings, load_settings
from .controller import SO101ArmController
from .ik import SO101IKTranslator

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Channel enum — the single source of truth for vector positions.
# Names match the physical controller layout; numeric values are vector indices.
# ---------------------------------------------------------------------------

class Channel(enum.IntEnum):
    """One-hot vector index for each controller input."""

    # Analog sticks (signed, bidirectional)
    LEFT_STICK_X  = 0
    LEFT_STICK_Y  = 1
    RIGHT_STICK_X = 2
    RIGHT_STICK_Y = 3

    # Face buttons (binary, unused for arm control)
    BUTTON_A = 4
    BUTTON_B = 5
    BUTTON_X = 6
    BUTTON_Y = 7

    # Bumpers (binary, unused for arm control)
    BUMPER_LEFT  = 8
    BUMPER_RIGHT = 9

    # Triggers (binary)
    TRIGGER_LEFT  = 10
    TRIGGER_RIGHT = 11


NUM_CHANNELS = len(Channel)

# Raw analog stick range from the controller hardware
_RAW_MAX = 32767.0

# ---------------------------------------------------------------------------
# Step magnitudes — mirror the defaults from ArmSettings.api_ui_step_sizes
# ---------------------------------------------------------------------------
_LINEAR_STEP: float = 0.03   # metres  (dx, dz)
_PAN_STEP:    float = 0.10   # radians (dy)
_ROLL_STEP:   float = 5.0    # degrees (d_rot)
_JAW_STEP:    float = 5.0    # jaw units (d_jaw)


# ---------------------------------------------------------------------------
# Abstract action — normalised, unit-free intermediate representation
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class AbstractAction:
    """Normalised control intent. All fields in [-1, 1]. Zero = no motion."""

    x:       float = 0.0   # shoulder pan   (left-stick horizontal)
    y:       float = 0.0   # radial reach   (right-stick vertical)
    z:       float = 0.0   # vertical       (left-stick vertical)
    theta:   float = 0.0   # tool roll      (right-stick horizontal)
    gripper: float = 0.0   # +1 open, -1 close

    def is_null(self) -> bool:
        return all(
            v == 0.0 for v in (self.x, self.y, self.z, self.theta, self.gripper)
        )


# ---------------------------------------------------------------------------
# Mapping module
# ---------------------------------------------------------------------------

def decode_vector(vec: list[float] | np.ndarray) -> AbstractAction:
    """Convert a one-hot controller vector into an AbstractAction.

    Parameters
    ----------
    vec:
        Array of length 12.  Exactly one element is non-zero.
        For analog sticks the value is signed (direction + intensity).
        For buttons/triggers the value is 0 or 32767.

    Returns
    -------
    AbstractAction with all fields in [-1, 1].
    """
    arr = np.asarray(vec, dtype=float)

    if arr.shape != (NUM_CHANNELS,):
        raise ValueError(
            f"Expected vector of length {NUM_CHANNELS}, got shape {arr.shape}."
        )

    active = np.flatnonzero(arr)

    if active.size == 0:
        return AbstractAction()  # no-op: hold position

    if active.size > 1:
        raise ValueError(
            f"Expected one-hot (exactly one non-zero); "
            f"got {active.size} active: {active.tolist()}."
        )

    idx = int(active[0])
    raw_value = float(arr[idx])

    # Normalise: analog sticks → [-1, 1], binary inputs → clamp to [-1, 1]
    normalised = np.clip(raw_value / _RAW_MAX, -1.0, 1.0)

    channel = Channel(idx)

    if channel is Channel.LEFT_STICK_X:
        return AbstractAction(x=normalised)

    if channel is Channel.LEFT_STICK_Y:
        return AbstractAction(z=normalised)

    if channel is Channel.RIGHT_STICK_X:
        return AbstractAction(theta=normalised)

    if channel is Channel.RIGHT_STICK_Y:
        return AbstractAction(y=normalised)

    if channel is Channel.TRIGGER_LEFT:
        return AbstractAction(gripper=+1.0)  # open

    if channel is Channel.TRIGGER_RIGHT:
        return AbstractAction(gripper=-1.0)  # close

    # Buttons A/B/X/Y and bumpers LB/RB — not mapped to arm motion
    return AbstractAction()


def action_to_command(
    action: AbstractAction,
    linear_step: float = _LINEAR_STEP,
    pan_step:    float = _PAN_STEP,
    roll_step:   float = _ROLL_STEP,
    jaw_step:    float = _JAW_STEP,
) -> EndEffectorDeltaCommand:
    """Scale an AbstractAction into a concrete EndEffectorDeltaCommand.

    The magnitude of each field (from the stick's deflection) scales the
    step size, so a half-deflected stick produces half the movement.
    """
    return EndEffectorDeltaCommand(
        dx    = action.y       * linear_step,   # right-stick Y → reach
        dy    = action.x       * pan_step,      # left-stick X  → pan
        dz    = action.z       * linear_step,   # left-stick Y  → height
        d_rot = action.theta   * roll_step,     # right-stick X → roll
        d_jaw = action.gripper * jaw_step,      # triggers      → jaw
    )


# ---------------------------------------------------------------------------
# Controller interface
# ---------------------------------------------------------------------------

class ArmController:
    """Integration boundary between the BCI decoder and the physical arm.

    Accepts raw one-hot controller vectors, normalises them, and dispatches
    EndEffectorDeltaCommands to the SO-101 API.
    """

    def __init__(self, settings: ArmSettings, urdf_path: str | None = None) -> None:
        self.settings = settings
        controller = SO101ArmController(settings)
        ik = SO101IKTranslator(
            urdf_path=urdf_path,
            use_degrees=settings.use_degrees,
            joint_names=settings.ik_joint_names,
            end_effector_link=settings.ik_end_effector_link,
        )
        self._api = SO101ArmAPI(
            controller=controller,
            ik_translator=ik,
            smoothing_alpha=settings.smoothing_alpha,
        )

    # -- Lifecycle ----------------------------------------------------------

    def connect(self) -> None:
        self._api.connect()
        logger.info("ArmController connected (dry_run=%s)", self.settings.dry_run)

    def disconnect(self) -> None:
        self._api.disconnect()
        logger.info("ArmController disconnected.")

    def __enter__(self) -> "ArmController":
        self.connect()
        return self

    def __exit__(self, *_: object) -> None:
        self.disconnect()

    # -- Named axis methods (magnitude in [-1, 1]) -------------------------

    def move_x(self, magnitude: float) -> None:
        """Pan the shoulder left (-) or right (+)."""
        self._send(AbstractAction(x=float(magnitude)))

    def move_y(self, magnitude: float) -> None:
        """Extend (+) or retract (-) the reach."""
        self._send(AbstractAction(y=float(magnitude)))

    def move_z(self, magnitude: float) -> None:
        """Raise (+) or lower (-) the end-effector."""
        self._send(AbstractAction(z=float(magnitude)))

    def rotate_theta(self, magnitude: float) -> None:
        """Roll the tool CW (+) or CCW (-)."""
        self._send(AbstractAction(theta=float(magnitude)))

    def open_gripper(self, magnitude: float = 1.0) -> None:
        self._send(AbstractAction(gripper=abs(float(magnitude))))

    def close_gripper(self, magnitude: float = 1.0) -> None:
        self._send(AbstractAction(gripper=-abs(float(magnitude))))

    # -- Core dispatch ------------------------------------------------------

    def execute(self, vec: list[float] | np.ndarray) -> None:
        """Decode a one-hot controller vector and send the resulting command."""
        action = decode_vector(vec)
        self._send(action)

    def _send(self, action: AbstractAction) -> None:
        if action.is_null():
            return
        cmd = action_to_command(action)
        result = self._api.send_end_effector_delta(cmd)
        logger.debug("sent=%s  merged=%s", result.sent_targets, result.merged_targets)


# ---------------------------------------------------------------------------
# Main control loop
# ---------------------------------------------------------------------------

def run_control_loop(
    input_stream: Iterator[list[float] | np.ndarray],
    settings: ArmSettings | None = None,
    loop_hz: float | None = None,
    urdf_path: str | None = None,
) -> None:
    """Process a stream of one-hot vectors and drive the arm in real time.

    Parameters
    ----------
    input_stream:
        Iterator yielding vectors of length 12.  Can be a BCI decoder
        output, a file replay, a test generator, etc.
    settings:
        ArmSettings instance.  Loaded from arm/local_config.py if None.
    loop_hz:
        Target control frequency.  Defaults to settings.loop_hz (20 Hz).
    urdf_path:
        Path to so101.urdf for IK.  Falls back to geometric defaults if None.
    """
    if settings is None:
        settings = load_settings()

    hz = loop_hz if loop_hz is not None else settings.loop_hz
    period = 1.0 / hz

    with ArmController(settings, urdf_path=urdf_path) as ctrl:
        for vec in input_stream:
            t0 = time.monotonic()
            try:
                ctrl.execute(vec)
            except ValueError as exc:
                logger.warning("Bad input vector: %s", exc)

            elapsed = time.monotonic() - t0
            sleep_for = period - elapsed
            if sleep_for > 0:
                time.sleep(sleep_for)


# ---------------------------------------------------------------------------
# Self-contained test / demo
# ---------------------------------------------------------------------------

def _demo_sequence() -> Iterator[np.ndarray]:
    """Yield a short sequence of controller vectors for testing."""
    labelled: list[tuple[str, Channel | None, float]] = [
        # Analog sticks: value carries direction + intensity
        ("pan right (full)",    Channel.LEFT_STICK_X,   32767),
        ("pan left (half)",     Channel.LEFT_STICK_X,  -16384),
        ("raise up",            Channel.LEFT_STICK_Y,   32767),
        ("lower down",          Channel.LEFT_STICK_Y,  -32767),
        ("rotate CW",           Channel.RIGHT_STICK_X,  32767),
        ("rotate CCW (gentle)", Channel.RIGHT_STICK_X, -8000),
        ("reach forward",       Channel.RIGHT_STICK_Y,  32767),
        ("reach back",          Channel.RIGHT_STICK_Y, -32767),
        # Triggers: binary
        ("open gripper",        Channel.TRIGGER_LEFT,   32767),
        ("close gripper",       Channel.TRIGGER_RIGHT,  32767),
        # Unused inputs → no-op
        ("A button (unused)",   Channel.BUTTON_A,       32767),
        ("LB bumper (unused)",  Channel.BUMPER_LEFT,    32767),
        # All-zero → hold position
        ("no-op (idle)",        None,                   0),
    ]
    for label, channel, value in labelled:
        vec = np.zeros(NUM_CHANNELS, dtype=float)
        if channel is not None:
            vec[channel] = value
        action = decode_vector(vec)
        print(f"  [{label:24s}]  ch={str(channel):20s}  val={value:+7.0f}  -> {action}")
        yield vec


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(levelname)s  %(message)s")

    # ------------------------------------------------------------------
    # 1. Mapping test — verify all channels (no hardware)
    # ------------------------------------------------------------------
    print("=== Channel mapping test ===")
    for ch in Channel:
        vec = np.zeros(NUM_CHANNELS, dtype=float)
        vec[ch] = 32767 if ch >= Channel.BUTTON_A else 16384  # half-deflection for sticks
        action  = decode_vector(vec)
        command = action_to_command(action)
        print(f"  {ch.name:16s}  action={action}")
        print(f"  {'':16s}  command={command}")

    # ------------------------------------------------------------------
    # 2. Error handling
    # ------------------------------------------------------------------
    print("\n=== Graceful error handling ===")
    for label, bad in [
        ("two active",  [32767, 32767, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]),
        ("wrong size",  [0] * 5),
    ]:
        try:
            decode_vector(bad)
        except ValueError as e:
            print(f"  {label}: {e}")

    # ------------------------------------------------------------------
    # 3. Dry-run control loop
    # ------------------------------------------------------------------
    print("\n=== Control loop (dry-run) ===")
    dry_settings = ArmSettings(dry_run=True, loop_hz=10.0)
    run_control_loop(_demo_sequence(), settings=dry_settings)
    print("Done.")
