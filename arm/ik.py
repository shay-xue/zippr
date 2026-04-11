"""ikpy-based inverse kinematics support for high-level arm control."""

from __future__ import annotations

import importlib.util
import logging
import math
import tempfile
import xml.etree.ElementTree as ET
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .commands import EndEffectorDeltaCommand
from .controller import extract_joint_positions

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ArmPose:
    """Cartesian end-effector pose in the robot base frame."""

    x: float
    y: float
    z: float
    tool_roll: float


@dataclass(frozen=True)
class ArmKinematics:
    """Approximate geometric dimensions for the SO-101 arm."""

    base_height: float = 0.075
    upper_arm_length: float = 0.16
    forearm_length: float = 0.16
    tool_length: float = 0.1

    @classmethod
    def from_urdf(
        cls, urdf_path: str | Path, joint_names: list[str]
    ) -> "ArmKinematics":
        root = ET.parse(urdf_path).getroot()
        joint_origins = {
            joint.attrib["name"]: _parse_origin_xyz(joint)
            for joint in root.findall("joint")
        }

        missing = [name for name in joint_names if name not in joint_origins]
        if missing:
            raise ValueError(
                "URDF does not contain the configured IK joint names. "
                f"Missing: {missing}. Available joints: {sorted(joint_origins)}"
            )

        return cls(
            base_height=abs(joint_origins[joint_names[0]][2]),
            upper_arm_length=_vector_length(joint_origins[joint_names[2]]),
            forearm_length=_vector_length(joint_origins[joint_names[3]]),
            tool_length=_vector_length(joint_origins[joint_names[4]]),
        )


@dataclass
class SO101IKTranslator:
    """Translate end-effector deltas into joint targets using ikpy IK."""

    urdf_path: str | None = None
    use_degrees: bool = True
    kinematics: ArmKinematics = field(default_factory=ArmKinematics)
    joint_names: list[str] = field(
        default_factory=lambda: [
            "shoulder_pan",
            "shoulder_lift",
            "elbow_flex",
            "wrist_flex",
            "wrist_roll",
        ]
    )
    end_effector_link: str = "gripper_link"
    # Workspace floor limits: commands that would move the end-effector below
    # these values are clamped so the arm never solves to a pose that would
    # crash into the table or fold behind the base.
    workspace_min_x: float = 0.0
    workspace_min_z: float = 0.0
    _backend: "_IKPyBackend" = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if not _ikpy_available():
            raise ImportError("ikpy is required for IK. Run `uv add ikpy`.")
        if self.urdf_path:
            self.kinematics = ArmKinematics.from_urdf(self.urdf_path, self.joint_names)
        self._backend = _IKPyBackend(
            kinematics=self.kinematics,
            urdf_path=self.urdf_path,
            use_degrees=self.use_degrees,
            joint_names=self.joint_names,
            end_effector_link=self.end_effector_link,
            workspace_min_x=self.workspace_min_x,
            workspace_min_z=self.workspace_min_z,
        )

    def forward(self, observation: Mapping[str, Any]) -> ArmPose:
        return self._backend.forward(observation)

    def translate(
        self,
        command: EndEffectorDeltaCommand,
        observation: Mapping[str, Any],
        last_targets: dict[str, float] | None = None,
    ) -> dict[str, float]:
        return self._backend.translate(command, observation, last_targets)


def _ikpy_available() -> bool:
    return importlib.util.find_spec("ikpy") is not None


@dataclass
class _IKPyBackend:
    """ikpy-backed FK/IK wrapper."""

    kinematics: ArmKinematics
    urdf_path: str | None
    use_degrees: bool
    joint_names: list[str]
    end_effector_link: str
    workspace_min_x: float = 0.0
    workspace_min_z: float = 0.0
    _tempdir: tempfile.TemporaryDirectory[str] | None = field(
        default=None, init=False, repr=False
    )
    _chain: Any = field(default=None, init=False, repr=False)
    # Maps each joint_name to its index in the ikpy joint array.
    # ikpy arrays have length = len(chain.links); index 0 is always the fixed
    # base link (value 0.0), and active joints start at index 1.
    _joint_chain_index: dict[str, int] = field(
        default_factory=dict, init=False, repr=False
    )

    def __post_init__(self) -> None:
        from ikpy.chain import Chain

        urdf_path = self.urdf_path or self._build_generated_urdf()
        root_link = _find_root_link(urdf_path)
        self._chain = Chain.from_urdf_file(urdf_path, base_elements=[root_link])
        self._joint_chain_index = self._build_joint_index(urdf_path)

    def _build_joint_index(self, urdf_path: str) -> dict[str, int]:
        """Map each joint_name to its index in the ikpy joint array.

        ikpy arrays are indexed by link position in the chain. We parse the
        URDF to find joint_name → child_link_name, then look up that link in
        the chain. For generated URDFs whose joint names are numeric ("1"-"5")
        and don't match joint_names, we fall back to positional ordering.
        """
        root = ET.parse(urdf_path).getroot()
        joint_to_child: dict[str, str] = {}
        for joint in root.findall("joint"):
            child = joint.find("child")
            if child is not None:
                joint_to_child[joint.attrib["name"]] = child.attrib["link"]

        link_name_to_idx = {link.name: i for i, link in enumerate(self._chain.links)}

        result: dict[str, int] = {}
        for pos, jname in enumerate(self.joint_names):
            child_link = joint_to_child.get(jname)
            if child_link is not None and child_link in link_name_to_idx:
                result[jname] = link_name_to_idx[child_link]
            else:
                # Fallback: positional — joint i is at chain index i+1 (after base).
                result[jname] = pos + 1
        return result

    def forward(self, observation: Mapping[str, Any]) -> ArmPose:
        q = self._make_joint_array(extract_joint_positions(observation))
        transform = self._chain.forward_kinematics(q)
        position = transform[:3, 3]
        return ArmPose(
            x=float(position[0]),
            y=float(position[1]),
            z=float(position[2]),
            tool_roll=float(extract_joint_positions(observation)["wrist_roll"]),
        )

    def translate(
        self,
        command: EndEffectorDeltaCommand,
        observation: Mapping[str, Any],
        last_targets: dict[str, float] | None = None,
    ) -> dict[str, float]:
        joints = extract_joint_positions(observation)
        current_pan_rad = self._to_radians(joints["shoulder_pan"])
        target_pan_rad = current_pan_rad + command.dy

        # For joints that accumulate deltas (wrist_roll, gripper), prefer last
        # commanded target as the base to prevent drift from chasing the observation.
        base = last_targets if last_targets else joints
        gripper = max(
            0.0, min(100.0, base.get("gripper", joints["gripper"]) + command.d_jaw)
        )
        wrist_roll_base = base.get("wrist_roll", joints["wrist_roll"])

        # Only x/radial reach and z/height commands require IK. Pan, roll, and
        # gripper updates can pass through directly without perturbing the arm chain.
        if command.dx == 0.0 and command.dz == 0.0:
            return {
                "shoulder_pan": self._from_radians(target_pan_rad),
                "shoulder_lift": float(
                    base.get("shoulder_lift", joints["shoulder_lift"])
                ),
                "elbow_flex": float(base.get("elbow_flex", joints["elbow_flex"])),
                "wrist_flex": float(base.get("wrist_flex", joints["wrist_flex"])),
                "wrist_roll": float(wrist_roll_base + command.d_rot),
                "gripper": float(gripper),
            }

        current_pose = self.forward(observation)
        current_reach = math.hypot(current_pose.x, current_pose.y)
        target_reach = max(0.0, current_reach + command.dx)
        target_x = max(self.workspace_min_x, target_reach * math.cos(target_pan_rad))
        target_z = max(self.workspace_min_z, current_pose.z + command.dz)
        target_position = [
            target_x,
            target_reach * math.sin(target_pan_rad),
            target_z,
        ]
        logger.info(
            "IK target_position=%s command=%s current_pose=%s target_pan_rad=%s target_reach=%s",
            target_position,
            command,
            current_pose,
            target_pan_rad,
            target_reach,
        )

        # Use current joint config as the initial guess so the solver stays near
        # the rest pose — equivalent to pybullet's restPoses null-space parameter.
        initial_q = self._make_joint_array(joints)
        initial_q[self._joint_chain_index["shoulder_pan"]] = target_pan_rad

        # Clip initial guess to joint bounds so scipy.optimize.least_squares
        # doesn't raise "Initial guess is outside of provided bounds".
        for i, link in enumerate(self._chain.links):
            bounds = getattr(link, "bounds", None)
            if bounds is not None:
                lo, hi = bounds
                if lo is not None and hi is not None:
                    initial_q[i] = max(lo, min(hi, initial_q[i]))

        solution = self._chain.inverse_kinematics(
            target_position=target_position,
            initial_position=initial_q,
        )

        targets = {
            "shoulder_pan": self._from_radians(target_pan_rad),
            "shoulder_lift": self._from_radians(
                float(solution[self._joint_chain_index["shoulder_lift"]])
            ),
            "elbow_flex": self._from_radians(
                float(solution[self._joint_chain_index["elbow_flex"]])
            ),
            "wrist_flex": self._from_radians(
                float(solution[self._joint_chain_index["wrist_flex"]])
            ),
            "wrist_roll": float(wrist_roll_base + command.d_rot),
            "gripper": float(gripper),
        }
        logger.info("IK solved targets=%s", targets)
        return targets

    def _make_joint_array(self, joints: Mapping[str, Any]) -> list[float]:
        """Build ikpy joint array with base=0 at index 0 and active joints by chain index."""
        q = [0.0] * len(self._chain.links)
        for jname in self.joint_names:
            q[self._joint_chain_index[jname]] = self._to_radians(float(joints[jname]))
        return q

    def _build_generated_urdf(self) -> str:
        self._tempdir = tempfile.TemporaryDirectory(prefix="so101-ikpy-")
        urdf_path = Path(self._tempdir.name) / "so101_generated.urdf"
        urdf_path.write_text(_generated_so101_urdf(self.kinematics), encoding="utf-8")
        return str(urdf_path)

    def _to_radians(self, angle: float) -> float:
        if self.use_degrees:
            return math.radians(angle)
        return float(angle)

    def _from_radians(self, angle: float) -> float:
        if self.use_degrees:
            return math.degrees(angle)
        return float(angle)


def _generated_so101_urdf(kinematics: ArmKinematics) -> str:
    return f"""<?xml version="1.0"?>
<robot name="so101_generated">
  <link name="base"/>
  <link name="pan_link"/>
  <link name="upper_link"/>
  <link name="forearm_link"/>
  <link name="tool_link"/>
  <link name="tool_tip"/>

  <joint name="shoulder_pan" type="revolute">
    <parent link="base"/>
    <child link="pan_link"/>
    <origin xyz="0 0 {kinematics.base_height}" rpy="0 0 0"/>
    <axis xyz="0 0 1"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="shoulder_lift" type="revolute">
    <parent link="pan_link"/>
    <child link="upper_link"/>
    <origin xyz="0 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="elbow_flex" type="revolute">
    <parent link="upper_link"/>
    <child link="forearm_link"/>
    <origin xyz="{kinematics.upper_arm_length} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="wrist_flex" type="revolute">
    <parent link="forearm_link"/>
    <child link="tool_link"/>
    <origin xyz="{kinematics.forearm_length} 0 0" rpy="0 0 0"/>
    <axis xyz="0 1 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>

  <joint name="wrist_roll" type="revolute">
    <parent link="tool_link"/>
    <child link="tool_tip"/>
    <origin xyz="{kinematics.tool_length} 0 0" rpy="0 0 0"/>
    <axis xyz="1 0 0"/>
    <limit lower="-3.14159" upper="3.14159" effort="1" velocity="1"/>
  </joint>
</robot>
"""


def _find_root_link(urdf_path: str) -> str:
    """Return the root (base) link name — the link that is not a child of any joint."""
    root = ET.parse(urdf_path).getroot()
    all_links = {link.attrib["name"] for link in root.findall("link")}
    child_links = {
        j.find("child").attrib["link"]
        for j in root.findall("joint")
        if j.find("child") is not None
    }
    roots = all_links - child_links
    if len(roots) != 1:
        raise ValueError(f"Expected exactly one root link, found: {roots}")
    return next(iter(roots))


def _parse_origin_xyz(joint: ET.Element) -> tuple[float, float, float]:
    origin = joint.find("origin")
    if origin is None:
        return (0.0, 0.0, 0.0)
    return _parse_xyz(origin.attrib.get("xyz", "0 0 0"))


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = value.split()
    if len(parts) != 3:
        raise ValueError(f"Expected three xyz components, got: {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def _vector_length(vector: tuple[float, float, float]) -> float:
    x, y, z = vector
    return math.sqrt(x * x + y * y + z * z)
