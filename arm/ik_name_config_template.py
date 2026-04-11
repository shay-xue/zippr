"""Template for SO-101 URDF naming configuration.

Fill these values from your actual URDF if the IK loader needs explicit joint/link names.
This file is only a template and is not read automatically by the current code.
"""

# Ordered from base to wrist roll.
IK_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
]

# Link name for the tool tip / end effector frame used for FK and IK.
IK_END_EFFECTOR_LINK = "gripper"

# Optional gripper joint names if your URDF exposes them explicitly.
GRIPPER_JOINT_NAMES = [
    "gripper",
]

# Optional notes while mapping from URDF names to the control stack:
#
# URDF joint name -> control joint name
# shoulder_pan_joint   -> shoulder_pan
# shoulder_lift_joint  -> shoulder_lift
# elbow_flex_joint     -> elbow_flex
# wrist_flex_joint     -> wrist_flex
# wrist_roll_joint     -> wrist_roll
