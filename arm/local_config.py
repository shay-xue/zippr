"""Local SO-101 arm settings template.

This file is gitignored. Fill in the serial port for your machine before running teleop.
"""

from arm.config import ArmSettings


SETTINGS = ArmSettings(
    # Replace with the real serial port for your SO-101 follower arm.
    port="/dev/tty.usbmodem5AB01580631",
    robot_id="so101-local",
    # Keep this False during teleop unless you explicitly want interactive calibration.
    calibrate_on_connect=False,
    # Replace these with the actual joint names from your URDF, in base-to-wrist order.
    ik_joint_names=[
        "shoulder_pan",
        "shoulder_lift",
        "elbow_flex",
        "wrist_flex",
        "wrist_roll",
    ],
    # Replace this with the link name used as the end-effector target in your URDF.
    ik_end_effector_link="gripper_link",
    # Set to True to exercise the control loop without hardware attached.
    dry_run=False,
)
