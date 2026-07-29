#!/usr/bin/env python3
"""Visualize and verify P1 IMU/body axis directions in MuJoCo.

Keys:
  1 / 2 : step rotate about +X / -X  (roll axis, forward)
  3 / 4 : step rotate about +Y / -Y  (pitch axis, left)
  5 / 6 : step rotate about +Z / -Z  (yaw axis, up)
  0     : zero angular velocity
  R     : reset orientation
  P     : print one sample
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "unitree_robots" / "p1" / "scene.xml"
GRAVITY_WORLD = np.array([0.0, 0.0, -1.0], dtype=float)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product for MuJoCo quaternions in wxyz order."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=float,
    )


def quat_conj(q: np.ndarray) -> np.ndarray:
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def axis_angle_quat(axis: int, angle: float) -> np.ndarray:
    q = np.zeros(4, dtype=float)
    q[0] = math.cos(0.5 * angle)
    q[axis + 1] = math.sin(0.5 * angle)
    return q


def rotate_vec(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    qv = np.array([0.0, v[0], v[1], v[2]], dtype=float)
    return quat_mul(quat_mul(q, qv), quat_conj(q))[1:]


def quat_to_rpy(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = 2.0 * (w * y - z * x)
    pitch = math.asin(max(-1.0, min(1.0, sin_pitch)))
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return np.array([roll, pitch, yaw], dtype=float)


def dominant_gravity_direction(projected_gravity: np.ndarray) -> str:
    axis_id = int(np.argmax(np.abs(projected_gravity)))
    sign = 1 if projected_gravity[axis_id] >= 0.0 else -1
    axis_names = {
        (0, 1): "+X(forward)",
        (0, -1): "-X(backward)",
        (1, 1): "+Y(left)",
        (1, -1): "-Y(right)",
        (2, 1): "+Z(up)",
        (2, -1): "-Z(down)",
    }
    return axis_names[(axis_id, sign)]


def sensor_slice(model: mujoco.MjModel, name: str) -> slice | None:
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sid < 0:
        return None
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return slice(adr, adr + dim)


def freejoint_addresses(model: mujoco.MjModel) -> tuple[int, int]:
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "root")
    if jid < 0:
        for i in range(model.njnt):
            if model.jnt_type[i] == mujoco.mjtJoint.mjJNT_FREE:
                jid = i
                break
    if jid < 0:
        raise RuntimeError("No free joint found; this script expects a floating-base model.")
    return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])


def print_sample(
    label: str,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    quat_sl: slice | None,
    gyro_sl: slice | None,
) -> None:
    quat = data.sensordata[quat_sl].copy() if quat_sl else data.qpos[3:7].copy()
    gyro = data.sensordata[gyro_sl].copy() if gyro_sl else np.zeros(3)
    projected_gravity = rotate_vec(quat_conj(quat), GRAVITY_WORLD)
    gravity_dir = dominant_gravity_direction(projected_gravity)
    rpy = quat_to_rpy(quat)
    print(
        f"{label:>8} | "
        f"gyro xyz: {gyro[0]: .3f} {gyro[1]: .3f} {gyro[2]: .3f} | "
        f"proj_g: {projected_gravity[0]: .3f} {projected_gravity[1]: .3f} {projected_gravity[2]: .3f} | "
        f"gravity_dir: {gravity_dir:<12} | "
        f"rpy rad: {rpy[0]: .3f} {rpy[1]: .3f} {rpy[2]: .3f} | "
        f"quat wxyz: {quat[0]: .3f} {quat[1]: .3f} {quat[2]: .3f} {quat[3]: .3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML model path")
    parser.add_argument("--angle-step", type=float, default=10.0, help="Rotation angle per key press, degrees")
    parser.add_argument("--print-period", type=float, default=0.25, help="Terminal print period in seconds")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    qpos_adr, dof_adr = freejoint_addresses(model)
    quat_sl = sensor_slice(model, "imu_quat")
    gyro_sl = sensor_slice(model, "imu_gyro")

    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    pending_step: tuple[int, float] | None = None
    label = "stopped"
    print_once = True

    axis_names = {0: "X roll", 1: "Y pitch", 2: "Z yaw"}

    def reset_pose() -> None:
        nonlocal pending_step, label, print_once
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        mujoco.mj_forward(model, data)
        pending_step = None
        label = "reset"
        print_once = True

    def key_callback(keycode: int) -> None:
        nonlocal pending_step, label, print_once
        glfw = mujoco.glfw.glfw
        key_map = {
            glfw.KEY_1: (0, +1.0),
            glfw.KEY_2: (0, -1.0),
            glfw.KEY_3: (1, +1.0),
            glfw.KEY_4: (1, -1.0),
            glfw.KEY_5: (2, +1.0),
            glfw.KEY_6: (2, -1.0),
        }
        if keycode in key_map:
            axis, spin_sign = key_map[keycode]
            pending_step = (axis, spin_sign)
            label = f"{'+' if spin_sign > 0 else '-'}{axis_names[axis]}"
            print_once = True
        elif keycode == glfw.KEY_0:
            pending_step = None
            data.qvel[:] = 0.0
            label = "zero vel"
            print_once = True
        elif keycode == glfw.KEY_R:
            reset_pose()
        elif keycode == glfw.KEY_P:
            print_once = True

    reset_pose()
    print("Loaded:", args.model)
    print("Body/IMU axes: +X forward, +Y left, +Z up. Positive rotation follows the right-hand rule.")
    print(f"Rotation step: {args.angle_step:.3f} deg per key press.")
    print("Keys: 1/2 step +/-X roll, 3/4 step +/-Y pitch, 5/6 step +/-Z yaw, 0 zero velocity, R reset, P print.")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_t = time.perf_counter()
        last_print = 0.0
        while viewer.is_running():
            now = time.perf_counter()
            dt = min(now - last_t, 0.05)
            last_t = now

            qpos_prev = data.qpos.copy()
            step_applied = False
            if pending_step is not None:
                step_axis, step_sign = pending_step
                dq_body = axis_angle_quat(step_axis, step_sign * math.radians(args.angle_step))
                quat_adr = qpos_adr + 3
                data.qpos[quat_adr : quat_adr + 4] = quat_normalize(
                    quat_mul(data.qpos[quat_adr : quat_adr + 4], dq_body)
                )
                pending_step = None
                step_applied = True

            data.qpos[qpos_adr : qpos_adr + 3] = initial_qpos[qpos_adr : qpos_adr + 3]
            mujoco.mj_differentiatePos(model, data.qvel, max(dt, 1e-6), qpos_prev, data.qpos)
            data.qvel[dof_adr : dof_adr + 3] = 0.0
            if not step_applied:
                data.qvel[dof_adr + 3 : dof_adr + 6] = 0.0

            mujoco.mj_forward(model, data)
            viewer.sync()

            if print_once or now - last_print >= args.print_period:
                print_sample(label, model, data, quat_sl, gyro_sl)
                print_once = False
                last_print = now

            time.sleep(0.005)


if __name__ == "__main__":
    main()
