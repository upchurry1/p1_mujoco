#!/usr/bin/env python3
"""Print simulated IMU state from rt/lowstate.

Run the simulator first, then open another terminal:

    cd simulate_python
    python3 print_imu.py

The projected gravity vector is world gravity [0, 0, -1] expressed in the
IMU/body frame using the LowState quaternion in wxyz order.
"""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import config


TOPIC_LOWSTATE = "rt/lowstate"
GRAVITY_WORLD = (0.0, 0.0, -1.0)


def quat_mul(
    q1: tuple[float, float, float, float],
    q2: tuple[float, float, float, float],
) -> tuple[float, float, float, float]:
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return (
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    )


def quat_conj(q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    return (q[0], -q[1], -q[2], -q[3])


def quat_normalize(q: Iterable[float]) -> tuple[float, float, float, float]:
    w, x, y, z = [float(v) for v in q]
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 0.0:
        return (1.0, 0.0, 0.0, 0.0)
    return (w / norm, x / norm, y / norm, z / norm)


def rotate_vec(
    q: tuple[float, float, float, float],
    v: tuple[float, float, float],
) -> tuple[float, float, float]:
    qv = (0.0, v[0], v[1], v[2])
    out = quat_mul(quat_mul(q, qv), quat_conj(q))
    return (out[1], out[2], out[3])


def projected_gravity(q_world_from_imu: tuple[float, float, float, float]) -> tuple[float, float, float]:
    return rotate_vec(quat_conj(q_world_from_imu), GRAVITY_WORLD)


def quat_to_rpy(q: tuple[float, float, float, float]) -> tuple[float, float, float]:
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    sin_pitch = max(-1.0, min(1.0, 2.0 * (w * y - z * x)))
    pitch = math.asin(sin_pitch)
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return (roll, pitch, yaw)


def dominant_axis(v: tuple[float, float, float]) -> str:
    axis = max(range(3), key=lambda i: abs(v[i]))
    sign = "+" if v[axis] >= 0.0 else "-"
    names = ("X", "Y", "Z")
    labels = {
        "+X": "forward",
        "-X": "backward",
        "+Y": "left",
        "-Y": "right",
        "+Z": "up",
        "-Z": "down",
    }
    key = f"{sign}{names[axis]}"
    return f"{key}({labels[key]})"


def fmt_vec(v: Iterable[float]) -> str:
    return "[" + ", ".join(f"{x: .4f}" for x in v) + "]"


def import_lowstate(idl: str):
    if idl == "hg":
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_

        return LowState_

    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_

    return LowState_


def auto_idl(robot: str) -> str:
    return "hg" if robot in ("g1", "p1") else "go"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", type=int, default=config.DOMAIN_ID, help="DDS domain id")
    parser.add_argument("--interface", default=config.INTERFACE, help="DDS interface")
    parser.add_argument("--robot", default=config.ROBOT, help="Robot name used for auto IDL selection")
    parser.add_argument("--idl", choices=("auto", "go", "hg"), default="auto", help="LowState IDL type")
    parser.add_argument("--rate", type=float, default=10.0, help="Print rate in Hz")
    parser.add_argument("--once", action="store_true", help="Print one sample and exit")
    args = parser.parse_args()

    from unitree_sdk2py.core.channel import ChannelFactoryInitialize, ChannelSubscriber

    idl = auto_idl(args.robot) if args.idl == "auto" else args.idl
    lowstate_type = import_lowstate(idl)

    min_period = 1.0 / args.rate if args.rate > 0.0 else 0.0
    last_print = 0.0
    got_sample = False

    def lowstate_handler(msg) -> None:
        nonlocal last_print, got_sample
        now = time.perf_counter()
        if not args.once and min_period > 0.0 and now - last_print < min_period:
            return

        imu = msg.imu_state
        quat = quat_normalize(imu.quaternion)
        gyro = tuple(float(v) for v in imu.gyroscope)
        acc = tuple(float(v) for v in imu.accelerometer)
        rpy = quat_to_rpy(quat)
        proj_g = projected_gravity(quat)

        print(
            "imu | "
            f"quat_wxyz={fmt_vec(quat)} | "
            f"rpy_rad={fmt_vec(rpy)} | "
            f"gyro={fmt_vec(gyro)} | "
            f"acc={fmt_vec(acc)} | "
            f"projected_gravity={fmt_vec(proj_g)} | "
            f"gravity_dir={dominant_axis(proj_g)}",
            flush=True,
        )
        last_print = now
        got_sample = True

    print(
        f"Subscribing {TOPIC_LOWSTATE} with {idl} LowState "
        f"(domain={args.domain}, interface={args.interface})."
    )
    ChannelFactoryInitialize(args.domain, args.interface)
    subscriber = ChannelSubscriber(TOPIC_LOWSTATE, lowstate_type)
    subscriber.Init(lowstate_handler, 10)

    try:
        while not (args.once and got_sample):
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
