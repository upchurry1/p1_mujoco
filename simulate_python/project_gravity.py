#!/usr/bin/env python3
"""Standalone projected-gravity calculator.

The deploy code computes:

    projected_gravity_body = conjugate(body_to_world_quat) rotate [0, 0, -1]

Quaternion order is w x y z. RPY order is roll, pitch, yaw about body/world
X, Y, Z axes with the usual yaw-pitch-roll composition.

Examples:
  python3 simulate_python/project_gravity.py
  python3 simulate_python/project_gravity.py --rpy 0 -90 0 --degrees
  python3 simulate_python/project_gravity.py --quat 0.7071068 0 -0.7071068 0
"""

from __future__ import annotations

import argparse
import math
from typing import Iterable


GRAVITY_WORLD = (0.0, 0.0, -1.0)


def quat_mul(q1: tuple[float, float, float, float], q2: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
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
        raise ValueError("Quaternion norm must be non-zero.")
    return (w / norm, x / norm, y / norm, z / norm)


def rotate_vec(q: tuple[float, float, float, float], v: tuple[float, float, float]) -> tuple[float, float, float]:
    qv = (0.0, v[0], v[1], v[2])
    out = quat_mul(quat_mul(q, qv), quat_conj(q))
    return (out[1], out[2], out[3])


def quat_from_axis_angle(axis: tuple[float, float, float], angle: float) -> tuple[float, float, float, float]:
    half = 0.5 * angle
    s = math.sin(half)
    return (math.cos(half), axis[0] * s, axis[1] * s, axis[2] * s)


def quat_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    qx = quat_from_axis_angle((1.0, 0.0, 0.0), roll)
    qy = quat_from_axis_angle((0.0, 1.0, 0.0), pitch)
    qz = quat_from_axis_angle((0.0, 0.0, 1.0), yaw)
    # body -> world, yaw-pitch-roll composition: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return quat_normalize(quat_mul(quat_mul(qz, qy), qx))


def projected_gravity(body_to_world_quat: tuple[float, float, float, float]) -> tuple[float, float, float]:
    return rotate_vec(quat_conj(body_to_world_quat), GRAVITY_WORLD)


def fmt_vec(v: Iterable[float]) -> str:
    return "[" + ", ".join(f"{x: .6f}" for x in v) + "]"


def print_result(name: str, q: tuple[float, float, float, float]) -> None:
    pg = projected_gravity(q)
    print(f"{name:<18} quat_wxyz={fmt_vec(q)} projected_gravity={fmt_vec(pg)}")


def print_examples() -> None:
    examples = [
        ("upright", (0.0, 0.0, 0.0)),
        ("fall_forward", (0.0, 90.0, 0.0)),
        ("fall_backward", (0.0, -90.0, 0.0)),
        ("fall_left", (90.0, 0.0, 0.0)),
        ("fall_right", (-90.0, 0.0, 0.0)),
        ("upside_down", (180.0, 0.0, 0.0)),
    ]
    print("Body axes: +X forward, +Y left, +Z up. World gravity direction: [0, 0, -1].")
    for name, rpy_deg in examples:
        rpy = tuple(math.radians(v) for v in rpy_deg)
        print_result(f"{name} rpy={rpy_deg}", quat_from_rpy(*rpy))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--rpy", nargs=3, type=float, metavar=("ROLL", "PITCH", "YAW"), help="Body-to-world RPY")
    group.add_argument("--quat", nargs=4, type=float, metavar=("W", "X", "Y", "Z"), help="Body-to-world quaternion")
    parser.add_argument("--degrees", action="store_true", help="Interpret --rpy as degrees")
    args = parser.parse_args()

    if args.rpy is None and args.quat is None:
        print_examples()
        return

    if args.rpy is not None:
        roll, pitch, yaw = args.rpy
        if args.degrees:
            roll, pitch, yaw = math.radians(roll), math.radians(pitch), math.radians(yaw)
        q = quat_from_rpy(roll, pitch, yaw)
        print_result("from_rpy", q)
    else:
        q = quat_normalize(args.quat)
        print_result("from_quat", q)


if __name__ == "__main__":
    main()
