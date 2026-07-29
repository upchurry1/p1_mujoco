#!/usr/bin/env python3
"""Quick MuJoCo check for the P1 A/B closed-chain ankle XML."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco


def joint_qpos_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"missing joint: {name}")
    return int(model.jnt_qposadr[joint_id])


def joint_qvel_addr(model: mujoco.MjModel, name: str) -> int:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
        raise RuntimeError(f"missing joint: {name}")
    return int(model.jnt_dofadr[joint_id])


def actuator_id(model: mujoco.MjModel, name: str) -> int:
    act_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if act_id < 0:
        raise RuntimeError(f"missing actuator: {name}")
    return int(act_id)


def tendon_id(model: mujoco.MjModel, name: str) -> int:
    tid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TENDON, name)
    if tid < 0:
        raise RuntimeError(f"missing tendon: {name}")
    return int(tid)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate A/B motor closure against passive pitch/roll ankle joints."
    )
    parser.add_argument(
        "--model",
        default="unitree_robots/p1_ankle_ik_check/scene_fixed_base.xml",
        help="Closed-chain MuJoCo XML path.",
    )
    parser.add_argument("--side", choices=("l", "r"), default="l")
    parser.add_argument(
        "--mode",
        choices=("pitch", "roll", "custom"),
        default="pitch",
        help="pitch uses A=target,B=-target; roll uses A=target,B=target.",
    )
    parser.add_argument("--target", type=float, default=0.05)
    parser.add_argument("--target-a", type=float, default=0.0)
    parser.add_argument("--target-b", type=float, default=0.0)
    parser.add_argument("--duration", type=float, default=1.0)
    parser.add_argument("--kp", type=float, default=50.0)
    parser.add_argument("--kd", type=float, default=2.0)
    parser.add_argument("--torque-limit", type=float, default=34.0)
    parser.add_argument("--gravity-z", type=float, default=0.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    model.opt.gravity[2] = args.gravity_z

    suffix = args.side
    pitch_name = f"ankle_pitch_{suffix}_joint"
    roll_name = f"ankle_roll_{suffix}_joint"
    motor_a_name = f"ankle_A_{suffix}_joint"
    motor_b_name = f"ankle_B_{suffix}_joint"

    qpos = {
        "pitch": joint_qpos_addr(model, pitch_name),
        "roll": joint_qpos_addr(model, roll_name),
        "A": joint_qpos_addr(model, motor_a_name),
        "B": joint_qpos_addr(model, motor_b_name),
    }
    qvel = {
        "A": joint_qvel_addr(model, motor_a_name),
        "B": joint_qvel_addr(model, motor_b_name),
    }
    act = {
        "A": actuator_id(model, motor_a_name),
        "B": actuator_id(model, motor_b_name),
    }
    tendons = [
        tendon_id(model, f"ankle_A_{suffix}_rod"),
        tendon_id(model, f"ankle_B_{suffix}_rod"),
    ]

    if args.mode == "pitch":
        target_a = args.target
        target_b = -args.target
    elif args.mode == "roll":
        target_a = args.target
        target_b = args.target
    else:
        target_a = args.target_a
        target_b = args.target_b

    mujoco.mj_forward(model, data)
    initial_lengths = [float(data.ten_length[tid]) for tid in tendons]
    steps = max(1, int(args.duration / model.opt.timestep))
    for _ in range(steps):
        qa = float(data.qpos[qpos["A"]])
        qb = float(data.qpos[qpos["B"]])
        dqa = float(data.qvel[qvel["A"]])
        dqb = float(data.qvel[qvel["B"]])
        ctrl_a = args.kp * (target_a - qa) - args.kd * dqa
        ctrl_b = args.kp * (target_b - qb) - args.kd * dqb
        data.ctrl[act["A"]] = max(-args.torque_limit, min(args.torque_limit, ctrl_a))
        data.ctrl[act["B"]] = max(-args.torque_limit, min(args.torque_limit, ctrl_b))
        mujoco.mj_step(model, data)

    final_lengths = [float(data.ten_length[tid]) for tid in tendons]
    print(f"model={model_path}")
    print(f"side={args.side} mode={args.mode} target_A={target_a:.6f} target_B={target_b:.6f}")
    print(
        "q_rad "
        f"pitch={float(data.qpos[qpos['pitch']]):.6f} "
        f"roll={float(data.qpos[qpos['roll']]):.6f} "
        f"A={float(data.qpos[qpos['A']]):.6f} "
        f"B={float(data.qpos[qpos['B']]):.6f}"
    )
    print(
        "rod_length_delta_m "
        f"A={final_lengths[0] - initial_lengths[0]:.9f} "
        f"B={final_lengths[1] - initial_lengths[1]:.9f}"
    )
    print("warnings=" + ",".join(str(int(w)) for w in data.warning.number[:3]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
