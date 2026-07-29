#!/usr/bin/env python3
"""Interactively verify P1 motor/joint positive rotation directions in MuJoCo.

This script bypasses DDS and writes MuJoCo actuator controls directly with a
simple PD controller:

    ctrl = kp * (q_target - q) - kd * dq

Keys:
  ] / [ : select next / previous motor
  = / - : increase / decrease selected motor target angle
  0     : zero selected motor target
  R     : reset all joint targets and pose
  P     : print one sample
  G     : toggle gravity
  F     : toggle base freeze
"""

from __future__ import annotations

import argparse
import math
import time
from dataclasses import dataclass
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = REPO_ROOT / "unitree_robots" / "p1" / "scene.xml"


@dataclass
class MotorInfo:
    actuator_id: int
    actuator_name: str
    joint_id: int
    joint_name: str
    qpos_adr: int
    dof_adr: int
    axis: np.ndarray
    joint_range: np.ndarray
    ctrl_range: np.ndarray


def find_root_freejoint(model: mujoco.MjModel) -> tuple[int, int] | None:
    for jid in range(model.njnt):
        if model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE:
            return int(model.jnt_qposadr[jid]), int(model.jnt_dofadr[jid])
    return None


def motor_infos(model: mujoco.MjModel) -> list[MotorInfo]:
    infos: list[MotorInfo] = []
    for aid in range(model.nu):
        joint_id = int(model.actuator_trnid[aid, 0])
        if joint_id < 0:
            continue
        actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid) or f"actuator_{aid}"
        joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or f"joint_{joint_id}"
        infos.append(
            MotorInfo(
                actuator_id=aid,
                actuator_name=actuator_name,
                joint_id=joint_id,
                joint_name=joint_name,
                qpos_adr=int(model.jnt_qposadr[joint_id]),
                dof_adr=int(model.jnt_dofadr[joint_id]),
                axis=model.jnt_axis[joint_id].copy(),
                joint_range=model.jnt_range[joint_id].copy(),
                ctrl_range=model.actuator_ctrlrange[aid].copy(),
            )
        )
    return infos


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def print_motor_table(infos: list[MotorInfo]) -> None:
    print("Motor order: MuJoCo actuator / SDK motor_cmd order")
    for info in infos:
        lo, hi = info.joint_range
        axis = info.axis
        print(
            f"  {info.actuator_id:2d}: {info.joint_name:<22} "
            f"range=[{lo: .3f}, {hi: .3f}] rad "
            f"axis=[{axis[0]: .0f}, {axis[1]: .0f}, {axis[2]: .0f}]"
        )


def print_sample(label: str, info: MotorInfo, data: mujoco.MjData, target: float, ctrl: float) -> None:
    q = float(data.qpos[info.qpos_adr])
    dq = float(data.qvel[info.dof_adr])
    err = target - q
    print(
        f"{label:>9} | motor {info.actuator_id:2d} {info.joint_name:<22} "
        f"target={target: .3f} q={q: .3f} dq={dq: .3f} "
        f"err={err: .3f} ctrl={ctrl: .2f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL, help="MuJoCo XML model path")
    parser.add_argument("--kp", type=float, default=140.0, help="PD position gain")
    parser.add_argument("--kd", type=float, default=6.0, help="PD damping gain")
    parser.add_argument("--step", type=float, default=0.15, help="Target angle increment per key press, rad")
    parser.add_argument("--suspend-height", type=float, default=1.15, help="Initial floating-base height, m")
    parser.add_argument("--print-period", type=float, default=0.25, help="Terminal print period, seconds")
    parser.add_argument("--keep-gravity", action="store_true", help="Keep model gravity enabled at startup")
    parser.add_argument("--free-base", action="store_true", help="Do not freeze the floating base at startup")
    args = parser.parse_args()

    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    infos = motor_infos(model)
    if not infos:
        raise RuntimeError("No joint motors found in the MuJoCo model.")

    root = find_root_freejoint(model)
    original_gravity = model.opt.gravity.copy()
    if not args.keep_gravity:
        model.opt.gravity[:] = 0.0

    mujoco.mj_resetData(model, data)
    if root is not None:
        qadr, _ = root
        data.qpos[qadr + 2] = args.suspend_height
    mujoco.mj_forward(model, data)
    initial_qpos = data.qpos.copy()
    initial_qvel = data.qvel.copy()
    targets = np.array([data.qpos[info.qpos_adr] for info in infos], dtype=float)
    selected = 0
    freeze_base = not args.free_base
    print_once = True
    label = "start"
    last_ctrl = np.zeros(len(infos), dtype=float)

    def reset_all() -> None:
        nonlocal targets, selected, print_once, label
        data.qpos[:] = initial_qpos
        data.qvel[:] = initial_qvel
        targets = np.array([data.qpos[info.qpos_adr] for info in infos], dtype=float)
        selected = 0
        mujoco.mj_forward(model, data)
        label = "reset"
        print_once = True

    def select(delta: int) -> None:
        nonlocal selected, print_once, label
        selected = (selected + delta) % len(infos)
        label = "select"
        print_once = True

    def change_target(delta: float) -> None:
        nonlocal print_once, label
        info = infos[selected]
        lo, hi = info.joint_range
        # Leave a small margin from hard limits to avoid fighting the constraint.
        margin = min(0.02, max(0.0, 0.05 * (hi - lo)))
        targets[selected] = clamp(float(targets[selected] + delta), lo + margin, hi - margin)
        label = "+target" if delta > 0 else "-target"
        print_once = True

    def key_callback(keycode: int) -> None:
        nonlocal freeze_base, print_once, label
        glfw = mujoco.glfw.glfw
        if keycode == glfw.KEY_RIGHT_BRACKET:
            select(+1)
        elif keycode == glfw.KEY_LEFT_BRACKET:
            select(-1)
        elif keycode in (glfw.KEY_EQUAL, glfw.KEY_KP_ADD):
            change_target(+args.step)
        elif keycode in (glfw.KEY_MINUS, glfw.KEY_KP_SUBTRACT):
            change_target(-args.step)
        elif keycode == glfw.KEY_0:
            targets[selected] = initial_qpos[infos[selected].qpos_adr]
            label = "zero"
            print_once = True
        elif keycode == glfw.KEY_R:
            reset_all()
        elif keycode == glfw.KEY_P:
            label = "print"
            print_once = True
        elif keycode == glfw.KEY_G:
            if np.linalg.norm(model.opt.gravity) > 1e-9:
                model.opt.gravity[:] = 0.0
                label = "grav off"
            else:
                model.opt.gravity[:] = original_gravity
                label = "grav on"
            print_once = True
        elif keycode == glfw.KEY_F:
            freeze_base = not freeze_base
            label = "freeze" if freeze_base else "freebase"
            print_once = True

    print("Loaded:", args.model)
    print_motor_table(infos)
    print()
    print("Positive target means q_target increases. The joint should move toward increasing q.")
    print("Keys: ]/[ select motor, =/- change target, 0 zero selected, R reset, G gravity, F freeze base, P print.")
    print(
        f"Startup: gravity={'on' if args.keep_gravity else 'off'}, "
        f"base={'free' if args.free_base else 'frozen'}, "
        f"suspend_height={args.suspend_height:.2f} m"
    )

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        last_print = 0.0
        while viewer.is_running():
            if freeze_base and root is not None:
                qadr, dadr = root
                data.qpos[qadr : qadr + 7] = initial_qpos[qadr : qadr + 7]
                data.qvel[dadr : dadr + 6] = 0.0

            for i, info in enumerate(infos):
                q = float(data.qpos[info.qpos_adr])
                dq = float(data.qvel[info.dof_adr])
                ctrl = args.kp * (targets[i] - q) - args.kd * dq
                if model.actuator_ctrllimited[info.actuator_id]:
                    lo, hi = info.ctrl_range
                    ctrl = clamp(ctrl, float(lo), float(hi))
                data.ctrl[info.actuator_id] = ctrl
                last_ctrl[i] = ctrl

            mujoco.mj_step(model, data)

            if freeze_base and root is not None:
                qadr, dadr = root
                data.qpos[qadr : qadr + 7] = initial_qpos[qadr : qadr + 7]
                data.qvel[dadr : dadr + 6] = 0.0
                mujoco.mj_forward(model, data)

            viewer.sync()

            now = time.perf_counter()
            if print_once or now - last_print >= args.print_period:
                print_sample(label, infos[selected], data, float(targets[selected]), float(last_ctrl[selected]))
                print_once = False
                last_print = now

            time.sleep(max(0.0, model.opt.timestep))


if __name__ == "__main__":
    main()
