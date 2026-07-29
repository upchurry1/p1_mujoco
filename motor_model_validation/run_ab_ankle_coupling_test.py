#!/usr/bin/env python3
"""Run a P1 A/B ankle coupling validation scaffold in MuJoCo.

The P1 XML still contains virtual ankle pitch/roll joints. This script inserts
an external A/B drive layer:
  pitch/roll target -> A/B target -> A/B MIT torque -> equivalent pitch/roll torque

Fill ab_ankle_config.yaml with the real P1 ankle mapping before using results
for sim2real conclusions.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from ab_ankle_mapping import (
    clip_ab_command,
    load_linear_mapping,
    load_motor_gains,
    load_motor_limits,
)
from analyze_motor_log import get_nested, resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent

CSV_COLUMNS = [
    "trial_id",
    "time",
    "side",
    "axis",
    "q_pitch_des",
    "q_roll_des",
    "dq_pitch_des",
    "dq_roll_des",
    "q_pitch_sim",
    "q_roll_sim",
    "dq_pitch_sim",
    "dq_roll_sim",
    "q_A_des",
    "q_B_des",
    "dq_A_des",
    "dq_B_des",
    "q_A_sim",
    "q_B_sim",
    "dq_A_sim",
    "dq_B_sim",
    "tau_A_before_clip",
    "tau_B_before_clip",
    "tau_A_after_clip",
    "tau_B_after_clip",
    "tau_pitch_equiv",
    "tau_roll_equiv",
    "sat_A",
    "sat_B",
]


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def parse_float(value: Any, default: float = math.nan) -> float:
    if value is None:
        return default
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text == "":
        return default
    try:
        return float(text)
    except ValueError:
        return default


def load_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Python package 'mujoco' is not installed in this environment. "
            "Run with the conda env that contains MuJoCo."
        ) from exc
    return mujoco


def finite_or_error(value: Any, name: str) -> float:
    value = parse_float(value)
    if not np.isfinite(value):
        raise ValueError(f"{name} must be finite")
    return value


class ABAnkleCouplingSimulator:
    def __init__(self, config: dict[str, Any], config_dir: Path) -> None:
        self.config = config
        self.config_dir = config_dir
        self.mujoco = load_mujoco()
        xml_path = resolve_path(str(get_nested(config, "model", "xml_path", default="../unitree_robots/p1/p1.xml")), config_dir)
        if xml_path is None or not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = self.mujoco.MjData(self.model)

        self.sim_dt = finite_or_error(get_nested(config, "simulation", "sim_dt", default=self.model.opt.timestep), "simulation.sim_dt")
        self.control_dt = finite_or_error(get_nested(config, "simulation", "control_dt", default=0.02), "simulation.control_dt")
        self.duration = finite_or_error(get_nested(config, "simulation", "duration", default=8.0), "simulation.duration")
        self.warmup_time = finite_or_error(get_nested(config, "simulation", "warmup_time", default=0.5), "simulation.warmup_time")
        self.base_z = parse_float(get_nested(config, "simulation", "base_z", default=math.nan))
        self.lock_free_base = bool(get_nested(config, "simulation", "lock_free_base", default=True))
        self.model.opt.timestep = self.sim_dt
        gravity = get_nested(config, "simulation", "gravity", default=None)
        if gravity is not None:
            self.model.opt.gravity[:] = np.asarray(gravity, dtype=float)
        ratio = self.control_dt / self.sim_dt
        self.substeps = int(round(ratio))
        if self.substeps < 1 or abs(ratio - self.substeps) > 1e-9:
            raise ValueError("simulation.control_dt must be an integer multiple of simulation.sim_dt")

    def _id(self, obj_type: Any, name: str) -> int:
        idx = self.mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise KeyError(name)
        return int(idx)

    def _joint_info(self, joint_name: str) -> tuple[int, int]:
        joint_id = self._id(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        return int(self.model.jnt_qposadr[joint_id]), int(self.model.jnt_dofadr[joint_id])

    def _actuator_id(self, actuator_name: str) -> int:
        return self._id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)

    def _root_state(self) -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]:
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                qadr = int(self.model.jnt_qposadr[joint_id])
                dadr = int(self.model.jnt_dofadr[joint_id])
                return qadr, dadr, np.array(self.data.qpos[qadr : qadr + 7], copy=True), np.array(self.data.qvel[dadr : dadr + 6], copy=True)
        return None, None, None, None

    def _lock_root(self, root_state: tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]) -> None:
        if not self.lock_free_base:
            return
        qadr, dadr, qpos, _ = root_state
        if qadr is None or dadr is None or qpos is None:
            return
        self.data.qpos[qadr : qadr + 7] = qpos
        self.data.qvel[dadr : dadr + 6] = 0.0

    def _reset(self, side_cfg: dict[str, Any]) -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]:
        self.mujoco.mj_resetData(self.model, self.data)
        if np.isfinite(self.base_z):
            for joint_id in range(self.model.njnt):
                if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                    self.data.qpos[int(self.model.jnt_qposadr[joint_id]) + 2] = self.base_z
                    break

        pitch_qadr, pitch_dadr = self._joint_info(side_cfg["pitch_joint"])
        roll_qadr, roll_dadr = self._joint_info(side_cfg["roll_joint"])
        q_pr_zero = np.asarray(side_cfg.get("q_pr_zero", [0.0, 0.0]), dtype=float)
        self.data.qpos[pitch_qadr] = q_pr_zero[0]
        self.data.qpos[roll_qadr] = q_pr_zero[1]
        self.data.qvel[pitch_dadr] = 0.0
        self.data.qvel[roll_dadr] = 0.0

        for joint_name, cfg in self.config.get("hold_joints", {}).items():
            qadr, dadr = self._joint_info(joint_name)
            self.data.qpos[qadr] = parse_float(cfg.get("q", 0.0), 0.0)
            self.data.qvel[dadr] = 0.0
        self.mujoco.mj_forward(self.model, self.data)
        return self._root_state()

    def _read_pr(self, side_cfg: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
        pitch_qadr, pitch_dadr = self._joint_info(side_cfg["pitch_joint"])
        roll_qadr, roll_dadr = self._joint_info(side_cfg["roll_joint"])
        q_pr = np.array([self.data.qpos[pitch_qadr], self.data.qpos[roll_qadr]], dtype=float)
        dq_pr = np.array([self.data.qvel[pitch_dadr], self.data.qvel[roll_dadr]], dtype=float)
        return q_pr, dq_pr

    def _hold_non_ankle_joints(self) -> None:
        for joint_name, cfg in self.config.get("hold_joints", {}).items():
            qadr, dadr = self._joint_info(joint_name)
            act_id = self._actuator_id(joint_name)
            q = float(self.data.qpos[qadr])
            dq = float(self.data.qvel[dadr])
            q_des = parse_float(cfg.get("q", 0.0), 0.0)
            kp = parse_float(cfg.get("kp", 0.0), 0.0)
            kd = parse_float(cfg.get("kd", 0.0), 0.0)
            tau_limit = parse_float(cfg.get("tau_limit", math.inf), math.inf)
            tau = kp * (q_des - q) - kd * dq
            self.data.ctrl[act_id] = float(np.clip(tau, -tau_limit, tau_limit))

    def _warmup(self, side_cfg: dict[str, Any], root_state: tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]) -> None:
        steps = int(round(self.warmup_time / self.sim_dt))
        if steps <= 0:
            return
        pitch_act = self._actuator_id(side_cfg["pitch_actuator"])
        roll_act = self._actuator_id(side_cfg["roll_actuator"])
        for _ in range(steps):
            self.data.ctrl[pitch_act] = 0.0
            self.data.ctrl[roll_act] = 0.0
            self._hold_non_ankle_joints()
            self._lock_root(root_state)
            self.mujoco.mj_step(self.model, self.data)
            self._lock_root(root_state)

    def run_position_trial(
        self,
        side_name: str,
        axis: str,
        amplitude: float,
        frequency: float,
        viewer: bool = False,
        keep_open: bool = False,
    ) -> list[dict[str, Any]]:
        side_cfg = self.config["ankles"][side_name]
        mapping = load_linear_mapping(side_name, side_cfg)
        gains = load_motor_gains(side_name, side_cfg)
        limits = load_motor_limits(side_name, side_cfg)
        root_state = self._reset(side_cfg)
        self._warmup(side_cfg, root_state)
        self.data.time = 0.0

        pitch_act = self._actuator_id(side_cfg["pitch_actuator"])
        roll_act = self._actuator_id(side_cfg["roll_actuator"])
        q_pr_zero = np.asarray(side_cfg.get("q_pr_zero", [0.0, 0.0]), dtype=float)
        dq_des_mode = str(get_nested(self.config, "tests", "position_tracking", "dq_des_mode", default="zero"))
        phase_roll = parse_float(get_nested(self.config, "tests", "position_tracking", "phase_offset_roll", default=0.0), 0.0)

        trial_id = f"ab_{side_name}_{axis}_A{amplitude:g}_f{frequency:g}"
        rows: list[dict[str, Any]] = []
        n_control_steps = int(round(self.duration / self.control_dt)) + 1

        viewer_handle = None
        wall_start = time.perf_counter()
        if viewer:
            import mujoco.viewer

            viewer_handle = mujoco.viewer.launch_passive(self.model, self.data)
            viewer_handle.sync()

        try:
            for step_idx in range(n_control_steps):
                if viewer_handle is not None and not viewer_handle.is_running():
                    break
                t = step_idx * self.control_dt
                q_pr_des = np.array(q_pr_zero, copy=True)
                dq_pr_des = np.zeros(2)
                omega = 2.0 * math.pi * frequency

                if axis in ("pitch", "both"):
                    q_pr_des[0] += amplitude * math.sin(omega * t)
                    if dq_des_mode == "trajectory":
                        dq_pr_des[0] = omega * amplitude * math.cos(omega * t)
                if axis in ("roll", "both"):
                    q_pr_des[1] += amplitude * math.sin(omega * t + phase_roll)
                    if dq_des_mode == "trajectory":
                        dq_pr_des[1] = omega * amplitude * math.cos(omega * t + phase_roll)

                q_pr, dq_pr = self._read_pr(side_cfg)
                q_ab = mapping.pr_to_ab(q_pr)
                dq_ab = mapping.dpr_to_dab(dq_pr)
                q_ab_des = mapping.pr_to_ab(q_pr_des)
                dq_ab_des = mapping.dpr_to_dab(dq_pr_des)
                tau_ab_before = gains.kp * (q_ab_des - q_ab) + gains.kd * (dq_ab_des - dq_ab)
                q_ab_des_clip, dq_ab_des_clip, tau_ab_after, sat = clip_ab_command(
                    q_ab_des,
                    dq_ab_des,
                    tau_ab_before,
                    limits,
                )
                tau_pr = mapping.tau_ab_to_pr(tau_ab_after)
                self.data.ctrl[pitch_act] = tau_pr[0]
                self.data.ctrl[roll_act] = tau_pr[1]
                self._hold_non_ankle_joints()
                self.mujoco.mj_forward(self.model, self.data)

                rows.append(
                    {
                        "trial_id": trial_id,
                        "time": t,
                        "side": side_name,
                        "axis": axis,
                        "q_pitch_des": q_pr_des[0],
                        "q_roll_des": q_pr_des[1],
                        "dq_pitch_des": dq_pr_des[0],
                        "dq_roll_des": dq_pr_des[1],
                        "q_pitch_sim": q_pr[0],
                        "q_roll_sim": q_pr[1],
                        "dq_pitch_sim": dq_pr[0],
                        "dq_roll_sim": dq_pr[1],
                        "q_A_des": q_ab_des_clip[0],
                        "q_B_des": q_ab_des_clip[1],
                        "dq_A_des": dq_ab_des_clip[0],
                        "dq_B_des": dq_ab_des_clip[1],
                        "q_A_sim": q_ab[0],
                        "q_B_sim": q_ab[1],
                        "dq_A_sim": dq_ab[0],
                        "dq_B_sim": dq_ab[1],
                        "tau_A_before_clip": tau_ab_before[0],
                        "tau_B_before_clip": tau_ab_before[1],
                        "tau_A_after_clip": tau_ab_after[0],
                        "tau_B_after_clip": tau_ab_after[1],
                        "tau_pitch_equiv": tau_pr[0],
                        "tau_roll_equiv": tau_pr[1],
                        "sat_A": sat[0],
                        "sat_B": sat[1],
                    }
                )

                if viewer_handle is not None:
                    viewer_handle.sync()
                if step_idx == n_control_steps - 1:
                    break
                for _ in range(self.substeps):
                    self._lock_root(root_state)
                    self.mujoco.mj_step(self.model, self.data)
                    self._lock_root(root_state)
                if viewer_handle is not None:
                    target_time = wall_start + (step_idx + 1) * self.control_dt
                    sleep_time = target_time - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)

            if viewer_handle is not None and keep_open:
                print("Trial finished. Close MuJoCo viewer to exit.", flush=True)
                while viewer_handle.is_running():
                    viewer_handle.sync()
                    time.sleep(0.02)
        finally:
            if viewer_handle is not None:
                viewer_handle.close()
        return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            out = {}
            for key in CSV_COLUMNS:
                value = row.get(key, "")
                if isinstance(value, float):
                    out[key] = f"{value:.10g}" if np.isfinite(value) else ""
                else:
                    out[key] = value
            writer.writerow(out)


def print_required_params(config: dict[str, Any]) -> None:
    print("Required parameters to fill in ab_ankle_config.yaml:")
    for side_name in ("left", "right"):
        print(f"- ankles.{side_name}.mapping.matrix: 2x2 finite matrix")
        print(f"- ankles.{side_name}.mapping.offset: [A0, B0]")
        for motor_name in ("A", "B"):
            prefix = f"- ankles.{side_name}.motors.{motor_name}"
            print(f"{prefix}.kp / kd")
            print(f"{prefix}.q_limit: [min, max] rad")
            print(f"{prefix}.dq_limit: rad/s")
            print(f"{prefix}.tau_limit: Nm")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "ab_ankle_config.yaml"))
    parser.add_argument("--side", choices=["left", "right"], default=None)
    parser.add_argument("--axis", choices=["pitch", "roll", "both"], default=None)
    parser.add_argument("--amplitude", type=float, default=None)
    parser.add_argument("--frequency", type=float, default=None)
    parser.add_argument("--viewer", action="store_true")
    parser.add_argument("--keep-open", action="store_true")
    parser.add_argument("--print-required-params", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    if args.print_required_params:
        print_required_params(config)
        return

    test_cfg = get_nested(config, "tests", "position_tracking", default={}) or {}
    side = args.side or str(test_cfg.get("side", "left"))
    axis = args.axis or str(test_cfg.get("axis", "pitch"))
    amplitude = args.amplitude if args.amplitude is not None else float(test_cfg.get("amplitudes", [0.05])[0])
    frequency = args.frequency if args.frequency is not None else float(test_cfg.get("frequencies", [0.5])[0])

    sim = ABAnkleCouplingSimulator(config, config_path.parent)
    rows = sim.run_position_trial(
        side_name=side,
        axis=axis,
        amplitude=amplitude,
        frequency=frequency,
        viewer=args.viewer,
        keep_open=args.keep_open,
    )
    results_dir = resolve_path(str(get_nested(config, "output", "results_dir", default="results_ab_ankle")), config_path.parent)
    assert results_dir is not None
    csv_path = results_dir / str(get_nested(config, "output", "csv", default="ab_ankle_coupling_response.csv"))
    write_csv(csv_path, rows)
    print(f"Wrote {len(rows)} rows to {csv_path}")


if __name__ == "__main__":
    main()
