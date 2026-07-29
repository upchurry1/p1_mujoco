#!/usr/bin/env python3
"""Run P1 MIT-mode motor response tests in MuJoCo."""

from __future__ import annotations

import argparse
import csv
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from analyze_motor_log import analyze_logs, canonical_joint_name, get_nested, resolve_path

SCRIPT_DIR = Path(__file__).resolve().parent

CSV_COLUMNS = [
    "trial_id",
    "test_type",
    "time",
    "joint_name",
    "q_des",
    "dq_des",
    "q_sim",
    "dq_sim",
    "kp",
    "kd",
    "tau_ff",
    "tau_cmd_before_clip",
    "tau_cmd_after_clip",
    "actuator_force",
    "q_error",
    "dq_error",
    "saturation_flag",
    "amplitude",
    "torque_amplitude",
    "frequency",
    "control_dt",
    "sim_dt",
    "tau_limit",
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


def get_joint_scalar(config: dict[str, Any], section: str, joint_name: str, default: float = 0.0) -> float:
    values = config.get(section, {})
    return parse_float(values.get(joint_name, values.get("default", default)), default)


def get_joint_gains(config: dict[str, Any], section: str, joint_name: str) -> tuple[float, float]:
    values = config.get(section, {})
    default = values.get("default", {})
    joint_values = values.get(joint_name, {})
    kp = parse_float(joint_values.get("kp", default.get("kp", 0.0)), 0.0)
    kd = parse_float(joint_values.get("kd", default.get("kd", 0.0)), 0.0)
    return kp, kd


def load_mujoco():
    try:
        import mujoco
    except ImportError as exc:
        raise SystemExit(
            "Python package 'mujoco' is not installed in this environment. "
            "Run with the conda env that contains MuJoCo, for example:\n"
            "/home/hr/miniconda3/bin/conda run -n my_robot_isaaclab "
            "python motor_model_validation/run_mujoco_motor_test.py"
        ) from exc
    return mujoco


class P1MitSimulator:
    def __init__(self, config: dict[str, Any], config_dir: Path) -> None:
        self.config = config
        self.config_dir = config_dir
        self.mujoco = load_mujoco()
        xml_path = resolve_path(str(get_nested(config, "model", "xml_path", default="../unitree_robots/p1/p1.xml")), config_dir)
        if xml_path is None or not xml_path.exists():
            raise FileNotFoundError(f"MuJoCo XML not found: {xml_path}")
        self.model = self.mujoco.MjModel.from_xml_path(str(xml_path))
        self.sim_dt = parse_float(get_nested(config, "simulation", "sim_dt", default=self.model.opt.timestep), self.model.opt.timestep)
        self.control_dt = parse_float(get_nested(config, "simulation", "control_dt", default=0.02), 0.02)
        self.duration = parse_float(get_nested(config, "simulation", "duration", default=8.0), 8.0)
        self.warmup_time = parse_float(get_nested(config, "simulation", "warmup_time", default=0.0), 0.0)
        self.base_z = parse_float(get_nested(config, "simulation", "base_z", default=math.nan), math.nan)
        self.lock_free_base = bool(get_nested(config, "simulation", "lock_free_base", default=True))
        self.hold_other_joints = bool(get_nested(config, "simulation", "hold_other_joints", default=True))
        self.saturation_threshold = parse_float(get_nested(config, "analysis", "saturation_threshold", default=0.98), 0.98)

        self.model.opt.timestep = self.sim_dt
        gravity = get_nested(config, "simulation", "gravity", default=None)
        if gravity is not None:
            self.model.opt.gravity[:] = np.asarray(gravity, dtype=float)
        ratio = self.control_dt / self.sim_dt
        self.substeps = int(round(ratio))
        if self.substeps < 1 or abs(ratio - self.substeps) > 1e-9:
            raise ValueError(f"control_dt ({self.control_dt}) must be an integer multiple of sim_dt ({self.sim_dt})")

        self.all_controlled_joints = [canonical_joint_name(name) for name in config.get("joints", [])]
        if not self.all_controlled_joints:
            raise ValueError("config.joints must contain at least one joint")
        self.joint_info = {joint: self._resolve_joint(joint) for joint in self.all_controlled_joints}
        self._apply_configured_torque_limits()

    def _name2id(self, obj_type: Any, name: str) -> int:
        idx = self.mujoco.mj_name2id(self.model, obj_type, name)
        if idx < 0:
            raise KeyError(name)
        return int(idx)

    def _resolve_joint(self, joint_name: str) -> dict[str, int]:
        joint_id = self._name2id(self.mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        actuator_id = self._name2id(self.mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name)
        actuator_joint_id = int(self.model.actuator_trnid[actuator_id, 0])
        if actuator_joint_id != joint_id:
            raise ValueError(f"Actuator {joint_name} is not attached to joint {joint_name}")
        if int(self.model.actuator_biastype[actuator_id]) != 0:
            raise ValueError(f"Actuator {joint_name} is not a pure torque motor; MIT mode requires <motor>, not position actuator")
        return {
            "joint_id": joint_id,
            "actuator_id": actuator_id,
            "qpos_adr": int(self.model.jnt_qposadr[joint_id]),
            "dof_adr": int(self.model.jnt_dofadr[joint_id]),
        }

    def _apply_configured_torque_limits(self) -> None:
        for joint_name, info in self.joint_info.items():
            tau_limit = get_joint_scalar(self.config, "torque_limits", joint_name, math.nan)
            if not np.isfinite(tau_limit) or tau_limit <= 0.0:
                raise ValueError(f"Invalid torque limit for {joint_name}: {tau_limit}")
            act_id = info["actuator_id"]
            self.model.actuator_ctrllimited[act_id] = 1
            self.model.actuator_ctrlrange[act_id, 0] = -tau_limit
            self.model.actuator_ctrlrange[act_id, 1] = tau_limit

    def _new_data(self):
        data = self.mujoco.MjData(self.model)
        self.mujoco.mj_resetData(self.model, data)
        self._set_initial_base_height(data)
        for joint_name, info in self.joint_info.items():
            data.qpos[info["qpos_adr"]] = get_joint_scalar(self.config, "q0", joint_name, 0.0)
            data.qvel[info["dof_adr"]] = 0.0
        self.mujoco.mj_forward(self.model, data)
        return data

    def _set_initial_base_height(self, data: Any) -> None:
        if not np.isfinite(self.base_z):
            return
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                qadr = int(self.model.jnt_qposadr[joint_id])
                data.qpos[qadr + 2] = self.base_z
                return

    def _root_state(self, data: Any) -> tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]:
        for joint_id in range(self.model.njnt):
            if int(self.model.jnt_type[joint_id]) == int(self.mujoco.mjtJoint.mjJNT_FREE):
                qadr = int(self.model.jnt_qposadr[joint_id])
                dadr = int(self.model.jnt_dofadr[joint_id])
                return qadr, dadr, np.array(data.qpos[qadr : qadr + 7], copy=True), np.array(data.qvel[dadr : dadr + 6], copy=True)
        return None, None, None, None

    def _lock_root(self, data: Any, root_state: tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]) -> None:
        if not self.lock_free_base:
            return
        qadr, dadr, root_qpos, root_qvel = root_state
        if qadr is None or dadr is None or root_qpos is None or root_qvel is None:
            return
        data.qpos[qadr : qadr + 7] = root_qpos
        data.qvel[dadr : dadr + 6] = 0.0 * root_qvel

    def _mit_command(
        self,
        data: Any,
        joint_name: str,
        q_des: float,
        dq_des: float,
        kp: float,
        kd: float,
        tau_ff: float,
    ) -> dict[str, float]:
        info = self.joint_info[joint_name]
        q = float(data.qpos[info["qpos_adr"]])
        dq = float(data.qvel[info["dof_adr"]])
        q_error = q_des - q
        dq_error = dq_des - dq
        tau_before = kp * q_error + kd * dq_error + tau_ff
        tau_limit = get_joint_scalar(self.config, "torque_limits", joint_name)
        tau_after = float(np.clip(tau_before, -tau_limit, tau_limit))
        return {
            "q": q,
            "dq": dq,
            "q_error": q_error,
            "dq_error": dq_error,
            "tau_limit": tau_limit,
            "tau_cmd_before_clip": tau_before,
            "tau_cmd_after_clip": tau_after,
            "saturation_flag": float(abs(tau_after) >= self.saturation_threshold * tau_limit),
        }

    def _set_all_controls(
        self,
        data: Any,
        tested_joint: str,
        tested_command: dict[str, float],
    ) -> None:
        for joint_name, info in self.joint_info.items():
            if joint_name == tested_joint:
                tau = tested_command["tau_cmd_after_clip"]
            elif self.hold_other_joints:
                q0 = get_joint_scalar(self.config, "q0", joint_name, 0.0)
                kp, kd = get_joint_gains(self.config, "hold_gains", joint_name)
                hold_command = self._mit_command(data, joint_name, q0, 0.0, kp, kd, 0.0)
                tau = hold_command["tau_cmd_after_clip"]
            else:
                tau = 0.0
            data.ctrl[info["actuator_id"]] = tau

    def _warmup(self, data: Any, root_state: tuple[int | None, int | None, np.ndarray | None, np.ndarray | None]) -> None:
        if self.warmup_time <= 0.0:
            return
        steps = int(round(self.warmup_time / self.sim_dt))
        for _ in range(steps):
            for joint_name, info in self.joint_info.items():
                q0 = get_joint_scalar(self.config, "q0", joint_name, 0.0)
                kp, kd = get_joint_gains(self.config, "hold_gains", joint_name)
                command = self._mit_command(data, joint_name, q0, 0.0, kp, kd, 0.0)
                data.ctrl[info["actuator_id"]] = command["tau_cmd_after_clip"]
            self._lock_root(data, root_state)
            self.mujoco.mj_step(self.model, data)
            self._lock_root(data, root_state)

    def run_position_trial(
        self,
        joint_name: str,
        amplitude: float,
        frequency: float,
        viewer: bool = False,
        realtime: bool = False,
        keep_open: bool = False,
    ) -> list[dict[str, Any]]:
        test_cfg = get_nested(self.config, "tests", "position_tracking", default={}) or {}
        dq_des_mode = str(test_cfg.get("dq_des_mode", "zero")).strip().lower()
        tau_ff = parse_float(test_cfg.get("tau_ff", 0.0), 0.0)
        kp, kd = get_joint_gains(self.config, "mit_gains", joint_name)
        q0 = get_joint_scalar(self.config, "q0", joint_name, 0.0)
        trial_id = f"position_{joint_name}_A{amplitude:g}_f{frequency:g}"
        return self._run_trial(
            trial_id=trial_id,
            test_type="position_tracking",
            tested_joint=joint_name,
            amplitude=amplitude,
            torque_amplitude=math.nan,
            frequency=frequency,
            kp=kp,
            kd=kd,
            viewer=viewer,
            realtime=realtime,
            keep_open=keep_open,
            trajectory=lambda t: (
                q0 + amplitude * math.sin(2.0 * math.pi * frequency * t),
                2.0 * math.pi * frequency * amplitude * math.cos(2.0 * math.pi * frequency * t)
                if dq_des_mode == "trajectory"
                else 0.0,
                tau_ff,
            ),
        )

    def run_torque_trial(
        self,
        joint_name: str,
        torque_amplitude: float,
        frequency: float,
        viewer: bool = False,
        realtime: bool = False,
        keep_open: bool = False,
    ) -> list[dict[str, Any]]:
        q0 = get_joint_scalar(self.config, "q0", joint_name, 0.0)
        trial_id = f"torque_{joint_name}_Atau{torque_amplitude:g}_f{frequency:g}"
        return self._run_trial(
            trial_id=trial_id,
            test_type="torque_response",
            tested_joint=joint_name,
            amplitude=math.nan,
            torque_amplitude=torque_amplitude,
            frequency=frequency,
            kp=0.0,
            kd=0.0,
            viewer=viewer,
            realtime=realtime,
            keep_open=keep_open,
            trajectory=lambda t: (
                q0,
                0.0,
                torque_amplitude * math.sin(2.0 * math.pi * frequency * t),
            ),
        )

    def _run_trial(
        self,
        trial_id: str,
        test_type: str,
        tested_joint: str,
        amplitude: float,
        torque_amplitude: float,
        frequency: float,
        kp: float,
        kd: float,
        trajectory: Any,
        viewer: bool = False,
        realtime: bool = False,
        keep_open: bool = False,
    ) -> list[dict[str, Any]]:
        data = self._new_data()
        root_state = self._root_state(data)
        self._warmup(data, root_state)
        data.time = 0.0

        rows: list[dict[str, Any]] = []
        n_control_steps = int(round(self.duration / self.control_dt)) + 1
        info = self.joint_info[tested_joint]
        tau_limit = get_joint_scalar(self.config, "torque_limits", tested_joint)

        viewer_handle = None
        wall_start = time.perf_counter()
        if viewer:
            try:
                import mujoco.viewer

                viewer_handle = mujoco.viewer.launch_passive(self.model, data)
                viewer_handle.sync()
            except Exception as exc:
                raise RuntimeError(
                    "Failed to open MuJoCo viewer. Check DISPLAY/OpenGL on this machine."
                ) from exc

        try:
            for step_idx in range(n_control_steps):
                if viewer_handle is not None and not viewer_handle.is_running():
                    break
                t = step_idx * self.control_dt
                q_des, dq_des, tau_ff = trajectory(t)
                command = self._mit_command(data, tested_joint, q_des, dq_des, kp, kd, tau_ff)
                self._set_all_controls(data, tested_joint, command)
                self.mujoco.mj_forward(self.model, data)
                rows.append(
                    {
                        "trial_id": trial_id,
                        "test_type": test_type,
                        "time": t,
                        "joint_name": tested_joint,
                        "q_des": q_des,
                        "dq_des": dq_des,
                        "q_sim": command["q"],
                        "dq_sim": command["dq"],
                        "kp": kp,
                        "kd": kd,
                        "tau_ff": tau_ff,
                        "tau_cmd_before_clip": command["tau_cmd_before_clip"],
                        "tau_cmd_after_clip": command["tau_cmd_after_clip"],
                        "actuator_force": float(data.actuator_force[info["actuator_id"]]),
                        "q_error": command["q_error"],
                        "dq_error": command["dq_error"],
                        "saturation_flag": command["saturation_flag"],
                        "amplitude": amplitude,
                        "torque_amplitude": torque_amplitude,
                        "frequency": frequency,
                        "control_dt": self.control_dt,
                        "sim_dt": self.sim_dt,
                        "tau_limit": tau_limit,
                    }
                )
                if viewer_handle is not None:
                    viewer_handle.sync()
                if step_idx == n_control_steps - 1:
                    break
                for _ in range(self.substeps):
                    self._lock_root(data, root_state)
                    self.mujoco.mj_step(self.model, data)
                    self._lock_root(data, root_state)
                if viewer_handle is not None and realtime:
                    target_time = wall_start + (step_idx + 1) * self.control_dt
                    sleep_time = target_time - time.perf_counter()
                    if sleep_time > 0.0:
                        time.sleep(sleep_time)
            if viewer_handle is not None and keep_open:
                print("Trial finished. Close the MuJoCo viewer window to exit.", flush=True)
                while viewer_handle.is_running():
                    viewer_handle.sync()
                    time.sleep(0.02)
        finally:
            if viewer_handle is not None:
                viewer_handle.close()
        return rows

    def run_visual_trial(
        self,
        joint_name: str | None,
        test_type: str,
        amplitude: float | None,
        torque_amplitude: float | None,
        frequency: float | None,
        keep_open: bool,
    ) -> list[dict[str, Any]]:
        joint = canonical_joint_name(joint_name or self.all_controlled_joints[0])
        if joint not in self.joint_info:
            raise ValueError(f"Unknown joint in config/model: {joint}")

        if test_type == "position_tracking":
            pos_cfg = get_nested(self.config, "tests", "position_tracking", default={}) or {}
            amp = float(amplitude if amplitude is not None else pos_cfg.get("amplitudes", [0.05])[0])
            freq = float(frequency if frequency is not None else pos_cfg.get("frequencies", [0.5])[0])
            print(f"Opening MuJoCo viewer: position_tracking {joint}, A={amp:g}, f={freq:g}Hz", flush=True)
            return self.run_position_trial(joint, amp, freq, viewer=True, realtime=True, keep_open=keep_open)

        torque_cfg = get_nested(self.config, "tests", "torque_response", default={}) or {}
        amp_tau = float(torque_amplitude if torque_amplitude is not None else torque_cfg.get("torque_amplitudes", [5.0])[0])
        freq = float(frequency if frequency is not None else torque_cfg.get("frequencies", [0.5])[0])
        print(f"Opening MuJoCo viewer: torque_response {joint}, Atau={amp_tau:g}, f={freq:g}Hz", flush=True)
        return self.run_torque_trial(joint, amp_tau, freq, viewer=True, realtime=True, keep_open=keep_open)

    def run_all(self, selected_joints: list[str] | None = None, selected_tests: list[str] | None = None) -> list[dict[str, Any]]:
        joints = [canonical_joint_name(name) for name in (selected_joints or self.all_controlled_joints)]
        unknown = [joint for joint in joints if joint not in self.joint_info]
        if unknown:
            raise ValueError(f"Unknown joints in config/model: {unknown}")
        selected_tests = selected_tests or ["position_tracking", "torque_response"]

        rows: list[dict[str, Any]] = []
        pos_cfg = get_nested(self.config, "tests", "position_tracking", default={}) or {}
        total_trials = 0
        for test_name in selected_tests:
            if test_name == "position_tracking" and bool(pos_cfg.get("enabled", True)):
                total_trials += len(joints) * len(pos_cfg.get("amplitudes", [])) * len(pos_cfg.get("frequencies", []))
            if test_name == "torque_response":
                torque_cfg_count = get_nested(self.config, "tests", "torque_response", default={}) or {}
                if bool(torque_cfg_count.get("enabled", True)):
                    total_trials += len(joints) * len(torque_cfg_count.get("torque_amplitudes", [])) * len(torque_cfg_count.get("frequencies", []))
        trial_index = 0
        print(
            f"Running {total_trials} headless batch trials. Use --viewer for one visual MuJoCo trial.",
            flush=True,
        )
        if "position_tracking" in selected_tests and bool(pos_cfg.get("enabled", True)):
            for joint_name in joints:
                for amplitude in pos_cfg.get("amplitudes", []):
                    for frequency in pos_cfg.get("frequencies", []):
                        trial_index += 1
                        print(
                            f"[{trial_index}/{total_trials}] position_tracking {joint_name}, "
                            f"A={float(amplitude):g}, f={float(frequency):g}Hz",
                            flush=True,
                        )
                        rows.extend(self.run_position_trial(joint_name, float(amplitude), float(frequency)))

        torque_cfg = get_nested(self.config, "tests", "torque_response", default={}) or {}
        if "torque_response" in selected_tests and bool(torque_cfg.get("enabled", True)):
            for joint_name in joints:
                for torque_amplitude in torque_cfg.get("torque_amplitudes", []):
                    for frequency in torque_cfg.get("frequencies", []):
                        trial_index += 1
                        print(
                            f"[{trial_index}/{total_trials}] torque_response {joint_name}, "
                            f"Atau={float(torque_amplitude):g}, f={float(frequency):g}Hz",
                            flush=True,
                        )
                        rows.extend(self.run_torque_trial(joint_name, float(torque_amplitude), float(frequency)))
        return rows


def write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_cell(row.get(key, "")) for key in CSV_COLUMNS})


def format_cell(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return value


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--real-csv", default=None, help="Optional hardware MIT log CSV to compare after simulation")
    parser.add_argument("--results-dir", default=None, help="Override output.results_dir")
    parser.add_argument("--joints", nargs="*", default=None, help="Optional subset of joints")
    parser.add_argument(
        "--test-types",
        nargs="*",
        choices=["position_tracking", "torque_response"],
        default=None,
        help="Optional subset of tests",
    )
    parser.add_argument("--viewer", action="store_true", help="Open MuJoCo viewer and run one realtime trial")
    parser.add_argument("--viewer-joint", default=None, help="Joint used by --viewer; defaults to the first configured joint")
    parser.add_argument(
        "--viewer-test",
        choices=["position_tracking", "torque_response"],
        default="position_tracking",
        help="Test type used by --viewer",
    )
    parser.add_argument("--amplitude", type=float, default=None, help="Position sine amplitude for --viewer position_tracking")
    parser.add_argument("--torque-amplitude", type=float, default=None, help="Torque sine amplitude for --viewer torque_response")
    parser.add_argument("--frequency", type=float, default=None, help="Sine frequency for --viewer")
    parser.add_argument("--keep-open", action="store_true", help="Keep MuJoCo viewer open after the visual trial finishes")
    parser.add_argument("--no-warmup", action="store_true", help="Start each trial exactly from config q0 instead of settling first")
    parser.add_argument("--warmup-time", type=float, default=None, help="Override simulation.warmup_time from config")
    parser.add_argument("--base-z", type=float, default=None, help="Override simulation.base_z root/pelvis height")
    parser.add_argument("--free-base", action="store_true", help="Do not lock the floating base during the motor test")
    parser.add_argument("--no-hold-other-joints", action="store_true", help="Set non-tested configured joint torques to zero")
    parser.add_argument("--no-analysis", action="store_true", help="Only write simulation CSV")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG plots during analysis")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    config_dir = config_path.parent

    results_dir = resolve_path(args.results_dir, config_dir) if args.results_dir else resolve_path(
        str(get_nested(config, "output", "results_dir", default="results")), config_dir
    )
    assert results_dir is not None
    sim_csv = results_dir / str(get_nested(config, "output", "sim_csv", default="sim_motor_response.csv"))
    metrics_csv = results_dir / str(get_nested(config, "output", "metrics_csv", default="motor_metrics.csv"))
    real_csv = resolve_path(args.real_csv, Path.cwd()) if args.real_csv else None

    simulator = P1MitSimulator(config, config_dir)
    if args.no_warmup:
        simulator.warmup_time = 0.0
    if args.warmup_time is not None:
        simulator.warmup_time = args.warmup_time
    if args.base_z is not None:
        simulator.base_z = args.base_z
    if args.free_base:
        simulator.lock_free_base = False
    if args.no_hold_other_joints:
        simulator.hold_other_joints = False

    print(
        "Mode: full P1 model with MIT torque actuators "
        f"(lock_free_base={simulator.lock_free_base}, "
        f"hold_other_joints={simulator.hold_other_joints}, "
        f"warmup_time={simulator.warmup_time:g}s, "
        f"base_z={simulator.base_z:g}m)",
        flush=True,
    )
    if args.viewer:
        rows = simulator.run_visual_trial(
            joint_name=args.viewer_joint or (args.joints[0] if args.joints else None),
            test_type=args.viewer_test,
            amplitude=args.amplitude,
            torque_amplitude=args.torque_amplitude,
            frequency=args.frequency,
            keep_open=args.keep_open,
        )
    else:
        rows = simulator.run_all(selected_joints=args.joints, selected_tests=args.test_types)
    write_rows(sim_csv, rows)
    print(f"Wrote {len(rows)} simulation rows to {sim_csv}")

    if not args.no_analysis:
        metrics = analyze_logs(
            sim_csv=sim_csv,
            real_csv=real_csv,
            config=config,
            results_dir=results_dir,
            metrics_csv=metrics_csv,
            make_plots=not args.no_plots,
        )
        print(f"Wrote {len(metrics)} metric rows to {metrics_csv}")


if __name__ == "__main__":
    main()
