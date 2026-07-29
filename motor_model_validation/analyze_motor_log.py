#!/usr/bin/env python3
"""Analyze P1 MIT motor response logs from MuJoCo and hardware."""

from __future__ import annotations

import argparse
import csv
import math
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import yaml

SCRIPT_DIR = Path(__file__).resolve().parent


ALIASES = {
    "left_hip_pitch_joint": "hip_pitch_l_joint",
    "left_knee_joint": "knee_pitch_l_joint",
    "left_ankle_pitch_joint": "ankle_pitch_l_joint",
    "left_ankle_roll_joint": "ankle_roll_l_joint",
    "right_hip_pitch_joint": "hip_pitch_r_joint",
    "right_knee_joint": "knee_pitch_r_joint",
    "right_ankle_pitch_joint": "ankle_pitch_r_joint",
    "right_ankle_roll_joint": "ankle_roll_r_joint",
}

NUMERIC_COLUMNS = {
    "time",
    "q_des",
    "dq_des",
    "q_sim",
    "dq_sim",
    "q_real",
    "dq_real",
    "kp",
    "kd",
    "tau_ff",
    "tau_est",
    "motor_current",
    "tau_real",
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
}

METRIC_COLUMNS = [
    "trial_id",
    "test_type",
    "joint_name",
    "amplitude",
    "torque_amplitude",
    "frequency",
    "kp",
    "kd",
    "tau_limit",
    "sim_position_rmse",
    "real_position_rmse",
    "sim_velocity_rmse",
    "real_velocity_rmse",
    "sim_max_position_error",
    "real_max_position_error",
    "q_amplitude_sim",
    "q_amplitude_real",
    "amplitude_ratio_real_over_sim",
    "phase_delay_ms_q_sim_vs_q_des",
    "phase_delay_ms_real_vs_sim",
    "torque_gain_a",
    "torque_gain_b",
    "sim_saturation_ratio",
    "real_saturation_ratio",
    "real_match_status",
]


def canonical_joint_name(name: Any) -> str:
    raw = str(name).strip()
    return ALIASES.get(raw, raw)


def load_config(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_path(path_text: str | None, base_dir: Path) -> Path | None:
    if not path_text:
        return None
    path = Path(path_text).expanduser()
    if path.is_absolute():
        return path
    if (base_dir / path).exists() or not (Path.cwd() / path).exists():
        return (base_dir / path).resolve()
    return (Path.cwd() / path).resolve()


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


def parse_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    return str(value).strip().lower() in {"1", "true", "yes", "y"}


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows: list[dict[str, Any]] = []
        for row in reader:
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key is None:
                    continue
                key = key.strip()
                if key in NUMERIC_COLUMNS:
                    parsed[key] = parse_float(value)
                else:
                    parsed[key] = value.strip() if isinstance(value, str) else value
            if "joint_name" in parsed:
                parsed["joint_name"] = canonical_joint_name(parsed["joint_name"])
            rows.append(parsed)
    return rows


def write_csv(path: Path, rows: Iterable[dict[str, Any]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in columns})


def format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.10g}"
    return value


def get_nested(config: dict[str, Any], *keys: str, default: Any = None) -> Any:
    value: Any = config
    for key in keys:
        if not isinstance(value, dict) or key not in value:
            return default
        value = value[key]
    return value


def get_torque_limit(config: dict[str, Any], joint_name: str) -> float:
    limits = config.get("torque_limits", {})
    value = limits.get(joint_name, limits.get("default", math.nan))
    return parse_float(value)


def get_current_to_torque(config: dict[str, Any], joint_name: str) -> float:
    gains = get_nested(config, "analysis", "motor_current_to_torque", default={}) or {}
    return parse_float(gains.get(joint_name, gains.get("default", 1.0)), 1.0)


def enrich_real_rows(rows: list[dict[str, Any]], config: dict[str, Any]) -> None:
    real_torque_column = get_nested(config, "analysis", "real_torque_column", default="auto")
    threshold = parse_float(get_nested(config, "analysis", "saturation_threshold", default=0.98), 0.98)
    for row in rows:
        joint_name = canonical_joint_name(row.get("joint_name", ""))
        row["joint_name"] = joint_name
        q_des = parse_float(row.get("q_des"))
        dq_des = parse_float(row.get("dq_des"), 0.0)
        q_real = parse_float(row.get("q_real"))
        dq_real = parse_float(row.get("dq_real"), 0.0)
        kp = parse_float(row.get("kp"), 0.0)
        kd = parse_float(row.get("kd"), 0.0)
        tau_ff = parse_float(row.get("tau_ff"), 0.0)
        tau_limit = get_torque_limit(config, joint_name)

        q_error = q_des - q_real
        dq_error = dq_des - dq_real
        tau_before = kp * q_error + kd * dq_error + tau_ff
        tau_after = float(np.clip(tau_before, -tau_limit, tau_limit)) if np.isfinite(tau_limit) else tau_before

        row.setdefault("dq_des", dq_des)
        row["q_error"] = q_error
        row["dq_error"] = dq_error
        row["tau_cmd_before_clip"] = tau_before
        row["tau_cmd_after_clip"] = tau_after
        row["tau_limit"] = tau_limit
        if np.isfinite(tau_limit) and tau_limit > 0.0:
            row["saturation_flag"] = float(abs(tau_after) >= threshold * tau_limit)
        else:
            row["saturation_flag"] = float(abs(tau_after - tau_before) > 1e-9)

        if real_torque_column == "tau_est" or (real_torque_column == "auto" and "tau_est" in row):
            row["tau_real"] = parse_float(row.get("tau_est"))
        elif real_torque_column == "motor_current" or (real_torque_column == "auto" and "motor_current" in row):
            row["tau_real"] = parse_float(row.get("motor_current")) * get_current_to_torque(config, joint_name)


def finite_array(rows: list[dict[str, Any]], column: str) -> np.ndarray:
    values = np.asarray([parse_float(row.get(column)) for row in rows], dtype=float)
    return values


def sorted_by_time(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: parse_float(row.get("time"), 0.0))


def trim_rows(rows: list[dict[str, Any]], transient_skip: float) -> list[dict[str, Any]]:
    if not rows or transient_skip <= 0.0:
        return rows
    first_time = parse_float(rows[0].get("time"), 0.0)
    return [row for row in rows if parse_float(row.get("time"), 0.0) >= first_time + transient_skip]


def rmse(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.sqrt(np.mean(values * values)))


def max_abs(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan
    return float(np.max(np.abs(values)))


def signal_amplitude(values: np.ndarray) -> float:
    values = values[np.isfinite(values)]
    if values.size < 2:
        return math.nan
    return float(0.5 * (np.max(values) - np.min(values)))


def saturation_ratio(rows: list[dict[str, Any]], threshold: float) -> float:
    if not rows:
        return math.nan
    flags = finite_array(rows, "saturation_flag")
    if np.isfinite(flags).any():
        return float(np.mean(flags[np.isfinite(flags)] > 0.5))

    before = finite_array(rows, "tau_cmd_before_clip")
    after = finite_array(rows, "tau_cmd_after_clip")
    finite = np.isfinite(before) & np.isfinite(after)
    if finite.any():
        return float(np.mean(np.abs(before[finite] - after[finite]) > 1e-9))

    tau = after
    limit = finite_array(rows, "tau_limit")
    finite = np.isfinite(tau) & np.isfinite(limit) & (limit > 0.0)
    if finite.any():
        return float(np.mean(np.abs(tau[finite]) >= threshold * limit[finite]))
    return math.nan


def unique_time_series(rows: list[dict[str, Any]], value_column: str) -> tuple[np.ndarray, np.ndarray]:
    time = finite_array(rows, "time")
    values = finite_array(rows, value_column)
    finite = np.isfinite(time) & np.isfinite(values)
    time = time[finite]
    values = values[finite]
    if time.size == 0:
        return time, values
    order = np.argsort(time)
    time = time[order]
    values = values[order]
    unique_time, indices = np.unique(time, return_index=True)
    return unique_time, values[indices]


def interpolate_series(rows: list[dict[str, Any]], value_column: str, target_time: np.ndarray) -> np.ndarray:
    time, values = unique_time_series(rows, value_column)
    if time.size < 2 or target_time.size == 0:
        return np.full_like(target_time, np.nan, dtype=float)
    valid = (target_time >= time[0]) & (target_time <= time[-1])
    out = np.full_like(target_time, np.nan, dtype=float)
    out[valid] = np.interp(target_time[valid], time, values)
    return out


def phase_delay_ms(
    sim_time: np.ndarray,
    sim_values: np.ndarray,
    real_values: np.ndarray,
    max_lag_ms: float,
) -> float:
    finite = np.isfinite(sim_time) & np.isfinite(sim_values) & np.isfinite(real_values)
    sim_time = sim_time[finite]
    sim_values = sim_values[finite]
    real_values = real_values[finite]
    if sim_time.size < 4:
        return math.nan
    dt = float(np.median(np.diff(sim_time)))
    if not np.isfinite(dt) or dt <= 0.0:
        return math.nan
    max_lag = max(1, int(round(max_lag_ms / 1000.0 / dt)))
    max_lag = min(max_lag, sim_values.size - 2)
    x = sim_values - np.mean(sim_values)
    y = real_values - np.mean(real_values)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return math.nan

    best_lag = 0
    best_corr = -np.inf
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            xs = x[:-lag]
            ys = y[lag:]
        elif lag < 0:
            xs = x[-lag:]
            ys = y[:lag]
        else:
            xs = x
            ys = y
        if xs.size < 4:
            continue
        denom = np.linalg.norm(xs) * np.linalg.norm(ys)
        if denom <= 1e-12:
            continue
        corr = float(np.dot(xs, ys) / denom)
        if corr > best_corr:
            best_corr = corr
            best_lag = lag
    return float(best_lag * dt * 1000.0)


def torque_gain(sim_tau: np.ndarray, real_tau: np.ndarray) -> tuple[float, float]:
    finite = np.isfinite(sim_tau) & np.isfinite(real_tau)
    sim_tau = sim_tau[finite]
    real_tau = real_tau[finite]
    if sim_tau.size < 2 or np.std(sim_tau) < 1e-12:
        return math.nan, math.nan
    a, b = np.polyfit(sim_tau, real_tau, 1)
    return float(a), float(b)


def key_value(row: dict[str, Any], column: str) -> Any:
    value = row.get(column, "")
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return round(value, 9)
    return canonical_joint_name(value) if column == "joint_name" else value


def make_key(row: dict[str, Any], columns: list[str]) -> tuple[Any, ...]:
    return tuple(key_value(row, column) for column in columns)


def group_rows(rows: list[dict[str, Any]]) -> dict[tuple[Any, ...], list[dict[str, Any]]]:
    if not rows:
        return {}
    if any(row.get("trial_id") for row in rows):
        columns = ["trial_id"]
    elif any(row.get("test_type") for row in rows):
        columns = ["test_type", "joint_name", "amplitude", "torque_amplitude", "frequency"]
    else:
        columns = ["joint_name"]
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[make_key(row, columns)].append(row)
    return dict(grouped)


def candidate_real_matches(
    sim_group: list[dict[str, Any]],
    real_rows: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str]:
    if not real_rows:
        return [], "no_real_log"
    first = sim_group[0]
    candidates = [
        ["trial_id"],
        ["test_type", "joint_name", "amplitude", "torque_amplitude", "frequency"],
        ["test_type", "joint_name", "amplitude", "frequency"],
        ["joint_name", "amplitude", "frequency"],
        ["joint_name"],
    ]
    for columns in candidates:
        if not all(first.get(column, "") not in ("", None) for column in columns):
            continue
        key = make_key(first, columns)
        matched = [row for row in real_rows if make_key(row, columns) == key]
        if matched:
            return sorted_by_time(matched), "matched_by_" + "_".join(columns)
    return [], "no_matching_real_rows"


def first_value(rows: list[dict[str, Any]], column: str) -> Any:
    for row in rows:
        value = row.get(column, "")
        if value not in ("", None):
            return value
    return ""


def compute_group_metrics(
    sim_rows_in: list[dict[str, Any]],
    real_rows_all: list[dict[str, Any]],
    config: dict[str, Any],
) -> dict[str, Any]:
    transient_skip = parse_float(get_nested(config, "analysis", "transient_skip", default=0.0), 0.0)
    max_lag_ms = parse_float(get_nested(config, "analysis", "max_phase_lag_ms", default=500.0), 500.0)
    sat_threshold = parse_float(get_nested(config, "analysis", "saturation_threshold", default=0.98), 0.98)
    sim_torque_column = get_nested(config, "analysis", "sim_torque_column", default="actuator_force")

    sim_rows = trim_rows(sorted_by_time(sim_rows_in), transient_skip)
    real_match_rows, match_status = candidate_real_matches(sim_rows_in, real_rows_all)
    real_rows = trim_rows(real_match_rows, transient_skip)

    q_des = finite_array(sim_rows, "q_des")
    dq_des = finite_array(sim_rows, "dq_des")
    q_sim = finite_array(sim_rows, "q_sim")
    dq_sim = finite_array(sim_rows, "dq_sim")

    row: dict[str, Any] = {
        "trial_id": first_value(sim_rows_in, "trial_id"),
        "test_type": first_value(sim_rows_in, "test_type"),
        "joint_name": first_value(sim_rows_in, "joint_name"),
        "amplitude": first_value(sim_rows_in, "amplitude"),
        "torque_amplitude": first_value(sim_rows_in, "torque_amplitude"),
        "frequency": first_value(sim_rows_in, "frequency"),
        "kp": first_value(sim_rows_in, "kp"),
        "kd": first_value(sim_rows_in, "kd"),
        "tau_limit": first_value(sim_rows_in, "tau_limit"),
        "sim_position_rmse": rmse(q_des - q_sim),
        "sim_velocity_rmse": rmse(dq_des - dq_sim),
        "sim_max_position_error": max_abs(q_des - q_sim),
        "q_amplitude_sim": signal_amplitude(q_sim),
        "sim_saturation_ratio": saturation_ratio(sim_rows, sat_threshold),
        "real_match_status": match_status,
    }
    sim_time_for_delay, q_des_for_delay = unique_time_series(sim_rows, "q_des")
    q_sim_for_delay = interpolate_series(sim_rows, "q_sim", sim_time_for_delay)
    row["phase_delay_ms_q_sim_vs_q_des"] = phase_delay_ms(
        sim_time_for_delay,
        q_des_for_delay,
        q_sim_for_delay,
        max_lag_ms,
    )

    if real_rows:
        q_real = finite_array(real_rows, "q_real")
        dq_real = finite_array(real_rows, "dq_real")
        q_des_real = finite_array(real_rows, "q_des")
        dq_des_real = finite_array(real_rows, "dq_des")
        row["real_position_rmse"] = rmse(q_des_real - q_real)
        row["real_velocity_rmse"] = rmse(dq_des_real - dq_real)
        row["real_max_position_error"] = max_abs(q_des_real - q_real)
        row["q_amplitude_real"] = signal_amplitude(q_real)
        row["real_saturation_ratio"] = saturation_ratio(real_rows, sat_threshold)
        if np.isfinite(row["q_amplitude_sim"]) and row["q_amplitude_sim"] > 1e-12:
            row["amplitude_ratio_real_over_sim"] = row["q_amplitude_real"] / row["q_amplitude_sim"]

        sim_time, sim_q = unique_time_series(sim_rows, "q_sim")
        real_q_on_sim = interpolate_series(real_rows, "q_real", sim_time)
        row["phase_delay_ms_real_vs_sim"] = phase_delay_ms(sim_time, sim_q, real_q_on_sim, max_lag_ms)

        sim_tau_time, sim_tau = unique_time_series(sim_rows, str(sim_torque_column))
        real_tau_on_sim = interpolate_series(real_rows, "tau_real", sim_tau_time)
        a, b = torque_gain(sim_tau, real_tau_on_sim)
        row["torque_gain_a"] = a
        row["torque_gain_b"] = b

    return row


def safe_name(text: Any) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", str(text)).strip("_")


def maybe_import_matplotlib(results_dir: Path):
    os.environ.setdefault("MPLCONFIGDIR", str((results_dir / ".mplconfig").resolve()))
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        return plt
    except Exception as exc:  # pragma: no cover - depends on optional runtime package
        print(f"[WARN] matplotlib unavailable, skip plots: {exc}")
        return None


def configure_time_axis(ax: Any, config: dict[str, Any], zoom: bool = False) -> None:
    try:
        from matplotlib.ticker import MultipleLocator
    except Exception:
        ax.grid(True, alpha=0.3)
        return

    if zoom:
        major = parse_float(get_nested(config, "analysis", "plot_zoom", "major_tick", default=0.1), 0.1)
        minor = parse_float(get_nested(config, "analysis", "plot_zoom", "minor_tick", default=0.02), 0.02)
    else:
        major = parse_float(get_nested(config, "analysis", "plot_time_major_tick", default=0.5), 0.5)
        minor = parse_float(get_nested(config, "analysis", "plot_time_minor_tick", default=0.1), 0.1)
    if np.isfinite(major) and major > 0.0:
        ax.xaxis.set_major_locator(MultipleLocator(major))
    if np.isfinite(minor) and minor > 0.0:
        ax.xaxis.set_minor_locator(MultipleLocator(minor))
    ax.grid(True, which="major", alpha=0.3)
    ax.grid(True, which="minor", alpha=0.12)


def plot_position_axis(
    ax: Any,
    t: np.ndarray,
    q_des: np.ndarray,
    q_sim: np.ndarray,
    real_rows: list[dict[str, Any]],
    title: str,
    config: dict[str, Any],
    zoom: bool = False,
) -> None:
    ax.plot(t, q_des, label="q_des", linewidth=1.4)
    ax.plot(t, q_sim, label="q_sim", linewidth=1.2)
    if real_rows:
        ax.plot(finite_array(real_rows, "time"), finite_array(real_rows, "q_real"), label="q_real", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("position [rad]")
    ax.set_title(title)
    configure_time_axis(ax, config, zoom=zoom)
    ax.legend()


def plot_group(
    sim_rows_in: list[dict[str, Any]],
    real_rows_all: list[dict[str, Any]],
    config: dict[str, Any],
    plot_dir: Path,
    results_dir: Path,
) -> None:
    plt = maybe_import_matplotlib(results_dir)
    if plt is None:
        return

    sim_rows = sorted_by_time(sim_rows_in)
    real_rows, _ = candidate_real_matches(sim_rows_in, real_rows_all)
    title = (
        f"{first_value(sim_rows, 'trial_id')} "
        f"{first_value(sim_rows, 'joint_name')} "
        f"{first_value(sim_rows, 'frequency')}Hz"
    )
    stem = safe_name(first_value(sim_rows, "trial_id") or title)
    plot_dir.mkdir(parents=True, exist_ok=True)

    t = finite_array(sim_rows, "time")
    q_des = finite_array(sim_rows, "q_des")
    q_sim = finite_array(sim_rows, "q_sim")
    dq_des = finite_array(sim_rows, "dq_des")
    dq_sim = finite_array(sim_rows, "dq_sim")
    tau_after = finite_array(sim_rows, "tau_cmd_after_clip")
    actuator_force = finite_array(sim_rows, "actuator_force")

    delay_rows = trim_rows(sim_rows, parse_float(get_nested(config, "analysis", "transient_skip", default=0.0), 0.0))
    delay_t, delay_q_des = unique_time_series(delay_rows, "q_des")
    delay_q_sim = interpolate_series(delay_rows, "q_sim", delay_t)
    delay_ms = phase_delay_ms(
        delay_t,
        delay_q_des,
        delay_q_sim,
        parse_float(get_nested(config, "analysis", "max_phase_lag_ms", default=500.0), 500.0),
    )
    delay_text = f", q_sim delay={delay_ms:.1f}ms" if np.isfinite(delay_ms) else ""

    fig, ax = plt.subplots(figsize=(11, 5))
    plot_position_axis(ax, t, q_des, q_sim, real_rows, title + delay_text, config)
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_position.png", dpi=150)
    plt.close(fig)

    zoom_cfg = get_nested(config, "analysis", "plot_zoom", default={}) or {}
    if bool(zoom_cfg.get("enabled", True)):
        zoom_start = parse_float(zoom_cfg.get("start", get_nested(config, "analysis", "transient_skip", default=1.0)), 1.0)
        zoom_duration = parse_float(zoom_cfg.get("duration", 2.0), 2.0)
        if np.isfinite(zoom_start) and np.isfinite(zoom_duration) and zoom_duration > 0.0:
            fig, ax = plt.subplots(figsize=(11, 5))
            plot_position_axis(ax, t, q_des, q_sim, real_rows, title + delay_text, config, zoom=True)
            ax.set_xlim(zoom_start, zoom_start + zoom_duration)
            fig.tight_layout()
            fig.savefig(plot_dir / f"{stem}_position_zoom.png", dpi=170)
            plt.close(fig)

    if not bool(get_nested(config, "analysis", "plot_error_and_torque", default=False)):
        for suffix in ("tracking_error", "torque"):
            stale_path = plot_dir / f"{stem}_{suffix}.png"
            if stale_path.exists():
                stale_path.unlink()
        return

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, q_des - q_sim, label="q_des - q_sim", linewidth=1.2)
    ax.plot(t, dq_des - dq_sim, label="dq_des - dq_sim", linewidth=1.0)
    if real_rows:
        tr = finite_array(real_rows, "time")
        ax.plot(tr, finite_array(real_rows, "q_des") - finite_array(real_rows, "q_real"), label="q_des - q_real", linewidth=1.0)
        ax.plot(tr, finite_array(real_rows, "dq_des") - finite_array(real_rows, "dq_real"), label="dq_des - dq_real", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("error")
    ax.set_title(title)
    configure_time_axis(ax, config)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_tracking_error.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(t, tau_after, label="tau_cmd_after_clip", linewidth=1.2)
    ax.plot(t, actuator_force, label="actuator_force", linewidth=1.0)
    if real_rows and "tau_real" in real_rows[0]:
        ax.plot(finite_array(real_rows, "time"), finite_array(real_rows, "tau_real"), label="tau_real", linewidth=1.0)
    ax.set_xlabel("time [s]")
    ax.set_ylabel("torque [Nm]")
    ax.set_title(title)
    configure_time_axis(ax, config)
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / f"{stem}_torque.png", dpi=150)
    plt.close(fig)


def analyze_logs(
    sim_csv: Path,
    real_csv: Path | None,
    config: dict[str, Any],
    results_dir: Path,
    metrics_csv: Path | None = None,
    make_plots: bool | None = None,
) -> list[dict[str, Any]]:
    sim_rows = read_csv_rows(sim_csv)
    real_rows = read_csv_rows(real_csv) if real_csv else []
    enrich_real_rows(real_rows, config)

    grouped_sim = group_rows(sim_rows)
    metrics = [
        compute_group_metrics(group, real_rows, config)
        for _, group in sorted(grouped_sim.items(), key=lambda item: str(item[0]))
    ]

    if metrics_csv is None:
        metrics_csv = results_dir / str(get_nested(config, "output", "metrics_csv", default="motor_metrics.csv"))
    write_csv(metrics_csv, metrics, METRIC_COLUMNS)

    if make_plots is None:
        make_plots = bool(get_nested(config, "analysis", "make_plots", default=True))
    if make_plots:
        plot_dir = results_dir / str(get_nested(config, "output", "plot_dir", default="plots"))
        max_trials = int(get_nested(config, "analysis", "plot_max_trials", default=200))
        for _, group in list(sorted(grouped_sim.items(), key=lambda item: str(item[0])))[:max_trials]:
            plot_group(group, real_rows, config, plot_dir, results_dir)

    return metrics


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"), help="Path to config.yaml")
    parser.add_argument("--sim-csv", required=True, help="MuJoCo simulation CSV from run_mujoco_motor_test.py")
    parser.add_argument("--real-csv", default=None, help="Optional hardware MIT log CSV")
    parser.add_argument("--results-dir", default=None, help="Directory for metrics and plots")
    parser.add_argument("--metrics-csv", default=None, help="Metrics CSV path")
    parser.add_argument("--no-plots", action="store_true", help="Skip PNG generation")
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config).expanduser().resolve()
    config = load_config(config_path)
    base_dir = config_path.parent
    results_dir = resolve_path(args.results_dir, base_dir) if args.results_dir else resolve_path(
        str(get_nested(config, "output", "results_dir", default="results")), base_dir
    )
    assert results_dir is not None
    sim_csv = resolve_path(args.sim_csv, Path.cwd()) or Path(args.sim_csv).resolve()
    real_csv = resolve_path(args.real_csv, Path.cwd()) if args.real_csv else None
    metrics_csv = resolve_path(args.metrics_csv, results_dir) if args.metrics_csv else None

    metrics = analyze_logs(
        sim_csv=sim_csv,
        real_csv=real_csv,
        config=config,
        results_dir=results_dir,
        metrics_csv=metrics_csv,
        make_plots=not args.no_plots,
    )
    print(f"Wrote {len(metrics)} metric rows to {metrics_csv or results_dir / get_nested(config, 'output', 'metrics_csv', default='motor_metrics.csv')}")


if __name__ == "__main__":
    main()
