#!/usr/bin/env python3
"""Plot real2sim_replay outputs."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


MOTOR_NAMES = [
    "L hip roll",
    "L hip pitch",
    "L hip yaw",
    "L knee",
    "L ankle pitch",
    "L ankle roll",
    "R hip roll",
    "R hip pitch",
    "R hip yaw",
    "R knee",
    "R ankle pitch",
    "R ankle roll",
]

MODEL_NAMES = [
    "L hip roll",
    "R hip roll",
    "L hip pitch",
    "R hip pitch",
    "L hip yaw",
    "R hip yaw",
    "L knee",
    "R knee",
    "L ankle pitch",
    "R ankle pitch",
    "L ankle roll",
    "R ankle roll",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot action and motor comparisons from real2sim_replay CSV."
    )
    parser.add_argument("csv", type=Path, help="real2sim_replay result CSV")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory, default is <csv stem>_plots next to the CSV",
    )
    parser.add_argument("--start-time", type=float, default=None)
    parser.add_argument("--duration", type=float, default=None)
    parser.add_argument("--max-points", type=int, default=6000)
    return parser.parse_args()


def parse_float(text: str) -> float:
    text = (text or "").strip()
    if not text:
        return math.nan
    try:
        value = float(text)
    except ValueError:
        return math.nan
    return value if math.isfinite(value) else math.nan


def load_csv(path: Path) -> tuple[list[str], list[dict[str, float]]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        header = [name.strip() for name in reader.fieldnames]
        rows: list[dict[str, float]] = []
        for row in reader:
            parsed = {key: parse_float(row.get(key, "")) for key in header}
            if math.isfinite(parsed.get("timestamp", math.nan)):
                rows.append(parsed)
        if not rows:
            raise RuntimeError(f"CSV has no numeric rows: {path}")
    return header, rows


def filter_rows(
    rows: list[dict[str, float]], start_time: float | None, duration: float | None
) -> list[dict[str, float]]:
    if start_time is None:
        start_time = rows[0]["timestamp"]
    end_time = start_time + duration if duration is not None else None
    out = []
    for row in rows:
        timestamp = row["timestamp"]
        if timestamp < start_time:
            continue
        if end_time is not None and timestamp > end_time:
            continue
        out.append(row)
    if not out:
        raise RuntimeError("selected time window contains no rows")
    return out


def downsample(rows: list[dict[str, float]], max_points: int) -> list[dict[str, float]]:
    if max_points <= 0 or len(rows) <= max_points:
        return rows
    step = math.ceil(len(rows) / max_points)
    return rows[::step]


def has_columns(header: list[str], prefix: str, count: int = 12) -> bool:
    names = set(header)
    return all(f"{prefix}{i}" in names for i in range(count))


def series(rows: list[dict[str, float]], name: str) -> list[float]:
    return [row.get(name, math.nan) for row in rows]


def time_axis(rows: list[dict[str, float]]) -> list[float]:
    t0 = rows[0]["timestamp"]
    return [row["timestamp"] - t0 for row in rows]


def finite_pair_count(a: list[float], b: list[float]) -> int:
    return sum(1 for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y))


def rmse(a: list[float], b: list[float]) -> float:
    values = [(x - y) ** 2 for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    return math.sqrt(sum(values) / len(values)) if values else math.nan


def mae(a: list[float], b: list[float]) -> float:
    values = [abs(x - y) for x, y in zip(a, b) if math.isfinite(x) and math.isfinite(y)]
    return sum(values) / len(values) if values else math.nan


def plot_grid(
    rows: list[dict[str, float]],
    output_path: Path,
    title: str,
    traces: list[tuple[str, str, str]],
    ylabel: str,
    names: list[str],
) -> None:
    t = time_axis(rows)
    fig, axes = plt.subplots(4, 3, figsize=(18, 11), sharex=True)
    axes_flat = axes.flatten()
    for i, ax in enumerate(axes_flat):
        for prefix, label, style in traces:
            y = series(rows, f"{prefix}{i}")
            if any(math.isfinite(v) for v in y):
                ax.plot(t, y, style, linewidth=1.0, label=label)
        ax.set_title(f"M{i} {names[i]}", fontsize=9)
        ax.grid(True, alpha=0.25)
        if i % 3 == 0:
            ax.set_ylabel(ylabel)
        if i >= 9:
            ax.set_xlabel("time (s)")
        if i == 0:
            ax.legend(loc="best", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def plot_error_bars(
    rows: list[dict[str, float]],
    output_path: Path,
    comparisons: list[tuple[str, str, str]],
) -> None:
    labels = [f"M{i}" for i in range(12)]
    x = list(range(12))
    width = 0.8 / max(1, len(comparisons))
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for c, (lhs_prefix, rhs_prefix, label) in enumerate(comparisons):
        rmses = []
        maes = []
        for i in range(12):
            lhs = series(rows, f"{lhs_prefix}{i}")
            rhs = series(rows, f"{rhs_prefix}{i}")
            rmses.append(rmse(lhs, rhs))
            maes.append(mae(lhs, rhs))
        offset = (c - (len(comparisons) - 1) / 2) * width
        axes[0].bar([v + offset for v in x], rmses, width=width, label=label)
        axes[1].bar([v + offset for v in x], maes, width=width, label=label)
    axes[0].set_ylabel("RMSE")
    axes[1].set_ylabel("MAE")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    for ax in axes:
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(loc="best")
    fig.suptitle("Real2sim comparison errors")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def write_summary(
    rows: list[dict[str, float]],
    path: Path,
    comparisons: list[tuple[str, str, str]],
) -> None:
    with path.open("w", encoding="utf-8") as f:
        f.write(f"rows={len(rows)}\n")
        f.write(f"duration_s={rows[-1]['timestamp'] - rows[0]['timestamp']:.9g}\n")
        for lhs_prefix, rhs_prefix, label in comparisons:
            f.write(f"\n[{label}]\n")
            for i in range(12):
                lhs = series(rows, f"{lhs_prefix}{i}")
                rhs = series(rows, f"{rhs_prefix}{i}")
                f.write(
                    f"M{i}: pairs={finite_pair_count(lhs, rhs)} "
                    f"rmse={rmse(lhs, rhs):.9g} mae={mae(lhs, rhs):.9g}\n"
                )


def main() -> int:
    args = parse_args()
    header, rows = load_csv(args.csv)
    rows = filter_rows(rows, args.start_time, args.duration)
    plot_rows = downsample(rows, args.max_points)
    out_dir = args.out_dir or args.csv.with_suffix("").parent / f"{args.csv.stem}_plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    comparisons: list[tuple[str, str, str]] = []
    if has_columns(header, "predicted_raw_action_") and has_columns(header, "logged_raw_action_"):
        plot_grid(
            plot_rows,
            out_dir / "policy_raw_action_compare.png",
            "Policy raw action: predicted vs logged real",
            [
                ("predicted_raw_action_", "predicted", "-"),
                ("logged_raw_action_", "logged real", "--"),
            ],
            "raw action",
            MODEL_NAMES,
        )
        comparisons.append(("predicted_raw_action_", "logged_raw_action_", "raw action"))

    if has_columns(header, "q_target_model_rad_") and has_columns(header, "real_q_target_"):
        plot_grid(
            plot_rows,
            out_dir / "model_joint_target_compare.png",
            "Model joint target: replay vs logged model target",
            [
                ("q_target_model_rad_", "replay model target", "-"),
                ("real_q_target_", "logged model target", "--"),
            ],
            "rad",
            MODEL_NAMES,
        )
        comparisons.append(("q_target_model_rad_", "real_q_target_", "model joint target"))
    if has_columns(header, "q_target_model_rad_") and has_columns(
        header, "real_target_q_model_from_motor_rad_"
    ):
        plot_grid(
            plot_rows,
            out_dir / "motor_fk_model_target_compare.png",
            "Model joint target: replay vs logged motor target after FK",
            [
                ("q_target_model_rad_", "replay model target", "-"),
                (
                    "real_target_q_model_from_motor_rad_",
                    "logged target_pos_rad after FK",
                    "--",
                ),
            ],
            "rad",
            MODEL_NAMES,
        )
        comparisons.append(
            (
                "q_target_model_rad_",
                "real_target_q_model_from_motor_rad_",
                "model target vs motor FK target",
            )
        )
    if has_columns(header, "q_target_motor_rad_M") and has_columns(header, "mujoco_q_rad_M"):
        comparisons.append(("q_target_motor_rad_M", "mujoco_q_rad_M", "target vs mujoco q"))

    if comparisons:
        plot_error_bars(plot_rows, out_dir / "comparison_error_summary.png", comparisons)
        write_summary(rows, out_dir / "comparison_summary.txt", comparisons)

    print(f"[INFO] Wrote plots to: {out_dir}")
    for path in sorted(out_dir.iterdir()):
        print(f"[INFO]   {path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
