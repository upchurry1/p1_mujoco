#!/usr/bin/env python3
"""A/B parallel ankle coupling utilities for P1 validation.

This module intentionally supports only a parameterized mapping scaffold. Fill
the real matrix/limits in ab_ankle_config.yaml after deriving them from CAD,
URDF, calibration, or the G1_ankle-style solver you choose to port.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


def _as_finite_vector(value: Any, name: str, size: int) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != (size,) or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite vector with shape ({size},)")
    return arr


def _as_finite_matrix(value: Any, name: str, shape: tuple[int, int]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.shape != shape or not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be a finite matrix with shape {shape}")
    return arr


@dataclass(frozen=True)
class ABMotorLimits:
    q_min: np.ndarray
    q_max: np.ndarray
    dq_max: np.ndarray
    tau_max: np.ndarray


@dataclass(frozen=True)
class ABMotorGains:
    kp: np.ndarray
    kd: np.ndarray


class LinearABAnkleMapping:
    """Linear local ankle mapping.

    Convention:
        q_ab = M @ q_pr + offset
        dq_ab = M @ dq_pr
        tau_pr = M.T @ tau_ab

    This is a local approximation. Replace this class or extend it when you
    have an angle-dependent Jacobian/LUT from real geometry.
    """

    def __init__(self, matrix: Any, offset: Any, name: str = "ankle") -> None:
        self.name = name
        self.M = _as_finite_matrix(matrix, f"{name}.mapping.matrix", (2, 2))
        self.offset = _as_finite_vector(offset, f"{name}.mapping.offset", 2)
        det = float(np.linalg.det(self.M))
        if abs(det) < 1e-9:
            raise ValueError(f"{name}.mapping.matrix is singular; det={det}")
        self.M_inv = np.linalg.inv(self.M)

    def pr_to_ab(self, q_pr: np.ndarray) -> np.ndarray:
        return self.M @ q_pr + self.offset

    def ab_to_pr(self, q_ab: np.ndarray) -> np.ndarray:
        return self.M_inv @ (q_ab - self.offset)

    def dpr_to_dab(self, dq_pr: np.ndarray) -> np.ndarray:
        return self.M @ dq_pr

    def dab_to_dpr(self, dq_ab: np.ndarray) -> np.ndarray:
        return self.M_inv @ dq_ab

    def tau_ab_to_pr(self, tau_ab: np.ndarray) -> np.ndarray:
        return self.M.T @ tau_ab

    def tau_pr_to_ab(self, tau_pr: np.ndarray) -> np.ndarray:
        return self.M_inv.T @ tau_pr


def load_linear_mapping(side_name: str, side_cfg: dict[str, Any]) -> LinearABAnkleMapping:
    mapping_cfg = side_cfg.get("mapping", {})
    mapping_type = str(mapping_cfg.get("type", "linear")).strip().lower()
    if mapping_type != "linear":
        raise NotImplementedError(
            f"{side_name}.mapping.type={mapping_type!r} is not implemented yet. "
            "Use 'linear' first, or add your nonlinear geometry/LUT solver here."
        )
    return LinearABAnkleMapping(
        matrix=mapping_cfg.get("matrix"),
        offset=mapping_cfg.get("offset"),
        name=side_name,
    )


def load_motor_gains(side_name: str, side_cfg: dict[str, Any]) -> ABMotorGains:
    motors = side_cfg.get("motors", {})
    kp = []
    kd = []
    for motor_name in ("A", "B"):
        cfg = motors.get(motor_name, {})
        kp.append(cfg.get("kp"))
        kd.append(cfg.get("kd"))
    return ABMotorGains(
        kp=_as_finite_vector(kp, f"{side_name}.motors.kp[A,B]", 2),
        kd=_as_finite_vector(kd, f"{side_name}.motors.kd[A,B]", 2),
    )


def load_motor_limits(side_name: str, side_cfg: dict[str, Any]) -> ABMotorLimits:
    motors = side_cfg.get("motors", {})
    q_min = []
    q_max = []
    dq_max = []
    tau_max = []
    for motor_name in ("A", "B"):
        cfg = motors.get(motor_name, {})
        q_limit = cfg.get("q_limit")
        if not isinstance(q_limit, (list, tuple)) or len(q_limit) != 2:
            raise ValueError(f"{side_name}.motors.{motor_name}.q_limit must be [min, max]")
        q_min.append(q_limit[0])
        q_max.append(q_limit[1])
        dq_max.append(cfg.get("dq_limit"))
        tau_max.append(cfg.get("tau_limit"))
    return ABMotorLimits(
        q_min=_as_finite_vector(q_min, f"{side_name}.motors.q_min[A,B]", 2),
        q_max=_as_finite_vector(q_max, f"{side_name}.motors.q_max[A,B]", 2),
        dq_max=_as_finite_vector(dq_max, f"{side_name}.motors.dq_limit[A,B]", 2),
        tau_max=_as_finite_vector(tau_max, f"{side_name}.motors.tau_limit[A,B]", 2),
    )


def clip_ab_command(
    q_ab_des: np.ndarray,
    dq_ab_des: np.ndarray,
    tau_ab: np.ndarray,
    limits: ABMotorLimits,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    q_clipped = np.clip(q_ab_des, limits.q_min, limits.q_max)
    dq_clipped = np.clip(dq_ab_des, -limits.dq_max, limits.dq_max)
    tau_clipped = np.clip(tau_ab, -limits.tau_max, limits.tau_max)
    sat = (
        (np.abs(q_clipped - q_ab_des) > 1e-12)
        | (np.abs(dq_clipped - dq_ab_des) > 1e-12)
        | (np.abs(tau_clipped - tau_ab) > 1e-12)
    )
    return q_clipped, dq_clipped, tau_clipped, sat.astype(float)
