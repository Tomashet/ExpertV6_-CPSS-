from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np


# ============================================================
# Safety parameters
# ============================================================

@dataclass
class SafetyParams:
    horizon_n: int = 10
    epsilon: float = 0.5


# ============================================================
# Conformal calibrator
# ============================================================

class ConformalCalibrator:
    """
    Lightweight conformal calibration module.

    Designed to be backward compatible with multiple call patterns:
        ConformalCalibrator(params=SafetyParams(...))
        ConformalCalibrator(safety_params=SafetyParams(...))
        ConformalCalibrator(SafetyParams(...))
    """

    def __init__(self, params=None, safety_params=None, **kwargs):

        if safety_params is None:
            safety_params = params

        if safety_params is None:
            raise TypeError(
                "ConformalCalibrator requires safety_params or params argument"
            )

        self.params = safety_params

        # simple running statistics
        self.residuals = []

    def update(self, value: float):
        self.residuals.append(value)

        if len(self.residuals) > 1000:
            self.residuals.pop(0)

    def quantile(self) -> float:

        if len(self.residuals) == 0:
            return 0.0

        return float(np.quantile(self.residuals, 1 - self.params.epsilon))


# ============================================================
# MPC-like safety shield
# ============================================================

class MPCLikeSafetyShield:
    """
    Simple MPC-style safety shield.

    The shield filters unsafe actions by checking
    predicted clearance and collision risk.

    Compatible with SafetyShieldWrapper.
    """

    def __init__(
        self,
        params: SafetyParams,
        action_space_type: str = "discrete",
        no_mpc: bool = False,
        no_conformal: bool = False,
        calibrator: Optional[ConformalCalibrator] = None,
    ):

        self.params = params
        self.action_space_type = action_space_type
        self.no_mpc = no_mpc
        self.no_conformal = no_conformal
        self.calibrator = calibrator

    # ========================================================
    # action filtering
    # ========================================================

    def filter_action(
        self,
        env,
        action,
        cur_ctx_id: int,
        eps_override: Optional[float] = None,
    ) -> Tuple[object, Dict]:

        """
        Returns:
            safe_action
            metadata dict
        """

        # base epsilon
        eps = self.params.epsilon

        # apply override from adjustment-speed constraint
        if eps_override is not None:
            eps = float(eps_override)

        inflate = 0.0

        # optional conformal calibration
        if not self.no_conformal and self.calibrator is not None:

            inflate = self.calibrator.quantile()
            eps = eps + inflate

        # ----------------------------------------------------
        # Safety heuristic
        # ----------------------------------------------------

        shield_used = False
        reason = ""

        try:
            # attempt to get vehicle states
            road = env.unwrapped.road
            ego = env.unwrapped.vehicle

            # simple clearance check
            min_dist = np.inf

            for veh in road.vehicles:
                if veh is ego:
                    continue

                dist = np.linalg.norm(veh.position - ego.position)

                if dist < min_dist:
                    min_dist = dist

            # safety threshold
            threshold = 5.0 + eps

            if min_dist < threshold:

                shield_used = True
                reason = "clearance_violation"

                # override action with braking / safe control
                if self.action_space_type == "discrete":
                    safe_action = 1  # typically IDLE or slow action
                else:
                    safe_action = np.zeros_like(action)

                return safe_action, {
                    "shield_used": True,
                    "shield_reason": reason,
                    "eps": eps,
                    "inflate": inflate,
                }

        except Exception:
            # if environment structure unexpected
            pass

        return action, {
            "shield_used": shield_used,
            "shield_reason": reason,
            "eps": eps,
            "inflate": inflate,
        }