# src/cpss/rollout/base.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np

class RiskToGoEstimator:
    def risk_to_go(self, obs, action, z_hat: Any, info: Optional[dict] = None) -> float:
        """Return estimated risk-to-go for candidate (obs, action) under context z_hat."""
        raise NotImplementedError