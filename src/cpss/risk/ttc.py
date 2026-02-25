# src/cpss/rollout/h1.py
from __future__ import annotations
from typing import Any, Optional
from .base import RiskToGoEstimator
from ..risk.base import RiskPredictor

class H1RiskToGo(RiskToGoEstimator):
    def __init__(self, risk: RiskPredictor):
        self.risk = risk

    def risk_to_go(self, obs, action, z_hat: Any, info: Optional[dict] = None) -> float:
        return float(self.risk.predict_prob(obs, action, z_hat, info=info))