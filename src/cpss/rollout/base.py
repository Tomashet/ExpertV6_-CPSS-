# src/cpss/risk/base.py
from __future__ import annotations
from typing import Any, Optional
import numpy as np

class RiskPredictor:
    def reset(self) -> None:
        pass

    def predict_prob(self, obs, action, z_hat: Any, info: Optional[dict] = None) -> float:
        """Return P(violation | s,a,z). Must be in [0,1]."""
        raise NotImplementedError