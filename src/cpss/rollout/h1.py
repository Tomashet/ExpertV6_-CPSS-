# src/cpss/context/oracle.py
from __future__ import annotations
from typing import Any, Optional
from .base import ContextEstimator

class OracleContext(ContextEstimator):
    def __init__(self, key: str = "context", default: Any = 0):
        self.key = key
        self.default = default

    def predict(self, obs, info: Optional[dict] = None) -> Any:
        if info is None:
            return self.default
        return info.get(self.key, self.default)