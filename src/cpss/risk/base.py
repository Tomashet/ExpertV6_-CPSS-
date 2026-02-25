# src/cpss/types.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional
import numpy as np

@dataclass
class ShieldDiag:
    z_hat: Any
    r_hat_bar: float
    r_hat_safe: float
    intervened: bool
    safe_set_size: Optional[int] = None
    extra: Optional[Dict[str, Any]] = None