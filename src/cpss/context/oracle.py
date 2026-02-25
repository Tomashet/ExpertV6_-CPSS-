# src/cpss/shield.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Optional, Sequence, Tuple
import numpy as np

from .types import ShieldDiag
from .budgets import BudgetFn
from .context.base import ContextEstimator
from .rollout.base import RiskToGoEstimator

@dataclass
class CPSSConfig:
    action_space_type: str  # "discrete" or "continuous"
    n_action_samples: int = 64          # for continuous sampling filter
    noise_std: float = 0.15             # for continuous local sampling
    fallback_to_min_risk: bool = True   # if no safe action exists
    deterministic_filter: bool = True   # for discrete: pick best among safe, not random

class CPSSShield:
    def __init__(
        self,
        context: ContextEstimator,
        risk_to_go: RiskToGoEstimator,
        budget: BudgetFn,
        cfg: CPSSConfig,
        action_space,  # gymnasium space
        q_value_fn: Optional[Callable[[np.ndarray, Sequence[int]], np.ndarray]] = None,
    ):
        """
        q_value_fn(obs, actions)->values optional:
          - for DQN: use Q-values to choose best safe action.
          - for PPO: can ignore and choose bar action if safe, else nearest safe.
        """
        self.context = context
        self.risk_to_go = risk_to_go
        self.budget = budget
        self.cfg = cfg
        self.action_space = action_space
        self.q_value_fn = q_value_fn

    def reset(self):
        self.context.reset()
        # risk_to_go may wrap risk predictor; it can have reset if needed

    def filter_action(self, obs, a_bar, info: Optional[dict] = None) -> Tuple[Any, ShieldDiag]:
        z_hat = self.context.predict(obs, info=info)
        B = float(self.budget(z_hat))

        if self.cfg.action_space_type == "discrete":
            return self._filter_discrete(obs, int(a_bar), z_hat, B, info)
        else:
            return self._filter_continuous(obs, np.asarray(a_bar, dtype=np.float32), z_hat, B, info)

    def _filter_discrete(self, obs, a_bar: int, z_hat: Any, B: float, info: Optional[dict]):
        n = int(self.action_space.n)
        actions = list(range(n))
        risks = np.array([self.risk_to_go.risk_to_go(obs, a, z_hat, info=info) for a in actions], dtype=np.float32)
        safe_mask = risks <= B
        safe_actions = np.where(safe_mask)[0]

        r_bar = float(risks[a_bar])

        if safe_mask[a_bar]:
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_bar, intervened=False, safe_set_size=int(safe_actions.size))
            return a_bar, diag

        if safe_actions.size > 0:
            # choose best safe action
            if self.q_value_fn is not None:
                qvals = self.q_value_fn(obs, safe_actions.tolist())
                a_safe = int(safe_actions[int(np.argmax(qvals))])
            else:
                # no q info: choose minimum risk among safe
                a_safe = int(safe_actions[int(np.argmin(risks[safe_actions]))])
            r_safe = float(risks[a_safe])
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_safe, intervened=True, safe_set_size=int(safe_actions.size))
            return a_safe, diag

        # no safe action exists
        if self.cfg.fallback_to_min_risk:
            a_safe = int(np.argmin(risks))
            r_safe = float(risks[a_safe])
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_safe, intervened=True, safe_set_size=0,
                              extra={"fallback": "min_risk"})
            return a_safe, diag

        # last resort: original action
        diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_bar, intervened=False, safe_set_size=0,
                          extra={"fallback": "none"})
        return a_bar, diag

    def _filter_continuous(self, obs, a_bar: np.ndarray, z_hat: Any, B: float, info: Optional[dict]):
        # sample candidates around a_bar
        candidates = [a_bar]
        for _ in range(self.cfg.n_action_samples - 1):
            noise = np.random.normal(0.0, self.cfg.noise_std, size=a_bar.shape).astype(np.float32)
            a = a_bar + noise
            # clip to action space bounds
            if hasattr(self.action_space, "low"):
                a = np.clip(a, self.action_space.low, self.action_space.high)
            candidates.append(a)

        risks = np.array([self.risk_to_go.risk_to_go(obs, a, z_hat, info=info) for a in candidates], dtype=np.float32)
        r_bar = float(risks[0])
        safe_mask = risks <= B

        if safe_mask[0]:
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_bar, intervened=False,
                              safe_set_size=int(np.sum(safe_mask)))
            return a_bar, diag

        safe_idx = np.where(safe_mask)[0]
        if safe_idx.size > 0:
            # choose safe candidate closest to a_bar
            dists = np.array([np.linalg.norm(candidates[i] - a_bar) for i in safe_idx], dtype=np.float32)
            j = int(safe_idx[int(np.argmin(dists))])
            a_safe = candidates[j]
            r_safe = float(risks[j])
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_safe, intervened=True,
                              safe_set_size=int(safe_idx.size))
            return a_safe, diag

        if self.cfg.fallback_to_min_risk:
            j = int(np.argmin(risks))
            a_safe = candidates[j]
            r_safe = float(risks[j])
            diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_safe, intervened=True, safe_set_size=0,
                              extra={"fallback": "min_risk"})
            return a_safe, diag

        diag = ShieldDiag(z_hat=z_hat, r_hat_bar=r_bar, r_hat_safe=r_bar, intervened=False, safe_set_size=0,
                          extra={"fallback": "none"})
        return a_bar, diag