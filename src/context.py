from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, List, Tuple
import numpy as np

Context = Tuple[str, str, str]  # (density, aggressiveness, noise)

DENSITY_LEVELS = ["low", "med", "high"]
AGGR_LEVELS = ["calm", "normal", "aggr"]
NOISE_LEVELS = ["clean", "noisy", "dropout"]

def all_contexts() -> List[Context]:
    return [(d, a, n) for d in DENSITY_LEVELS for a in AGGR_LEVELS for n in NOISE_LEVELS]

ALL_CTX: List[Context] = all_contexts()
CTX_TO_ID: Dict[Context, int] = {c: i for i, c in enumerate(ALL_CTX)}
ID_TO_CTX: Dict[int, Context] = {i: c for c, i in CTX_TO_ID.items()}

def build_markov_transition(num_states: int, p_stay: float) -> np.ndarray:
    assert 0.0 <= p_stay <= 1.0
    P = np.full((num_states, num_states), (1.0 - p_stay) / max(num_states - 1, 1), dtype=np.float64)
    np.fill_diagonal(P, p_stay)
    return P

@dataclass
class MarkovContextScheduler:
    seed: int = 0
    p_stay: float = 0.8

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)
        self.P = build_markov_transition(len(ALL_CTX), self.p_stay)
        self.cur_id = int(self.rng.integers(0, len(ALL_CTX)))

    def current(self) -> Context:
        return ID_TO_CTX[self.cur_id]

    def step_episode(self) -> Context:
        probs = self.P[self.cur_id]
        self.cur_id = int(self.rng.choice(len(probs), p=probs))
        return ID_TO_CTX[self.cur_id]

def context_to_highway_config(ctx: Context) -> Dict:
    density, aggr, noise = ctx
    vehicles_count = {"low": 15, "med": 30, "high": 50}[density]
    speed_limit = {"calm": 25, "normal": 30, "aggr": 35}[aggr]
    obs_noise_std = {"clean": 0.0, "noisy": 0.05, "dropout": 0.05}[noise]
    dropout_prob = {"clean": 0.0, "noisy": 0.0, "dropout": 0.15}[noise]
    return {
        "vehicles_count": vehicles_count,
        "duration": 60,
        "policy_frequency": 1,
        "simulation_frequency": 15,
        "lanes_count": 4,
        "reward_speed_range": [20, speed_limit],
        "collision_reward": -10,
        "high_speed_reward": 0.4,
        "right_lane_reward": 0.1,
        "observation": {
            "type": "Kinematics",
            "vehicles_count": 10,
            "features": ["presence", "x", "y", "vx", "vy"],
            "absolute": False,
        },
        "_ctx_obs_noise_std": float(obs_noise_std),
        "_ctx_dropout_prob": float(dropout_prob),
        "_ctx_id": int(CTX_TO_ID[ctx]),
        "_ctx_tuple": ctx,
    }
