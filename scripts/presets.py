from __future__ import annotations
from typing import Dict, Any

# Simple presets for highway-env task variants.
# These are deliberately conservative and stable defaults.
# You can tweak them to match your machine and desired training budget.

PRESETS: Dict[str, Dict[str, Any]] = {
    # -------------------------
    # highway-v0 (discrete)
    # -------------------------
    "highway_discrete_default": {
        "env_id": "highway-v0",
        "action_space_type": "discrete",
        "sb3": {
            "dqn": dict(
                learning_rate=3e-4,
                buffer_size=100_000,
                learning_starts=10_000,
                batch_size=64,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1_000,
            ),
            "ppo": dict(
                learning_rate=3e-4,
                n_steps=2048,
                batch_size=64,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.0,
            ),
        },
        "safety": dict(horizon_n=10, epsilon=0.5, delta_nearmiss=1.0, d0=2.0, h=1.2),
        "nonstationarity": dict(p_stay=0.8),
    },

    # -------------------------
    # merge-v0 (discrete) - tighter traffic geometry, more cut-ins.
    # Tune to be slightly more conservative and train longer.
    # -------------------------
    "merge_discrete_default": {
        "env_id": "merge-v0",
        "action_space_type": "discrete",
        "sb3": {
            "dqn": dict(
                learning_rate=2e-4,
                buffer_size=150_000,
                learning_starts=15_000,
                batch_size=64,
                gamma=0.99,
                train_freq=4,
                target_update_interval=1_500,
            ),
            "ppo": dict(
                learning_rate=2.5e-4,
                n_steps=4096,
                batch_size=128,
                gamma=0.99,
                gae_lambda=0.95,
                ent_coef=0.0,
            ),
        },
        "safety": dict(horizon_n=10, epsilon=0.7, delta_nearmiss=1.2, d0=2.2, h=1.3),
        "nonstationarity": dict(p_stay=0.85),
    },

    # -------------------------
    # highway-v0 (continuous) SAC
    # -------------------------
    "highway_continuous_default": {
        "env_id": "highway-v0",
        "action_space_type": "continuous",
        "sb3": {
            "sac": dict(
                learning_rate=3e-4,
                buffer_size=200_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
            ),
        },
        "safety": dict(horizon_n=10, epsilon=0.5, delta_nearmiss=1.0, d0=2.0, h=1.2),
        "nonstationarity": dict(p_stay=0.8),
    },

    # -------------------------
    # merge-v0 (continuous) SAC - more conservative defaults
    # -------------------------
    "merge_continuous_default": {
        "env_id": "merge-v0",
        "action_space_type": "continuous",
        "sb3": {
            "sac": dict(
                learning_rate=2e-4,
                buffer_size=300_000,
                batch_size=256,
                gamma=0.99,
                tau=0.005,
                train_freq=1,
                gradient_steps=1,
            ),
        },
        "safety": dict(horizon_n=10, epsilon=0.8, delta_nearmiss=1.2, d0=2.2, h=1.3),
        "nonstationarity": dict(p_stay=0.85),
    },
}

def get_preset(name: str) -> Dict[str, Any]:
    if name not in PRESETS:
        raise KeyError(f"Unknown preset '{name}'. Available: {list(PRESETS.keys())}")
    return PRESETS[name].copy()
