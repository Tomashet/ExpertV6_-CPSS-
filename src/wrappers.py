from __future__ import annotations
import numpy as np
import gymnasium as gym
from typing import Any, Dict, Optional

from .context import MarkovContextScheduler, context_to_highway_config, CTX_TO_ID
from .safety import SafetyParams, MPCLikeSafetyShield, ConformalCalibrator, clearance_margin

class ContextNonstationaryWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, scheduler: MarkovContextScheduler):
        super().__init__(env)
        self.scheduler = scheduler
        self.ctx = scheduler.current()
        self.ctx_id = CTX_TO_ID[self.ctx]
        self.last_config: Dict[str, Any] = {}
        self._first_reset = True

    def reset(self, **kwargs):
        # advance context once per episode (except very first reset)
        if not self._first_reset:
            self.scheduler.step_episode()
        else:
            self._first_reset = False
        self.ctx = self.scheduler.current()
        self.ctx_id = CTX_TO_ID[self.ctx]
        cfg = context_to_highway_config(self.ctx)
        self.last_config = cfg
        if hasattr(self.env.unwrapped, "configure"):
            self.env.unwrapped.configure(cfg)
        obs, info = self.env.reset(**kwargs)
        info = dict(info)
        info.update({"ctx_tuple": self.ctx, "ctx_id": self.ctx_id})
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        info = dict(info)
        info.update({"ctx_tuple": self.ctx, "ctx_id": self.ctx_id})
        return obs, reward, terminated, truncated, info

    def next_episode(self):
        self.ctx = self.scheduler.step_episode()
        self.ctx_id = CTX_TO_ID[self.ctx]

class ObservationNoiseWrapper(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, seed: int = 0):
        super().__init__(env)
        self.rng = np.random.default_rng(seed)

    def observation(self, observation):
        cfg = getattr(self.env, "last_config", {})
        noise_std = float(cfg.get("_ctx_obs_noise_std", 0.0))
        dropout_prob = float(cfg.get("_ctx_dropout_prob", 0.0))
        obs = np.array(observation, dtype=np.float32, copy=True)
        if obs.ndim != 2 or obs.shape[1] < 2:
            return obs
        presence = obs[:, 0]
        if dropout_prob > 0:
            drop_mask = (self.rng.random(size=obs.shape[0]) < dropout_prob)
            drop_mask[0] = False
            obs[drop_mask, :] = 0.0
            obs[:, 0] = presence * (~drop_mask)
        if noise_std > 0:
            mask = obs[:, 0] > 0.5
            obs[mask, 1:] += self.rng.normal(0.0, noise_std, size=obs[mask, 1:].shape).astype(np.float32)
        return obs

class SafetyShieldWrapper(gym.Wrapper):
    def __init__(self, env: gym.Env, params: SafetyParams, action_space_type: str,
                 no_mpc: bool, no_conformal: bool, calibrator: Optional[ConformalCalibrator] = None):
        super().__init__(env)
        self.params = params
        self.calibrator = calibrator
        self.shield = MPCLikeSafetyShield(params=params, action_space_type=action_space_type,
                                          no_mpc=no_mpc, no_conformal=no_conformal, calibrator=calibrator)

    def step(self, action):
        ctx_id = getattr(self.env, "ctx_id", -1)
        filtered_action, shield_info = self.shield.filter_action(self.env, action, ctx_id)
        obs, reward, terminated, truncated, info = self.env.step(filtered_action)
        d = clearance_margin(self.env, self.params)
        viol = (np.isfinite(d) and d <= 0.0)
        near = (np.isfinite(d) and d <= self.params.delta_nearmiss)
        info = dict(info)
        info.update(shield_info)
        info.update({"clearance": float(d) if np.isfinite(d) else np.nan,
                     "violation": bool(viol),
                     "near_miss": bool(near),
                     "filtered_action": filtered_action})
        return obs, reward, terminated, truncated, info