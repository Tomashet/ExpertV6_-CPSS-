from __future__ import annotations
import gymnasium as gym
from stable_baselines3.common.monitor import Monitor
from src.context import MarkovContextScheduler
from src.wrappers import ContextNonstationaryWrapper, ObservationNoiseWrapper, SafetyShieldWrapper
from src.safety import SafetyParams, ConformalCalibrator

def make_env(env_id: str, seed: int, action_space_type: str, p_stay: float,
             no_mpc: bool, no_conformal: bool, safety_params: SafetyParams):
    env = gym.make(env_id, render_mode=None, disable_env_checker=True)
    if hasattr(env.unwrapped, "configure"):
        cfg = {"action": {"type": "DiscreteMetaAction" if action_space_type == "discrete" else "ContinuousAction"}}
        env.unwrapped.configure(cfg)
    scheduler = MarkovContextScheduler(seed=seed, p_stay=p_stay)
    env = ContextNonstationaryWrapper(env, scheduler=scheduler)
    env = ObservationNoiseWrapper(env, seed=seed)
    calibrator = ConformalCalibrator(alpha=0.1, window=200, seed=seed)
    env = SafetyShieldWrapper(env, params=safety_params, action_space_type=action_space_type,
                              no_mpc=no_mpc, no_conformal=no_conformal, calibrator=calibrator)
    env = Monitor(env)
    return env, scheduler, calibrator
