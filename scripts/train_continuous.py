# scripts/train_continuous.py
from __future__ import annotations

import argparse
import os
from typing import Any, Dict

import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure

from src.safety import SafetyParams
from src.logging_utils import ensure_dir, save_json, append_csv
from .common import make_env

# Optional preset support (same pattern as train_discrete.py)
try:
    from .presets import get_preset
except Exception:
    get_preset = None


def _normalize_env_id(env: str) -> str:
    """Accepts merge/merge-v0/highway/highway-v0 and returns a valid gym id."""
    if env is None:
        return "highway-v0"
    e = str(env).strip()
    if e == "merge":
        return "merge-v0"
    if e == "highway":
        return "highway-v0"
    return e


class TrainLoggerCallback(BaseCallback):
    def __init__(self, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = os.path.join(run_dir, "train_monitor.csv")

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[-1]
            append_csv(
                self.csv_path,
                {
                    "timestep": int(self.num_timesteps),
                    "clearance": info.get("clearance", np.nan),
                    "violation": int(bool(info.get("violation", False))),
                    "near_miss": int(bool(info.get("near_miss", False))),
                    "shield_used": int(bool(info.get("shield_used", False))),
                    "shield_reason": info.get("shield_reason", ""),
                    "eps": info.get("eps", np.nan),
                    "inflate": info.get("inflate", np.nan),
                    "ctx_id": info.get("ctx_id", -1),
                    # ---- Adjustment-speed diagnostics (new) ----
                    "adj_risk": info.get("adj_risk", np.nan),
                    "adj_unsafe": int(bool(info.get("adj_unsafe", False))),
                    "adj_s_env": info.get("adj_s_env", np.nan),
                    "adj_s_agent": info.get("adj_s_agent", np.nan),
                    "adj_eps_override": info.get("adj_eps_override", np.nan),
                },
            )
        return True


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()

    ap.add_argument("--preset", default="", help="Optional preset name")
    ap.add_argument("--env", default=None, help="highway-v0 / merge-v0 (default from preset or highway-v0)")
    ap.add_argument("--total_steps", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--p_stay", type=float, default=None)

    ap.add_argument("--no_tier2", action="store_true")
    ap.add_argument("--no_conformal", action="store_true")
    ap.add_argument("--no_mpc", action="store_true")

    ap.add_argument("--run_dir", default="")

    # ---- Adjustment-speed monitor flags (new) ----
    ap.add_argument("--adjust_speed", action="store_true", help="Enable adjustment-speed safety monitor")
    ap.add_argument("--adj_shift_window", type=int, default=200, help="Window for shift-speed estimation")
    ap.add_argument("--adj_metric", default="discrete", choices=["discrete", "l2"], help="Shift-speed metric")
    ap.add_argument("--adj_adapt_window", type=int, default=20, help="Window for adaptation-speed estimation")
    ap.add_argument("--adj_margin", type=float, default=0.0, help="Feasibility margin: unsafe if s_env > s_agent + margin")
    ap.add_argument("--adj_temp", type=float, default=10.0, help="Sigmoid temperature for risk score")

    return ap


def _load_preset(name: str) -> Dict[str, Any]:
    if not name:
        return {}
    if get_preset is None:
        raise RuntimeError(
            "Preset requested but scripts/presets.py:get_preset not found. "
            "Either add get_preset or remove --preset."
        )
    cfg = get_preset(name)
    if not isinstance(cfg, dict):
        raise RuntimeError(f"Preset {name} did not return a dict.")
    return cfg


def main() -> None:
    ap = _parser()
    args = ap.parse_args()

    preset_cfg: Dict[str, Any] = {}
    if args.preset:
        preset_cfg = _load_preset(args.preset)

    # Apply preset defaults where CLI omitted
    preset_env = preset_cfg.get("env_id", preset_cfg.get("env", None))
    if args.env is None and preset_env is not None:
        args.env = preset_env

    if args.total_steps is None:
        args.total_steps = int(preset_cfg.get("total_steps", 300_000))
    if args.seed is None:
        args.seed = int(preset_cfg.get("seed", 0))

    ns = preset_cfg.get("nonstationarity", {}) or {}
    if args.p_stay is None:
        args.p_stay = float(ns.get("p_stay", 0.8))

    args.env = _normalize_env_id(args.env)

    print("\n=== TRAIN CONTINUOUS DIAGNOSTICS ===")
    print("preset:", args.preset)
    print("env:", args.env)
    print("seed:", args.seed)
    print("total_steps:", args.total_steps)
    print("p_stay:", args.p_stay)
    print("no_mpc:", args.no_mpc, "no_conformal:", args.no_conformal)
    print("adjust_speed:", args.adjust_speed)
    print("===================================\n")

    # Safety params
    safety_kwargs = (preset_cfg.get("safety", {}) if preset_cfg else {}) or {}
    if not safety_kwargs:
        safety_kwargs = {"horizon_n": 10, "epsilon": 0.5}
    safety_params = SafetyParams(**safety_kwargs)

    # Run directory
    run_name = args.run_dir or f"{args.env}_continuous_sac_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    ensure_dir(run_dir)
    ensure_dir(os.path.join(run_dir, "models"))

    # Make env (continuous)
    env, _, _ = make_env(
        args.env,
        args.seed,
        "continuous",
        args.p_stay,
        args.no_mpc,
        args.no_conformal,
        safety_params,
    )

    # Save config
    save_json(
        os.path.join(run_dir, "config.json"),
        {**vars(args), "preset_cfg": preset_cfg, "safety_params": safety_params.__dict__},
    )

    # SB3 logger
    sb3_logger = sb3_configure(run_dir, ["stdout", "csv", "tensorboard"])

    hp = (preset_cfg.get("sb3", {}) if preset_cfg else {}) or {}
    sac_hp = hp.get("sac", {}) if isinstance(hp, dict) else {}

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=sac_hp.get("learning_rate", 3e-4),
        buffer_size=sac_hp.get("buffer_size", 200_000),
        learning_starts=sac_hp.get("learning_starts", 10_000),
        batch_size=sac_hp.get("batch_size", 256),
        gamma=sac_hp.get("gamma", 0.99),
        tau=sac_hp.get("tau", 0.005),
        train_freq=sac_hp.get("train_freq", 1),
        gradient_steps=sac_hp.get("gradient_steps", 1),
        ent_coef=sac_hp.get("ent_coef", "auto"),
        verbose=1,
        seed=args.seed,
    )

    model.set_logger(sb3_logger)

    # ---- Callbacks (logger + optional adjust-speed monitor) ----
    callbacks = [TrainLoggerCallback(run_dir)]

    if args.adjust_speed:
        from src.adjust_speed import (
            ShiftSpeedConfig,
            ShiftSpeedEstimator,
            AdaptSpeedConfig,
            AdaptationSpeedEstimator,
            FeasibilityConfig,
            FeasibilityMonitor,
            AdjustSpeedSafetyCallback,
        )

        shift = ShiftSpeedEstimator(
            ShiftSpeedConfig(window=args.adj_shift_window, metric=args.adj_metric)
        )
        adapt = AdaptationSpeedEstimator(
            AdaptSpeedConfig(window_updates=args.adj_adapt_window)
        )
        mon = FeasibilityMonitor(
            FeasibilityConfig(margin=args.adj_margin, temperature=args.adj_temp)
        )
        callbacks.append(AdjustSpeedSafetyCallback(shift, adapt, mon))

    model.learn(
        total_timesteps=int(args.total_steps),
        callback=callbacks,
        log_interval=1,
        progress_bar=True,
    )

    model.save(os.path.join(run_dir, "models", "final_model"))
    env.close()
    print(f"Saved model to {os.path.join(run_dir, 'models', 'final_model.zip')}")


if __name__ == "__main__":
    main()