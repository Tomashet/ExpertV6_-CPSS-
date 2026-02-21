from __future__ import annotations
import argparse, os
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.logger import configure as sb3_configure
from src.safety import SafetyParams
from src.logging_utils import ensure_dir, save_json, append_csv
from .common import make_env
from .presets import get_preset

class TrainLoggerCallback(BaseCallback):
    def __init__(self, run_dir: str, verbose: int = 0):
        super().__init__(verbose)
        self.csv_path = os.path.join(run_dir, "train_monitor.csv")
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        if infos:
            info = infos[-1]
            append_csv(self.csv_path, {
                "timestep": int(self.num_timesteps),
                "clearance": info.get("clearance", np.nan),
                "violation": int(bool(info.get("violation", False))),
                "near_miss": int(bool(info.get("near_miss", False))),
                "shield_used": int(bool(info.get("shield_used", False))),
                "shield_reason": info.get("shield_reason", ""),
                "eps": info.get("eps", np.nan),
                "inflate": info.get("inflate", np.nan),
                "ctx_id": info.get("ctx_id", -1),
            })
        return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="highway-v0")
    ap.add_argument("--preset", default="", help="Optional preset name, e.g. merge_continuous_default")
    ap.add_argument("--total_steps", type=int, default=400_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--p_stay", type=float, default=0.8)
    ap.add_argument("--no_tier2", action="store_true")
    ap.add_argument("--no_conformal", action="store_true")
    ap.add_argument("--no_mpc", action="store_true")
    ap.add_argument("--run_dir", default="")
    args = ap.parse_args()
    preset_cfg = None
    if args.preset:
        preset_cfg = get_preset(args.preset)
        args.env = preset_cfg.get("env_id", args.env)
        args.p_stay = float(preset_cfg.get("nonstationarity", {}).get("p_stay", args.p_stay))


    safety_params = SafetyParams(**(preset_cfg.get('safety', {}) if preset_cfg else {'horizon_n':10,'epsilon':0.5}))
    run_name = args.run_dir or f"{args.env}_continuous_sac_seed{args.seed}"
    run_dir = os.path.join("runs", run_name)
    ensure_dir(run_dir); ensure_dir(os.path.join(run_dir, "models"))
    env, _, _ = make_env(args.env, args.seed, "continuous", args.p_stay, args.no_mpc, args.no_conformal, safety_params)

    save_json(os.path.join(run_dir,"config.json"), {**vars(args), "safety_params": safety_params.__dict__})
    hp = (preset_cfg.get('sb3', {}) if preset_cfg else {})

    sb3_logger = sb3_configure(run_dir, ["stdout","csv","tensorboard"])

    model = SAC(
        "MlpPolicy",
        env,
        learning_rate=hp.get("sac", {}).get("learning_rate", 3e-4),
        buffer_size=hp.get("sac", {}).get("buffer_size", 200_000),
        batch_size=hp.get("sac", {}).get("batch_size", 256),
        gamma=hp.get("sac", {}).get("gamma", 0.99),
        tau=hp.get("sac", {}).get("tau", 0.005),
        train_freq=hp.get("sac", {}).get("train_freq", 1),
        gradient_steps=hp.get("sac", {}).get("gradient_steps", 1), verbose=1, seed=args.seed)
    model.set_logger(sb3_logger)
    model.learn(total_timesteps=args.total_steps, callback=TrainLoggerCallback(run_dir), log_interval=1, progress_bar=True)
    model.save(os.path.join(run_dir,"models","final_model"))
    env.close()
    print(f"Saved model to {os.path.join(run_dir,'models','final_model.zip')}")
if __name__ == "__main__":
    main()
