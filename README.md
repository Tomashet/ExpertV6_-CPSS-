# Proactive Context-Dependent Safety Experiments (highway-env + SB3)

This repo is a **runnable experiment package** for episodic nonstationarity in `highway-env`, with:
- **Discrete control**: SB3 **DQN** and **PPO**
- **Continuous control**: SB3 **SAC**
- **Episode-level nonstationarity** via a **Markov context scheduler**
- Logging to **CSV** + automatic **plots**
- Ablation flags: `--no_tier2`, `--no_conformal`, `--no_mpc`

> This package is designed to be stable and easy to run. It uses **oracle contexts** (episode regimes) to drive the scheduler and optional forecasting proxies. You can later swap in learned Tier-1/Tier-2/Tier-3 modules.

## Install
```bash
pip install -r requirements.txt
```

## Discrete (DQN / PPO)
Train:
```bash
python -m scripts.train_discrete --algo dqn --env highway-v0 --total_steps 200000 --seed 0
python -m scripts.train_discrete --algo ppo --env highway-v0 --total_steps 300000 --seed 0
```

Evaluate + plot:
```bash
python -m scripts.eval --env highway-v0 --run_dir runs/highway-v0_discrete_dqn_seed0 --episodes 200
python -m scripts.plot_results --run_dir runs/highway-v0_discrete_dqn_seed0
```

## Continuous (SAC)
Train:
```bash
python -m scripts.train_continuous --env highway-v0 --total_steps 400000 --seed 0
```

Evaluate + plot:
```bash
python -m scripts.eval --env highway-v0 --run_dir runs/highway-v0_continuous_sac_seed0 --episodes 200
python -m scripts.plot_results --run_dir runs/highway-v0_continuous_sac_seed0
```

## Ablations
- `--no_tier2`: placeholder (kept for parity); context is oracle in this repo
- `--no_conformal`: disable conformal inflation
- `--no_mpc`: disable safety shield

## Output
Each run writes to `runs/<name>/`:
- `config.json`
- `train_monitor.csv`
- `eval_metrics.csv`
- `plots/*.png`
- saved SB3 model
## Presets (including merge-v0)
Presets live in `scripts/presets.py` and `configs/*.json`.

Examples:
```bash
# Merge-v0 discrete with tuned defaults
python -m scripts.train_discrete --preset merge_discrete_default --algo dqn --total_steps 300000 --seed 0

# Merge-v0 continuous SAC
python -m scripts.train_continuous --preset merge_continuous_default --total_steps 600000 --seed 0
```

## Multi-seed sweeps
Run 3 seeds for merge-v0 discrete DQN (train + eval + plots):
```bash
python -m scripts.run_sweep --preset merge_discrete_default --algo dqn --seeds 0,1,2 --total_steps 300000 --episodes 200
```

## LaTeX table export
After you have multiple `runs/*/eval_metrics.csv`, export an Overleaf-ready table:
```bash
python -m scripts.export_latex --pattern "runs/merge_discrete_default_dqn_seed*" --out results_merge_dqn.tex
```

## Full suite (all methods + ablations + LaTeX export)
Run everything (DQN/PPO/SAC × full/no\_mpc/no\_conformal/no\_mpc+no\_conformal) over multiple seeds,
then export LaTeX tables into `paper_tables/`:

```bash
python -m scripts.run_all --seeds 0,1,2 --steps_discrete 300000 --steps_continuous 600000 --episodes 200
```

You can restrict to one environment:
```bash
python -m scripts.run_all --only merge
```

## Paper-friendly table export (discrete vs continuous)
This creates two tables with a fixed, nicer order:
- `paper_tables/results_discrete.tex`
- `paper_tables/results_continuous.tex`
and also per-environment splits if you pass `--split_by_env`.

```bash
python -m scripts.export_paper_tables --pattern "runs/*_seed*" --out_dir paper_tables --split_by_env
```



## Troubleshooting (Windows / runs appear stuck)

- First run a tiny smoke test:
  ```bash
  python -m scripts.train_discrete --preset merge_discrete_default --algo dqn --total_steps 2000 --no_mpc --no_conformal
  ```
  If this works, re-enable components one at a time.

- If a run appears to "hang" with no prints, the updated scripts now enable SB3 `progress_bar=True` and `log_interval=1`.

- If you see a frozen/blank window, ensure you are not using `render_mode="human"` (we force `render_mode=None`).
