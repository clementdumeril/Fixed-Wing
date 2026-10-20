# Training on Savio

Not run yet: the smoke test is the first check of `examples/train_sac.py`.

1. Log in, then clone the repository in your scratch space and install (Python 3.10+, same module as your other projects):
   ```bash
   cd /global/scratch/users/$USER
   git clone https://github.com/clementdumeril/Fixed-Wing.git && cd Fixed-Wing/python
   python -m venv ~/.venvs/fixed-wing && source ~/.venvs/fixed-wing/bin/activate
   pip install -e ".[rl]"
   mkdir -p runs
   ```
2. Replace `REPLACE_ME` in the two `.slurm` files with your account, partition and QoS.
3. Smoke test, a few minutes: `sbatch savio/smoke_test.slurm`, then read `runs/slurm-smoke-<job>.out`.
4. Training, 5 seeds in parallel: `sbatch savio/train_sac.slurm`. Follow with `squeue -u $USER`.

Each run writes `runs/<name>/`: `progress.csv` (learning curves), `checkpoints/`, `model.zip`, `evaluation.json`
(same metrics as `examples/pid_baseline.py`, on the same 100 test scenarios).
