# Training on Savio

The job files use the `ic_piml` account on `savio4_htc` (billed per core). Change them if you use another account.

## First time

```bash
module load python/3.11.6-gcc-11.4.0
cd /global/scratch/users/$USER
git clone https://github.com/clementdumeril/Fixed-Wing.git && cd Fixed-Wing/python
python -m venv ~/.venvs/fixed-wing && source ~/.venvs/fixed-wing/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu   # CPU build, much smaller
pip install -e ".[rl]"
mkdir -p runs
```

## Every session

```bash
cd /global/scratch/users/$USER/Fixed-Wing/python
module load python/3.11.6-gcc-11.4.0 && source ~/.venvs/fixed-wing/bin/activate
git pull   # if job files were edited by hand before: git checkout -- savio/ first
```

## Runs

- Smoke test, a few minutes: `sbatch savio/smoke_test.slurm`
- One training: `sbatch --array=0 savio/train_sac.slurm`; seeds 1 to 4: `sbatch --array=1-4 savio/train_sac.slurm`
- Follow: `squeue -u $USER`, `grep ep_len_mean runs/slurm-<job>_0.out | tail`
- Result: `cat runs/sac_<job>_seed0/evaluation.json`, same metrics and test scenarios as `examples/pid_baseline.py`

Jobs keep running after you log out. Each run writes `runs/sac_<job>_seed<k>/`: `progress.csv`, `checkpoints/`,
`model.zip`, `evaluation.json`.

## Plots, on your computer

Copy the runs back (from your computer, through the data transfer node), then:

```bash
scp -r <user>@dtn.brc.berkeley.edu:/global/scratch/users/<user>/Fixed-Wing/python/runs .
python examples/plot_training.py runs/sac_<job>_seed0 --out training.png
python examples/compare_flights.py --model runs/sac_<job>_seed0/model.zip --seeds 0 1 2 --out flights.png
python examples/ui_policy.py --model runs/sac_<job>_seed0/model.zip --seed 0   # 3D, with ui-server running
```
