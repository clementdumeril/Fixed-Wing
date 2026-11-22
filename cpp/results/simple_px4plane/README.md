# Simple reward in wind and thermals (PX4 advanced_plane, tracking)

Savio job 41645214, `slurm/tracking_px4plane_simple.slurm`:
`train_sac_c128_b256_tracking_g995_px4plane_i10a20_fine_wind`, `--reward simple`, constant 35, energy weights 0, 0.1
and 0.3, 4 seeds each, 8 M steps. Tables and discussion: [docs/RESULTS.md](../../../docs/RESULTS.md). Method:
[docs/method.md](../../../docs/method.md).

`energy<w>/seed<k>/`, for each of the 12 runs:

| File | Content |
| --- | --- |
| `config.json` | every setting of the run |
| `evaluation_best_disturbed.json` | best checkpoint, 100 disturbed test flights of 120 s: `all`, `with_thermals` (42 flights), `without_thermals` (58) |
| `evaluation_best_calm.json` | best checkpoint, the same 100 flights in calm air |
| `evaluation_disturbed.json`, `evaluation_calm.json` | the same tests for the last actor |
| `checkpoints.csv` | validation success, RMSE and score every 50 k steps |
| `progress.csv` | training curves: episode length and return, position error, energy weight, time |
| `reward_check.txt` | stored against recomputed rewards of the replay buffer |

Fields of the evaluation files: `success_rate` (share of flights completed), `position_rmse` [m] and `energy_kj`
(per completed flight), `mean_power_w`, `mean_action_change` (mean absolute change of each command per step) and
`failures` (why the failed flights ended: `airspeed`, `alpha`, `position`).

Not committed (size): the actors (`actor.h`, `best_actor.h`) and the recorded flights (`flights_*.csv`).
