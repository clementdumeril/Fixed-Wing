# First Eratix model, 10 seeds (Savio, 6 November 2026)

`slurm/tracking_eratix.slurm` (job 40975471): the tracking task on the first model of the E-flite UMX Eratix
([`docs/eratix_model.md`](../../../docs/eratix_model.md)), SAC, critic 128, batch 256, critics every 10 steps and
actor every 20 (L2F), our tanh reward, 4 M steps, best checkpoint every 50 k steps.

| 10 seeds | Test A | Test B | Ahead | Position RMS | Velocity RMS | Command change |
| --- | --- | --- | --- | --- | --- | --- |
| Last actor | 85.9 ± 22.0 % | 84.7 % | 84.3 % | 11.9 m | 1.64 m/s | 0.421 |
| **Best checkpoint** | **99.2 ± 0.7 %** | 96.7 % | 97.3 % | **7.9 ± 2.8 m** | 1.11 m/s | 0.244 |

18 min per run; flies (mean episode length at 90 % of the maximum) after 570 k steps, 2.4 min. Failures of the best
checkpoints: a few stalls (0–2 of 100 per seed).

- The Eratix model learns as the advanced_plane does: 99 % of the test flights, 8 m of position error at its lower
  speeds (the reference flies at about cruise speed, 8 m/s).
- **Its commands are rough**: 0.24 per step for the best checkpoints against 0.036 on the advanced_plane with the same
  recipe. The small airplane reacts fast (servo time constant 45 ms, ±40–50° throws); the L2F reward form, which made
  the advanced_plane's commands smoother, is the next run.
- The last actors are unsteady (one seed collapses): the best checkpoint is needed here.

`seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `evaluation*.json`.
