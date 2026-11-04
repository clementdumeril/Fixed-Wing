# The L2F critic size and TD3 on PX4's advanced_plane (Savio, 4 November 2026)

`slurm/tracking_px4plane_l2f.slurm` (job 40857996): the tracking task on the advanced_plane with the batch (256) and
update intervals (critics every 10 steps, actor every 20) of the L2F paper, 4 M steps, seeds 0 to 9, a validation
every 50 k steps and the best checkpoint kept. Three settings in the same job:

- `sac_c128_b256_i10a20_4M`: SAC, critic 128, the reference of [`l2f_intervals_px4plane`](../l2f_intervals_px4plane/README.md);
- `sac_c64_b256_i10a20_4M`: SAC, critic 64 as L2F;
- `td3_c64_b256_i10a20_4M`: TD3 as L2F (critic 64, target-action noise 0.2 clipped at 0.5, exploration noise 0.1).

| 10 seeds, mean | Training time | Flies (90 % of the max episode length) | Test A | Position RMS | Best checkpoint: A, B, ahead, position | Command change, best |
| --- | --- | --- | --- | --- | --- | --- |
| SAC, critic 128 | 17.5 min | 515 k steps, 2.2 min | 94.9 % | 19.9 m | 99.7 %, 98.3 %, 99.4 %, 12.9 m | 0.036 |
| SAC, critic 64 | 9.8 min | 687 k steps, 1.6 min | 92.6 % | 20.8 m | 98.6 %, 97.8 %, 98.1 %, 12.7 m | 0.105 |
| **TD3, critic 64** | **9.0 min** | **293 k steps, 0.6 min** | 90.5 % | 18.8 m | 98.5 %, 98.4 %, 99.3 %, 13.3 m | **0.569** |

Command change: squared change of the commands from one step to the next, mean over the 4 commands, the steps and the completed flights of test A (`mean_action_change`).

## What it shows

- **The runs are reproducible**: the SAC critic-128 runs give exactly the same results as in job 40855038, 17.5
  instead of 19.8 min (the speed of the nodes).
- **Critic 64 halves the time** (9.8 against 17.5 min) for the same precision of the best checkpoints (12.7 against
  12.9 m), with a little less margin (93 to 100 % of test A per seed instead of 99 to 100 %).
- **TD3 learns about ten times faster in time**: it flies after 293 k steps, 37 s, as L2F (300 k steps); the best
  validation checkpoint within the first minute already completes 95.5 % of the validation flights with 15 m of
  error, against 13 % for SAC critic 128 (validation flights only: the test flights of those early checkpoints were
  not flown).
- **But TD3's commands are bang-bang**: the surfaces switch between the stops at about the control rate
  (`figures/commands.png`), with 16 times the command change of SAC (0.57 against 0.036; 0.12 to 0.90 per seed). In
  simulation the actuator lag filters it; on real servos it is not flyable. SAC's commands stay smooth.
- **TD3 also drifts after its peak**: validation at 91 % at 400 k steps, 83 % at 2 M, 87 % at 4 M; its best
  checkpoints are between 0.5 and 3.1 M steps.

## Figures

`figures/commands.png`: aileron, elevator and throttle over the first 10 s of test flight 1, seed 0, best checkpoint,
SAC critic 128 against TD3. `figures/compare_time.png`: the validation flights of the three settings against the
training time. `figures/episode_length*.png`, `episode_return.png`, `validation.png`: TD3, in the format of Fig. 3
and 4 of the L2F paper (`tools/plot_l2f_style.py`).

## Files

`<setting>/seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `evaluation{,_random,_ahead}.json` (last
actor) and `evaluation_best{,_random,_ahead}.json` (kept actor).
