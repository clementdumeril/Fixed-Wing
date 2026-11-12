# Training time against precision (PX4 advanced_plane, tracking)

Job 41205677, `slurm/tracking_px4plane_speed.slurm`. The best recipe of [`fine_px4plane`](../fine_px4plane/README.md)
(third run, `fine2m_bonus4`: fine position observation, adaptive closeness bonus of weight 4, precise checkpoint
choice; 1.8 m, 16 min per run with critic 128 and updates every 10 steps) with one change per variant, 10 seeds each.

Where the time goes (`--profile`, 100k steps on one laptop core): 312 µs per step, of which the two critics' updates
75 %, the actor's 19 %, the batch 4 %, and the airplane simulation 2 % (5.4 µs). The variants shrink the networks or
update them less often, as in RLtools' L2F zoo (`rl/zoo/l2f/td3.h` at commit b32d9985: critics of 32, updates every
16 steps), or change the number of steps.

| 10 seeds, best checkpoint | Time per step (laptop) | Run time (Savio, 4M steps) | Test A: done | Position | Median, range | Seeds above 6 m | Command change |
| --- | --- | --- | --- | --- | --- | --- | --- |
| critic 128, every 10 steps (`fine2m_bonus4`, reference) | 312 µs | 16.1 min | 97.6 % | **1.8 ± 0.5 m** | 1.6 m, 1.3–2.6 | 0 | 0.041 |
| `c64`: critic 64 | 130 µs | 9.2 min | 97.4 % | 3.1 ± 1.5 m | 2.8 m, 1.3–6.2 | 1 | 0.050 |
| `c32`: critic 32 | 86 µs | 5.9 min | 96.9 % | 15.9 ± 12.8 m | 12.2 m, 2.4–43.5 | 7 | 0.072 |
| `c64_i16`: critic 64, every 16 steps | 81 µs | 6.3 min | 97.5 % | 11.0 ± 12.8 m | 3.5 m, 1.2–39.3 | 4 | 0.050 |
| `c32_i16`: critic 32, every 16 steps | 55 µs | 4.7 min | 95.4 % | 28.5 ± 15.8 m | 22.7 m, 12.3–55.1 | 10 | 0.088 |
| `c128_gate50k`: curriculum gate every 50k steps | 312 µs | 16.1 min | 97.6 % | 2.4 ± 1.7 m | 2.0 m, 1.0–7.1 | 1 | 0.042 |
| `c128_8M`: 8M steps | 312 µs | 31.9 min | **98.4 %** | **1.4 ± 0.35 m** | **1.3 m, 1.1–2.3** | 0 | 0.046 |

Position: RMSE on test A (`evaluation_best.json`), mean ± std over the seeds. Run time: `time/time_elapsed` of the last
row of `progress.csv`, including the 100 validation flights every 50k steps.

- **Smaller critics cost precision, and below 64 the training fails.** Critic 64 runs 1.7 times faster for 3.1 m
  instead of 1.8 m; critic 32 and the updates every 16 steps leave most seeds tens of meters from the point. Those
  actors still complete 95–98 % of the flights: they fly safely but do not follow the point. Four of the `c32` seeds
  and six of the `c32_i16` seeds never reach 97 % of validation flights after 1M steps. L2F's sizes, enough for a
  quadrotor hovering at a point, are not enough here.
- **The faster curriculum gate does not bring the precision earlier**: the best checkpoints are still at 2–4M steps,
  and the error is 2.4 m, within what a rerun changes.
- **Training longer helps: 1.4 ± 0.35 m at 8M steps**, every seed between 1.1 and 2.3 m, 98.4 % of test A completed.
  The 8M runs repeat the 4M runs step for step up to 4M (same validation results at every checkpoint, seeds 0 and 3
  checked), so the gain is from the extra steps alone. Half the seeds keep a checkpoint after 5M steps.
- The airplane simulation is 2 % of the time: the training time is the network updates, and only fewer or cheaper
  updates would shorten it.
