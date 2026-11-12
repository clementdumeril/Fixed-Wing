# The reward with norms instead of squares (PX4 advanced_plane, tracking)

Job 41220553, `slurm/tracking_px4plane_reward.slurm`, `--reward l2f-norm`. The reward of RLtools' L2F environment at
b32d9985 (`l2f/parameters/reward_functions/squared`) takes the norm of each error (distance to the point, velocity
error, command change), not its square. Our L2F form squared them, so one meter near the point cost almost nothing
(2.4 × (1 / 20)² = 0.006 at the last stage); with the norm it costs 2.4 / 20 = 0.12, the same per meter near the
point as far from it. Same floor (−2) and termination penalty (2 / (1 − γ)) as `l2f-penalty`; every other setting is
the best recipe of [`fine_px4plane`](../fine_px4plane/README.md) (fine observation at 2 m, K = 0.3 /s, precise
checkpoint choice, critic 128, 4M steps). The stored rewards recompute to 3e-7 (`reward_check.txt`).

| 10 seeds, best checkpoint, 4M steps | Test A: done | Random starts: done | Position | Median, range | Last actor | Command change | Steady offset (behind, side, altitude) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| squares, closeness bonus 4 (`fine2m_bonus4`, reference) | 97.6 % | 96.8 % | 1.8 ± 0.5 m | 1.6 m, 1.3–2.6 | 3.5 m | 0.041 | 1.0, 0.8, 0.7 m |
| `norm_bonus4`: norms, closeness bonus 4 | 98.2 % | 95.7 % | **1.4 ± 0.45 m** | **1.3 m, 0.8–2.4** | **2.2 m** | 0.060 | 0.9, 0.7, 0.5 m |
| `norm`: norms, no bonus | 97.6 % | 96.8 % | 3.9 ± 0.6 m | 3.7 m, 3.1–4.9 | 4.8 m | 0.041 | 2.0, 1.6, 2.0 m |
| `norm_scale10`: norms, no bonus, position scale 10 m | 97.9 % | 96.2 % | 3.0 ± 0.7 m | 2.9 m, 2.0–4.1 | 3.8 m | 0.046 | 1.4, 1.4, 1.4 m |

Position: RMSE on test A (`evaluation_best.json`), mean ± std over the seeds. Last actor: `evaluation.json`. Steady
offset: mean absolute error per axis over the second half of the recorded test flights.

- **With the bonus, the norms reach 1.4 m in 4M steps**, what the squares reach in 8M
  ([`speed_px4plane`](../speed_px4plane/README.md), `c128_8M`): 16.7 min of training instead of 32. Four seeds are
  below 1.1 m, the best at 0.84 m. The difference with the squares at 4M (0.4 m) is at the edge of what a rerun changes
  (Welch t-test p = 0.09, Mann-Whitney p = 0.05).
- **The last actors are steadier**: 2.2 m against 3.5 m, so the checkpoint choice matters less.
- **The norms do not replace the bonus**: without it, 3.9 m (3.0 m with twice the cost per meter), better than the
  squares without a bonus (6.0–6.6 m, `fine_px4plane` and `precision_px4plane`) but not near 1 m.
- The commands are rougher (0.060 per step against 0.041), and 1 % fewer flights from random starts complete; the
  failures are still the angle-of-attack limit (42 of 61 ends) and low airspeed.

Next: the norms with the bonus at 8M steps.
