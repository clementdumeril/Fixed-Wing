# C++ port: check against Python and first trainings (23 October 2026)

## Same physics and task as Python

`compare_with_python.txt`: 11 flights recorded with the Python environment (10 with random commands around trim, one
full nose-up flight that stalls), replayed in C++ from the same starts with the same commands, 7136 steps in total.
States agree within 7e-13, rewards within 8e-15, terminations exactly, observations within 1.2e-7 (Python stores them
as float32). Made with `tools/compare_with_python.py`.

## First trainings (calm air, nominal airplane, one seed)

Both evaluated on the 100 test scenarios of the Python environment (`tools/export_scenarios.py`), so the numbers compare
directly with `python/results/sac_gated_curriculum`. One core of a Windows laptop with Intel MKL, the two runs sharing
the machine for most of the time.

| 100 test scenarios | C++, critic 256, batch 256, 300 000 steps | C++, critic 64, batch 64, 1 000 000 steps | Python, 1 000 000 steps (mean of 3 seeds) |
| --- | --- | --- | --- |
| Training time | 1 h 42 on 1 core | 28 min on 1 core | about 3.5 h on 8 cores |
| Complete flights | 88 | 83 | 90 |
| Mean return (max 1500) | 1114 | 738 | 1163 |
| Airspeed RMS error | 0.51 m/s | 1.22 m/s | 0.55 m/s |
| Altitude RMS error | 2.31 m | 2.41 m | 2.33 m |
| Course RMS error | 8.1° | 28.0° | 4.4° |

- With the network and batch sizes of the Python run, the C++ training learns like the Python one: at 300 000 steps
  it is where the Python runs were at 300 000 steps (local check: 90 % of 20 flights, 8.6° of course error).
- With the L2F sizes it runs 10 times faster per step but never learns the course: the error stays above the 15° of
  the curriculum gate, so the reward weights never grow (`training.png`). For this task the critic or the batch size
  matters; the sweep between the two is in [`sweep/`](sweep/README.md): the critic size decides.
- The curriculum stayed at its initial weights during both trainings: the 256 run went under 15° after 140 000 steps,
  and the gate averages over 100 000-step periods.

## Files

- `large_300k/` (`train_sac`), `small_1M/` (`train_sac_small`, critic 64 and batch 64, in tag `experiments-2026-11`):
  `evaluation.json`, `progress.csv`. In `large_300k/progress.csv` the episode length and
  return are estimated over a 10 000-step window, which can exceed 1500 steps; later runs count only the episodes that
  end inside a 50 000-step window.
- `training.png`: made with `python/examples/plot_training.py`.

## Result folders

- [`sweep/`](sweep/README.md): network and batch sizes on the first task.
- [`long_px4plane/`](long_px4plane/README.md): 3 M steps on the advanced_plane with the best checkpoint, and the
  figure of the PX4 SITL flight against this simulator.
- [`l2f_reward_px4plane/`](l2f_reward_px4plane/README.md): the L2F reward form with squared errors, 9.8 m.
- [`precision_px4plane/`](precision_px4plane/README.md): closing the steady lag, velocity guidance K = 0.3 /s, 6.6 m.
- [`fine_px4plane/`](fine_px4plane/README.md): the fine-scale observation and the closeness bonus, 4.9, 2.4 and 1.8 m.
- [`speed_px4plane/`](speed_px4plane/README.md): training time against precision, 1.4 m in 8 M steps.
- [`reward_px4plane/`](reward_px4plane/README.md): the norms instead of the squares, 1.4 m in 4 M steps: the precision
  recipe of `slurm/tracking_px4plane_reward.slurm`.
- [`reference_suite/`](reference_suite/README.md), [`reference_suite_fine/`](reference_suite_fine/README.md): points
  that turn or change speed.
- [`eratix/`](eratix/README.md): the first Eratix model, `slurm/tracking_eratix.slurm`.

The folders of the other variants (batch 256, the L2F update intervals, TD3 and its smoothing, near starts) are in tag
`experiments-2026-11`.
