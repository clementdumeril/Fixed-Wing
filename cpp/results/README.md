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
  matters; the next step is a sweep between the two.
- The curriculum stayed at its initial weights during both trainings: the 256 run went under 15° after 140 000 steps,
  and the gate averages over 100 000-step periods.

## Files

- `large_300k/`, `small_1M/`: `evaluation.json`, `progress.csv`. In `large_300k/progress.csv` the episode length and
  return are estimated over a 10 000-step window, which can exceed 1500 steps; later runs count only the episodes that
  end inside a 50 000-step window.
- `training.png`: made with `python/examples/plot_training.py`.
