# The reward form of Learning to Fly in Seconds on PX4's advanced_plane (lab PC, 5 November 2026)

Same task and recipe as [`l2f_td3_px4plane`](../l2f_td3_px4plane/README.md) (SAC, critic 128, batch 256, critics
every 10 steps and actor every 20, 4 M steps, best checkpoint every 50 k steps), with only the reward changed
(`--reward`, `RewardConfig::l2f_form` in `include/fixed_wing/fixed_wing.h`): survival 2 minus weighted squared scaled
errors (position / 20 m, velocity toward the point / 4 m/s, sideslip / 5°, command change, stall margin), no tanh
cap, and the L2F schedule: every 100 k steps the position weight x1.2 and the velocity and command weights x1.4, up to
caps, with the stored rewards recomputed. Seeds 0 to 3 on the lab PC (MSVC); the reference ran on Savio (gcc).

- `l2f`: the reward clipped at 0, so that ending a flight never pays.
- `l2f-penalty`: clipped at −2, and the step that ends a flight gets −2 / (1 − γ) = −400, as bad as staying at the
  floor for the rest of the flight.

L2F's orientation term is left out (an airplane banks to turn) and its action-magnitude term is replaced by the
command change (no trim prior here). The velocity guidance, stall margin, γ = 0.995 and the task are unchanged.

| 4 seeds, best checkpoint | Test A | Test B | Position RMS | Velocity RMS | Command change | Flies after |
| --- | --- | --- | --- | --- | --- | --- |
| tanh (ours, seeds 0–3 of the 10 on Savio) | 99.2 ± 0.4 % | 98.2 % | 14.1 ± 2.4 m | 1.95 m/s | 0.045 | 488 k steps |
| `l2f` (floor 0) | 96.0 ± 2.7 % | 95.5 % | 56.6 ± 32.8 m | 2.46 m/s | 0.109 | 450 k |
| **`l2f-penalty`** | **99.2 ± 0.4 %** | 97.8 % | **11.5 ± 1.8 m** | **0.68 m/s** | **0.031** | **295 k** |

Velocity RMS per seed: `l2f-penalty` 0.31, 0.42, 0.56, 1.45 m/s; tanh (10 seeds) 0.35 to 3.95 m/s. The recomputed
rewards match the stored ones to 4e-7 (`reward_check.txt`).

## What it shows

- **With a floor at 0 the L2F form is worse**: past about 18 m the reward is flat at the final weights, the
  position error of the kept checkpoints is 49–90 m in 3 of 4 seeds. The saturation that the tanh had, made sharper.
- **With the floor below 0 and the termination penalty it matches or beats ours** on 4 seeds: the same success, a
  little less position error (within the spread), a velocity error about 3 times smaller, smoother commands, and it
  flies after fewer steps. The quadratic keeps a slope everywhere, and the penalty keeps crashing from paying.
- 4 seeds, another machine and compiler than the reference: to be confirmed on 10 seeds, and with TD3, whose
  bang-bang commands may come from the flat part of the tanh.

## Files

`<variant>_seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `reward_check.txt`, `evaluation*.json`.
