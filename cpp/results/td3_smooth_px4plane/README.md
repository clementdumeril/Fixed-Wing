# Smoothing TD3's commands (Savio, 4 November 2026)

`slurm/tracking_px4plane_td3_smooth.slurm` (job 40876567): TD3, critic 64, batch 256 and the L2F intervals, 4 M
steps, seeds 0 to 9, best checkpoint every 50 k steps, as `td3_c64_b256_i10a20_4M` in
[`l2f_td3_px4plane`](../l2f_td3_px4plane/README.md), whose policies switch the surfaces between their limits:

- `td3_c64_i10a20_change10_4M`: 10 times the weight of the command-change cost (`--action-change-scale 10`);
- `td3_c64_i10a20_noise005_4M`: half the exploration noise (0.05 instead of 0.1);
- `td3_c64_i10a20_noise005_change10_4M`: both.

| 10 seeds, best checkpoint | Training time | Test A | Test B | Position RMS | Command change |
| --- | --- | --- | --- | --- | --- |
| TD3 (l2f_td3_px4plane) | 9.0 min | 98.5 ± 2.3 % | 98.4 % | 13.3 ± 4.0 m | 0.569 ± 0.281 |
| TD3, command cost x10 | 9.3 min | 98.7 ± 1.2 % | 97.3 % | 12.3 ± 4.7 m | 0.540 ± 0.157 |
| TD3, noise 0.05 | 8.3 min | 98.3 ± 1.3 % | 97.8 % | 16.1 ± 4.1 m | 0.604 ± 0.197 |
| TD3, both | 6.6 min | 99.0 ± 0.8 % | 99.2 % | 14.2 ± 3.2 m | 0.592 ± 0.166 |
| *SAC, critic 128 (l2f_td3_px4plane)* | *17.5 min* | *99.7 ± 0.5 %* | *98.3 %* | *12.9 ± 4.4 m* | *0.036 ± 0.023* |

## What it shows

- **Neither change smooths the commands**: 0.54 to 0.60 in every variant, against 0.57 before and 0.036 for SAC. The
  tracking stays as good (98–99 % of test A, 12–16 m).
- **A likely reason, not tested yet: the command cost sits inside the capped tanh of the reward.** The reward is
  1 - 0.5 tanh(c / 0.5) - ..., and the command change enters c as its weight times the sum of the 4 squared changes.
  For a bang-bang policy that sum is about 4 x 0.57 = 2.3, so c > 2 and the tanh is flat: its slope is about 2000
  times smaller than for SAC's smooth commands (sum about 0.14). Once a policy chatters, the cost no longer tells it
  that smoother would be better, and multiplying the weight by 10 only pushes further into the flat part. The margin
  to the stall had the same problem and was moved outside the cap for the same reason.
- **Next**: the command cost outside the cap, bounded so that the reward of a step stays above 0 away from the stall.

## Files

`<setting>/seed<k>/`: `config.json`, `progress.csv`, `checkpoints.csv`, `evaluation{,_random,_ahead}.json` (last
actor) and `evaluation_best{,_random,_ahead}.json` (kept actor).
