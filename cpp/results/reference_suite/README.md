# Tracking a point that accelerates (lab laptop, 6 November 2026)

The 10 best actors of SAC 128 with the L2F reward (Savio job 40965874, [`l2f_reward_px4plane`](../l2f_reward_px4plane/README.md):
98.8 % of test A completed, 9.8 m RMS) trained with a point that always flies straight at constant velocity. Here
the point follows other paths (`include/fixed_wing/reference_motion.h`), each with its acceleration swept past what
the policy can follow (`src/evaluate_reference.cpp`):

- **circle**: constant speed V (12–18 m/s, as sampled for the test), centripetal acceleration a, so R = V²/a
  (R below is at 15 m/s); turns alternate left and right.
- **helix**: the same circle, climbing or descending at 2 m/s (twice the largest climb of training).
- **figure-eight**: the Lissajous figure of L2F Fig. 5, x = A sin ωt, y = A/2 sin 2ωt, constant altitude, mean speed
  15 m/s (the speed along it spans 10.2–21.9 m/s); a is its peak lateral acceleration; the start along the figure is
  spread over the flights.
- **speed 12–18**: straight line, the speed oscillates sinusoidally over 12–18 m/s with a peak acceleration a.
- **speed at 2 m/s²**: the same at 2 m/s² over wider ranges, down to 6 m/s, below the 7 m/s at which a flight ends.

"Bank" is atan(a/g), the bank of a coordinated turn. Each motion starts where the test puts the point (position,
direction of its velocity), so only the path changes, and the velocity the actor observes is the analytic one at every
step. Two kinds of starts, the first 100 flights of each test (seed 12345), the same for every actor and level:

- **level**: test A, trimmed flight perturbed by up to 20° of roll and 10° of pitch, neutral commands;
- **random**: test B, the training distribution (`--start random` in the training runs): roll ±60°, pitch ±20°,
  rates ±0.5 rad/s, airspeed 11–19 m/s, α −4…10°, β ±8°, random commands.

On the straight line the program flies the tests themselves: it reproduces `evaluation_best.json` and
`evaluation_best_random.json` of all 10 Savio runs (success equal, RMSE within 1e-4 m). 84 levels x 100 flights per
actor, 50 s per actor on 7 laptop threads.

```bash
build/evaluate_reference <run>/best_actor.h runs/reference_suite/<name> --flights 100 --record 3
python tools/plot_reference_suite.py results/reference_suite runs/reference_suite/<name>...
```

## Results (10 seeds, mean ± std; RMSE over the second half of the completed flights)

| Motion | a [m/s²] | Bank | At 15 m/s | Completed, level | Completed, random | Within 20 m, level | RMSE level [m] | RMSE random [m] |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| straight | 0 | 0° | – | 99 ± 1 % | 96 ± 3 % | 96 % | 9.8 ± 2.3 | 9.7 ± 2.3 |
| circle | 0.75 | 4° | R 300 m | 99 ± 2 % | 96 ± 3 % | 97 % | 9.8 ± 1.9 | 9.9 ± 2.2 |
| circle | 1.9 | 11° | R 120 m | 98 ± 1 % | 96 ± 3 % | 97 % | 10.9 ± 1.8 | 11.1 ± 1.8 |
| circle | 3.75 | 21° | R 60 m | 98 ± 2 % | 96 ± 3 % | 96 % | 12.2 ± 1.6 | 12.2 ± 1.6 |
| circle | 6.4 | 33° | R 35 m | 98 ± 3 % | 96 ± 3 % | 95 % | 12.6 ± 1.8 | 12.7 ± 1.8 |
| circle | 9 | 43° | R 25 m | 97 ± 3 % | 96 ± 3 % | 92 % | 13.2 ± 2.2 | 13.2 ± 2.2 |
| circle | 12.5 | 52° | R 18 m | 97 ± 4 % | 94 ± 5 % | 86 % | 14.5 ± 3.0 | 14.5 ± 3.0 |
| circle | 17.3 | 60° | R 13 m | 92 ± 9 % | 90 ± 9 % | 71 % | 16.0 ± 3.4 | 15.9 ± 3.4 |
| circle | 25 | 69° | R 9 m | 72 ± 19 % | 72 ± 16 % | 52 % | 17.4 ± 3.1 | 17.7 ± 3.0 |
| circle | 34.6 | 74° | R 6.5 m | 38 ± 24 % | 43 ± 24 % | 23 % | 19.7 ± 4.2 | 19.1 ± 3.5 |
| circle | 50 | 79° | R 4.5 m | 12 ± 18 % | 15 ± 21 % | 4 % | 24.8 ± 6.8 | 22.4 ± 5.2 |
| helix | 3.75 | 21° | R 60 m | 98 ± 2 % | 96 ± 3 % | 96 % | 12.5 ± 1.6 | 12.5 ± 1.5 |
| helix | 12.5 | 52° | R 18 m | 97 ± 3 % | 93 ± 5 % | 86 % | 14.5 ± 2.8 | 14.7 ± 2.8 |
| helix | 25 | 69° | R 9 m | 71 ± 19 % | 72 ± 15 % | 52 % | 17.4 ± 2.8 | 17.3 ± 2.9 |
| figure-eight | 0.75 | 4° | period 272 s | 97 ± 2 % | 95 ± 3 % | 95 % | 9.8 ± 2.0 | 9.9 ± 2.1 |
| figure-eight | 3.75 | 21° | period 54 s | 97 ± 2 % | 95 ± 4 % | 96 % | 10.8 ± 1.4 | 10.9 ± 1.4 |
| figure-eight | 9 | 43° | period 23 s | 95 ± 3 % | 94 ± 4 % | 95 % | 12.7 ± 1.2 | 12.6 ± 1.2 |
| figure-eight | 17.3 | 60° | period 12 s | 93 ± 4 % | 91 ± 6 % | 88 % | 15.2 ± 1.8 | 15.4 ± 1.9 |
| figure-eight | 25 | 69° | period 8 s | 79 ± 19 % | 76 ± 17 % | 53 % | 18.6 ± 3.4 | 18.7 ± 3.6 |
| figure-eight | 34.6 | 74° | period 6 s | 61 ± 25 % | 58 ± 26 % | 34 % | 21.9 ± 5.9 | 21.7 ± 5.6 |
| figure-eight | 50 | 79° | period 4 s | 16 ± 22 % | 18 ± 22 % | 6 % | 24.2 ± 4.7 | 24.7 ± 4.9 |
| speed 12–18 | 0.25 | – | period 75 s | 99 ± 2 % | 96 ± 3 % | 97 % | 9.6 ± 2.1 | 9.5 ± 2.2 |
| speed 12–18 | 2 | – | period 9.4 s | 99 ± 2 % | 96 ± 3 % | 99 % | 8.4 ± 1.8 | 8.3 ± 1.8 |
| speed 12–18 | 8 | – | period 2.4 s | 99 ± 2 % | 96 ± 3 % | 99 % | 7.8 ± 2.0 | 7.8 ± 2.1 |
| speed at 2 m/s² | 2 | – | 10–20 m/s | 98 ± 2 % | 96 ± 3 % | 98 % | 8.5 ± 1.6 | 8.4 ± 1.5 |
| speed at 2 m/s² | 2 | – | 8–22 m/s | 94 ± 5 % | 91 ± 6 % | 94 % | 9.3 ± 1.5 | 9.4 ± 1.5 |
| speed at 2 m/s² | 2 | – | 6–24 m/s | 75 ± 11 % | 72 ± 13 % | 70 % | 11.8 ± 1.9 | 11.7 ± 1.8 |

Every level, with the along-track, cross-track and vertical offsets, velocity RMSE, bank, α and failure causes:
`summary.csv`; per seed: `per_seed/*.json`. Figures: `success_rmse.png` (success and RMSE against bank or
acceleration), `trajectories.png` (top-down flights at an easy, a medium and a breaking level, seed 4),
`circle_limits.png` (bank and α flown, and how the flights end, on the circles).

## What it shows

- **Up to about 45° of equivalent bank (9 m/s²) the policy follows circles, helices and figure-eights** as well as
  it follows the straight line: 97–98 % completed, the RMSE rising from 9.8 to 13 m. The extra error is a lag and a
  height offset: on circles the airplane flies 6–10 m behind the point and, from 30° of bank on, 5–12 m above it.
- **It degrades from 60° (17 m/s²) and breaks at 69° (25 m/s², R 9 m at 15 m/s)**: 72 % completed, half the flights
  within 20 m, and a large spread between seeds (36 to 98 % completed on circles). At 74° 38 % complete, at 79° 12 %.
  The figure-eight breaks at the same peak lateral acceleration, a little later as it reaches it only on part of the
  figure. The 2 m/s climb of the helix changes nothing.
- **Not the airplane's limit in steady flight**: from the lift curve of the model, a level turn at 25 m/s² needs α of
  5–10° at 12–18 m/s, and 34.6 m/s² is possible from about 12.5 m/s (`circle_limits.png`, dotted). The completed
  flights fly the bank of the coordinated turn and an α close to the steady one. The flights end at the 18° α limit
  (70–80 % of the ends) or below 7 m/s of airspeed (pulling up with the throttle near idle, in the recorded
  flights), half to two thirds of them within the first 5 s, while the airplane joins a point that already turns hard.
- **The policy never saw such turns in training**: the point flies straight, so steady turns only happen while
  joining it, and the random starts stop at 60° of roll. The breaking point is where the needed bank leaves that range.
- **Speed changes within 12–18 m/s never break it**, at any rate up to 8 m/s²; the RMSE even falls (7.8 m), as fast
  oscillations move the point by at most 3²/a m around a 15 m/s line and the along-track lag of the straight line
  goes away. It breaks when the point slows to near the 7 m/s limit: over 6–24 m/s 75 % complete, the flights end on
  the airspeed limit (189 of 247 ends): the policy follows the point's speed down instead of letting it go ahead.
- **Level and random starts give the same curves**; random starts lose 2–3 % more at every level, in the first
  seconds, independently of the motion.

## Next step: training with accelerating references (not run)

The obvious next step is to train with these motions: an option such as `--reference-motion mixed` that draws a
motion per flight (straight, circle, figure-eight, speed oscillation) with accelerations up to about 25 m/s², which
needs the motion and the time since the start in `State` and `simulate_task` moving the point with `reference_at`. The
policy observes the point's velocity but not its acceleration; adding the acceleration (or the point a second ahead)
to the observation would let it anticipate the turn, at the cost of a new observation size and a new target.
