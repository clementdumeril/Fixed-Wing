# Limit tests of the best tracking policy (advanced_plane, simple reward, energy weight 0, seed 1)

Actor: the best actor of run energy 0, seed 1 of job 41645214 ([`cpp/results/simple_px4plane/energy0/seed1`](../../cpp/results/simple_px4plane/energy0/seed1), 65 inputs).
Simulator: our C++ model of PX4's `advanced_plane` (`cpp/include/fixed_wing/`), the same physics as training.
Tool: `cpp/src/evaluate_limits.cpp` (target `evaluate_limits`), tables and figures by `cpp/tools/analyze_limit_tests.py`,
videos by `cpp/tools/render_flight_video.py`. The videos are not stored in the repository; the commands under
Reproduce make them again.

## Protocol

- **Flights.** Each severity level is flown by **50 flights** (50 seeds: start state, point speed 12-18 m/s, point
  heading within 20 deg, point within 5 m at the start, as in the tests of training). The same 50 seeds are used at
  every level of a scenario, so the levels differ only by the severity. 60 s per flight (90 s for the motor cut), the
  event at **t = 10 s**, after the approach.
- **Background.** Calm air, nominal airplane, 1 step (20 ms) of latency, a point flying straight, so that each test
  isolates one effect. The motor cut is also flown on top of the full training disturbances (`motor_cut_disturbed`:
  wind 0-6 m/s, Dryden up to moderate, gusts, thermals in half the flights, randomized airplane, latency 0-2).
- **Two outcomes per flight.**
  - *Completed*: the flight never meets a real termination rule of training and the tests (more than 30 m from the
    point, angle of attack outside -15/18 deg, airspeed outside 7/30 m/s, below 20 m, more than 50 m above or below the
    point). This is the success rate of the tests (85 % disturbed, 100 % calm for this actor).
  - *Recovered*: the simulation goes on past the tracking rules (30 m, 300 m, 50 m of altitude error) and stops only on
    the physical ones (stall/angle of attack, airspeed, 20 m floor, divergence). The flight is recovered when it is
    still flying at the end and has been back **within 12 m** of the point for at least the last 5 s (12 m is about
    3.5 times the calm RMSE; the largest distance of a calm baseline flight is 11 m at the 90th percentile). The
    *recovery time* is the time from the end of the event (motor restart, end of the gust, wind step, jump) to the last
    moment the distance was above 12 m. For changes that last the whole flight (mass, thrust, latency, ...) the
    second rate is *survived* (no stall, overspeed or crash).
- **Measured per flight** (`raw/summary.csv`): termination reason and time under both rule sets, RMSE of the 3D
  distance over the flight (as the tests), maximum distance after the event, recovery time, minimum airspeed,
  maximum angle of attack and bank, minimum altitude; for the motor cut, altitude lost, mean airspeed, angle of attack,
  pitch and sink rate while gliding (after the first 2 s of the cut), worst airspeed and angle of attack in the cut,
  glide ratio (horizontal distance / height lost during the cut).
- **Recorded flights** (`raw/flights/<scenario>_<index>_<level>.csv`, the first 2 flights of every level, every 20 ms):
  position, true roll, pitch, heading, airspeed, angle of attack, sideslip, ground and vertical speed, the point, the
  distance, the local wind, the commanded actions, the applied throttle, the event flag and both termination reasons.
- **Breaking point**: the first level where fewer than 90 % of the flights complete.

### Check of the tool against training

The tool flies the actor from its weights in its own loop (latency queue, motor cut, sensor noise). Replaying the 100
test flights of training (`reproduce_*`, same seeds, 120 s):

| test | training (`evaluation_best_*.json`) | this tool |
| --- | --- | --- |
| calm | 100 %, RMSE 3.633 m | 100 %, RMSE 3.632 m |
| disturbed | 85 %, RMSE 4.22 m (airspeed 8, alpha 4, distance 3) | 85 %, RMSE 4.50 m (airspeed 9, alpha 3, distance 3) |

Calm air matches to the millimetre. In turbulence a few flights diverge after some time (float summation order of the
network, then chaotic flight), with the same success rate.

Baselines of this protocol: calm 100 % (RMSE 3.4 m), training disturbances 86 % (RMSE 4.9 m).

## Breaking points

Training range in brackets. "Safe up to" is the last level with at least 90 % completed.

| test | training | safe up to | breaks at (completed) | < 50 % at | how it fails | meaning |
| --- | --- | --- | --- | --- | --- | --- |
| **Motor cut** (calm) | never | **5 s** (100 %) | 10 s (68 %) | 15 s | falls > 30 m behind, then stalls | 3-5 s: altitude lost 5-8 m, back within 12 m in < 5 s. Beyond 10 s it stalls trying to hold altitude |
| Motor cut (training disturbances) | never | 5 s (90 %, baseline 90 %) | 10 s (70 %) | 20 s | same | the disturbances barely change the limit |
| Motor cut, 2 s soft restart | never | 5 s (100 %) | 10 s (52 %) | 15 s | same | a gentler restart does not help (below) |
| Headwind step | 0-6 any direction | 10 m/s | 12 m/s (68 %) | 14 m/s | airspeed > 30 m/s | needs point speed + wind > 28 m/s |
| Headwind ramp (10 s) | | 10 m/s | 12 m/s (76 %) | 14 m/s | airspeed > 30 m/s | same: the steady state, not the step, sets the limit |
| Tailwind step | | 2 m/s | 4 m/s (86 %) | 8 m/s | airspeed < 7 m/s | needs point speed - wind < 10 m/s: the slow points (12 m/s) are lost first |
| Tailwind ramp (10 s) | | 2 m/s | 4 m/s (84 %) | 8 m/s | airspeed < 7 m/s | same |
| Crosswind step | | 10 m/s | 12 m/s (88 %) | not reached (66 % at 20) | angle of attack | crab angle about 50 deg at 20 m/s |
| Head gust (1-cos, 2 s) | up to 3 m/s | 10 m/s | 13 m/s (86 %) | 16 m/s | airspeed > 30 m/s | |
| Tail gust | | 5 m/s | 7 m/s (80 %) | 10 m/s | airspeed < 7 m/s | |
| Side gust | | 20 m/s | not reached (98 % at 20) | | | |
| Down gust (sink) | | 20 m/s | not reached (96 % at 20) | | | |
| Up gust | | 16 m/s | 20 m/s (70 %) | | angle of attack | |
| Dryden turbulence | up to 1 x moderate | 1 x moderate (92 %) | **1.5 x** (66 %) | 2 x (34 %) | stall and airspeed, after roll upsets | 2 x moderate is about "severe": 0 % at 3 x |
| Thermal core updraft | up to 3 m/s | 5 m/s (94 %) | 6 m/s (86 %) | not reached (62 % at 10) | stall, distance | the sink ring and the updraft core |
| Mass, heavier | +-10 % | 2 x (94 %) | 2.5 x (80 %) | not reached (66 % at 3x) | distance, stall | RMSE 3.4 -> 7.6 m at 2 x |
| Mass, lighter | | 0.4 x (100 %) | not reached | | | |
| Thrust | +-15 % | **0.18 x** (98 %) | 0.12 x (44 %) | 0.12 x | falls behind, stalls | 0.12 x = 2.6 N, about the drag at 18 m/s |
| Control effectiveness, weaker | +-10 % | 0.25 x (96 %) | 0.15 x (74 %) | not reached (54 % at 0.1) | distance | a quarter of the control power still flies |
| Control effectiveness, stronger | | 1 x | **1.5 x** (86 %) | 3 x (34 %) | angle of attack | too much gain: pitch oscillation into the alpha limit |
| Latency | 0-2 steps (0-40 ms) | 4 steps = 80 ms (98 %) | **5 steps = 100 ms** (78 %) | 6 steps = 120 ms (20 %) | angle of attack | pitch limit cycle |
| Sensor noise (white, on the observed state) | none | 12 x base (100 %) | not reached | | | RMSE 3.4 -> 8.1 m at 12 x (position 6 m, attitude 6 deg, gyro 0.24 rad/s) |
| Point jumps ahead / behind / side / down / up | never | 20 m | 30 m (14-46 %) | 30 m | the 30 m rule | below 30 m: 98-100 % back within 12 m in 1.3-1.6 s (median) |
| Point heading jump | never (point turns within 30 deg of bank) | 60 deg (98 %) | 90 deg (58 %) | 120 deg | angle of attack in the turn | |

Full per-level tables (RMSE, max distance, recovery time, all failure reasons): `tables.md`, `levels.csv`,
`breaking_points.csv`.

## The motor cut

Throttle forced to 0 at t = 10 s for 3, 5, 10, 15, 20, 30, 45 s and for the rest of the flight (80 s), start at
100-120 m, point flying level at 12-18 m/s, 50 flights per duration.

| cut | completed | recovered | recovery after restart, median / p90 | altitude lost, median / worst | glide (airspeed, alpha mean / max, sink) | glide ratio |
| --- | --- | --- | --- | --- | --- | --- |
| 0 s | 100 % | 100 % | - | 0.4 / 5.3 m | - | - |
| 3 s | 100 % | 100 % | 0 / 4.7 s | 4.9 / 12.3 m | 14.2 m/s, 2.6 / 3.6 deg, 0.60 m/s | 21 |
| 5 s | 100 % | 100 % | 0 / 5.1 s | 8.0 / 14.6 m | 13.8 m/s, 3.6 / 5.0 deg, 1.09 m/s | 16 |
| 10 s | 68 % | 74 % | 2.2 / 4.1 s | 9.4 / 22.0 m | 13.2 m/s, 4.3 / 8.1 deg, 0.83 m/s | 16 |
| 15 s | 28 % | 34 % | 2.6 / 3.8 s | 11.0 / 23.8 m | 12.8 m/s, 5.0 / 17.0 deg, 0.71 m/s | 19 |
| 20 s | 12 % | 12 % | 4.6 / 4.8 s | 11.5 / 23.8 m | 12.6 m/s, 5.2 / 18.3 deg, 0.68 m/s | 19 |
| 30 s and more | 0 % | 0 % | - | - | stall in every flight | |

In the training disturbances: 90 / 92 / 90 / 70 / 50 / 18 / 8 % completed for 0 / 3 / 5 / 10 / 15 / 20 / 30 s.

What the policy does (`figures/timeline_motor_cut_*.png`, `videos/`):

1. At the cut it pushes the nose down (pitch to -15 deg), trades height for speed and keeps 13-14 m/s: a glide at
   about 0.7-1 m/s of sink, a glide ratio of 16-20, close to the airplane's best $L/D$. It keeps commanding full throttle
   (it cannot know the motor is off; it only sees its own commands).
2. A 3-5 s cut costs 5-8 m of height and 8-12 m of distance; at the restart it climbs back and is within 12 m of the
   point at once or within 5 s. **100 % recovery.**
3. The glide is slower than the point (12-18 m/s), so the distance grows. Once the airplane is 12-15 m below the point
   it stops descending and pulls the nose up to hold altitude, as it always could with a motor. The airspeed then
   decays and it stalls (angle of attack limit) 9 to 28 s after the cut (median 14 s), whatever the cut duration.
   The faster the point, the earlier: 10 s cut, 93 % completed with a point at 12-14 m/s, 57 % at 16-18 m/s.
4. The restart itself is a second failure mode. For the 10 s cut, 10 of the 13 physical failures are an overspeed
   (above 30 m/s) 1.2-1.8 s after the restart. These flights are 15-20 m below and behind the point when the motor
   comes back, a state training never produced, and the policy is already manoeuvring hard (flight 1: pull-up to
   38 deg of pitch, roll to 100 deg, dive at 8 m/s). It has commanded full throttle for 10 s, so the motor returns at
   21.9 N for 1 kg (2.2 g) in the middle of the pull-up, and the airspeed goes from 16 to 30 m/s in about 1.3 s.
5. A 2 s soft restart (the applied throttle capped by a ramp from 0 to full, as an ESC soft start;
   `motor_cut_soft_restart`) does not cure it: 52 % completed at 10 s instead of 68 %. The same 10 flights still
   overspeed and 6 more do; the airplane simply stays slow and low for longer. The problem is the state the policy
   reaches during the cut, not the throttle step.

So the breaking point of the motor cut is **between 5 and 10 s**, and it is a behaviour limit, not a physical one:
with a glide ratio of 17, a 20 s cut at 15 m/s costs about 18 m of height, which the airplane has (it starts at
100-120 m, the floor is 20 m). A policy that kept gliding instead of holding altitude would survive the cut and only
lose the 30 m tracking rule. It never saw a motor failure in training.

## Other findings

- **Wind: the limit is the airspeed band, not the policy.** Pooling the head and tail wind runs by the airspeed the task
  requires (point speed + headwind - tailwind): 98-100 % completed between 10 and 28 m/s, 56 % between 8 and 10, 0-2 %
  below 8 m/s, 20-40 % between 28 and 30 m/s, 0 % above 30 (`figures/wind_required_airspeed.png`). The step and the
  10 s ramp give the same curve, so the sudden change itself costs nothing. The airspeed rules (7 and 30 m/s) are the
  envelope.
- **Far from the point is unknown territory.** The 30 m rule ended every training flight that went farther, so the
  policy never flew beyond 30 m. When the rule is lifted (relaxed runs), a point 40-80 m away makes it overspeed
  (point ahead) or stall in a tight turn (point behind): 0-16 % survive a 60 m jump. Below 30 m every jump is
  recovered in 1.3-1.6 s (median).
- **Latency and control gain** fail the same way (angle of attack, a growing pitch oscillation): the policy tolerates
  4 steps (80 ms) of latency, twice its training range, but not 1.5 x the control power. Weaker controls are much less
  of a problem (0.25 x still flies).
- **Turbulence** is the tightest margin: 92 % at the training maximum, 66 % at 1.5 x. Thermals are tolerated up to
  5 m/s of core updraft (training 3).
- **Sensor noise** (white noise on the observed state, which the policy never saw) does not make any flight fail up to
  12 x; it only costs precision (RMSE 8 m at 12 x).
- **Turbulence failures start with roll upsets**: in the 1.5 x flight of the video the airplane rolls past 90 deg
  twice (inverted for 1.8 s in total, pitch down to -32 deg) before it hits the angle of attack limit at 36 s
  (`figures/timeline_turbulence_1p5x_flight0.png`).
- In calm air, with or without any event, some flights settle into a small sustained pitch and throttle oscillation
  (pitch standard deviation about 4-5 deg, throttle between 0 and 0.45, about 2.5 s period) while tracking within 3 m
  (visible after the recovery in `figures/timeline_motor_cut_10s_flight0.png`; the same seed without a cut and the
  calm baseline show it too).

## Figures

- `figures/success_vs_severity.png`: completed (solid) and recovered or survived (dashed) against the severity, all
  tests, training range shaded.
- `figures/rmse_vs_severity.png`: RMSE of the completed flights against the severity.
- `figures/motor_cut_summary.png`: motor cut (calm, disturbed, soft restart), completed and recovered, altitude lost,
  recovery time.
- `figures/wind_required_airspeed.png`: wind runs against the airspeed the task requires.
- `figures/timeline_motor_cut_10s_flight0.png` (recovered), `timeline_motor_cut_10s_flight1.png` (fails, point at
  17 m/s: upset, then overspeed after the restart), `timeline_motor_cut_20s_flight0.png` (stall during the cut),
  `timeline_turbulence_1p5x_flight0.png`: distance, altitude, airspeed, angle of attack, pitch, roll, vertical speed,
  applied and commanded throttle.

## Videos

- `videos/motor_cut_10s_glide_restart.mp4` (45 s, x2 speed): motor off from 10 to 20 s in calm air, glide, restart,
  back on the point; completed, RMSE of the second half 2.6 m. Frame checked (`videos/frame_motor_cut_10s.png`,
  t = 15 s): motor off, pitch -3 deg, airspeed 14.1 m/s, alpha 3.1 deg, 8.6 m from the point.
- `videos/motor_cut_20s_stall_failure.mp4`: the limit, a 20 s cut; the airplane glides, then pulls up and stalls at
  29.4 s, before the restart. Frame checked (`videos/frame_motor_cut_20s.png`, t = 27 s): motor off, pitch +9 deg,
  airspeed down to 10.7 m/s, alpha 8.2 deg and rising, 21.8 m from the point.
- `videos/turbulence_1p5x_moderate_failure.mp4`: Dryden turbulence at 1.5 x moderate (first level above training),
  roll upsets, stall at 36.4 s. Frame checked (`videos/frame_turbulence_1p5x.png`, t = 32 s): the airplane inverted
  (bank -174 deg, pitch -22 deg), which the CSV confirms (roll through -93 and 178 deg at 31.5-32 s).

The motor-off or turbulence period is shaded on the side plots, and the info line shows the true pitch, bank,
airspeed and angle of attack.

## Limits of these tests

- Our simulator only: the same model the policy was trained in. A failure here is a failure of the policy; a success
  says nothing yet about the model error to the real airplane or Gazebo.
- One airplane change at a time, in calm air (except the disturbed motor cut). Combined effects are likely worse.
- The mass change keeps the inertia; the control change scales the force and moment coefficients of all surfaces;
  sensor noise is white (no bias, no GPS random walk); the latency is a pure delay of whole 20 ms steps.
- 50 flights per level: a rate of 90 % has a 95 % interval of about +-8 %.

## Reproduce

```bash
build.bat <worktree>/cpp <build> evaluate_limits          # laptop build, MKL, as cleanup-verify
evaluate_limits best_actor.h limit_tests/raw --flights 50 --record 2 --threads 2     # 3 min on 2 laptop cores
python cpp/tools/analyze_limit_tests.py limit_tests
python cpp/tools/analyze_limit_tests.py --timeline limit_tests/raw/flights/motor_cut_3_10.csv 0 "<title>" <out.png> "motor off"
python cpp/tools/render_flight_video.py limit_tests/raw/flights/motor_cut_3_10.csv out.mp4 --flight 0 --event-label "motor off"
```

`evaluate_limits --list` prints every scenario with its levels; `--scenarios a,b` runs a subset;
`--recovery-radius R` changes the 12 m.
