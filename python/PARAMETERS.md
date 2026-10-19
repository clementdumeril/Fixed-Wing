# Parameters: values and where they come from

"Guess" means the value still has to be identified. As of 10 Oct 2026 the aircraft is still the Aerosonde placeholder; the Strix model has been requested from the lab.

## Aircraft (`aerosonde()` in `fixed_wing/airplane/parameters.py`)

| Parameter | Value | Source |
| --- | --- | --- |
| Mass, inertia, wing geometry, air density, 28 aerodynamic coefficients | see `parameters.py` | Beard & McLain, *Small Unmanned Aircraft*, appendix E (Aerosonde) |
| Maximum thrust `thrust_max` | 40 N | Guess |
| Actuator time constants | 0.05 s (surfaces), 0.2 s (throttle) | Guess |
| Surface limits, throttle range | ±30°, [0, 1] | Same as the lab NMPC (Bobbili et al. 2026, Table I) |

## Simulation

| Parameter | Value | Source |
| --- | --- | --- |
| Control period `integration.dt` | 0.01 s (100 Hz) | Same as L2F; external actuator commands to PX4 are limited to < ~50 Hz (Bobbili et al. 2026) |
| Substeps `integration.n_substeps` | 1 | Choice (L2F `N_SUBSTEPS`) |
| Command latency `latency.action_steps` | 0 | Unknown; asked to the lab |
| Thermals | off; w* = 2.56 m/s, z_i = 1401 m when used | Allen 2006, AIAA 2006-1510, table 1 (yearly mean); shape constants from its appendix B code |
| Turbulence | off; presets such as σ = 1.06, 1.06, 0.7 m/s, L = 200, 200, 50 m | Beard & McLain, table 4.1 (MIL-F-8785C), light turbulence at 50 m |

## Strix: what is known

| Quantity | Value | Source |
| --- | --- | --- |
| Mass, wing span | 4 kg, 1.7 m | Lim & Loianno 2026, arXiv 2607.15863 |
| Best-glide airspeed, sink rate | ~12 m/s, 1.51 m/s | Morando et al. 2026, arXiv 2605.15619 |
| Level-flight throttle | 0.376 at ~12 m/s | Morando et al. 2026 |
| Trim | throttle 0.60, elevator 8° | Bobbili et al. 2026, arXiv 2608.20655 |
| Climb / descent rate at 15 m/s | 4.6 / 2.26 m/s in flight | Lim & Loianno 2026 |
| Aero coefficients, inertia, wing area, T_max | not published | requested |

## Placeholder vs Strix (`examples/validate.py`)

| Quantity | Strix | Aerosonde model |
| --- | --- | --- |
| Best-glide airspeed | 12.0 m/s | 16.6 m/s |
| Sink rate at best glide | 1.51 m/s | 1.06 m/s |
| Best glide ratio | 7.9 | 15.7 |
| Level-flight throttle at 12 m/s | 0.376 | 0.287 |
| Climb rate at 15 m/s, full throttle | 4.6 m/s | 3.25 m/s |
| Descent rate at 15 m/s, throttle off | 2.26 m/s | 0.97 m/s |

The Strix has about twice the drag of the placeholder. `calibrate_drag_polar` will fit its polar once the wing area is known.

## Simulation speed (`examples/benchmark.py`)

About 335 steps/s on one core for one airplane in pure Python, so 1 M steps take ~50 min. Batching (NumPy) or the C++ port (RLtools) comes later.
