# Parameters: values and where they come from

Every number the simulator uses, with its source. "Guess" means no source: it must be identified or replaced.
Status on 10 Oct 2026: the aircraft is still the Aerosonde placeholder; the Strix model has been requested from the lab.

## Aircraft (`aerosonde()` in `fixed_wing/airplane/parameters.py`)

| Parameter | Value | Source |
| --- | --- | --- |
| Mass | 13.5 kg | Beard & McLain, *Small Unmanned Aircraft*, appendix E (Aerosonde) |
| Inertia Jx, Jy, Jz, Jxz | 0.8244, 1.135, 1.759, 0.1204 kg·m² | Beard & McLain, appendix E |
| Wing area, span, mean chord | 0.55 m², 2.8956 m, 0.18994 m | Beard & McLain, appendix E |
| Air density | 1.2682 kg/m³ | Beard & McLain, appendix E |
| 28 aerodynamic coefficients | see `parameters.py` | Beard & McLain, appendix E (aerospace FRD convention) |
| Maximum thrust `thrust_max` | 40 N | **Guess** (the book uses a propeller model instead) |
| Actuator time constants | 0.05 s (surfaces), 0.2 s (throttle) | **Guess** |
| Surface limits | ±30° | Choice; same bounds as the lab NMPC (Bobbili et al. 2026, Table I) |
| Throttle range | [0, 1] | Convention, as in the lab NMPC |

## Simulation

| Parameter | Value | Source |
| --- | --- | --- |
| Control period `integration.dt` | 0.01 s (100 Hz) | Same as L2F; **to revisit**: external actuator commands to PX4 are limited to < ~50 Hz (Bobbili et al. 2026) |
| Substeps `integration.n_substeps` | 1 | Choice (L2F `N_SUBSTEPS`); use > 1 when the control period grows |
| Command latency `latency.action_steps` | 0 | **Unknown**; Orin NX → PX4 latency asked to the lab |
| Wind, gust | none | Set per experiment |

## Strix: what is known

| Quantity | Value | Source |
| --- | --- | --- |
| Mass | 4 kg | Lim & Loianno 2026, arXiv 2607.15863 |
| Wing span | 1.7 m (stock StratoSurfer: 1.5 m, so the lab airframe is modified) | Lim & Loianno 2026; Flite Test spec sheet |
| Best-glide airspeed, sink rate | ~12 m/s, 1.51 m/s (glide ratio ~7.9) | Morando et al. 2026, arXiv 2605.15619 (CFD and flight) |
| Level-flight throttle | 0.376 at ~12 m/s | Morando et al. 2026 |
| Trim | throttle 0.60, elevator 8° (airspeed not given) | Bobbili et al. 2026, arXiv 2608.20655 (AVL model) |
| Climb / descent rate at 15 m/s | 4.6 / 2.26 m/s in flight (6 / 3 m/s in their simulation) | Lim & Loianno 2026 |
| Aero coefficients, inertia, wing area, chord, T_max | **not published** | AVL model of the NMPC, requested |

## Placeholder vs Strix (`examples/validate.py`)

| Quantity | Strix | Aerosonde model |
| --- | --- | --- |
| Best-glide airspeed | 12.0 m/s | 16.6 m/s |
| Sink rate at best glide | 1.51 m/s | 1.06 m/s |
| Best glide ratio | 7.9 | 15.7 |
| Level-flight throttle at 12 m/s | 0.376 | 0.287 |
| Climb rate at 15 m/s, full throttle | 4.6 m/s | 3.25 m/s |
| Descent rate at 15 m/s, throttle off | 2.26 m/s | 0.97 m/s |

The Strix has about twice the drag of the placeholder. Its glide data imply a parabolic polar with an Oswald efficiency
of ~0.77 whatever the wing area, and a parasitic drag coefficient of 0.062–0.093 for a wing area of 0.45–0.30 m²
(Aerosonde: 0.044). `calibrate_drag_polar` computes them once the wing area is known.

## Simulation speed (`examples/benchmark.py`)

About 335 steps/s on one core (3.4 s of flight per second) for one airplane in pure Python. A first baseline of 1 M
steps takes ~50 min; a domain-randomization study needs much more, hence batching over airplanes (NumPy) or the C++
port (RLtools) later.
