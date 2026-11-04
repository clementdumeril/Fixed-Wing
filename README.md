# Fixed-Wing: low-level SAC flight control of a fixed-wing UAV

[Python results](python/results/sac_gated_curriculum/README.md) · [C++ port and PX4 SITL](cpp/README.md) ·
[Reward and checks](docs/method.md) · [Components and figures](docs/reproduce.md) ·
Related: [WingJEPA](https://github.com/clementdumeril/WingJEPA), [GeneticWing](https://github.com/clementdumeril/GeneticWing)

![Overview: task, simulator and asymmetric SAC actor-critic, PID baseline, and results on 100 test flights](docs/figures/overview.png)

A NumPy and C++ (RLtools) fixed-wing simulator in the structure of L2F, in which SAC policies fly the four control
surfaces at 50 Hz. In wind and turbulence, SAC holds airspeed better than a cascaded PID on all 3 seeds (0.59 against
0.78 m/s RMS) but completes fewer of the 100 test flights on average (92 against 97). MEng capstone at UC Berkeley,
ARPL (Prof. Giuseppe Loianno), in simulation only.

## Method

The question behind the project: for sim-to-real transfer of a low-level fixed-wing policy, how much does a faithful
model help compared with wide domain randomization? This repository is the simulator and training side. The
structure (parameters, state, RK4, observe / reward / terminate) follows L2F from RLtools; the fixed-wing physics,
wind models, task, PID baseline, C++ port and PX4 SITL deployment are mine. SAC uses an asymmetric actor-critic: the
actor sees only what PX4 estimates, the critics also see angle of attack, sideslip and the true wind. Reward,
gated curriculum and the three reward fixes: [docs/method.md](docs/method.md).

## Results

From a perturbed start, hold an airspeed (20 to 30 m/s), an altitude (±20 m) and a course (±90°) for 30 s. Every
pilot flies the same 100 test scenarios. Windy: constant wind up to 5 m/s, light Dryden turbulence, sensor noise. SAC
rows: mean ± std over 3 seeds, 1 M steps each.

| Pilot | Air | Complete / 100 | Return (max 1500) | Airspeed RMS (m/s) | Altitude RMS (m) | Course RMS (deg) |
| --- | --- | --- | --- | --- | --- | --- |
| PID | calm | 100 | 1322 | 0.37 | 0.68 | 0.0 |
| SAC, Python | calm | 90 ± 3 | 1163 ± 21 | 0.55 ± 0.22 | 2.33 ± 0.15 | 4.4 ± 2.1 |
| SAC, C++ critic 256 | calm | 97 ± 1 | 1335 ± 19 | 0.23 ± 0.06 | 0.99 ± 0.89 | 1.7 ± 1.0 |
| SAC, C++ critic 64 | calm | 69 ± 49 | 642 ± 184 | 2.30 ± 1.02 | 15.54 ± 12.70 | 37.3 ± 19.8 |
| PID | windy | 97 | 1238 | 0.78 | 1.34 | 0.1 |
| SAC, Python | windy | 92 ± 7 | 1197 ± 70 | 0.59 ± 0.05 | 1.73 ± 0.45 | 3.6 ± 0.9 |

Printed by `python examples/results_table.py` from the committed `evaluation.json` files of
[`python/results/sac_gated_curriculum`](python/results/sac_gated_curriculum/README.md) and
[`cpp/results/sweep`](cpp/results/sweep/README.md). On average the PID is still the better pilot; most SAC failures
are stalls in the first 5 s, from banking 50 to 80° where the PID limits its bank to 30°.

![SAC (orange) and PID (blue) turning to a new course and altitude in the same wind and turbulence, seen from above](python/results/sac_gated_curriculum/rl_vs_pid_windy_scenario1.gif)

*Best windy SAC seed (seed 2) and the PID on test scenario 1, same wind and turbulence.*

## Install and quick start

Tested with Python 3.11 on Windows 10 and Linux. Dependencies are not pinned.

```bash
cd python
python -m venv ~/.venvs/fixed-wing && source ~/.venvs/fixed-wing/bin/activate
pip install -e ".[dev,rl]"
pytest                                   # 51 passed, about 1 min
python examples/pid_baseline.py          # PID on the 100 test scenarios, about 6 min
python examples/train_sac.py --steps 1000000 --n-envs 8 --out runs/sac   # about 4 h on 8 cores
```

The C++ build (CMake, fetches a fixed RLtools commit, Intel MKL or OpenBLAS), Slurm job files and the ROS 2 node
(Ubuntu 24.04, ROS 2 Jazzy, PX4 SITL) are in [`cpp/README.md`](cpp/README.md). More commands and figure scripts:
[docs/reproduce.md](docs/reproduce.md).

## Code structure

| Path | Content |
| --- | --- |
| `python/fixed_wing/airplane/` | simulator, wind, Gymnasium task and reward, PID baseline, asymmetric SAC |
| `python/examples/` | training, evaluation, diagnostics, plots, GIFs, 3D viewer |
| `python/tests/` | 51 tests (trim, control signs, wind statistics, reward recomputation) |
| `cpp/include/fixed_wing/`, `cpp/src/` | C++ port in the RLtools MDP form, SAC training, check against Python |
| `cpp/deploy/ros2/fw_policy/` | ROS 2 node that flies an exported actor on PX4 SITL over uXRCE-DDS |
| `python/slurm/`, `cpp/slurm/` | Slurm job files |

## Limitations

- Placeholder airframes (Aerosonde, PX4 `advanced_plane`), linear aerodynamics without stall: the angle-of-attack
  limit ends the episode instead. No parameter randomization yet, so the research question is not answered yet.
- Three seeds per condition; success varies by up to 14 flights between seeds, so small differences are not
  significant.
- The C++ port is trained in calm air only: wind, turbulence and sensor noise are not ported.
- The C++ tracking task and the PX4 SITL flights have no committed evaluation files or logs, so no number from them
  is reported here.
- No real flight.

## Credits

L2F and RLtools: J. Eschmann, D. Albani, G. Loianno, "Learning to Fly in Seconds", RA-L 2024
([arXiv:2311.13081](https://arxiv.org/abs/2311.13081)), [rl-tools/rl-tools](https://github.com/rl-tools/rl-tools).
Aerosonde and Dryden parameters: Beard and McLain, *Small Unmanned Aircraft*, 2012. Thermals: M. J. Allen,
AIAA 2006-1510. `advanced_plane`: PX4 Gazebo models. SAC: [stable-baselines3](https://github.com/DLR-RM/stable-baselines3).

Clément Dumeril, [clement.dumeril.net](https://clement.dumeril.net). MIT license ([LICENSE](LICENSE)); citation in
[CITATION.cff](CITATION.cff).
