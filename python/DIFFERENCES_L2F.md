# What comes from L2F, what was adapted, what is new

This file compares this simulator with Jonas Eschmann's **L2F** quadrotor simulator
(*Learning to Fly in Seconds*, RLtools C++ library).

**Reference used for the comparison:** the copy of rl-tools bundled with the `ui-server` 0.0.15 package,
folder `rl-tools/include/rl_tools/rl/environments/l2f/`. The rl-tools main branch may have changed since.

**Legend**

| Status | Meaning |
| --- | --- |
| **Translated** | Same logic as L2F, rewritten from C++ to Python |
| **Simplified** | Exists in L2F, deliberately left out or reduced here |
| **Adapted** | Same role as in L2F, changed for a fixed-wing aircraft |
| **New** | Does not exist in L2F |

---

## 1. File-by-file mapping

| Python file | Original L2F file | Status |
| --- | --- | --- |
| `fixed_wing/quaternion_helper.py` | `l2f/quaternion_helper.h` | Translated (+ 2 functions, see 2.1) |
| `fixed_wing/integrators.py` | `utils/generic/integrators.h` + `l2f/operations_generic/50_state_algebra.h` | Translated |
| `fixed_wing/multirotor/multirotor.py` | `l2f/multirotor.h` (`StateBase` + `StateRotors`, base parameters) | Translated, simplified |
| `fixed_wing/multirotor/parameters.py` | `l2f/parameters/dynamics/arpl.h` | Translated |
| `fixed_wing/multirotor/operations.py` | `l2f/operations_generic.h` + `20_initial_state.h`, `60_dynamics.h`, `70_post_integration.h` | Translated, simplified |
| `fixed_wing/airplane/airplane.py` | counterpart of `l2f/multirotor.h` | Adapted |
| `fixed_wing/airplane/parameters.py` | counterpart of `l2f/parameters/dynamics/*.h` | Adapted (Aerosonde instead of a quadrotor) |
| `fixed_wing/airplane/operations.py` | counterpart of `operations_generic.h` + `60_dynamics.h` + `70_post_integration.h` | Adapted + New |
| `fixed_wing/airplane/trim.py` | none | New |
| `fixed_wing/airplane/ui.py`, `ui.js` | `get_ui` in `l2f/operations_cpu.h` + `ui.h` | Adapted (same protocol, airplane rendering) |
| `examples/ui.py` | `l2f/examples/ui_min.py` | Adapted |
| `tests/` | none | New |

---

## 2. Translated from L2F

### 2.1 Quaternions (`quaternion_helper.py`)

- **Kept:** `quaternion_derivative`, `rotate_vector_by_quaternion`, `quaternion_to_rotation_matrix`.
  Same convention: `[w, x, y, z]`, body to world rotation.
- **Added:**
  - `quaternion_multiply`: in L2F the product is written inline in `quaternion_derivative`; here it is its own function.
  - `rotate_vector_by_quaternion_inverse`: needed to bring the wind from the world frame to the body frame in `air_data`.

### 2.2 Integrators (`integrators.py`)

- **Kept:** `euler` and `rk4`. As in L2F, RK4 holds the action **constant** over the step.
- **Implementation difference:**
  - In C++, L2F defines one addition function per state type (`50_state_algebra.h`).
  - In Python, a single `add_scaled` walks over the fields of any dataclass (`dataclasses.fields`).
  - The result is the same: the state derivative has the type of the state, and the integrator does not depend on the vehicle.

### 2.3 Quadrotor dynamics (`multirotor/operations.py`)

- **Kept as is:**
  - per-rotor thrust `c0 + c1·rpm + c2·rpm²`;
  - torque = rotor drag torque + lever arm;
  - Newton-Euler `R·F/m + g` and `J⁻¹(τ − ω × Jω)`;
  - first-order motor lag with rising / falling time constants.
- **Implementation difference:** the C++ `for i_rotor < 4` loops are replaced by vectorized NumPy operations, which work for N rotors.
- `post_integration`: quaternion renormalization and `rpm` clamping, as in L2F (non `CLOSED_FORM` mode).
- `step`: scale the action from [−1, 1] to physical limits, then RK4, then `post_integration`, as in L2F.
- `parameters.py`: values of the ARPL quadrotor copied from `arpl.h`.

The quadrotor is only used as a **reference system**: its tests check that the Python skeleton (quaternions,
RK4, actuator lag) behaves like L2F before trusting the airplane built on top of it.

---

## 3. In L2F but deliberately left out of this port

These are the "RL environment" parts of L2F. They have not been ported yet; this is the job of the
upcoming RL wrapper.

| L2F element | Where in L2F | Why it is left out for now |
| --- | --- | --- |
| Action noise (`action_noise`) | `operations_generic.h`, `step` | Useful for RL robustness, not needed to validate the physics |
| Integration substeps (`N_SUBSTEPS`) | `operations_generic.h`, `step` | 100 Hz is enough for now; to add if the physics must run faster than the controller |
| Safety bounds on position, velocity, angular velocity (`STATE_LIMIT_*`) | `70_post_integration.h` | To add together with the termination conditions |
| Exact motor model (`CLOSED_FORM`, exponential) | `70_post_integration.h` | The RK4-integrated version is accurate enough at 100 Hz |
| Stackable state components (`StateLastAction`, `StateRandomForce`, histories, delays…) | `multirotor.h` | The Python port has a single flat `State`; to extend when the observation needs it |
| Random force / torque (`Disturbances::random_force`) | `multirotor.h` | For the airplane, the physical disturbance is the wind (section 5.4) |
| Parameter randomization | `10_sample_initial_parameters.h` | Planned after the nominal baseline |
| Initial state sampling | `30_sample_initial_state.h` | To be done in the RL wrapper (around the trim) |
| Observation, reward, termination, NaN detection | `40_observe.h`, `parameters/reward_functions`, `parameters/termination`, `05_state_is_nan.h` | RL wrapper |
| Random number generator (`rng`) passed to `step` | everywhere | No randomness in the simulator yet |

---

## 4. Adapted for the airplane

| Element | In L2F | Here | Why |
| --- | --- | --- | --- |
| Actuators in the state | `rpm`: 4 identical motors | `actuators`: aileron, elevator, rudder (rad) + throttle [0, 1] | An airplane has control surfaces, not 4 motors |
| Action limits | a single `action_limit = (min, max)` | one pair per actuator, `actuator_limits` (4 × 2) | Surfaces go from −30° to +30°, throttle from 0 to 1 |
| Actuator lag | rising and falling τ per rotor | one τ per actuator: 0.05 s (surfaces), 0.2 s (throttle) | A servo is symmetric; values are **guesses** |
| `post_integration` | clamps `rpm` | clamps each actuator to its own range | Same idea, different limits |
| Action scaling | `scale_action` | per-actuator `scale_action` + `normalize_action` (the inverse) | The inverse converts the trim into a normalized action |
| `Disturbances` | random force and torque | constant wind + gust (section 5.4) | An airplane is disturbed through the air, not by an external force |
| Vehicle parameters | rotors, mass, inertia | mass, inertia, wing geometry, 28 aerodynamic coefficients, `thrust_max` | Different physics |
| Visualizer | quadrotor | box airplane, animated surfaces, trail, wind arrow, HUD | Same ui-server protocol |

---

## 5. New (not in L2F)

### 5.1 Two frame conventions and their conversion

- The state, the dynamics and the quaternion stay in **FLU** (x forward, y left, z up), as in L2F. World frame: z up.
- The aerodynamic coefficients are in **FRD** (x forward, y right, z down), the aerospace convention.
- `FRD_FLU = diag(1, −1, −1)` (in `airplane/parameters.py`) does the conversion, in three places only:
  - `air_data`;
  - `airplane_aerodynamics`;
  - the inertia in `aerosonde()`.

### 5.2 Aerodynamics (`airplane/operations.py`)

- `air_data`: air-relative velocity = ground velocity − wind, rotated to body FRD, then:
  - `V_a = ‖v‖`;
  - `α = atan2(w, u)`;
  - `β = arcsin(v / V_a)`.
- `airplane_aerodynamics`:
  - quasi-steady **linear** model with 6 coefficients (C_L, C_D, C_Y, C_l, C_m, C_n), coefficients from Beard & McLain;
  - parabolic drag polar `C_D = C_D_p + C_L² / (π e AR)`;
  - non-dimensional angular rates;
  - wind-to-body rotation with cos α / sin α.
  - Guard: if V_a < 1 mm/s, no aerodynamic force (avoids 0/0).
- **Propulsion**: `thrust = throttle × thrust_max` along the body x axis. No propeller model, no motor torque. `thrust_max = 40 N` is a **guess**.
- **Not modeled**: stall, ground effect, the ground itself, varying air density.

### 5.3 Airplane parameters (`airplane/parameters.py`)

- `aerosonde()`: placeholder airframe, values from Beard & McLain, *Small Unmanned Aircraft*, appendix E.
- To be replaced by an `arpl()` function once the identified ARPL parameters are available.

### 5.4 Wind and gust (`wind()` in `airplane/operations.py`)

- **Constant wind**: `Disturbances.wind`, world-frame vector.
- **"1 − cos" gust**, added on October 7, 2026:
  - **What it does:** a slab of space, normal to `gust_direction`, starts at `gust_start` meters and is `gust_length` meters thick. Inside the slab the wind is:

    ```text
    wind = wind + gust × ½ (1 − cos(2π s)),    s = (gust_direction · position − gust_start) / gust_length
    ```

    Outside the slab, `wind = wind`. The gust rises smoothly from 0 to `gust` in the middle of the slab, then goes back to 0.
  - **Why in space and not in time:**
    - `airplane_dynamics(parameters, state, action)` does not know the time, and RK4 evaluates the wind at intermediate positions within the step.
    - A gust fixed in space keeps the dynamics and the API unchanged.
    - It is also the classic shape of the discrete gust used in certification, defined by a distance.
  - **Assumption:** the slab is fixed relative to the ground; it is not convected by the mean wind.
  - **Default:** `gust = 0`, no gust. All previous behavior is unchanged.
  - **Files changed:**
    - `airplane.py`: 4 fields added to `Disturbances`;
    - `operations.py`: `wind()`;
    - `ui.py`: the local wind is sent to the visualizer;
    - `ui.js`: the HUD shows the local wind, gust included;
    - `examples/ui.py`: 6 m/s updraft between x = 375 and 425 m (around t = 15-17 s);
    - `tests/test_airplane.py`: 2 tests.

### 5.5 Trim (`airplane/trim.py`)

- **Goal:** find the settings that make the airplane fly straight without accelerating.
- **Powered flight:** 3 unknowns (pitch, elevator, throttle) and 3 equations (horizontal acceleration, vertical acceleration and pitch angular acceleration, all zero).
- **Glide:** throttle is 0, and the flight path angle becomes the 3rd unknown.
- **Solver:** Newton's method, with a finite-difference Jacobian.
- **Result for the Aerosonde at 25 m/s:** pitch 4.72°, elevator −6.26°, throttle 28.5 %.
- **Use:** initial state only. It gives no answer to the RL policy.

### 5.6 Other additions

- `specific_energy`: `g·h + ½‖v‖²`, for the tests and the HUD.
- `initial_state`: kept for symmetry with L2F, but useless for an airplane (zero velocity).

### 5.7 Tests (`tests/`, 19 tests, all passing)

| Group | What it checks |
| --- | --- |
| Multirotor (3) | The skeleton translated from L2F: hover stays still, exact free fall, roll sign |
| Trim (4) | Exact equilibrium, holds for 30 s, stability after a perturbation, consistent glide |
| Controls (4) | Signs of elevator, aileron, rudder, throttle |
| Actuators (1) | Exact first-order lag and saturation |
| Wind (5) | Exact V_a, α, β for 4 winds, and invariance under a constant wind |
| Gust (2) | Shape of the 1 − cos profile, and physical response: α rises, the nose pitches down, energy increases |

---

## 6. One-sentence summary

We kept the L2F **skeleton** (parameters / state / dynamics → RK4 → post-integration, normalized action),
put **fixed-wing physics** in it (aerodynamics, wind, gust, trim), and have **not yet ported** the RL
environment side of L2F (noise, randomization, observation, reward, termination).
