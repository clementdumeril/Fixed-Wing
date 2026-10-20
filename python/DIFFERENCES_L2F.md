# Differences with L2F

Compared against the `l2f` sources bundled with ui-server 0.0.15.

## 1. What is reused from L2F

- The structure: Parameters / State / dynamics → RK4 → post_integration, with actions in [-1, 1] scaled inside `step`.
- `quaternion_helper` and the `rk4` / `euler` integrators, translated to Python.
- The ARPL quadrotor (`multirotor/`), with tests that check the port behaves like L2F.

## 2. What changes for the airplane

| | L2F (quadrotor) | Here (airplane) |
| --- | --- | --- |
| Actuators in the state | 4 rotor speeds (`rpm`) | aileron, elevator, rudder (rad) and throttle [0, 1] (`actuators`) |
| Actuator limits | one range for all rotors | one range per actuator (±30° for the surfaces, [0, 1] for the throttle) |
| Actuator lag | first order, rising and falling time constants | first order, one time constant per actuator |
| Forces | sum of the rotor thrusts | aerodynamics + one thrust along the body x axis |
| Disturbances | random force and torque | wind (constant, gust, thermals, turbulence), entering only through the airspeed |

Wind is subtracted from the ground velocity (v_air = v − wind), so it acts through the airspeed, α and β.

## 3. What is new

- Aerodynamics: quasi-steady linear model (lift, parabolic drag polar, side force, 3 moments) with the Aerosonde coefficients (Beard & McLain, *Small Unmanned Aircraft*, appendix E) as a placeholder.
- Frames: L2F conventions in the state; the aerodynamics are computed in the aerospace frame and converted with diag(1, −1, −1).
- Gust: optional "1 − cos" wind burst at a fixed location in space, since the dynamics have no time argument.
- Thermals: Allen's updraft model (NASA, AIAA 2006-1510), following its reference code: updrafts sized by height, a downdraft ring high up, sinking air in between. Fixed in space; `sample_thermals` places them at random.
- Turbulence: Dryden model (MIL-F-8785C, Beard & McLain chapter 4) in the body frame, driven by the distance flown through the air and discretized exactly. It is random, so `step` takes an optional `rng`, as in L2F; the filter states are in `State.turbulence`.
- Trim: Newton's method with a finite-difference Jacobian for pitch, elevator and throttle. Aerosonde at 25 m/s: pitch 4.72°, elevator −6.26°, throttle 28.5 %.
- Command latency: `latency.action_steps` delays each command by whole control periods.
- Validation: `validation.py` computes the figures the lab measured on the Strix and calibrates a drag polar from a measured best glide. See [PARAMETERS.md](PARAMETERS.md).
- Integration substeps as in L2F (`integration.n_substeps`).
- 25 airplane tests, including the Allen check case and the Dryden statistics.

## 4. RL environment (`environment.py`)

Same pieces as the L2F environment, rewritten for the airplane task: hold an airspeed, an altitude and a course.

- `sample_initial_parameters`: domain randomization, every factor uniform in [1 − r, 1 + r] (mass, inertia, thrust,
  actuator time constants, the 28 coefficients), plus optional wind, turbulence and thermals.
- `sample_initial_state`: trim at a random airspeed, then attitude and rate perturbations; the setpoint.
- `observe_actor`: only what PX4 estimates (errors to the setpoint, rotation matrix, rates, body velocity, airspeed if
  there is a Pitot tube, past actions). `observe_critic` adds α, β, the true wind, the actuators and the randomization
  factors, for an asymmetric actor-critic.
- `reward`: survival bonus minus a quadratic cost, capped at half the bonus: every step in flight earns something, so ending the episode never
  pays (L2F's survival bonus against "learning to terminate").
- `terminated`: below 20 m (the safety pilot takes over), airspeed limits, altitude error, and α outside [−10°, 15°] since the model has no stall. Start and setpoint altitudes stay within 50–120 m, where the lab flies the Strix.
  The time limit is a truncation.
- `FixedWingEnv`: Gymnasium wrapper (`pip install -e ".[rl]"`). `evaluate` runs any controller on the same scenarios.
- `pid.py`: PID baseline (successive loop closure, Beard & McLain chapter 6), using the same measurements as the actor.

Not ported: the closed-form motor model and the state clamps (`STATE_LIMIT_*`).

## 5. Simplifications to revisit

- No stall: lift keeps growing linearly with α, which a policy could exploit.
- Thrust = throttle × 40 N, independent of airspeed; 40 N is a guess.
- Actuator time constants are guesses, to identify on the real servos and motor.
- No ground: episodes must end on altitude.
- Constant air density (1.2682 kg/m³).
