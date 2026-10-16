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
| Disturbances | random force and torque | wind, entering only through the airspeed |

Wind is subtracted from the ground velocity (v_air = v − wind), so it acts through the airspeed, α and β.

## 3. What is new

- Aerodynamics: quasi-steady linear model (lift, parabolic drag polar, side force, 3 moments) with the Aerosonde coefficients (Beard & McLain, *Small Unmanned Aircraft*, appendix E) as a placeholder.
- Frames: L2F conventions in the state; the aerodynamics are computed in the aerospace frame and converted with diag(1, −1, −1).
- Gust: optional "1 − cos" wind burst at a fixed location in space, since the dynamics have no time argument.
- Trim: Newton's method with a finite-difference Jacobian for pitch, elevator and throttle. Aerosonde at 25 m/s: pitch 4.72°, elevator −6.26°, throttle 28.5 %.
- Command latency: `latency.action_steps` delays each command by whole control periods.
- Validation: `validation.py` computes the figures the lab measured on the Strix and calibrates a drag polar from a measured best glide. See [PARAMETERS.md](PARAMETERS.md).
- Integration substeps as in L2F (`integration.n_substeps`).
- 20 airplane tests.

## 4. Not ported yet (needed for closed-loop RL)

Observation, reward, termination, parameter and initial-state randomization, action noise, state clamps
(`STATE_LIMIT_*`) and the closed-form motor model. So there is no closed-loop controller yet.

## 5. Simplifications to revisit

- No stall: lift keeps growing linearly with α, which a policy could exploit.
- Thrust = throttle × 40 N, independent of airspeed; 40 N is a guess.
- Actuator time constants are guesses, to identify on the real servos and motor.
- No ground: episodes must end on altitude.
- Constant air density (1.2682 kg/m³).
