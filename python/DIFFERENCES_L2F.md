# Differences with L2F

**In short:** the simulator reuses the L2F structure unchanged and replaces the quadrotor physics with fixed-wing
physics. The RL environment side of L2F (observation, reward, termination, randomization) is not ported yet, so
there is no closed-loop controller yet: the airplane either flies at trim or follows scripted commands.

The comparison is against the `l2f` sources bundled with ui-server 0.0.15.

## 1. What is reused from L2F

- **The structure:** Parameters / State / dynamics → RK4 → post_integration, with actions normalized to [-1, 1]
  and scaled to physical units inside `step`.
- **The math helpers:** `quaternion_helper` and the `rk4` / `euler` integrators, translated to Python.
- **The ARPL quadrotor** (`multirotor/`), translated as a reference: its tests (hover, free fall, roll sign) check
  that the Python port behaves like L2F before the airplane is built on top of it.

## 2. What changes for the airplane

| | L2F (quadrotor) | Here (airplane) |
| --- | --- | --- |
| Actuators in the state | 4 rotor speeds (`rpm`) | aileron, elevator, rudder (rad) and throttle [0, 1] (`actuators`) |
| Actuator limits | one range for all rotors | one range per actuator (±30° for the surfaces, [0, 1] for the throttle) |
| Actuator lag | first order, rising and falling time constants | first order, one time constant per actuator |
| Forces | sum of the rotor thrusts | aerodynamics + one thrust along the body x axis |
| Disturbances | random force and torque | wind, entering only through the airspeed |

**Why the wind is not a force:** an airplane only feels the air moving over it. The wind is subtracted from the
ground velocity (v_air = v − wind), which changes the airspeed, α and β, and therefore the aerodynamic forces. A
constant wind then only shifts the trajectory, as it should.

## 3. What is new

- **Aerodynamics.** Quasi-steady linear model: lift, drag (parabolic polar), side force and the 3 moments, each a
  linear function of α, β, the angular rates and the surface deflections. The coefficients are the Aerosonde's
  (Beard & McLain, *Small Unmanned Aircraft*, appendix E), a placeholder until the identified ARPL model.
- **Frames.** The state keeps the L2F conventions (world z up, body x forward / y left / z up). The aerodynamics are
  computed in the aerospace frame (x forward / y right / z down), because the published coefficients use it, and
  converted back with diag(1, −1, −1) inside the aerodynamics function only.
- **Gust.** An optional short wind burst with a smooth "1 − cos" profile, placed at a fixed location in space: the
  airplane feels it while flying through it. It depends on position rather than time because the dynamics have no
  time argument and RK4 evaluates the wind at intermediate positions within a step.
- **Trim.** Finds the pitch, elevator and throttle for steady flight (all accelerations equal to zero). There is no
  closed-form solution, so it uses Newton's method with a finite-difference Jacobian; it converges in a few
  iterations. Used to start simulations from a realistic state. Example, Aerosonde at 25 m/s: pitch 4.72°,
  elevator −6.26°, throttle 28.5 %.
- **Tests.** 16 airplane tests: trim, control signs, actuator lag and saturation, wind, gust.

## 4. Not ported yet (needed for closed-loop RL)

Observation, reward, termination, parameter and initial-state randomization, action noise, integration substeps
(`N_SUBSTEPS`), state clamps (`STATE_LIMIT_*`) and the closed-form motor model.

## 5. Simplifications to revisit

| Simplification | Consequence |
| --- | --- |
| No stall: lift keeps growing linearly with α | A policy could exploit unrealistically high lift at large α |
| Thrust = throttle × 40 N, independent of airspeed | Too much thrust at high speed; 40 N is a guess |
| Actuator time constants 0.05 s (surfaces), 0.2 s (throttle) | Guesses, to identify on the real servos and motor |
| No ground | The airplane can fly below z = 0; episodes must end on altitude |
| Constant air density (1.2682 kg/m³) | Negligible at a few hundred meters of altitude |
