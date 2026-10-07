# Differences with L2F

This simulator follows the structure of the L2F environment in RLtools: Parameters / State / dynamics → RK4 →
post_integration, with actions normalized to [-1, 1]. The comparison below is against the `l2f` sources
bundled with ui-server 0.0.15.

## Ported as is

`quaternion_helper`, `rk4` / `euler`, and the ARPL quadrotor (`multirotor/`). The quadrotor is only there as a
reference to check the Python port (hover, free fall, roll sign) before building the airplane on top of it.

## Adapted for the airplane

- `rpm` becomes `actuators`: aileron, elevator, rudder (rad) and throttle [0, 1], each with its own limits, and
  one first-order time constant per actuator instead of rising / falling constants.
- `Disturbances` holds the wind instead of a random force / torque. The wind only enters through the
  air-relative velocity (v_air = v − wind → V_a, α, β), so a constant wind just shifts the trajectory.
- `normalize_action` was added (the inverse of the action scaling) to turn the trim into an action.

## New

- **Aerodynamics:** quasi-steady linear model (C_L, C_D with a parabolic polar, C_Y, C_l, C_m, C_n). The
  coefficients are the Aerosonde's (Beard & McLain, appendix E), as a placeholder until the identified ARPL model.
- **Frames:** the state keeps the L2F conventions (world z up, body FLU). The aero is computed in FRD, because
  the published coefficients are in FRD, and converted with diag(1, −1, −1) inside the aero function only.
- **Gust:** optional "1 − cos" discrete gust, defined in space (a slab along a direction) rather than in time,
  because the dynamics have no time argument and RK4 evaluates the wind at intermediate positions.
- **Trim:** Newton solver with a finite-difference Jacobian, for level flight or glide. Used for initial states.
- **Tests:** 16 airplane tests (trim, control signs, actuator lag and saturation, wind, gust).

## Not ported yet

Action noise, `N_SUBSTEPS`, the `STATE_LIMIT_*` clamps, the closed-form motor model, parameter and initial-state
sampling, observation, reward and termination. These will come with the RL wrapper.

## Assumptions to revisit

- Thrust = throttle × 40 N along body x, independent of airspeed. The actuator time constants (0.05 s for the
  surfaces, 0.2 s for the throttle) are guesses.
- No stall: lift grows linearly with α.
- No ground, constant air density.
