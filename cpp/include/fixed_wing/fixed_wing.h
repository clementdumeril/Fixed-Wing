// Fixed-wing airplane and low-level RL task, C++ port of python/fixed_wing/airplane for RLtools.
//
// World frame: z up. Body frame: x forward, y left, z up (same as L2F). The aerodynamic coefficients keep the
// aerospace convention (x forward, y right, z down); the conversion happens in aerodynamics() only.
// The dynamics are computed from the air-relative velocity: the wind is a world-frame vector field (zero for now).
#pragma once

#include <rl_tools/rl/environments/environments.h>

// 1: the time integrals of the airspeed, altitude and course errors are part of the observation (train_sac_*_integral)
#ifndef FW_ERROR_INTEGRAL
#define FW_ERROR_INTEGRAL 0
#endif
// 1: position and velocity tracking (train_sac_*_tracking): a reference point moves in a straight line at constant
// velocity, and the airplane has to reach it and follow it. 0: airspeed, altitude and course setpoints (Python task).
#ifndef FW_TRACKING
#define FW_TRACKING 0
#endif

namespace fixed_wing{
    // order of the actions and of Body::actuators
    constexpr int AILERON = 0, ELEVATOR = 1, RUDDER = 2, THROTTLE = 3;
    constexpr int N_ACTUATORS = 4;
    constexpr int N_COEFFICIENTS = 28;
    constexpr int MAX_LATENCY = 4;
    constexpr int ACTION_HISTORY = 10;
    constexpr int N_INTEGRALS = FW_ERROR_INTEGRAL ? 3 : 0;
    // errors at the start of the observation: airspeed, altitude, sin and cos of course; or position and velocity
    constexpr int N_TASK_ERRORS = FW_TRACKING ? 6 : 4;

    // aerodynamic coefficients, in the order of airplane.Aerodynamics in Python (randomization factors follow it)
    enum Coefficient{
        C_L_0, C_L_alpha, C_L_q, C_L_delta_e,
        C_D_p, oswald_efficiency,
        C_Y_0, C_Y_beta, C_Y_p, C_Y_r, C_Y_delta_a, C_Y_delta_r,
        C_l_0, C_l_beta, C_l_p, C_l_r, C_l_delta_a, C_l_delta_r,
        C_m_0, C_m_alpha, C_m_q, C_m_delta_e,
        C_n_0, C_n_beta, C_n_p, C_n_r, C_n_delta_a, C_n_delta_r
    };

    template <typename T>
    struct Parameters{
        T mass;
        T gravity[3];  // world frame
        T J[3][3];  // body frame (x forward, y left, z up)
        T J_inv[3][3];
        T air_density;
        T wing_area;
        T wing_span;
        T mean_chord;
        T aerodynamics[N_COEFFICIENTS];
        T thrust_max;  // thrust = throttle * thrust_max, along body x
        T actuator_time_constants[N_ACTUATORS];
        T actuator_limits[N_ACTUATORS][2];  // rad for the surfaces, [0, 1] for the throttle
        T dt;  // control period
        int n_substeps;  // RK4 steps per control period, the command is held over all of them
        int latency_steps;  // control periods between a command and its application
        T wind[3];  // world frame, constant over an episode
        T error_integral_limits[3];  // m, m s, rad s (TaskConfig)
        // randomization factors, kept for the critic: mass, inertia, thrust, actuator time, then the 28 coefficients
        T randomization[4 + N_COEFFICIENTS];
    };

    // what is integrated
    template <typename T>
    struct Body{
        T position[3];  // world frame
        T orientation[4];  // quaternion [w, x, y, z], body to world
        T linear_velocity[3];  // world frame, ground-relative
        T angular_velocity[3];  // body frame
        T actuators[N_ACTUATORS];  // actual deflections [rad] and throttle [0, 1]
    };

    template <typename T>
    struct State{
        Body<T> body;
        T action_buffer[MAX_LATENCY][N_ACTUATORS];  // commands sent but not applied yet, oldest first, physical units
        // setpoint, constant over an episode (tracking: the speed, altitude and course of the reference, which move)
        T setpoint_airspeed;
        T setpoint_altitude;
        T setpoint_course;  // rad, direction of the ground velocity, counterclockwise from world x
        T reference_position[3];  // tracking only: the point to reach, world frame, moves at reference_velocity
        T reference_velocity[3];  // world frame, constant over an episode
        T trim_action[N_ACTUATORS];  // normalized commands of steady level flight at the setpoint airspeed
        T action_history[ACTION_HISTORY][N_ACTUATORS];  // normalized, oldest first; the last row is the previous action
        T error_integral[3];  // time integrals of the airspeed, altitude and course errors, clipped (simulate_task)
    };

    template <typename T>
    struct RewardConfig{
        // reward = survival - cap * tanh(cost / cap) - far, cap = cost_cap * survival (see environment.py)
        T survival = 1;
        T cost_cap = 0.5;
        T airspeed_scale = 2;  // m/s
        T altitude_scale = 10;  // m
        T course_scale = T(0.5235987755982988);  // 30 deg
        T airspeed_weight = 0.3;
        T altitude_weight = 0.3;
        T course_weight = 0.3;
        T action_change_weight = 0.1;
        T sideslip_scale = T(0.08726646259971647);  // 5 deg
        T sideslip_weight = 0.1;
        T trim_weight = 0.05;
        T far_weight = 0.1;
        T far_airspeed_scale = 10;  // m/s
        T far_altitude_scale = 50;  // m
        T far_course_scale = T(1.5707963267948966);  // 90 deg
        // margin to the stall: a cost once the angle of attack is within 4 deg of the termination limits (-10, 15 deg),
        // so that the policy sees the stall coming instead of only the end of the flight. Off by default (not in Python).
        T alpha_margin_weight = 0;
        T alpha_soft_min = T(-0.10471975511965977);  // -6 deg
        T alpha_soft_max = T(0.19198621771937624);  // 11 deg
        T alpha_margin_scale = T(0.06981317007977318);  // 4 deg: costs the weight at the limit
        // 0: squared airspeed, altitude and course errors. > 0: Huber (smooth L1) on the scaled errors, quadratic below
        // huber_delta and linear above, so that small errors still cost (the square is flat near 0)
        T huber_delta = 0;
        // tracking task: errors of position and ground velocity to the reference, replacing airspeed, altitude, course
        // 50 m: the cost keeps growing up to about 80 m. With 20 m it saturated (tanh) beyond about 26 m, and the
        // policy learned to fly at the reference velocity about 55 m away instead of reaching the point
        T position_scale = 50;  // m
        // a cost linear in the distance to the reference, outside the saturation: weight x min(distance / 100 m, 1).
        // At most 0.3: survival 1 - cap 0.5 - far 0.2 - 0.3 keeps the reward of a step >= 0, so ending a flight never pays
        T position_linear_weight = 0;
        // > 0: the velocity cost is on the error to the reference velocity plus a correction toward the reference
        // point, gain x position error, at most velocity_guidance_max. With the reference velocity alone, flying at
        // that velocity some 30 to 50 m behind the point was a stable optimum: closing the gap costs velocity error
        T velocity_guidance_gain = 0;  // 1/s
        T velocity_guidance_max = 5;  // m/s
        T velocity_scale = 4;  // m/s
        T position_weight = 0.6;
        T velocity_weight = 0.3;
        T far_position_scale = 100;  // m
        T far_velocity_scale = 15;  // m/s
    };

    // how an episode starts (sample_initial_state)
    enum StartMode{
        TRIMMED = 0,  // trimmed flight, perturbed in attitude and rates (environment.py)
        NEUTRAL,  // the same flight with neutral commands: surfaces at 0, half throttle (no trim prior)
        RANDOM  // wide attitude, airspeed, angle of attack, sideslip and commands, never stalled (no trim prior)
    };

    template <typename T>
    struct TaskConfig{
        T frequency = 50;  // Hz
        int n_substeps = 2;
        int latency_steps = 1;
        T altitude_range[2] = {50, 120};  // m
        T airspeed_range[2] = {20, 30};  // m/s
        T altitude_change_max = 20;  // m
        T course_change_max = T(1.5707963267948966);  // 90 deg
        T roll_max = T(0.3490658503988659);  // 20 deg
        T pitch_max = T(0.17453292519943295);  // 10 deg
        T rate_max = 0.3;  // rad/s
        T alpha_range[2] = {T(-0.17453292519943295), T(0.2617993877991494)};  // -10, 15 deg
        T airspeed_limits[2] = {15, 40};  // m/s
        T altitude_error_max = 50;  // m
        T altitude_floor = 20;  // m
        T randomization = 0;  // every factor uniform in [1 - r, 1 + r]
        T wind_speed_max = 0;  // m/s, constant horizontal wind, random direction
        StartMode start = TRIMMED;
        // RANDOM start: alpha and beta are sampled directly, so the airplane never starts stalled
        T random_roll_max = T(1.0471975511965976);  // 60 deg
        T random_pitch_max = T(0.3490658503988659);  // 20 deg
        T random_rate_max = 0.5;  // rad/s
        T random_airspeed_range[2] = {18, 32};  // m/s
        T random_alpha_range[2] = {T(-0.06981317007977318), T(0.17453292519943295)};  // -4, 10 deg
        T random_beta_max = T(0.13962634015954636);  // 8 deg
        // clip of the error integrals, also their observation scale: about 20 s of 1 m/s, 5 m and 14 deg
        T error_integral_limits[3] = {20, 100, 5};  // m, m s, rad s
        // tracking task: the reference starts up to 50 m away horizontally and 15 m vertically, flies at 20-28 m/s
        // within 90 deg of the airplane's heading, and climbs or descends at up to 1 m/s inside 40-140 m of altitude
        T reference_offset_max = 50;  // m
        T reference_altitude_offset_max = 15;  // m
        T reference_speed_range[2] = {20, 28};  // m/s, horizontal
        T reference_climb_max = 1;  // m/s
        T reference_altitude_range[2] = {40, 140};  // m, over the 30 s of an episode
        T position_error_max = 300;  // m: farther from the reference, the flight ends
        RewardConfig<T> reward;
    };

    // observation tags for RLtools: what the actor sees (PX4 estimates + last actions) and what the critic sees
    template <typename TI>
    struct ActorObservation{
        static constexpr TI DIM = N_TASK_ERRORS + 16 + N_INTEGRALS + N_ACTUATORS * ACTION_HISTORY;
    };
    template <typename TI>
    struct CriticObservation{
        static constexpr TI DIM = ActorObservation<TI>::DIM + 3 + 3 + N_ACTUATORS + 4 + N_COEFFICIENTS + N_ACTUATORS;
    };

    template <typename T_T, typename T_TI>
    struct Specification{
        using T = T_T;
        using TI = T_TI;
    };

    template <typename T_SPEC>
    struct Environment: rl_tools::rl::environments::Environment<typename T_SPEC::T, typename T_SPEC::TI>{
        using SPEC = T_SPEC;
        using T = typename SPEC::T;
        using TI = typename SPEC::TI;
        using Parameters = fixed_wing::Parameters<T>;
        using State = fixed_wing::State<T>;
        using Observation = ActorObservation<TI>;
        using ObservationPrivileged = CriticObservation<TI>;
        static constexpr TI OBSERVATION_DIM = Observation::DIM;
        static constexpr TI ACTION_DIM = N_ACTUATORS;
        static constexpr TI EPISODE_STEP_LIMIT = 1500;  // 30 s at 50 Hz
        TaskConfig<T> config;  // shared by all episodes; the curriculum changes config.reward
    };
}
