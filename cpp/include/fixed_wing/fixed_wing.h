// Fixed-wing airplane and low-level RL task, C++ port of python/fixed_wing/airplane for RLtools.
//
// World frame: z up. Body frame: x forward, y left, z up (same as L2F). The aerodynamic coefficients keep the
// aerospace convention (x forward, y right, z down); the conversion happens in aerodynamics() only.
// The dynamics are computed from the air-relative velocity: the wind is a world-frame vector field (zero for now).
#pragma once

#include <rl_tools/rl/environments/environments.h>

namespace fixed_wing{
    // order of the actions and of Body::actuators
    constexpr int AILERON = 0, ELEVATOR = 1, RUDDER = 2, THROTTLE = 3;
    constexpr int N_ACTUATORS = 4;
    constexpr int N_COEFFICIENTS = 28;
    constexpr int MAX_LATENCY = 4;
    constexpr int ACTION_HISTORY = 10;

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
        // setpoint, constant over an episode
        T setpoint_airspeed;
        T setpoint_altitude;
        T setpoint_course;  // rad, direction of the ground velocity, counterclockwise from world x
        T trim_action[N_ACTUATORS];  // normalized commands of steady level flight at the setpoint airspeed
        T action_history[ACTION_HISTORY][N_ACTUATORS];  // normalized, oldest first; the last row is the previous action
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
        RewardConfig<T> reward;
    };

    // observation tags for RLtools: what the actor sees (PX4 estimates + last actions) and what the critic sees
    template <typename TI>
    struct ActorObservation{
        static constexpr TI DIM = 20 + N_ACTUATORS * ACTION_HISTORY;
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
