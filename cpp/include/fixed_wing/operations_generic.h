// RLtools MDP interface of the fixed-wing task (same functions as the pendulum of rl-tools/example and as L2F).
#pragma once

#include <rl_tools/rl/environments/operations_generic.h>

#include "fixed_wing.h"
#include "dynamics.h"

namespace rl_tools{
    template <typename DEVICE, typename SPEC>
    void malloc(DEVICE&, fixed_wing::Environment<SPEC>&){}
    template <typename DEVICE, typename SPEC>
    void free(DEVICE&, fixed_wing::Environment<SPEC>&){}
    template <typename DEVICE, typename SPEC>
    void init(DEVICE&, fixed_wing::Environment<SPEC>& env){
        // RLtools allocates its environments without constructing them: set the default task here
        env.config = fixed_wing::TaskConfig<typename SPEC::T>{};
    }

    template <typename DEVICE, typename SPEC>
    void initial_parameters(DEVICE&, const fixed_wing::Environment<SPEC>& env, typename fixed_wing::Environment<SPEC>::Parameters& p){
        fixed_wing::task_parameters(env.config, p);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_parameters(DEVICE& device, const fixed_wing::Environment<SPEC>& env, typename fixed_wing::Environment<SPEC>::Parameters& p, RNG& rng){
        using T = typename SPEC::T;
        fixed_wing::task_parameters(env.config, p);
        T r = env.config.randomization;
        for(int i = 0; i < 4 + fixed_wing::N_COEFFICIENTS; i++){
            p.randomization[i] = r > 0 ? random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), 1 - r, 1 + r, rng) : T(1);
        }
        fixed_wing::apply_randomization(p);
        T direction = random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), T(-fixed_wing::PI), T(fixed_wing::PI), rng);
        T speed = env.config.wind_speed_max > 0 ? random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), T(0), env.config.wind_speed_max, rng) : T(0);
        p.wind[0] = speed * std::cos(direction);
        p.wind[1] = speed * std::sin(direction);
        p.wind[2] = 0;
    }

    template <typename DEVICE, typename SPEC>
    void initial_state(DEVICE&, const fixed_wing::Environment<SPEC>& env, const typename fixed_wing::Environment<SPEC>::Parameters& p, typename fixed_wing::Environment<SPEC>::State& s){
        // level flight along x at 25 m/s, 85 m, setpoint = current flight
        using T = typename SPEC::T;
        fixed_wing::Body<T> body = fixed_wing::trim(p, T(25));
        body.position[2] = T(85);
        fixed_wing::start_state(p, body, T(25), T(85), T(0), s);
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    void sample_initial_state(DEVICE& device, const fixed_wing::Environment<SPEC>& env, const typename fixed_wing::Environment<SPEC>::Parameters& p, typename fixed_wing::Environment<SPEC>::State& s, RNG& rng){
        // TRIMMED: trimmed flight at a random airspeed, perturbed in attitude and rates, and a setpoint (environment.py);
        // NEUTRAL: the same flight with neutral commands; RANDOM: wide flight conditions, never stalled (TaskConfig)
        using T = typename SPEC::T;
        const auto& c = env.config;
        auto uniform = [&](T lo, T hi){ return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), lo, hi, rng); };
        fixed_wing::Body<T> body{};
        T R[3][3], air_body[3], course, rate_max;
        if(c.start == fixed_wing::RANDOM){
            T airspeed = uniform(c.random_airspeed_range[0], c.random_airspeed_range[1]);
            T alpha = uniform(c.random_alpha_range[0], c.random_alpha_range[1]);
            T beta = uniform(-c.random_beta_max, c.random_beta_max);
            // air-relative velocity in the body frame (x forward, y left, z up), inverse of air_data()
            air_body[0] = airspeed * std::cos(alpha) * std::cos(beta);
            air_body[1] = -airspeed * std::sin(beta);
            air_body[2] = -airspeed * std::sin(alpha) * std::cos(beta);
            course = uniform(T(-fixed_wing::PI), T(fixed_wing::PI));
            fixed_wing::euler_quaternion(uniform(-c.random_roll_max, c.random_roll_max), uniform(-c.random_pitch_max, c.random_pitch_max), course, body.orientation);
            T command[fixed_wing::N_ACTUATORS];
            for(int i = 0; i < fixed_wing::N_ACTUATORS; i++) command[i] = uniform(T(-1), T(1));
            fixed_wing::scale_action(p, command, body.actuators);
            rate_max = c.random_rate_max;
        }
        else{
            fixed_wing::Body<T> trimmed = fixed_wing::trim(p, uniform(c.airspeed_range[0], c.airspeed_range[1]));
            T air[3];
            fixed_wing::rotation_matrix(trimmed.orientation, R);
            for(int i = 0; i < 3; i++) air[i] = trimmed.linear_velocity[i] - p.wind[i];
            fixed_wing::rotate_inverse(R, air, air_body);  // the perturbation keeps alpha and beta at their trim values
            T pitch = std::asin(fixed_wing::clip(R[2][0], T(-1), T(1)));
            course = uniform(T(-fixed_wing::PI), T(fixed_wing::PI));
            body = trimmed;
            fixed_wing::euler_quaternion(uniform(-c.roll_max, c.roll_max), pitch + uniform(-c.pitch_max, c.pitch_max), course, body.orientation);
            if(c.start == fixed_wing::NEUTRAL){
                T neutral[fixed_wing::N_ACTUATORS] = {};
                fixed_wing::scale_action(p, neutral, body.actuators);
            }
            rate_max = c.rate_max;
        }
        body.position[0] = 0;
        body.position[1] = 0;
        body.position[2] = uniform(c.altitude_range[0], c.altitude_range[1]);
        fixed_wing::rotation_matrix(body.orientation, R);
        fixed_wing::rotate(R, air_body, body.linear_velocity);
        for(int i = 0; i < 3; i++){
            body.linear_velocity[i] += p.wind[i];
            body.angular_velocity[i] = uniform(-rate_max, rate_max);
        }
        if constexpr(FW_TRACKING){
            // a reference point near the airplane, flying straight at constant velocity; the setpoints follow it
            T heading = fixed_wing::wrap(course + uniform(-c.course_change_max, c.course_change_max));
            T speed = uniform(c.reference_speed_range[0], c.reference_speed_range[1]);
            // one draw either way: with the default, the same starts as before the option existed
            T bearing = c.reference_bearing_max >= T(fixed_wing::PI) ? uniform(T(-fixed_wing::PI), T(fixed_wing::PI))
                      : fixed_wing::wrap(course + uniform(-c.reference_bearing_max, c.reference_bearing_max));
            T distance = uniform(T(0), c.reference_offset_max);
            T altitude = fixed_wing::clip(body.position[2] + uniform(-c.reference_altitude_offset_max, c.reference_altitude_offset_max),
                                          c.reference_altitude_range[0], c.reference_altitude_range[1]);
            T duration = fixed_wing::Environment<SPEC>::EPISODE_STEP_LIMIT * p.dt;
            T climb = fixed_wing::clip(uniform(-c.reference_climb_max, c.reference_climb_max),
                                       (c.reference_altitude_range[0] - altitude) / duration, (c.reference_altitude_range[1] - altitude) / duration);
            fixed_wing::start_state(p, body, std::sqrt(speed * speed + climb * climb), altitude, heading, s);
            s.reference_position[0] = body.position[0] + distance * std::cos(bearing);
            s.reference_position[1] = body.position[1] + distance * std::sin(bearing);
            s.reference_position[2] = altitude;
            s.reference_velocity[0] = speed * std::cos(heading);
            s.reference_velocity[1] = speed * std::sin(heading);
            s.reference_velocity[2] = climb;
        }
        else{
            T setpoint_airspeed = uniform(c.airspeed_range[0], c.airspeed_range[1]);
            T setpoint_altitude = fixed_wing::clip(body.position[2] + uniform(-c.altitude_change_max, c.altitude_change_max), c.altitude_range[0], c.altitude_range[1]);
            T setpoint_course = fixed_wing::wrap(course + uniform(-c.course_change_max, c.course_change_max));
            fixed_wing::start_state(p, body, setpoint_airspeed, setpoint_altitude, setpoint_course, s);
        }
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE&, const fixed_wing::Environment<SPEC>&, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State& s, const Matrix<ACTION_SPEC>& action, typename fixed_wing::Environment<SPEC>::State& next, RNG&){
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == fixed_wing::N_ACTUATORS);
        using T = typename SPEC::T;
        T a[fixed_wing::N_ACTUATORS];
        for(int i = 0; i < fixed_wing::N_ACTUATORS; i++) a[i] = get(action, 0, i);
        fixed_wing::simulate_task(p, s, a, next);
        return p.dt;
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T reward(DEVICE&, const fixed_wing::Environment<SPEC>& env, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State& s, const Matrix<ACTION_SPEC>& action, const typename fixed_wing::Environment<SPEC>::State& next, RNG&){
        using T = typename SPEC::T;
        T a[fixed_wing::N_ACTUATORS];
        for(int i = 0; i < fixed_wing::N_ACTUATORS; i++) a[i] = get(action, 0, i);
        return fixed_wing::reward(p, s, a, next, env.config);
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    void observe(DEVICE&, const fixed_wing::Environment<SPEC>&, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State& s, const fixed_wing::ActorObservation<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG&){
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == fixed_wing::ActorObservation<OBS_TYPE_SPEC>::DIM);
        fixed_wing::observe_actor(p, s, [&](int k, typename SPEC::T value){ set(observation, 0, k, value); });
    }

    template <typename DEVICE, typename SPEC, typename OBS_TYPE_SPEC, typename OBS_SPEC, typename RNG>
    void observe(DEVICE&, const fixed_wing::Environment<SPEC>&, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State& s, const fixed_wing::CriticObservation<OBS_TYPE_SPEC>&, Matrix<OBS_SPEC>& observation, RNG&){
        static_assert(OBS_SPEC::ROWS == 1 && OBS_SPEC::COLS == fixed_wing::CriticObservation<OBS_TYPE_SPEC>::DIM);
        fixed_wing::observe_critic(p, s, [&](int k, typename SPEC::T value){ set(observation, 0, k, value); });
    }

    template <typename DEVICE, typename SPEC, typename RNG>
    bool terminated(DEVICE&, const fixed_wing::Environment<SPEC>& env, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State s, RNG&){
        return fixed_wing::termination(p, s, env.config) != fixed_wing::FLYING;
    }
}
