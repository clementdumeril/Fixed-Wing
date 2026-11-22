// RLtools MDP interface of the fixed-wing task (same functions as the pendulum of rl-tools/example and as L2F).
#pragma once

#include <algorithm>
#include <cmath>

#include <rl_tools/rl/environments/operations_generic.h>

#include "fixed_wing.h"
#include "dynamics.h"
#include "reference_motion.h"

namespace fixed_wing{
    template <typename T>
    T episode_duration(const TaskConfig<T>& c, T dt, T step_limit){
        return c.episode_duration > 0 ? c.episode_duration : step_limit * dt;
    }

    template <typename T, typename UNIFORM>
    void sample_disturbances(const TaskConfig<T>& c, T duration, Parameters<T>& p, UNIFORM uniform){
        const T moderate_sigma[3] = {T(2.12), T(2.12), T(1.4)};
        const T moderate_length_scale[3] = {200, 200, 50};
        if(c.turbulence_intensity_max > 0){
            T intensity = uniform(T(0), c.turbulence_intensity_max);
            for(int i = 0; i < 3; i++){
                p.turbulence_sigma[i] = intensity * moderate_sigma[i];
                p.turbulence_length_scale[i] = moderate_length_scale[i];
            }
        }
        if(c.gust_rate > 0){
            T probability = std::fmin(c.gust_rate * duration / 60 / MAX_GUSTS, T(1));
            p.gust_count = 0;
            for(int g = 0; g < MAX_GUSTS; g++){
                bool active = uniform(T(0), T(1)) < probability;
                T gust_duration = uniform(c.gust_duration_range[0], c.gust_duration_range[1]);
                T latest_start = std::fmax(duration - gust_duration, T(1));
                T start = uniform(T(1), latest_start);
                T vertical = uniform(T(-1), T(1));
                T azimuth = uniform(T(-PI), T(PI));
                T speed = uniform(T(0), c.gust_speed_max);
                if(!active) continue;
                T horizontal = std::sqrt(1 - vertical * vertical);
                Gust<T>& gust = p.gusts[p.gust_count++];
                gust.start = start;
                gust.duration = gust_duration;
                gust.peak[0] = speed * horizontal * std::cos(azimuth);
                gust.peak[1] = speed * horizontal * std::sin(azimuth);
                gust.peak[2] = speed * vertical;
            }
        }
        if(c.thermal_fraction > 0 && uniform(T(0), T(1)) < c.thermal_fraction){
            Thermals<T>& t = p.thermals;
            t.downdraft_ring = c.thermal_downdraft_ring;
            T wind_speed = std::sqrt(p.wind[0] * p.wind[0] + p.wind[1] * p.wind[1]);
            T half_size = (c.reference_speed_range[1] + wind_speed) * duration / 2 + c.thermal_field_margin;
            T size = 2 * half_size;
            t.area = size * size;
            T height = (c.reference_altitude_range[0] + c.reference_altitude_range[1]) / 2;
            T mean_radius = std::fmax(allen_mean_radius(height, t.z_i), T(10));
            T mean_updraft = allen_mean_updraft(height, t.z_i, t.w_star);
            int count = int(std::lround(T(0.6) * size * size / (t.z_i * mean_radius)));
            t.count = std::min(std::max(count, 1), MAX_THERMALS);
            for(int k = 0; k < t.count; k++){
                T spread = k == 0 ? T(0) : half_size;
                T core = uniform(c.thermal_core_range[0], c.thermal_core_range[1]);
                T radius = uniform(c.thermal_radius_range[0], c.thermal_radius_range[1]);
                T offset = k == 0 ? radius : T(0);
                t.centers[k][0] = uniform(-spread - offset, spread + offset);
                t.centers[k][1] = uniform(-spread - offset, spread + offset);
                t.r_gain[k] = radius / allen_mean_radius(height, t.z_i);
                t.w_gain[k] = core / (mean_updraft * allen_core_factor(radius));
            }
        }
    }
}

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
        const auto& c = env.config;
        auto uniform = [&](T lo, T hi){ return random::uniform_real_distribution(typename DEVICE::SPEC::RANDOM(), lo, hi, rng); };
        fixed_wing::task_parameters(c, p);
        for(int i = 0; i < 4 + fixed_wing::N_COEFFICIENTS; i++) p.randomization[i] = T(1);
        auto factor = [&](T range){ return range > 0 ? uniform(1 - range, 1 + range) : T(1); };
        if(c.randomization_mass > 0) p.randomization[0] = factor(c.randomization_mass);
        if(c.randomization_thrust > 0) p.randomization[2] = factor(c.randomization_thrust);
        if(c.randomization_aerodynamics > 0){
            for(int i = 0; i < fixed_wing::N_COEFFICIENTS; i++) p.randomization[4 + i] = factor(c.randomization_aerodynamics);
        }
        fixed_wing::apply_randomization(p);
        T direction = uniform(T(-fixed_wing::PI), T(fixed_wing::PI));
        T speed = c.wind_speed_max > 0 ? uniform(T(0), c.wind_speed_max) : T(0);
        p.wind[0] = speed * std::cos(direction);
        p.wind[1] = speed * std::sin(direction);
        p.wind[2] = 0;
        if(c.latency_steps_random_max >= 0){
            int steps = int(uniform(T(0), T(c.latency_steps_random_max + 1)));
            p.latency_steps = std::min(steps, std::min(c.latency_steps_random_max, fixed_wing::MAX_LATENCY));
        }
        T duration = fixed_wing::episode_duration(c, p.dt, T(fixed_wing::Environment<SPEC>::EPISODE_STEP_LIMIT));
        fixed_wing::sample_disturbances(c, duration, p, uniform);
    }

    template <typename DEVICE, typename SPEC>
    void initial_state(DEVICE&, const fixed_wing::Environment<SPEC>& env, const typename fixed_wing::Environment<SPEC>::Parameters& p, typename fixed_wing::Environment<SPEC>::State& s){
        // level flight along x at the middle of the airspeed range (25 m/s for the Aerosonde), 85 m, setpoint = current flight
        using T = typename SPEC::T;
        T airspeed = (env.config.airspeed_range[0] + env.config.airspeed_range[1]) / 2;
        fixed_wing::Body<T> body = fixed_wing::trim(p, airspeed);
        body.position[2] = T(85);
        fixed_wing::start_state(p, body, airspeed, T(85), T(0), s);
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
            // one draw either way, so that the other draws do not depend on the bearing range
            T bearing = c.reference_bearing_max >= T(fixed_wing::PI) ? uniform(T(-fixed_wing::PI), T(fixed_wing::PI))
                      : fixed_wing::wrap(course + uniform(-c.reference_bearing_max, c.reference_bearing_max));
            // the draw for a near start only when the option is on: without it, the same starts as before
            bool near_start = c.reference_near_fraction > 0 && uniform(T(0), T(1)) < c.reference_near_fraction;
            T distance = uniform(T(0), near_start ? c.reference_near_offset_max : c.reference_offset_max);
            T altitude_offset_max = near_start ? std::fmin(c.reference_near_offset_max, c.reference_altitude_offset_max) : c.reference_altitude_offset_max;
            T altitude = fixed_wing::clip(body.position[2] + uniform(-altitude_offset_max, altitude_offset_max),
                                          c.reference_altitude_range[0], c.reference_altitude_range[1]);
            T duration = fixed_wing::episode_duration(c, p.dt, T(fixed_wing::Environment<SPEC>::EPISODE_STEP_LIMIT));
            T climb = fixed_wing::clip(uniform(-c.reference_climb_max, c.reference_climb_max),
                                       (c.reference_altitude_range[0] - altitude) / duration, (c.reference_altitude_range[1] - altitude) / duration);
            fixed_wing::start_state(p, body, std::sqrt(speed * speed + climb * climb), altitude, heading, s);
            s.reference_position[0] = body.position[0] + distance * std::cos(bearing);
            s.reference_position[1] = body.position[1] + distance * std::sin(bearing);
            s.reference_position[2] = altitude;
            s.reference_velocity[0] = speed * std::cos(heading);
            s.reference_velocity[1] = speed * std::sin(heading);
            s.reference_velocity[2] = climb;
            fixed_wing::ReferenceMotion<T>& motion = s.reference_motion;
            for(int i = 0; i < 3; i++) motion.origin[i] = s.reference_position[i];
            motion.heading = heading;
            motion.speed = speed;
            motion.climb = climb;
            if(c.reference_turn_fraction > 0 && uniform(T(0), T(1)) < c.reference_turn_fraction){
                T bank = uniform(c.reference_turn_bank_range[0], c.reference_turn_bank_range[1]);
                motion.kind = fixed_wing::HELIX;
                motion.turn = uniform(T(-1), T(1)) < 0 ? T(-1) : T(1);
                motion.acceleration = std::abs(p.gravity[2]) * std::tan(bank);
                T position[3], velocity[3], acceleration[3];
                fixed_wing::reference_at(motion, T(0), position, velocity, acceleration);
                for(int i = 0; i < 3; i++){
                    s.reference_velocity[i] = velocity[i];
                    s.reference_acceleration[i] = acceleration[i];
                }
            }
            if(c.reference_air_relative){
                for(int i = 0; i < 2; i++){
                    motion.drift[i] = p.wind[i];
                    s.reference_velocity[i] += p.wind[i];
                }
            }
            if(p.thermals.count > 0){
                T position[3], velocity[3], acceleration[3];
                fixed_wing::reference_at(motion, duration / 2, position, velocity, acceleration);
                T shift[2] = {position[0] - p.wind[0] * duration / 2, position[1] - p.wind[1] * duration / 2};
                for(int i = 0; i < 2; i++){
                    s.body.position[i] -= shift[i];
                    s.reference_position[i] -= shift[i];
                    motion.origin[i] -= shift[i];
                }
            }
        }
        else{
            T setpoint_airspeed = uniform(c.airspeed_range[0], c.airspeed_range[1]);
            T setpoint_altitude = fixed_wing::clip(body.position[2] + uniform(-c.altitude_change_max, c.altitude_change_max), c.altitude_range[0], c.altitude_range[1]);
            T setpoint_course = fixed_wing::wrap(course + uniform(-c.course_change_max, c.course_change_max));
            fixed_wing::start_state(p, body, setpoint_airspeed, setpoint_altitude, setpoint_course, s);
        }
        if(fixed_wing::turbulent(p)){
            for(int i = 0; i < fixed_wing::N_TURBULENCE; i++){
                s.body.turbulence[i] = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), T(0), T(1), rng);
            }
        }
    }

    template <typename DEVICE, typename SPEC, typename ACTION_SPEC, typename RNG>
    typename SPEC::T step(DEVICE&, const fixed_wing::Environment<SPEC>&, const typename fixed_wing::Environment<SPEC>::Parameters& p, const typename fixed_wing::Environment<SPEC>::State& s, const Matrix<ACTION_SPEC>& action, typename fixed_wing::Environment<SPEC>::State& next, RNG& rng){
        static_assert(ACTION_SPEC::ROWS == 1 && ACTION_SPEC::COLS == fixed_wing::N_ACTUATORS);
        using T = typename SPEC::T;
        T a[fixed_wing::N_ACTUATORS];
        for(int i = 0; i < fixed_wing::N_ACTUATORS; i++) a[i] = get(action, 0, i);
        T normals[fixed_wing::N_TURBULENCE] = {};
        if(fixed_wing::turbulent(p)){
            for(int i = 0; i < fixed_wing::N_TURBULENCE; i++){
                normals[i] = random::normal_distribution::sample(typename DEVICE::SPEC::RANDOM(), T(0), T(1), rng);
            }
        }
        fixed_wing::simulate_moving_task(p, s, a, normals, next);
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
