#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>
#include <fixed_wing/reference_motion.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <deque>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <mutex>
#include <random>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

namespace rlt = rl_tools;
namespace fw = fixed_wing;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;
using ENVIRONMENT = fw::Environment<fw::Specification<T, TI>>;
using State = ENVIRONMENT::State;
using Parameters = ENVIRONMENT::Parameters;
using Config = fw::TaskConfig<T>;

static_assert(FW_TRACKING && FW_AIRPLANE == 1 && FW_DISTURBANCES, "build with FW_TRACKING=1 FW_AIRPLANE=1 FW_DISTURBANCES=1");

constexpr T DEG = 180 / fw::PI;

struct Actor{
    static constexpr int INPUT = ENVIRONMENT::Observation::DIM, HIDDEN = 64, OUTPUT = 2 * fw::N_ACTUATORS;
    std::vector<float> W0, b0, W1, b1, W2, b2;

    void act(const float observation[INPUT], T action[fw::N_ACTUATORS]) const {
        float h0[HIDDEN], h1[HIDDEN];
        for(int i = 0; i < HIDDEN; i++){
            float x = b0[i];
            for(int j = 0; j < INPUT; j++) x += W0[i * INPUT + j] * observation[j];
            h0[i] = x > 0 ? x : 0;
        }
        for(int i = 0; i < HIDDEN; i++){
            float x = b1[i];
            for(int j = 0; j < HIDDEN; j++) x += W1[i * HIDDEN + j] * h0[j];
            h1[i] = x > 0 ? x : 0;
        }
        for(int i = 0; i < fw::N_ACTUATORS; i++){
            float x = b2[i];
            for(int j = 0; j < HIDDEN; j++) x += W2[i * HIDDEN + j] * h1[j];
            action[i] = std::tanh(x);
        }
    }
};

Actor load_actor(const std::string& path){
    std::ifstream file(path);
    if(!file) throw std::runtime_error("cannot read " + path);
    std::stringstream buffer;
    buffer << file.rdbuf();
    std::string source = buffer.str();
    std::vector<std::vector<float>> arrays;
    const std::string marker = "constexpr float memory[] = {";
    for(size_t at = source.find(marker); at != std::string::npos; at = source.find(marker, at)){
        at += marker.size();
        size_t end = source.find('}', at);
        std::vector<float> values;
        const char* p = source.c_str() + at;
        const char* stop = source.c_str() + end;
        while(p < stop){
            char* next;
            float v = std::strtof(p, &next);
            if(next == p) throw std::runtime_error("cannot parse the weights of " + path);
            values.push_back(v);
            p = next;
            while(p < stop && (*p == 'f' || *p == ',' || std::isspace(static_cast<unsigned char>(*p)))) p++;
        }
        arrays.push_back(values);
        at = end;
    }
    const size_t expected[6] = {Actor::HIDDEN * Actor::INPUT, Actor::HIDDEN, Actor::HIDDEN * Actor::HIDDEN, Actor::HIDDEN,
                                Actor::OUTPUT * Actor::HIDDEN, Actor::OUTPUT};
    bool ok = arrays.size() == 6;
    for(size_t i = 0; ok && i < 6; i++) ok = arrays[i].size() == expected[i];
    if(!ok) throw std::runtime_error("unexpected actor layout in " + path + " (65 inputs, 64 x 64, 8 outputs expected)");
    return Actor{arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]};
}

enum Kind{BASELINE, MOTOR, WIND, GUST, TURBULENCE, THERMAL, MASS, THRUST, CONTROL, LATENCY, NOISE, JUMP, HEADING, REPRODUCE};
enum Direction{AHEAD = 0, BEHIND, SIDE, DOWN, UP};

struct Scenario{
    std::string name;
    Kind kind;
    std::string unit;
    std::vector<T> levels;
    bool disturbed;
    T duration;
    int direction;
    std::string description;
    T ramp = 0;
};

constexpr T EVENT_TIME = 10;
constexpr T GUST_DURATION = 2;

std::vector<Scenario> scenarios(){
    std::vector<Scenario> list;
    list.push_back({"baseline_calm", BASELINE, "-", {0}, false, 60, 0, "no event, calm air"});
    list.push_back({"baseline_disturbed", BASELINE, "-", {0}, true, 60, 0, "no event, training disturbances"});
    std::vector<T> cut{0, 3, 5, 10, 15, 20, 30, 45, 80};
    list.push_back({"motor_cut", MOTOR, "s", cut, false, 90, 0, "throttle forced to 0 at t = 10 s for the given time, calm air"});
    list.push_back({"motor_cut_disturbed", MOTOR, "s", cut, true, 90, 0, "throttle forced to 0 at t = 10 s, training disturbances"});
    list.push_back({"motor_cut_soft_restart", MOTOR, "s", cut, false, 90, 0, "throttle forced to 0 at t = 10 s, then capped by a ramp from 0 to full over 2 s, calm air", 2});
    std::vector<T> wind{0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20};
    list.push_back({"wind_head", WIND, "m/s", wind, false, 60, BEHIND, "wind steps from 0 at t = 10 s, against the point's direction"});
    list.push_back({"wind_tail", WIND, "m/s", wind, false, 60, AHEAD, "wind steps from 0 at t = 10 s, along the point's direction"});
    list.push_back({"wind_head_ramp", WIND, "m/s", wind, false, 60, BEHIND, "wind grows linearly from 0 over 10 s from t = 10 s, against the point's direction", 10});
    list.push_back({"wind_tail_ramp", WIND, "m/s", wind, false, 60, AHEAD, "wind grows linearly from 0 over 10 s from t = 10 s, along the point's direction", 10});
    list.push_back({"wind_cross", WIND, "m/s", wind, false, 60, SIDE, "wind steps from 0 at t = 10 s, across the point's direction"});
    std::vector<T> gust{0, 2, 3, 5, 7, 10, 13, 16, 20};
    list.push_back({"gust_head", GUST, "m/s", gust, false, 60, BEHIND, "one 1-cos gust of 2 s at t = 10 s, headwind"});
    list.push_back({"gust_tail", GUST, "m/s", gust, false, 60, AHEAD, "one 1-cos gust of 2 s at t = 10 s, tailwind"});
    list.push_back({"gust_side", GUST, "m/s", gust, false, 60, SIDE, "one 1-cos gust of 2 s at t = 10 s, crosswind"});
    list.push_back({"gust_down", GUST, "m/s", gust, false, 60, DOWN, "one 1-cos gust of 2 s at t = 10 s, downdraft"});
    list.push_back({"gust_up", GUST, "m/s", gust, false, 60, UP, "one 1-cos gust of 2 s at t = 10 s, updraft"});
    list.push_back({"turbulence", TURBULENCE, "x moderate", {0, 0.5, 1, 1.5, 2, 2.5, 3, 4, 5}, false, 60, 0,
                    "Dryden turbulence from t = 10 s (state starting at 0), sigma = level x (2.12, 2.12, 1.4) m/s"});
    list.push_back({"thermal", THERMAL, "m/s", {0, 1, 2, 3, 4, 5, 6, 8, 10}, false, 60, 0,
                    "Allen thermal field in every flight, core updraft of every thermal = level"});
    list.push_back({"mass_heavy", MASS, "x mass", {1, 1.1, 1.25, 1.5, 1.75, 2, 2.5, 3}, false, 60, 0, "mass times the level"});
    list.push_back({"mass_light", MASS, "x mass", {1, 0.9, 0.75, 0.6, 0.5, 0.4}, false, 60, 0, "mass times the level"});
    list.push_back({"thrust", THRUST, "x thrust", {1, 0.85, 0.7, 0.5, 0.35, 0.25, 0.18, 0.12, 0.08, 0.05}, false, 60, 0,
                    "maximum thrust times the level"});
    list.push_back({"control_weak", CONTROL, "x control", {1, 0.9, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1}, false, 60, 0,
                    "force and moment coefficients of the control surfaces times the level"});
    list.push_back({"control_strong", CONTROL, "x control", {1, 1.5, 2, 3, 4}, false, 60, 0,
                    "force and moment coefficients of the control surfaces times the level"});
    list.push_back({"latency", LATENCY, "steps of 20 ms", {0, 1, 2, 3, 4, 5, 6, 8, 10, 12, 15}, false, 60, 0,
                    "control periods between a command and its application (nominal 1)"});
    list.push_back({"noise", NOISE, "x base", {0, 0.5, 1, 2, 3, 5, 8, 12}, false, 60, 0,
                    "white noise on the observed state, base sigma: position 0.5 m, velocity 0.1 m/s, attitude 0.5 deg, rates 0.02 rad/s, airspeed 0.3 m/s"});
    std::vector<T> jump{0, 5, 10, 15, 20, 30, 40, 60, 80};
    list.push_back({"jump_ahead", JUMP, "m", jump, false, 60, AHEAD, "the point jumps forward at t = 10 s"});
    list.push_back({"jump_behind", JUMP, "m", jump, false, 60, BEHIND, "the point jumps backward at t = 10 s"});
    list.push_back({"jump_side", JUMP, "m", jump, false, 60, SIDE, "the point jumps sideways at t = 10 s"});
    list.push_back({"jump_down", JUMP, "m", jump, false, 60, DOWN, "the point jumps down at t = 10 s"});
    list.push_back({"jump_up", JUMP, "m", jump, false, 60, UP, "the point jumps up at t = 10 s"});
    list.push_back({"heading_jump", HEADING, "deg", {0, 15, 30, 45, 60, 90, 120, 150, 180}, false, 60, 0,
                    "the point's direction turns at once at t = 10 s"});
    list.push_back({"reproduce_disturbed", REPRODUCE, "-", {0}, true, 120, 0, "the 100 disturbed test flights of train_sac"});
    list.push_back({"reproduce_calm", REPRODUCE, "-", {0}, false, 120, 0, "the 100 calm test flights of train_sac"});
    return list;
}

Config base_config(bool disturbed, T duration){
    Config c{};
    c.start = fw::TRIMMED;
    c.reference_near_fraction = 1;
    c.reference_near_offset_max = 5;
    c.course_change_max = 20 / DEG;
    c.reference_distance_max = 30;
    c.episode_duration = duration;
    c.fine_position_scale = 2;
    if(disturbed){
        c.wind_speed_max = 6;
        c.turbulence_intensity_max = 1;
        c.gust_rate = 1;
        c.thermal_fraction = T(0.5);
        c.reference_turn_fraction = T(0.5);
        c.randomization_mass = T(0.1);
        c.randomization_thrust = T(0.15);
        c.randomization_aerodynamics = T(0.1);
        c.latency_steps_random_max = 2;
    }
    return c;
}

Config relaxed_config(Config c){
    c.reference_distance_max = 0;
    c.position_error_max = 1e12;
    c.altitude_error_max = 1e12;
    return c;
}

struct Setup{
    Parameters p;
    State s;
    int latency;
    std::uint64_t seed;
};

void sample(DEVICE& device, const Config& config, RNG& rng, Setup& setup){
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config = config;
    rlt::sample_initial_parameters(device, env, setup.p, rng);
    rlt::sample_initial_state(device, env, setup.p, setup.s, rng);
    setup.latency = setup.p.latency_steps;
}

Config scenario_config(const Scenario& sc, T level){
    Config c = base_config(sc.disturbed, sc.duration);
    if(sc.kind == MOTOR){
        c.altitude_range[0] = 100;
        c.altitude_range[1] = 120;
        c.reference_climb_max = 0;
    }
    if(sc.kind == THERMAL && level > 0){
        c.thermal_fraction = 1;
        c.thermal_core_range[0] = level;
        c.thermal_core_range[1] = level;
    }
    return c;
}

void unit_directions(const State& s, T u[3], T side[3]){
    T vh = std::hypot(s.reference_velocity[0], s.reference_velocity[1]);
    u[0] = s.reference_velocity[0] / vh;
    u[1] = s.reference_velocity[1] / vh;
    u[2] = 0;
    side[0] = -u[1];
    side[1] = u[0];
    side[2] = 0;
}

void direction_vector(const State& s, int direction, TI k, T out[3]){
    T u[3], side[3];
    unit_directions(s, u, side);
    T sign = k % 2 == 0 ? T(1) : T(-1);
    for(int i = 0; i < 3; i++) out[i] = 0;
    if(direction == AHEAD) for(int i = 0; i < 3; i++) out[i] = u[i];
    else if(direction == BEHIND) for(int i = 0; i < 3; i++) out[i] = -u[i];
    else if(direction == SIDE) for(int i = 0; i < 3; i++) out[i] = sign * side[i];
    else if(direction == DOWN) out[2] = -1;
    else out[2] = 1;
}

void scale_controls(Parameters& p, T factor){
    for(int c: {fw::C_L_delta_e, fw::C_Y_delta_a, fw::C_Y_delta_r, fw::C_l_delta_a, fw::C_l_delta_r, fw::C_m_delta_e,
                fw::C_n_delta_a, fw::C_n_delta_r}) p.randomization[4 + c] *= factor;
}

Setup make_setup(DEVICE& device, const Scenario& sc, T level, TI k){
    Setup setup;
    setup.seed = 7000 + k;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, setup.seed);
    sample(device, scenario_config(sc, level), rng, setup);
    rlt::free(device, rng);
    Parameters& p = setup.p;
    if(sc.kind == GUST && level > 0){
        T direction[3];
        direction_vector(setup.s, sc.direction, k, direction);
        p.gust_count = 1;
        p.gusts[0].start = EVENT_TIME;
        p.gusts[0].duration = GUST_DURATION;
        for(int i = 0; i < 3; i++) p.gusts[0].peak[i] = level * direction[i];
    }
    if(sc.kind == MASS) p.mass *= level;
    if(sc.kind == THRUST) p.thrust_max *= level;
    if(sc.kind == CONTROL) scale_controls(p, level);
    if(sc.kind == LATENCY) setup.latency = int(std::lround(level));
    return setup;
}

std::vector<Setup> reproduce_setups(DEVICE& device, bool disturbed, TI n){
    Config c = base_config(true, 120);
    if(!disturbed){
        c.wind_speed_max = 0;
        c.turbulence_intensity_max = 0;
        c.gust_rate = 0;
        c.thermal_fraction = 0;
    }
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 12345);
    std::vector<Setup> setups(n);
    for(TI k = 0; k < n; k++){
        sample(device, c, rng, setups[k]);
        setups[k].seed = std::uint64_t(12345) * 1000 + k;
    }
    rlt::free(device, rng);
    return setups;
}

void advance(const Parameters& p, const State& state, const T action[fw::N_ACTUATORS], std::deque<std::array<T, fw::N_ACTUATORS>>& queue,
             int latency, T throttle_cap, const T normals[fw::N_TURBULENCE], State& next){
    next = state;
    T clipped[fw::N_ACTUATORS], command[fw::N_ACTUATORS], applied[fw::N_ACTUATORS];
    for(int i = 0; i < fw::N_ACTUATORS; i++) clipped[i] = fw::clip(action[i], T(-1), T(1));
    fw::scale_action(p, clipped, command);
    if(latency == 0){
        for(int i = 0; i < fw::N_ACTUATORS; i++) applied[i] = command[i];
    }
    else{
        for(int i = 0; i < fw::N_ACTUATORS; i++) applied[i] = queue.front()[i];
        queue.pop_front();
        std::array<T, fw::N_ACTUATORS> c;
        for(int i = 0; i < fw::N_ACTUATORS; i++) c[i] = command[i];
        queue.push_back(c);
    }
    if(throttle_cap < 1){
        T lo = p.actuator_limits[fw::THROTTLE][0], hi = p.actuator_limits[fw::THROTTLE][1];
        applied[fw::THROTTLE] = std::min(applied[fw::THROTTLE], lo + std::max(throttle_cap, T(0)) * (hi - lo));
    }
    T h = p.dt / p.n_substeps;
    fw::Body<T> x = state.body;
    for(int k = 0; k < p.n_substeps; k++){
        fw::Body<T> y;
        fw::rk4(p, x, applied, h, y);
        x = y;
    }
    fw::post_integration(p, x);
    next.body = x;
    for(int k = 0; k + 1 < fw::ACTION_HISTORY; k++) for(int i = 0; i < fw::N_ACTUATORS; i++) next.action_history[k][i] = state.action_history[k + 1][i];
    for(int i = 0; i < fw::N_ACTUATORS; i++) next.action_history[fw::ACTION_HISTORY - 1][i] = clipped[i];
    if(state.reference_motion.kind == fw::STRAIGHT){
        for(int i = 0; i < 3; i++) next.reference_position[i] = state.reference_position[i] + state.reference_velocity[i] * p.dt;
        next.setpoint_altitude = next.reference_position[2];
    }
    else{
        T position[3], velocity[3], acceleration[3];
        fw::reference_at(state.reference_motion, state.body.time + p.dt, position, velocity, acceleration);
        for(int i = 0; i < 3; i++){
            next.reference_position[i] = position[i];
            next.reference_velocity[i] = velocity[i];
            next.reference_acceleration[i] = acceleration[i];
        }
        next.setpoint_airspeed = fw::norm3(velocity);
        next.setpoint_course = std::atan2(velocity[1], velocity[0]);
        next.setpoint_altitude = position[2];
    }
    fw::advance_turbulence(p, state, normals, next);
}

void observe(const Parameters& p, const State& s, T noise, std::mt19937_64& generator, float observation[Actor::INPUT]){
    if(noise <= 0){
        fw::observe_actor(p, s, [&](int j, T v){ observation[j] = float(v); });
        return;
    }
    std::normal_distribution<T> normal(0, 1);
    State noisy = s;
    const T sigma_position = T(0.5), sigma_velocity = T(0.1), sigma_attitude = T(0.5) / DEG, sigma_rate = T(0.02), sigma_airspeed = T(0.3);
    for(int i = 0; i < 3; i++){
        noisy.body.position[i] += noise * sigma_position * normal(generator);
        noisy.body.linear_velocity[i] += noise * sigma_velocity * normal(generator);
        noisy.body.angular_velocity[i] += noise * sigma_rate * normal(generator);
    }
    T r[3];
    for(int i = 0; i < 3; i++) r[i] = noise * sigma_attitude * normal(generator);
    T dq[4] = {1, r[0] / 2, r[1] / 2, r[2] / 2};
    T n = std::sqrt(dq[0] * dq[0] + dq[1] * dq[1] + dq[2] * dq[2] + dq[3] * dq[3]);
    for(T& v: dq) v /= n;
    fw::quaternion_multiply(s.body.orientation, dq, noisy.body.orientation);
    fw::observe_actor(p, noisy, [&](int j, T v){ observation[j] = float(v); });
    constexpr int AIRSPEED_INDEX = fw::N_TASK_ERRORS + 9 + 3 + 3;
    observation[AIRSPEED_INDEX] += float(noise * sigma_airspeed * normal(generator) / 25);
}

struct Result{
    int strict_reason = 0, hard_reason = 0;
    double strict_time = -1, hard_time = -1, flown = 0;
    double rmse_strict = 0, rmse_relaxed = 0, rmse_tail = -1;
    double max_distance = 0, max_distance_event = 0, last_exceed = -1, recover_time = -1;
    bool recovered = false;
    double min_airspeed = 1e9, max_alpha = -1e9, max_bank = 0, min_altitude = 1e9, start_altitude = 0, reference_speed = 0;
    double event_altitude = 0, alt_lost_total = 0, alt_lost_event = 0;
    double glide_airspeed = 0, glide_alpha = 0, glide_alpha_max = -1e9, glide_airspeed_min = 1e9, glide_sink = 0, glide_pitch = 0, glide_ratio = -1;
};

struct Event{ double start = -1, end = -1; };

Event event_window(const Scenario& sc, T level){
    if(sc.kind == MOTOR) return level > 0 ? Event{EVENT_TIME, std::min(double(EVENT_TIME + level), double(sc.duration))} : Event{EVENT_TIME, EVENT_TIME};
    if(sc.kind == GUST) return {EVENT_TIME, EVENT_TIME + GUST_DURATION};
    if(sc.kind == WIND) return {EVENT_TIME, EVENT_TIME + sc.ramp};
    if(sc.kind == TURBULENCE) return {EVENT_TIME, EVENT_TIME};
    if(sc.kind == JUMP || sc.kind == HEADING) return {EVENT_TIME, EVENT_TIME};
    return {0, 0};
}

Result fly(const Actor& actor, const Scenario& sc, T level, TI k, Setup setup, T radius, std::ostream* csv){
    Parameters p = setup.p;
    State s = setup.s;
    const Config config = scenario_config(sc, level);
    const Config relaxed = relaxed_config(config);
    const T dt = p.dt;
    const TI n_steps = TI(std::lround(sc.duration / dt));
    const Event ev = event_window(sc, level);
    const TI event_step = TI(std::lround(EVENT_TIME / dt));
    T wind_target[3] = {0, 0, 0};
    if(sc.kind == WIND){
        T d[3];
        direction_vector(s, sc.direction, k, d);
        for(int i = 0; i < 3; i++) wind_target[i] = level * d[i];
        wind_target[2] = 0;
    }
    std::deque<std::array<T, fw::N_ACTUATORS>> queue;
    for(int j = 0; j < setup.latency; j++){
        std::array<T, fw::N_ACTUATORS> c;
        for(int i = 0; i < fw::N_ACTUATORS; i++) c[i] = s.body.actuators[i];
        queue.push_back(c);
    }
    std::mt19937_64 turbulence_generator(setup.seed);
    std::mt19937_64 noise_generator(setup.seed * 31 + 11);
    std::normal_distribution<T> standard_normal(0, 1);
    const T noise = sc.kind == NOISE ? level : 0;
    Result r;
    r.start_altitude = s.body.position[2];
    r.reference_speed = std::hypot(s.reference_velocity[0], s.reference_velocity[1]);
    double sum_strict = 0, sum_relaxed = 0, sum_tail = 0;
    long n_strict = 0, n_relaxed = 0, n_tail = 0, n_glide = 0;
    double glide_distance = 0;
    T previous_position[2] = {s.body.position[0], s.body.position[1]};
    auto record = [&](T t, const State& x, const T a[4], int event, int strict, int hard){
        if(!csv) return;
        auto m = fw::measurements(p, x.body);
        auto air = fw::air_data(p, x.body);
        T wind[3];
        fw::local_wind(p, x.body, m.R, wind);
        T e_position[3], e_velocity[3];
        fw::tracking_errors(x, e_position, e_velocity);
        *csv << k << "," << t;
        for(int i = 0; i < 3; i++) *csv << "," << x.body.position[i];
        *csv << "," << std::atan2(m.R[2][1], m.R[2][2]) * DEG << "," << std::asin(fw::clip(m.R[2][0], T(-1), T(1))) * DEG
             << "," << std::atan2(m.R[1][0], m.R[0][0]) * DEG << "," << air.airspeed << "," << air.alpha * DEG << ","
             << air.beta * DEG << "," << std::hypot(x.body.linear_velocity[0], x.body.linear_velocity[1]) << "," << x.body.linear_velocity[2];
        for(int i = 0; i < 3; i++) *csv << "," << x.reference_position[i];
        *csv << "," << fw::norm3(e_position);
        for(int i = 0; i < 3; i++) *csv << "," << wind[i];
        for(int i = 0; i < 4; i++) *csv << "," << a[i];
        *csv << "," << x.body.actuators[fw::THROTTLE] << "," << event << "," << strict << "," << hard << "\n";
    };
    T previous[4];
    for(int i = 0; i < 4; i++) previous[i] = s.action_history[fw::ACTION_HISTORY - 1][i];
    record(0, s, previous, 0, 0, 0);
    for(TI step = 0; step < n_steps; step++){
        T t = T(step) * dt;
        if(sc.kind == TURBULENCE && level > 0 && step == event_step){
            const T moderate_sigma[3] = {T(2.12), T(2.12), T(1.4)};
            const T moderate_length_scale[3] = {200, 200, 50};
            for(int i = 0; i < 3; i++){
                p.turbulence_sigma[i] = level * moderate_sigma[i];
                p.turbulence_length_scale[i] = moderate_length_scale[i];
            }
            for(int i = 0; i < fw::N_TURBULENCE; i++) s.body.turbulence[i] = 0;
        }
        if(sc.kind == WIND && step >= event_step){
            T fraction = sc.ramp > 0 ? std::min(T(1), (t - EVENT_TIME) / sc.ramp) : T(1);
            for(int i = 0; i < 3; i++) p.wind[i] = fraction * wind_target[i];
        }
        if(step == event_step && sc.kind == JUMP && level > 0){
            T d[3];
            direction_vector(s, sc.direction, k, d);
            for(int i = 0; i < 3; i++){
                s.reference_position[i] += level * d[i];
                s.reference_motion.origin[i] += level * d[i];
            }
            s.setpoint_altitude = s.reference_position[2];
        }
        if(step == event_step && sc.kind == HEADING && level > 0){
            T angle = (k % 2 == 0 ? T(1) : T(-1)) * level / DEG;
            T c = std::cos(angle), sn = std::sin(angle);
            T vx = s.reference_velocity[0], vy = s.reference_velocity[1];
            s.reference_velocity[0] = c * vx - sn * vy;
            s.reference_velocity[1] = sn * vx + c * vy;
            s.reference_motion.heading += angle;
            s.setpoint_course = std::atan2(s.reference_velocity[1], s.reference_velocity[0]);
        }
        bool motor_off = sc.kind == MOTOR && level > 0 && t >= ev.start - 1e-9 && t < ev.end - 1e-9;
        if(step == event_step) r.event_altitude = s.body.position[2];
        float observation[Actor::INPUT];
        observe(p, s, noise, noise_generator, observation);
        T a[4];
        actor.act(observation, a);
        T normals[fw::N_TURBULENCE] = {};
        if(fw::turbulent(p)) for(auto& v: normals) v = standard_normal(turbulence_generator);
        State next;
        T throttle_cap = motor_off ? T(0) : T(1);
        if(sc.kind == MOTOR && level > 0 && sc.ramp > 0 && t >= ev.end - 1e-9 && t < ev.end + sc.ramp) throttle_cap = (t - ev.end) / sc.ramp;
        advance(p, s, a, queue, setup.latency, throttle_cap, normals, next);
        T t_next = T(step + 1) * dt;
        int strict = fw::termination(p, next, config);
        int hard = fw::termination(p, next, relaxed);
        if(strict != 0 && r.strict_reason == 0){ r.strict_reason = strict; r.strict_time = t_next; }
        T e_position[3], e_velocity[3];
        fw::tracking_errors(next, e_position, e_velocity);
        double d = fw::norm3(e_position);
        if(r.strict_reason == 0 || r.strict_time == t_next){ sum_strict += d * d; n_strict++; }
        sum_relaxed += d * d;
        n_relaxed++;
        if(t_next > sc.duration - 20){ sum_tail += d * d; n_tail++; }
        r.max_distance = std::max(r.max_distance, d);
        if(ev.start >= 0 && t_next >= ev.start) r.max_distance_event = std::max(r.max_distance_event, d);
        if(t_next > ev.end && d > radius) r.last_exceed = t_next;
        auto m = fw::measurements(p, next.body);
        auto air = fw::air_data(p, next.body);
        r.min_airspeed = std::min(r.min_airspeed, double(air.airspeed));
        r.max_alpha = std::max(r.max_alpha, double(air.alpha * DEG));
        r.max_bank = std::max(r.max_bank, std::abs(std::atan2(m.R[2][1], m.R[2][2])) * DEG);
        r.min_altitude = std::min(r.min_altitude, double(next.body.position[2]));
        if(sc.kind == MOTOR && t_next > ev.start) r.alt_lost_total = std::max(r.alt_lost_total, r.event_altitude - next.body.position[2]);
        if(motor_off){
            r.glide_alpha_max = std::max(r.glide_alpha_max, double(air.alpha * DEG));
            r.glide_airspeed_min = std::min(r.glide_airspeed_min, double(air.airspeed));
            if(t >= ev.start + 2 || ev.end - ev.start <= 4){
                r.glide_airspeed += air.airspeed;
                r.glide_alpha += air.alpha * DEG;
                r.glide_sink += -next.body.linear_velocity[2];
                r.glide_pitch += std::asin(fw::clip(m.R[2][0], T(-1), T(1))) * DEG;
                n_glide++;
            }
            glide_distance += std::hypot(next.body.position[0] - previous_position[0], next.body.position[1] - previous_position[1]);
            r.alt_lost_event = r.event_altitude - next.body.position[2];
        }
        previous_position[0] = next.body.position[0];
        previous_position[1] = next.body.position[1];
        int event = 0;
        if(sc.kind == MOTOR) event = motor_off ? 1 : 0;
        else if(sc.kind == GUST) event = level > 0 && t_next >= ev.start && t_next < ev.end ? 1 : 0;
        else if(sc.kind == WIND || sc.kind == JUMP || sc.kind == HEADING || sc.kind == TURBULENCE) event = level > 0 && t_next >= ev.start ? 1 : 0;
        record(t_next, next, a, event, r.strict_reason, hard);
        s = next;
        r.flown = t_next;
        if(hard != 0){ r.hard_reason = hard; r.hard_time = t_next; break; }
    }
    r.rmse_strict = std::sqrt(sum_strict / std::max(n_strict, 1L));
    r.rmse_relaxed = std::sqrt(sum_relaxed / std::max(n_relaxed, 1L));
    if(r.hard_reason == 0 && n_tail > 0) r.rmse_tail = std::sqrt(sum_tail / n_tail);
    if(n_glide > 0){
        r.glide_airspeed /= n_glide;
        r.glide_alpha /= n_glide;
        r.glide_sink /= n_glide;
        r.glide_pitch /= n_glide;
    }
    if(r.alt_lost_event > 0.5) r.glide_ratio = glide_distance / r.alt_lost_event;
    if(r.hard_reason == 0 && ev.end <= sc.duration - 5 && r.last_exceed <= sc.duration - 5){
        r.recovered = true;
        r.recover_time = r.last_exceed < 0 ? 0 : r.last_exceed - ev.end;
    }
    return r;
}

std::string level_tag(T level){
    std::ostringstream out;
    out << level;
    std::string s = out.str();
    std::replace(s.begin(), s.end(), '.', 'p');
    return s;
}

int run(int argc, char** argv){
    if(argc < 3){
        std::cerr << "usage: evaluate_limits <best_actor.h> <out_dir> [--flights N] [--record K] [--threads N] [--scenarios a,b,...]"
                     " [--recovery-radius R] [--list]" << std::endl;
        return 1;
    }
    std::vector<Scenario> all = scenarios();
    for(int i = 1; i < argc; i++) if(std::string(argv[i]) == "--list"){
        for(const auto& sc: all){
            std::cout << sc.name << " [" << sc.unit << "]:";
            for(T l: sc.levels) std::cout << " " << l;
            std::cout << "  " << sc.description << "\n";
        }
        return 0;
    }
    Actor actor = load_actor(argv[1]);
    std::filesystem::path out_dir = argv[2];
    TI n_flights = 40, n_recorded = 2;
    unsigned n_threads = 2;
    std::string selected;
    T radius = 12;
    for(int i = 3; i < argc; i++){
        std::string option = argv[i];
        if(option == "--flights" && i + 1 < argc) n_flights = std::stoul(argv[++i]);
        else if(option == "--record" && i + 1 < argc) n_recorded = std::stoul(argv[++i]);
        else if(option == "--threads" && i + 1 < argc) n_threads = std::stoul(argv[++i]);
        else if(option == "--scenarios" && i + 1 < argc) selected = "," + std::string(argv[++i]) + ",";
        else if(option == "--recovery-radius" && i + 1 < argc) radius = std::stod(argv[++i]);
        else throw std::runtime_error("unknown option " + option);
    }
    std::vector<Scenario> chosen;
    for(const auto& sc: all) if(selected.empty() || selected.find("," + sc.name + ",") != std::string::npos) chosen.push_back(sc);
    if(chosen.empty()) throw std::runtime_error("no scenario selected");
    std::filesystem::create_directories(out_dir / "flights");

    struct Job{ size_t scenario; size_t level; };
    std::vector<Job> jobs;
    for(size_t s = 0; s < chosen.size(); s++) for(size_t l = 0; l < chosen[s].levels.size(); l++) jobs.push_back({s, l});
    std::vector<std::vector<Result>> results(jobs.size());
    std::vector<std::vector<Setup>> reproduce(2);
    {
        DEVICE device;
        rlt::init(device);
        reproduce[0] = reproduce_setups(device, true, 100);
        reproduce[1] = reproduce_setups(device, false, 100);
    }
    std::atomic<size_t> next_job{0};
    std::mutex print;
    auto worker = [&](){
        DEVICE device;
        rlt::init(device);
        for(size_t j = next_job++; j < jobs.size(); j = next_job++){
            const Scenario& sc = chosen[jobs[j].scenario];
            T level = sc.levels[jobs[j].level];
            std::ofstream csv(out_dir / "flights" / (sc.name + "_" + std::to_string(jobs[j].level) + "_" + level_tag(level) + ".csv"));
            csv << std::setprecision(8) << "flight,t,x,y,altitude,roll_deg,pitch_deg,heading_deg,airspeed,alpha_deg,beta_deg,"
                   "ground_speed,vertical_speed,reference_x,reference_y,reference_altitude,distance,wind_x,wind_y,wind_z,"
                   "aileron,elevator,rudder,throttle,throttle_applied,event,strict_termination,termination\n";
            TI n = sc.kind == REPRODUCE ? 100 : n_flights;
            for(TI k = 0; k < n; k++){
                Setup setup = sc.kind == REPRODUCE ? reproduce[sc.disturbed ? 0 : 1][k] : make_setup(device, sc, level, k);
                results[j].push_back(fly(actor, sc, level, k, setup, radius, k < n_recorded ? &csv : nullptr));
            }
            std::lock_guard<std::mutex> lock(print);
            int completed = 0, recovered = 0;
            for(const auto& r: results[j]){ completed += r.strict_reason == 0; recovered += r.recovered; }
            std::cout << sc.name << " " << level << " " << sc.unit << ": completed " << completed << "/" << results[j].size()
                      << ", recovered " << recovered << "/" << results[j].size() << std::endl;
        }
    };
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for(auto& t: threads) t.join();

    std::ofstream summary(out_dir / "summary.csv");
    summary << std::setprecision(8) << "scenario,kind,unit,background,level_index,level,flight,start_altitude,reference_speed,"
               "strict_reason,strict_time,hard_reason,hard_time,flown,completed,survived,recovered,recover_time,rmse_strict,"
               "rmse_relaxed,rmse_tail,max_distance,max_distance_event,min_airspeed,max_alpha,max_bank,min_altitude,"
               "event_altitude,alt_lost_total,alt_lost_event,glide_airspeed,glide_alpha,glide_alpha_max,glide_airspeed_min,"
               "glide_sink,glide_pitch,glide_ratio\n";
    const char* kinds[] = {"baseline", "motor", "wind", "gust", "turbulence", "thermal", "mass", "thrust", "control", "latency",
                           "noise", "jump", "heading", "reproduce"};
    for(size_t j = 0; j < jobs.size(); j++){
        const Scenario& sc = chosen[jobs[j].scenario];
        for(size_t k = 0; k < results[j].size(); k++){
            const Result& r = results[j][k];
            summary << sc.name << "," << kinds[sc.kind] << "," << sc.unit << "," << (sc.disturbed ? "disturbed" : "calm") << ","
                    << jobs[j].level << "," << sc.levels[jobs[j].level] << "," << k << "," << r.start_altitude << ","
                    << r.reference_speed << "," << r.strict_reason << "," << r.strict_time << "," << r.hard_reason << ","
                    << r.hard_time << "," << r.flown << "," << (r.strict_reason == 0) << "," << (r.hard_reason == 0) << ","
                    << r.recovered << "," << r.recover_time << "," << r.rmse_strict << "," << r.rmse_relaxed << ","
                    << r.rmse_tail << "," << r.max_distance << "," << r.max_distance_event << "," << r.min_airspeed << ","
                    << r.max_alpha << "," << r.max_bank << "," << r.min_altitude << "," << r.event_altitude << ","
                    << r.alt_lost_total << "," << r.alt_lost_event << "," << r.glide_airspeed << "," << r.glide_alpha << ","
                    << (r.glide_alpha_max > -1e8 ? r.glide_alpha_max : 0) << "," << (r.glide_airspeed_min < 1e8 ? r.glide_airspeed_min : 0)
                    << "," << r.glide_sink << "," << r.glide_pitch << "," << r.glide_ratio << "\n";
        }
    }
    std::ofstream list(out_dir / "scenarios.csv");
    list << "scenario,unit,duration_s,background,description\n";
    for(const auto& sc: chosen) list << sc.name << "," << sc.unit << "," << sc.duration << "," << (sc.disturbed ? "disturbed" : "calm")
                                     << ",\"" << sc.description << "\"\n";
    std::cout << "wrote " << (out_dir / "summary.csv").string() << " (" << jobs.size() << " levels)" << std::endl;
    return 0;
}

int main(int argc, char** argv){
    try{
        return run(argc, argv);
    }
    catch(const std::exception& e){
        std::cerr << "error: " << e.what() << std::endl;
        return 2;
    }
}
