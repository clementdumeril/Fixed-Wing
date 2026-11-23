#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>
#include <fixed_wing/reference_motion.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iomanip>
#include <iostream>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;
namespace fw = fixed_wing;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = double;
using TI = typename DEVICE::index_t;
using ENVIRONMENT = fw::Environment<fw::Specification<T, TI>>;
using State = ENVIRONMENT::State;

static_assert(FW_TRACKING && FW_AIRPLANE == 1 && FW_DISTURBANCES && FW_FINE_POSITION, "build with the flags of the _wind target (CMakeLists.txt)");

struct Actor{
    static constexpr int INPUT = ENVIRONMENT::Observation::DIM, HIDDEN = 64, OUTPUT = 2 * fw::N_ACTUATORS;
    std::vector<float> W0, b0, W1, b1, W2, b2;

    void act(const float observation[INPUT], T action[fw::N_ACTUATORS], T noise[fw::N_ACTUATORS]) const {
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
            float x = b2[i], y = b2[fw::N_ACTUATORS + i];
            for(int j = 0; j < HIDDEN; j++){
                x += W2[i * HIDDEN + j] * h1[j];
                y += W2[(fw::N_ACTUATORS + i) * HIDDEN + j] * h1[j];
            }
            action[i] = std::tanh(x);
            noise[i] = (1 - action[i] * action[i]) * std::exp(std::clamp(T(y), T(-20), T(2)));
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

struct Flight{
    ENVIRONMENT::Parameters parameters;
    State start;
    std::uint64_t turbulence_seed;
};

std::vector<Flight> sample_flights(DEVICE& device, const fw::TaskConfig<T>& config, TI seed, TI n){
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config = config;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    std::vector<Flight> flights(n);
    for(TI k = 0; k < n; k++){
        rlt::sample_initial_parameters(device, env, flights[k].parameters, rng);
        rlt::sample_initial_state(device, env, flights[k].parameters, flights[k].start, rng);
        flights[k].turbulence_seed = std::uint64_t(seed) * 1000 + k;
    }
    rlt::free(device, rng);
    return flights;
}

struct Accumulator{
    double n = 0, sum = 0, sum2 = 0, lo = 1e300, hi = -1e300;
    void add(double x){ n++; sum += x; sum2 += x * x; lo = std::min(lo, x); hi = std::max(hi, x); }
    double mean() const { return n > 0 ? sum / n : 0; }
    double rms() const { return n > 0 ? std::sqrt(sum2 / n) : 0; }
    double stdev() const { return n > 1 ? std::sqrt(std::max(sum2 / n - mean() * mean(), 0.0)) : 0; }
};

const std::vector<std::string> METRICS = {
    "position_rmse", "along_rmse", "cross_rmse", "vertical_rmse", "velocity_error_rms",
    "roll_std_deg", "roll_abs_max_deg", "pitch_std_deg", "roll_rate_rms_dps", "pitch_rate_rms_dps", "yaw_rate_rms_dps",
    "rate_norm_rms_dps", "roll_minus_required_rms_deg", "required_bank_abs_max_deg", "required_turn_rate_rms_dps",
    "action_rate_rms", "aileron_rate_rms", "elevator_rate_rms", "rudder_rate_rms", "throttle_rate_rms",
    "action_change_norm_mean", "mean_action_change",
    "airspeed_std", "airspeed_min", "airspeed_max", "required_airspeed_std", "required_airspeed_min", "required_airspeed_max",
    "throttle_std", "distance_mean", "training_noise_std", "training_noise_action_change", "sideslip_abs_mean_deg", "sideslip_abs_max_deg"};

struct Result{
    int reason = 0;
    bool thermals = false, helix = false;
    double wind = 0;
    std::map<std::string, double> values;
};

Result fly(const Actor& actor, const Flight& flight, const fw::TaskConfig<T>& config, TI n_steps){
    const T deg = 180 / fw::PI, g = 9.81;
    const auto& p = flight.parameters;
    State s = flight.start;
    std::mt19937_64 generator(flight.turbulence_seed);
    std::normal_distribution<T> normal(0, 1);
    Result r;
    r.thermals = p.thermals.count > 0;
    r.helix = s.reference_motion.kind != fw::STRAIGHT;
    r.wind = std::hypot(p.wind[0], p.wind[1]);
    Accumulator distance, along, cross, vertical, velocity, roll, pitch, rate[3], rate_norm, roll_excess, required_bank,
                required_turn, action_rate[fw::N_ACTUATORS], action_norm, action_abs, airspeed, required_airspeed, throttle,
                noise, noise_change, sideslip;
    for(TI step = 0; step < n_steps; step++){
        float observation[Actor::INPUT];
        fw::observe_actor(p, s, [&](int j, T v){ observation[j] = float(v); });
        T a[fw::N_ACTUATORS], sigma[fw::N_ACTUATORS], normals[fw::N_TURBULENCE] = {};
        actor.act(observation, a, sigma);
        T sigma2 = 0;
        for(int i = 0; i < fw::N_ACTUATORS; i++) sigma2 += sigma[i] * sigma[i];
        noise.add(std::sqrt(sigma2 / fw::N_ACTUATORS));
        noise_change.add(std::sqrt(2 * sigma2));
        for(int i = 0; i < fw::N_ACTUATORS; i++) a[i] = fw::clip(a[i], T(-1), T(1));
        if(fw::turbulent(p)) for(auto& v: normals) v = normal(generator);
        State next;
        fw::simulate_moving_task(p, s, a, normals, next);
        r.reason = fw::termination(p, next, config);
        T e_position[3], e_velocity[3];
        fw::tracking_errors(next, e_position, e_velocity);
        distance.add(fw::norm3(e_position));
        velocity.add(fw::norm3(e_velocity));
        T vh = std::hypot(next.reference_velocity[0], next.reference_velocity[1]);
        T u[2] = {next.reference_velocity[0] / vh, next.reference_velocity[1] / vh};
        along.add(e_position[0] * u[0] + e_position[1] * u[1]);
        cross.add(-e_position[0] * u[1] + e_position[1] * u[0]);
        vertical.add(e_position[2]);
        auto m = fw::measurements(p, next.body);
        auto air = fw::air_data(p, next.body);
        T phi = std::atan2(m.R[2][1], m.R[2][2]);
        roll.add(phi * deg);
        pitch.add(std::asin(fw::clip(m.R[2][0], T(-1), T(1))) * deg);
        for(int i = 0; i < 3; i++) rate[i].add(next.body.angular_velocity[i] * deg);
        rate_norm.add(fw::norm3(next.body.angular_velocity) * deg);
        T va[3] = {next.reference_velocity[0] - p.wind[0], next.reference_velocity[1] - p.wind[1], next.reference_velocity[2] - p.wind[2]};
        T vah = std::hypot(va[0], va[1]);
        T lateral = r.helix ? (va[0] * next.reference_acceleration[1] - va[1] * next.reference_acceleration[0]) / std::max(vah, T(1e-6)) : T(0);
        T bank = -std::atan(lateral / g);
        required_bank.add(std::abs(bank) * deg);
        required_turn.add(std::abs(lateral) / std::max(vah, T(1e-6)) * deg);
        roll_excess.add((phi - bank) * deg);
        T change2 = 0, change_abs = 0;
        for(int i = 0; i < fw::N_ACTUATORS; i++){
            T d = a[i] - s.action_history[fw::ACTION_HISTORY - 1][i];
            action_rate[i].add(d);
            change2 += d * d;
            change_abs += std::abs(d);
        }
        action_norm.add(std::sqrt(change2));
        action_abs.add(change_abs / fw::N_ACTUATORS);
        airspeed.add(air.airspeed);
        sideslip.add(std::abs(air.beta) * deg);
        required_airspeed.add(fw::norm3(va));
        throttle.add(a[fw::THROTTLE]);
        s = next;
        if(r.reason != fw::FLYING) break;
    }
    double action_rate_ms = 0;
    for(int i = 0; i < fw::N_ACTUATORS; i++) action_rate_ms += action_rate[i].sum2 / std::max(action_rate[i].n, 1.0) / fw::N_ACTUATORS;
    auto& v = r.values;
    v["position_rmse"] = distance.rms();
    v["along_rmse"] = along.rms();
    v["cross_rmse"] = cross.rms();
    v["vertical_rmse"] = vertical.rms();
    v["velocity_error_rms"] = velocity.rms();
    v["roll_std_deg"] = roll.stdev();
    v["roll_abs_max_deg"] = std::max(std::abs(roll.lo), std::abs(roll.hi));
    v["pitch_std_deg"] = pitch.stdev();
    v["roll_rate_rms_dps"] = rate[0].rms();
    v["pitch_rate_rms_dps"] = rate[1].rms();
    v["yaw_rate_rms_dps"] = rate[2].rms();
    v["rate_norm_rms_dps"] = rate_norm.rms();
    v["roll_minus_required_rms_deg"] = roll_excess.rms();
    v["required_bank_abs_max_deg"] = required_bank.hi;
    v["required_turn_rate_rms_dps"] = required_turn.rms();
    v["action_rate_rms"] = std::sqrt(action_rate_ms);
    v["aileron_rate_rms"] = action_rate[fw::AILERON].rms();
    v["elevator_rate_rms"] = action_rate[fw::ELEVATOR].rms();
    v["rudder_rate_rms"] = action_rate[fw::RUDDER].rms();
    v["throttle_rate_rms"] = action_rate[fw::THROTTLE].rms();
    v["action_change_norm_mean"] = action_norm.mean();
    v["mean_action_change"] = action_abs.mean();
    v["airspeed_std"] = airspeed.stdev();
    v["airspeed_min"] = airspeed.lo;
    v["airspeed_max"] = airspeed.hi;
    v["required_airspeed_std"] = required_airspeed.stdev();
    v["required_airspeed_min"] = required_airspeed.lo;
    v["required_airspeed_max"] = required_airspeed.hi;
    v["throttle_std"] = throttle.stdev();
    v["distance_mean"] = distance.mean();
    v["training_noise_std"] = noise.mean();
    v["training_noise_action_change"] = noise_change.mean();
    v["sideslip_abs_mean_deg"] = sideslip.mean();
    v["sideslip_abs_max_deg"] = sideslip.hi;
    return r;
}

std::string group_json(const std::vector<Result>& results, const std::function<bool(const Result&)>& in_group){
    const char* names[] = {"", "diverged", "low_altitude", "alpha", "airspeed", "altitude", "position"};
    int flights = 0, completed = 0;
    std::map<std::string, int> failures;
    std::map<std::string, double> sums;
    for(const auto& r: results){
        if(!in_group(r)) continue;
        flights++;
        if(r.reason != 0){ failures[names[r.reason]]++; continue; }
        completed++;
        for(const auto& name: METRICS) sums[name] += r.values.at(name);
    }
    std::ostringstream out;
    out << std::setprecision(6) << "{\"flights\": " << flights << ", \"success_rate\": " << completed / double(std::max(flights, 1));
    for(const auto& name: METRICS){
        out << ", \"" << name << "\": ";
        if(completed > 0) out << sums[name] / completed;
        else out << "null";
    }
    out << ", \"failures\": {";
    bool first = true;
    for(auto& [name, count]: failures){ out << (first ? "" : ", ") << "\"" << name << "\": " << count; first = false; }
    out << "}}";
    return out.str();
}

std::string evaluate(const Actor& actor, const std::vector<Flight>& flights, const fw::TaskConfig<T>& config, TI n_steps,
                     std::ostream& csv, const std::string& set){
    std::vector<Result> results;
    for(const auto& f: flights) results.push_back(fly(actor, f, config, n_steps));
    for(size_t k = 0; k < results.size(); k++){
        const auto& r = results[k];
        csv << set << "," << k << "," << r.reason << "," << r.thermals << "," << r.helix << "," << r.wind;
        for(const auto& name: METRICS) csv << "," << r.values.at(name);
        csv << "\n";
    }
    std::ostringstream json;
    json << "{\n    \"all\": " << group_json(results, [](const Result&){ return true; })
         << ",\n    \"helix\": " << group_json(results, [](const Result& r){ return r.helix; })
         << ",\n    \"straight\": " << group_json(results, [](const Result& r){ return !r.helix; })
         << ",\n    \"with_thermals\": " << group_json(results, [](const Result& r){ return r.thermals; })
         << ",\n    \"without_thermals\": " << group_json(results, [](const Result& r){ return !r.thermals; }) << "\n  }";
    return json.str();
}

int run(int argc, char** argv){
    if(argc < 3){
        std::cerr << "usage: evaluate_smoothness <actor.h> <out_prefix> [--duration S] [--flights N] [--distance-max D] [--reference-heading-max DEG] [--near-start F D] [--reference-air-relative] [--calm-only]" << std::endl;
        return 1;
    }
    Actor actor = load_actor(argv[1]);
    std::string prefix = argv[2];
    T duration = 120, distance_max = 30, heading_max_deg = 20, near_fraction = 1, near_offset = 5;
    TI n_flights = 100;
    bool air_relative = false;
    for(int i = 3; i < argc; i++){
        std::string option = argv[i];
        if(option == "--duration" && i + 1 < argc) duration = std::stod(argv[++i]);
        else if(option == "--flights" && i + 1 < argc) n_flights = std::stoul(argv[++i]);
        else if(option == "--distance-max" && i + 1 < argc) distance_max = std::stod(argv[++i]);
        else if(option == "--reference-heading-max" && i + 1 < argc) heading_max_deg = std::stod(argv[++i]);
        else if(option == "--near-start" && i + 2 < argc){ near_fraction = std::stod(argv[++i]); near_offset = std::stod(argv[++i]); }
        else if(option == "--reference-air-relative") air_relative = true;
        else throw std::runtime_error("unknown option " + option);
    }
    DEVICE device;
    rlt::init(device);
    fw::TaskConfig<T> config{};
    config.start = fw::TRIMMED;
    config.reference_near_fraction = near_fraction;
    config.reference_near_offset_max = near_offset;
    config.course_change_max = heading_max_deg * fw::PI / 180;
    config.reference_distance_max = distance_max;
    config.wind_speed_max = 6;
    config.turbulence_intensity_max = 1;
    config.gust_rate = 1;
    config.thermal_fraction = T(0.5);
    config.reference_turn_fraction = T(0.5);
    config.reference_air_relative = air_relative;
    config.randomization_mass = T(0.1);
    config.randomization_thrust = T(0.15);
    config.randomization_aerodynamics = T(0.1);
    config.latency_steps_random_max = 2;
    config.episode_duration = duration;
    fw::TaskConfig<T> calm = config;
    calm.wind_speed_max = 0;
    calm.turbulence_intensity_max = 0;
    calm.gust_rate = 0;
    calm.thermal_fraction = 0;
    const TI n_steps = TI(std::lround(duration * config.frequency));
    std::ofstream csv(prefix + "_flights.csv");
    csv << "set,flight,termination,thermals,helix,wind";
    for(const auto& name: METRICS) csv << "," << name;
    csv << "\n";
    std::string disturbed = evaluate(actor, sample_flights(device, config, 12345, n_flights), config, n_steps, csv, "disturbed");
    std::string calm_result = evaluate(actor, sample_flights(device, calm, 12345, n_flights), calm, n_steps, csv, "calm");
    std::ofstream(prefix + ".json") << "{\n  \"actor\": \"" << std::filesystem::path(argv[1]).generic_string() << "\",\n  \"duration_s\": " << duration
                                    << ",\n  \"reference_air_relative\": " << (air_relative ? "true" : "false")
                                    << ",\n  \"disturbed\": " << disturbed << ",\n  \"calm\": " << calm_result << "\n}\n";
    std::cout << "disturbed: " << disturbed << "\ncalm: " << calm_result << std::endl;
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
