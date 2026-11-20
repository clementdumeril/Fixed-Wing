#include <algorithm>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>

namespace rlt = rl_tools;
using T = double;
using namespace fixed_wing;

int compare(const std::string& cases_path, const std::string& out_path){
    std::ifstream in(cases_path);
    std::ofstream out(out_path);
    if(!in || !out) throw std::runtime_error("cannot open " + cases_path + " or " + out_path);
    out << std::setprecision(17);
    TaskConfig<T> config;
    int n_wind_cases;
    in >> n_wind_cases;
    for(int c = 0; c < n_wind_cases; c++){
        Parameters<T> p;
        task_parameters(config, p);
        for(T& v: p.wind) in >> v;
        for(T& v: p.turbulence_sigma) in >> v;
        for(T& v: p.turbulence_length_scale) in >> v;
        in >> p.gust_count;
        for(int g = 0; g < p.gust_count; g++){
            Gust<T>& gust = p.gusts[g];
            in >> gust.start >> gust.duration;
            for(T& v: gust.peak) in >> v;
        }
        Thermals<T>& t = p.thermals;
        int sink;
        in >> t.count >> t.w_star >> t.z_i >> t.area >> sink >> t.downdraft_ring;
        t.sink = sink != 0;
        if(t.count > MAX_THERMALS) throw std::runtime_error("too many thermals");
        for(int k = 0; k < t.count; k++) in >> t.centers[k][0] >> t.centers[k][1];
        for(int k = 0; k < t.count; k++) in >> t.w_gain[k];
        for(int k = 0; k < t.count; k++) in >> t.r_gain[k];
        int n_queries;
        in >> n_queries;
        for(int q = 0; q < n_queries; q++){
            Body<T> x{};
            for(T& v: x.position) in >> v;
            for(T& v: x.orientation) in >> v;
            in >> x.time;
            for(T& v: x.turbulence) in >> v;
            T R[3][3], wind[3];
            rotation_matrix(x.orientation, R);
            local_wind(p, x, R, wind);
            T updraft = thermal_updraft(t, x.position[0] - p.wind[0] * x.time, x.position[1] - p.wind[1] * x.time, x.position[2]);
            out << wind[0] << " " << wind[1] << " " << wind[2] << " " << updraft << "\n";
        }
    }
    int n_sequences;
    in >> n_sequences;
    for(int c = 0; c < n_sequences; c++){
        T length_scale[3], state[N_TURBULENCE];
        for(T& v: length_scale) in >> v;
        for(T& v: state) in >> v;
        int n_steps;
        in >> n_steps;
        for(int k = 0; k < n_steps; k++){
            T distance, normals[N_TURBULENCE], next[N_TURBULENCE];
            in >> distance;
            for(T& v: normals) in >> v;
            advance_turbulence_states(length_scale, state, distance, normals, next);
            for(int i = 0; i < N_TURBULENCE; i++){
                state[i] = next[i];
                out << state[i] << (i + 1 < N_TURBULENCE ? " " : "\n");
            }
        }
    }
    if(!in) throw std::runtime_error("cases file ended early");
    return 0;
}

int sample(int n_episodes){
    using DEVICE = rlt::devices::DEVICE_FACTORY<>;
    using ENVIRONMENT = Environment<Specification<T, DEVICE::index_t>>;
    DEVICE device;
    ENVIRONMENT env;
    rlt::init(device, env);
    auto& c = env.config;
    c.wind_speed_max = 6;
    c.turbulence_intensity_max = 1;
    c.gust_rate = 1;
    c.thermal_fraction = T(0.5);
    c.reference_turn_fraction = T(0.5);
    c.randomization_mass = T(0.1);
    c.randomization_thrust = T(0.15);
    c.randomization_aerodynamics = T(0.1);
    c.latency_steps_random_max = 2;
    c.reference_near_fraction = 1;
    c.reference_near_offset_max = 5;
    c.episode_duration = 120;
    DEVICE::SPEC::RANDOM::ENGINE<> rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, 7);
    int with_thermals = 0, turning = 0, gusts = 0, latency[3] = {0, 0, 0};
    double thermal_count = 0, core_min = 1e9, core_max = -1e9, ring_min = 0, wind_max = 0, sigma_max = 0;
    double start_distance_max = 0, center_distance_sum = 0;
    for(int e = 0; e < n_episodes; e++){
        ENVIRONMENT::Parameters p;
        ENVIRONMENT::State s;
        rlt::sample_initial_parameters(device, env, p, rng);
        rlt::sample_initial_state(device, env, p, s, rng);
        latency[p.latency_steps]++;
        gusts += p.gust_count;
        turning += s.reference_motion.kind != STRAIGHT;
        wind_max = std::max(wind_max, std::hypot(p.wind[0], p.wind[1]));
        sigma_max = std::max(sigma_max, p.turbulence_sigma[0]);
        T e_position[3], e_velocity[3];
        tracking_errors(s, e_position, e_velocity);
        start_distance_max = std::max(start_distance_max, double(norm3(e_position)));
        if(p.thermals.count > 0){
            with_thermals++;
            thermal_count += p.thermals.count;
            for(int k = 0; k < p.thermals.count; k++){
                T core = thermal_updraft(p.thermals, p.thermals.centers[k][0], p.thermals.centers[k][1], T(90));
                core_min = std::min(core_min, double(core));
                core_max = std::max(core_max, double(core));
            }
            T r2 = allen_mean_radius(T(90), p.thermals.z_i) * p.thermals.r_gain[0];
            for(int j = 0; j <= 400; j++){
                T r = 3 * r2 * j / 400;
                ring_min = std::min(ring_min, double(thermal_updraft(p.thermals, p.thermals.centers[0][0] + r, p.thermals.centers[0][1], T(90))));
            }
            center_distance_sum += std::hypot(s.body.position[0], s.body.position[1]);
        }
    }
    std::cout << n_episodes << " sampled episodes (120 s), every start trimmed\n"
              << "  with thermals " << with_thermals << ", mean count " << thermal_count / std::max(with_thermals, 1)
              << ", updraft at the centres at 90 m between " << core_min << " and " << core_max << " m/s, lowest value across the first one "
              << ring_min << " m/s, mean start distance from the field centre " << center_distance_sum / std::max(with_thermals, 1) << " m\n"
              << "  turning references " << turning << ", gusts " << gusts << ", latency 0/1/2 steps " << latency[0] << "/" << latency[1] << "/" << latency[2]
              << "\n  largest wind " << wind_max << " m/s, largest longitudinal turbulence sigma " << sigma_max
              << " m/s, largest start distance to the point " << start_distance_max << " m" << std::endl;
    rlt::free(device, rng);
    return 0;
}

int run(int argc, char** argv){
    if(argc == 3 && std::string(argv[1]) == "--sample") return sample(std::stoi(argv[2]));
    if(argc != 3){
        std::cerr << "usage: check_wind <cases.txt> <out.txt> | check_wind --sample N" << std::endl;
        return 1;
    }
    return compare(argv[1], argv[2]);
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
