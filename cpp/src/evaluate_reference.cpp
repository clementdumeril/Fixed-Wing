// Flies a trained tracking actor (FW_TRACKING, FW_AIRPLANE=1) when the reference point accelerates: circles, helices,
// the figure-eight of Learning to Fly in Seconds and speed oscillations along a straight line (reference_motion.h), each
// with its acceleration swept from easy to beyond what the airplane can follow, next to the straight line at constant
// velocity of training and the tests. Every motion is flown from two kinds of starts:
//   level: test A of train_sac (trimmed attitude perturbed by up to 20 deg of roll and 10 of pitch, neutral commands)
//   random: test B, the training distribution (roll +-60 deg, pitch +-20, rates +-0.5 rad/s, airspeed 11-19 m/s, alpha
//   -4 to 10 deg, sideslip +-8, random commands)
// The starts are the first N flights of those tests (seed 12345), the same for every actor and every motion; only the
// path of the point changes. With the straight line the flights are those of the tests themselves.
//
// evaluate_reference <actor.h> <out_dir> [--flights N] [--record K] [--threads N] [--motions m1,m2,...]
//   actor.h: the actor written by train_sac (actor.h or best_actor.h, RLtools code export); its weights are read from
//   the file at run time, so that one build evaluates any run
//   out_dir gets reference_suite.json (one entry per start, motion and level), flights_summary.csv (one row per flight)
//   and flights/<start>_<motion>_<level>.csv (the first K flights of each level, step by step, for plots)
//   --motions: a subset of straight, circle, helix, figure_eight, speed, speed_range (default all)
//   --fine-position-scale: the S of the run (config.json, fine_position_scale), with evaluate_reference_fine for the
//   actors of the _fine targets (default 2)
#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>
#include <fixed_wing/reference_motion.h>

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
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

static_assert(FW_TRACKING && FW_AIRPLANE == 1, "build with FW_TRACKING=1 FW_AIRPLANE=1 (CMakeLists.txt)");

// ----- the actor: the MLP of train_sac (two hidden layers of 64, ReLU), tanh of the action means -----------------------
struct Actor{
    static constexpr int INPUT = ENVIRONMENT::Observation::DIM, HIDDEN = 64, OUTPUT = 2 * fw::N_ACTUATORS;
    std::vector<float> W0, b0, W1, b1, W2, b2;  // row-major W[out][in]

    void act(const float observation[INPUT], T action[fw::N_ACTUATORS]) const {
        // the same operations as tools/export_policy.py: deterministic action, as the evaluations of train_sac
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
        for(int i = 0; i < fw::N_ACTUATORS; i++){  // the first half of the outputs: the means (then the log std)
            float x = b2[i];
            for(int j = 0; j < HIDDEN; j++) x += W2[i * HIDDEN + j] * h1[j];
            action[i] = std::tanh(x);
        }
    }
};

Actor load_actor(const std::string& path){
    // the parameter arrays of the RLtools code export, in order: weights then biases of the input, hidden and output
    // layers, as hexadecimal float literals ("constexpr float memory[] = {0X1.10A4DCP-3f, ...};")
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
            float v = std::strtof(p, &next);  // reads the hexadecimal literal, stops at its 'f' suffix
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
    if(!ok) throw std::runtime_error("unexpected actor layout in " + path + " (62 inputs, 64 x 64, 8 outputs expected)");
    return Actor{arrays[0], arrays[1], arrays[2], arrays[3], arrays[4], arrays[5]};
}

// ----- the suite ---------------------------------------------------------------------------------------------------
struct Level{
    std::string motion;  // straight, circle, helix, figure_eight, speed
    fw::MotionKind kind;
    T acceleration;  // m/s^2, the difficulty (0 for the straight line)
    T climb;  // m/s (helix)
    T speed_amplitude;  // m/s, speed oscillations: the speed stays within 15 +- this
};

std::vector<Level> suite(){
    // circles: the radii of a 15 m/s turn from 300 m (0.75 m/s^2, 4 deg of bank) to 4.5 m (50 m/s^2, 79 deg); the
    // acceleration is the same at every speed of the point, the radius scales with its speed squared. The airplane
    // has a thrust of 2.2 times its weight and stalls near 6 m/s, so its own limit is far down this list
    const T radii[] = {300, 200, 120, 80, 60, 45, 35, 25, 18, 13, 9, T(6.5), T(4.5)};
    std::vector<Level> levels{{"straight", fw::STRAIGHT, 0, 0, 0}};
    for(T R: radii) levels.push_back({"circle", fw::CIRCLE, 225 / R, 0, 0});
    for(T R: {T(120), T(60), T(35), T(18), T(9)}) levels.push_back({"helix", fw::HELIX, 225 / R, 2, 0});  // twice the training climb
    for(T R: radii) levels.push_back({"figure_eight", fw::FIGURE_EIGHT, 225 / R, 0, 0});  // peak lateral acceleration
    // speed oscillations across the speed range of the task (12-18 m/s), faster and faster; then at 2 m/s^2 across wider
    // ranges, down to 6 m/s, below the lowest airspeed of a flight (7 m/s, TaskConfig::airspeed_limits)
    for(T a: {T(0.25), T(0.5), T(1), T(2), T(4), T(8)}) levels.push_back({"speed", fw::SPEED_OSCILLATION, a, 0, 3});
    for(T amplitude: {T(3), T(5), T(7), T(9)}) levels.push_back({"speed_range", fw::SPEED_OSCILLATION, 2, 0, amplitude});
    return levels;
}

std::vector<State> sampled_starts(DEVICE& device, TI seed, fw::StartMode start, TI n){
    // as train_sac's sampled_starts: the first n flights of its tests
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config.start = start;
    ENVIRONMENT::Parameters p;
    fw::task_parameters(env.config, p);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    std::vector<State> states(n);
    for(auto& s: states) rlt::sample_initial_state(device, env, p, s, rng);
    rlt::free(device, rng);
    return states;
}

fw::ReferenceMotion<T> motion_of(const Level& level, const State& start, TI k, const fw::TaskConfig<T>& c){
    // the motion of flight k from the point of its sampled start; turns alternate left and right
    fw::ReferenceMotion<T> m;
    m.kind = level.kind;
    for(int i = 0; i < 3; i++) m.origin[i] = start.reference_position[i];
    m.heading = std::atan2(start.reference_velocity[1], start.reference_velocity[0]);
    m.speed = std::hypot(start.reference_velocity[0], start.reference_velocity[1]);
    m.acceleration = level.acceleration;
    m.turn = k % 2 == 0 ? 1 : -1;
    m.speed_mid = (c.reference_speed_range[0] + c.reference_speed_range[1]) / 2;
    m.speed_amplitude = level.speed_amplitude;
    if(level.kind == fw::FIGURE_EIGHT){
        // mean speed at the middle of the range (15 m/s): the speed along the eight spans 0.68 to 1.46 times the mean,
        // 10.2 to 21.9 m/s, already wider than the range of the task; the start along it is spread over the flights
        m.speed = m.speed_mid;
        m.phase = T(2 * fw::PI) * std::fmod(T(k) * T(0.6180339887498949), T(1));
    }
    // helix: climb from the lower half of the altitude range, descend from the upper half (60 m over the flight)
    T middle = (c.reference_altitude_range[0] + c.reference_altitude_range[1]) / 2;
    m.climb = level.kind == fw::HELIX ? (m.origin[2] < middle ? level.climb : -level.climb) : 0;
    return m;
}

struct Flight{
    int reason = 0, steps = 0;
    double position_rmse = 0, velocity_rmse = 0, along = 0, cross = 0, vertical = 0;  // second half of the flight
    double mean_bank = 0, max_bank = 0, max_alpha = -1e9;  // deg, second half of the flight
    double action_change = 0, flight_max_alpha = -1e9, min_airspeed = 1e9, throttle_full = 0, surface_limit = 0;  // whole flight
};

struct Description{ double bank_deg, radius_15, period_15, peak_speed, min_speed, peak_total, peak_tangential; };

Description describe(const Level& level, const fw::TaskConfig<T>& c){
    // the motion at the middle of the speed range (15 m/s), for the tables
    const double g = 9.8, V = (c.reference_speed_range[0] + c.reference_speed_range[1]) / 2;
    Description d{std::atan(level.acceleration / g) * 180 / fw::PI, 0, 0, V, V, level.acceleration, 0};
    if(level.kind == fw::CIRCLE || level.kind == fw::HELIX){
        d.radius_15 = V * V / level.acceleration;
        d.period_15 = 2 * fw::PI * d.radius_15 / V;
    }
    else if(level.kind == fw::FIGURE_EIGHT){
        static const fw::FigureEightFactors f = fw::figure_eight_factors();
        double w = level.acceleration * f.mean_speed / (V * f.peak_lateral), A = V / (w * f.mean_speed);
        d.period_15 = 2 * fw::PI / w;
        d.radius_15 = A;  // the half-width of the eight
        d.peak_speed = A * w * f.max_speed;
        d.min_speed = A * w * f.min_speed;
        d.peak_total = A * w * w * f.peak_total;
        d.peak_tangential = A * w * w * f.peak_tangential;
    }
    else if(level.kind == fw::SPEED_OSCILLATION){
        d.bank_deg = 0;
        d.period_15 = 2 * fw::PI * level.speed_amplitude / level.acceleration;
        d.peak_speed = V + level.speed_amplitude;
        d.min_speed = V - level.speed_amplitude;
        d.peak_tangential = level.acceleration;
    }
    return d;
}

Flight fly(const Actor& actor, const ENVIRONMENT::Parameters& p, const fw::TaskConfig<T>& config, const Level& level,
           State s, TI k, std::ostream* csv){
    const T deg = 180 / fw::PI, dt = p.dt;
    fw::ReferenceMotion<T> motion = motion_of(level, s, k, config);
    T position[3], velocity[3], acceleration[3];
    if(level.kind != fw::STRAIGHT){
        // the point at t = 0 of its motion: the velocity (observed) and the setpoints, as start_state sets them
        fw::reference_at(motion, T(0), position, velocity, acceleration);
        for(int i = 0; i < 3; i++){ s.reference_position[i] = position[i]; s.reference_velocity[i] = velocity[i]; }
        s.setpoint_airspeed = fw::norm3(velocity);
        s.setpoint_course = std::atan2(velocity[1], velocity[0]);
        s.setpoint_altitude = position[2];
    }
    else{
        for(int i = 0; i < 3; i++) acceleration[i] = 0;
    }
    auto record = [&](T t, const State& x, const T a[4], int reason){
        if(!csv) return;
        auto m = fw::measurements(p, x.body);
        auto air = fw::air_data(p, x.body);
        *csv << k << "," << t;
        for(int i = 0; i < 3; i++) *csv << "," << x.body.position[i];
        *csv << "," << std::atan2(m.R[2][1], m.R[2][2]) * deg << "," << std::asin(fw::clip(m.R[2][0], T(-1), T(1))) * deg
             << "," << std::atan2(m.R[1][0], m.R[0][0]) * deg << "," << air.airspeed << "," << air.alpha * deg << ","
             << air.beta * deg << "," << fw::norm3(x.body.linear_velocity);
        for(int i = 0; i < 3; i++) *csv << "," << x.reference_position[i];
        for(int i = 0; i < 3; i++) *csv << "," << x.reference_velocity[i];
        for(int i = 0; i < 4; i++) *csv << "," << a[i];
        *csv << "," << reason << "\n";
    };
    T previous[4];
    for(int i = 0; i < 4; i++) previous[i] = s.action_history[fw::ACTION_HISTORY - 1][i];
    record(0, s, previous, 0);
    Flight f;
    struct Error{ double position, velocity, along, cross, vertical, bank, alpha; };
    std::vector<Error> errors;
    errors.reserve(ENVIRONMENT::EPISODE_STEP_LIMIT);
    for(TI step = 0; step < ENVIRONMENT::EPISODE_STEP_LIMIT; step++){
        float observation[Actor::INPUT];
        fw::observe_actor(p, s, [&](int j, T v){ observation[j] = float(v); });
        T a[4];
        actor.act(observation, a);
        State next;
        if(level.kind == fw::STRAIGHT) fw::simulate_task(p, s, a, next);  // the tests themselves
        else{
            fw::reference_at(motion, T(step + 1) * dt, position, velocity, acceleration);
            fw::simulate_task(p, s, a, position, velocity, next);
        }
        f.reason = fw::termination(p, next, config);
        // error to the point in its own frame: along its horizontal velocity (> 0: the airplane is ahead of it), across
        // it (> 0: on the side the point turns to, left when it flies straight), and vertical (> 0: above it)
        T e_position[3], e_velocity[3];
        fw::tracking_errors(next, e_position, e_velocity);
        T vh = std::hypot(next.reference_velocity[0], next.reference_velocity[1]);
        T u[2] = {next.reference_velocity[0] / vh, next.reference_velocity[1] / vh};
        T lateral = u[0] * acceleration[1] - u[1] * acceleration[0];
        T side = lateral < -1e-9 ? -1 : 1;
        auto m = fw::measurements(p, next.body);
        auto air = fw::air_data(p, next.body);
        errors.push_back({fw::norm3(e_position), fw::norm3(e_velocity), e_position[0] * u[0] + e_position[1] * u[1],
                          side * (-e_position[0] * u[1] + e_position[1] * u[0]), e_position[2],
                          std::abs(std::atan2(m.R[2][1], m.R[2][2])) * deg, air.alpha * deg});
        f.flight_max_alpha = std::max(f.flight_max_alpha, air.alpha * deg);
        f.min_airspeed = std::min(f.min_airspeed, air.airspeed);
        double change = 0;
        for(int i = 0; i < 4; i++){ change += std::abs(a[i] - previous[i]); previous[i] = a[i]; }
        f.action_change += change / 4;
        f.throttle_full += a[fw::THROTTLE] > T(0.98);
        f.surface_limit += std::abs(a[fw::AILERON]) > T(0.98) || std::abs(a[fw::ELEVATOR]) > T(0.98) || std::abs(a[fw::RUDDER]) > T(0.98);
        record(T(step + 1) * dt, next, a, f.reason);
        s = next;
        if(f.reason != fw::FLYING) break;
    }
    f.steps = int(errors.size());
    f.action_change /= f.steps;
    f.throttle_full /= f.steps;
    f.surface_limit /= f.steps;
    // the second half of the flight, as the evaluations of train_sac: the approach from the start is over (the bank
    // of the first turn toward the point, up to 60-70 deg even with a straight reference, is left out)
    size_t half = errors.size() / 2;
    double n = double(errors.size() - half);
    for(size_t t = half; t < errors.size(); t++){
        f.position_rmse += errors[t].position * errors[t].position / n;
        f.velocity_rmse += errors[t].velocity * errors[t].velocity / n;
        f.along += errors[t].along / n;
        f.cross += errors[t].cross / n;
        f.vertical += errors[t].vertical / n;
        f.mean_bank += errors[t].bank / n;
        f.max_bank = std::max(f.max_bank, errors[t].bank);
        f.max_alpha = std::max(f.max_alpha, errors[t].alpha);
    }
    f.position_rmse = std::sqrt(f.position_rmse);
    f.velocity_rmse = std::sqrt(f.velocity_rmse);
    return f;
}

constexpr double TRACKED_RMSE = 20;  // m: a completed flight that stays within this RMS distance follows the point

std::string level_json(const std::string& start, const Level& level, const Description& d, const std::vector<Flight>& flights){
    const char* names[] = {"", "diverged", "low_altitude", "alpha", "airspeed", "altitude", "position"};
    std::map<std::string, int> failures;
    int complete = 0, tracked = 0, early = 0;
    // means over the completed flights (the errors of a flight cut short are not comparable), and over all of them
    double sum[11] = {}, all[4] = {};
    for(const auto& f: flights){
        all[0] += f.steps * T(0.02);
        all[1] += f.flight_max_alpha;
        all[2] += f.min_airspeed;
        all[3] += f.throttle_full;
        if(f.reason != 0){
            failures[names[f.reason]]++;
            early += f.steps <= 250;
            continue;
        }
        complete++;
        tracked += f.position_rmse < TRACKED_RMSE;
        double v[11] = {f.position_rmse, f.velocity_rmse, f.along, f.cross, f.vertical, f.action_change, f.mean_bank,
                        f.max_bank, f.max_alpha, f.surface_limit, f.throttle_full};
        for(int i = 0; i < 11; i++) sum[i] += v[i];
    }
    double n = std::max(complete, 1), N = double(flights.size());
    std::ostringstream json;
    json << std::setprecision(6) << "    {\"start\": \"" << start << "\", \"motion\": \"" << level.motion
         << "\", \"acceleration\": " << level.acceleration << ", \"equivalent_bank_deg\": " << d.bank_deg
         << ", \"climb\": " << level.climb << ", \"speed_amplitude\": " << level.speed_amplitude << ", \"radius_at_15\": " << d.radius_15 << ", \"period_at_15\": " << d.period_15
         << ", \"peak_speed_at_15\": " << d.peak_speed << ", \"min_speed_at_15\": " << d.min_speed
         << ", \"peak_total_acceleration_at_15\": " << d.peak_total << ", \"peak_tangential_acceleration_at_15\": " << d.peak_tangential
         << ",\n     \"flights\": " << flights.size() << ", \"success_rate\": " << complete / N << ", \"tracked_rate\": " << tracked / N
         << ", \"failures\": {";
    bool first = true;
    for(auto& [name, count]: failures){ json << (first ? "" : ", ") << "\"" << name << "\": " << count; first = false; }
    json << "}, \"failures_within_5_s\": " << early
         << ",\n     \"position_rmse\": " << sum[0] / n << ", \"velocity_rmse\": " << sum[1] / n
         << ", \"along_track_mean\": " << sum[2] / n << ", \"cross_track_inside_mean\": " << sum[3] / n
         << ", \"vertical_mean\": " << sum[4] / n << ", \"mean_action_change\": " << sum[5] / n
         << ",\n     \"second_half_mean_bank_deg\": " << sum[6] / n << ", \"second_half_max_bank_deg\": " << sum[7] / n
         << ", \"second_half_max_alpha_deg\": " << sum[8] / n << ", \"surface_limit_fraction\": " << sum[9] / n
         << ", \"throttle_full_fraction\": " << sum[10] / n
         << ",\n     \"all_flights_mean_duration_s\": " << all[0] / N << ", \"all_flights_max_alpha_deg\": " << all[1] / N
         << ", \"all_flights_min_airspeed\": " << all[2] / N << ", \"all_flights_throttle_full_fraction\": " << all[3] / N << "}";
    return json.str();
}

int run(int argc, char** argv){
    if(argc < 3){
        std::cerr << "usage: evaluate_reference <actor.h> <out_dir> [--flights N] [--record K] [--threads N] [--motions m1,m2,...]"
                     " [--fine-position-scale S]" << std::endl;
        return 1;
    }
    Actor actor = load_actor(argv[1]);
    std::filesystem::path out_dir = argv[2];
    TI n_flights = 50, n_recorded = 2;
    unsigned n_threads = std::max(1u, std::thread::hardware_concurrency());
    std::string motions = "straight,circle,helix,figure_eight,speed,speed_range";
    T fine_scale = 2;
    for(int i = 3; i < argc; i++){
        std::string option = argv[i];
        if(option == "--flights" && i + 1 < argc) n_flights = std::stoul(argv[++i]);
        else if(option == "--record" && i + 1 < argc) n_recorded = std::stoul(argv[++i]);
        else if(option == "--threads" && i + 1 < argc) n_threads = std::stoul(argv[++i]);
        else if(option == "--motions" && i + 1 < argc) motions = "," + std::string(argv[++i]) + ",";
        else if(option == "--fine-position-scale" && i + 1 < argc) fine_scale = std::stod(argv[++i]);
        else throw std::runtime_error("unknown option " + option);
    }
    if(!FW_FINE_POSITION && fine_scale != 2) throw std::runtime_error("--fine-position-scale needs evaluate_reference_fine");
    if(motions.front() != ',') motions = "," + motions + ",";
    std::filesystem::create_directories(out_dir / "flights");

    DEVICE device;
    rlt::init(device);
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config.fine_position_scale = fine_scale;
    ENVIRONMENT::Parameters p;
    fw::task_parameters(env.config, p);
    const std::pair<std::string, fw::StartMode> starts[] = {{"level", fw::NEUTRAL}, {"random", fw::RANDOM}};
    std::vector<std::vector<State>> start_states;
    for(auto& [name, mode]: starts) start_states.push_back(sampled_starts(device, 12345, mode, n_flights));
    std::vector<Level> levels;
    for(const Level& level: suite()) if(motions.find("," + level.motion + ",") != std::string::npos) levels.push_back(level);

    // one job per start and level, spread over the threads; each writes its own results and its own flights file
    struct Job{ int start, level, index; };
    std::vector<Job> jobs;
    std::map<std::string, int> index_in_motion;
    std::vector<int> level_index(levels.size());
    for(size_t l = 0; l < levels.size(); l++) level_index[l] = index_in_motion[levels[l].motion]++;
    for(int s = 0; s < 2; s++) for(size_t l = 0; l < levels.size(); l++) jobs.push_back({s, int(l), level_index[l]});
    std::vector<std::vector<Flight>> results(jobs.size());
    std::atomic<size_t> next_job{0};
    auto worker = [&](){
        for(size_t j = next_job++; j < jobs.size(); j = next_job++){
            const Job& job = jobs[j];
            const Level& level = levels[job.level];
            std::ofstream csv(out_dir / "flights" / (starts[job.start].first + "_" + level.motion + "_" + std::to_string(job.index) + ".csv"));
            csv << std::setprecision(8) << "flight,t,x,y,altitude,roll_deg,pitch_deg,heading_deg,airspeed,alpha_deg,beta_deg,"
                   "ground_speed,reference_x,reference_y,reference_altitude,reference_vx,reference_vy,reference_vz,"
                   "aileron,elevator,rudder,throttle,termination\n";
            for(TI k = 0; k < n_flights; k++){
                results[j].push_back(fly(actor, p, env.config, level, start_states[job.start][k], k, k < n_recorded ? &csv : nullptr));
            }
        }
    };
    std::vector<std::thread> threads;
    for(unsigned t = 0; t < n_threads; t++) threads.emplace_back(worker);
    for(auto& t: threads) t.join();

    std::ofstream json(out_dir / "reference_suite.json");
    json << "{\n  \"actor\": \"" << std::filesystem::path(argv[1]).filename().string() << "\", \"flights_per_level\": " << n_flights
         << ", \"tracked_rmse_m\": " << TRACKED_RMSE << ",\n  \"levels\": [\n";
    std::ofstream summary(out_dir / "flights_summary.csv");
    summary << "start,motion,level,acceleration,flight,termination,steps,position_rmse,velocity_rmse,along_track,"
               "cross_track_inside,vertical,action_change,mean_bank_deg,max_bank_deg,max_alpha_deg,flight_max_alpha_deg,"
               "min_airspeed,throttle_full_fraction\n";
    for(size_t j = 0; j < jobs.size(); j++){
        const Level& level = levels[jobs[j].level];
        const std::string& start = starts[jobs[j].start].first;
        json << level_json(start, level, describe(level, env.config), results[j]) << (j + 1 < jobs.size() ? ",\n" : "\n");
        for(TI k = 0; k < results[j].size(); k++){
            const Flight& f = results[j][k];
            summary << start << "," << level.motion << "," << jobs[j].index << "," << level.acceleration << "," << k << ","
                    << f.reason << "," << f.steps << "," << f.position_rmse << "," << f.velocity_rmse << "," << f.along << ","
                    << f.cross << "," << f.vertical << "," << f.action_change << "," << f.mean_bank << "," << f.max_bank << ","
                    << f.max_alpha << "," << f.flight_max_alpha << "," << f.min_airspeed << "," << f.throttle_full << "\n";
        }
    }
    json << "  ]\n}\n";
    std::cout << "wrote " << (out_dir / "reference_suite.json").string() << " (" << jobs.size() << " levels x " << n_flights << " flights)" << std::endl;
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
