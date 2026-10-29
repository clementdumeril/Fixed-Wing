// SAC on the fixed-wing task with RLtools: same task, network sizes and hyperparameters as python/examples/train_sac.py
// (asymmetric actor-critic, gated reward curriculum with replay-buffer recomputation), then an evaluation on the
// 100 test scenarios of the Python environment and on 100 random starts.
//
// train_sac <scenarios.txt> <out_dir> [seed] [steps] [--alpha-weight W] [--start trimmed|neutral|random] [--no-trim-reward]
//           [--huber D] [--position-scale S] [--position-linear W]
//   scenarios.txt: tools/export_scenarios.py; out_dir gets config.json, progress.csv, evaluation.json (Python
//   scenarios) and evaluation_random.json (random starts, the same for every run), flights.csv and flights_random.csv
//   (the first 10 test flights of each, step by step) and actor.h (the trained actor as C++ code)
//   --alpha-weight: weight of the reward term that keeps a margin to the stall (RewardConfig::alpha_margin_weight)
//   --start: how the training episodes start (TaskConfig::start)
//   --no-trim-reward: no trim term in the reward, so that the policy finds trim on its own
//   --huber: Huber instead of square on the scaled tracking errors (RewardConfig::huber_delta)
//   --position-scale, --position-linear: tracking task, scale of the position cost and weight of the linear distance
//   cost (RewardConfig::position_scale, position_linear_weight)
// The train_sac_*_integral targets also observe the time integrals of the errors (FW_ERROR_INTEGRAL).
// The train_sac_*_tracking targets learn the position and velocity tracking task (FW_TRACKING): scenarios.txt is
// then not used, and test A is 100 normal flights (neutral start) instead of the Python scenarios.
#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>

#include <rl_tools/nn/optimizers/adam/instance/operations_generic.h>
#include <rl_tools/nn/operations_cpu_mux.h>
#include <rl_tools/nn/layers/sample_and_squash/operations_generic.h>
#include <rl_tools/nn_models/mlp/operations_generic.h>
#include <rl_tools/nn_models/random_uniform/operations_generic.h>
#include <rl_tools/nn_models/sequential/operations_generic.h>
#include <rl_tools/nn/optimizers/adam/operations_generic.h>

#include <rl_tools/rl/algorithms/sac/loop/core/config.h>
#include <rl_tools/rl/algorithms/sac/loop/core/operations_generic.h>

// export of the trained actor as C++ code (weights included), the checkpoint format of L2F
#include <rl_tools/numeric_types/persist_code.h>
#include <rl_tools/containers/tensor/persist_code.h>
#include <rl_tools/nn/layers/dense/persist_code.h>
#include <rl_tools/nn/layers/sample_and_squash/persist_code.h>
#include <rl_tools/nn_models/mlp/persist_code.h>
#include <rl_tools/nn_models/sequential/persist_code.h>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

namespace rlt = rl_tools;

using DEVICE = rlt::devices::DEVICE_FACTORY<>;
using RNG = DEVICE::SPEC::RANDOM::ENGINE<>;
using T = float;  // networks
using T_ENV = double;  // simulation
using TYPE_POLICY = rlt::numeric_types::Policy<T>;
using TI = typename DEVICE::index_t;
using ENVIRONMENT = fixed_wing::Environment<fixed_wing::Specification<T_ENV, TI>>;

// RLtools stops training silently past STEP_LIMIT: keep it above any step count passed on the command line
#ifndef STEP_LIMIT_MAX
#define STEP_LIMIT_MAX 10000000
#endif
// network and batch sizes: the Python run (critic 256, batch 256) by default; L2F uses about 64 and 64
#ifndef FW_CRITIC_HIDDEN
#define FW_CRITIC_HIDDEN 256
#endif
#ifndef FW_BATCH
#define FW_BATCH 256
#endif

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr T GAMMA = 0.99;
        static constexpr TI ACTOR_BATCH_SIZE = FW_BATCH;
        static constexpr TI CRITIC_BATCH_SIZE = FW_BATCH;
        static constexpr T ALPHA = 1.0;  // initial entropy coefficient, as stable-baselines3 "auto"
    };
    static constexpr TI N_ENVIRONMENTS = 1;
    static constexpr TI N_WARMUP_STEPS = 10000;  // uniform random actions, as learning_starts
    static constexpr TI N_WARMUP_STEPS_CRITIC = 10000;
    static constexpr TI N_WARMUP_STEPS_ACTOR = 10000;
    static constexpr TI STEP_LIMIT = STEP_LIMIT_MAX;
    static constexpr TI REPLAY_BUFFER_CAP = 1000000;
    static constexpr TI ACTOR_HIDDEN_DIM = 64;
    static constexpr TI ACTOR_NUM_LAYERS = 3;  // two hidden layers
    static constexpr TI CRITIC_HIDDEN_DIM = FW_CRITIC_HIDDEN;
    static constexpr TI CRITIC_NUM_LAYERS = 3;
    static constexpr auto ACTOR_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    static constexpr auto CRITIC_ACTIVATION_FUNCTION = rlt::nn::activation_functions::ActivationFunction::RELU;
    struct ACTOR_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr T ALPHA = 3e-4;
    };
    struct CRITIC_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr T ALPHA = 3e-4;
    };
    struct ALPHA_OPTIMIZER_PARAMETERS: rlt::nn::optimizers::adam::DEFAULT_PARAMETERS_TENSORFLOW<TYPE_POLICY>{
        static constexpr T ALPHA = 3e-4;
    };
};
using LOOP_CONFIG = rlt::rl::algorithms::sac::loop::core::Config<TYPE_POLICY, TI, RNG, ENVIRONMENT, LOOP_CORE_PARAMETERS>;
using LOOP_STATE = LOOP_CONFIG::State<LOOP_CONFIG>;

// ----- reward curriculum (environment.RewardCurriculum, gated as asymmetric_sac.CurriculumCallback) ---------------------
struct Curriculum{
    fixed_wing::RewardConfig<T_ENV> initial, target;
    TI period = 100000;
    int n_stages = 5;
    // distance to the reference [m], or course [rad]; the mean distance includes the approach from up to 50 m away
    T_ENV advance_error = FW_TRACKING ? 40 : 15 * fixed_wing::PI / 180;
    Curriculum(){
        target.airspeed_weight = 1.2;
        target.altitude_weight = 1.2;
        target.course_weight = 2.7;
        target.position_weight = 3.0;  // tracking task: about the altitude and course weights together
        target.velocity_weight = 1.2;
        target.action_change_weight = 1.0;
        target.sideslip_weight = 1.0;
        target.trim_weight = 0.3;
    }
    fixed_wing::RewardConfig<T_ENV> at_stage(int stage) const {
        T_ENV x = T_ENV(std::min(stage, n_stages)) / n_stages;
        // a weight kept constant (0 for --no-trim-reward) stays as it is: 0 * (0 / 0)^x would be NaN
        auto geometric = [x](T_ENV a, T_ENV b){ return a == b ? a : a * std::pow(b / a, x); };
        fixed_wing::RewardConfig<T_ENV> c = initial;
        c.airspeed_weight = geometric(initial.airspeed_weight, target.airspeed_weight);
        c.altitude_weight = geometric(initial.altitude_weight, target.altitude_weight);
        c.course_weight = geometric(initial.course_weight, target.course_weight);
        c.position_weight = geometric(initial.position_weight, target.position_weight);
        c.velocity_weight = geometric(initial.velocity_weight, target.velocity_weight);
        c.sideslip_weight = geometric(initial.sideslip_weight, target.sideslip_weight);
        c.action_change_weight = geometric(initial.action_change_weight, target.action_change_weight);
        c.trim_weight = geometric(initial.trim_weight, target.trim_weight);
        return c;
    }
};

template <typename BUFFER>
TI filled(const BUFFER& rb){ return rb.full ? BUFFER::CAPACITY : rb.position; }

template <typename BUFFER>
void recompute_rewards(DEVICE& device, BUFFER& rb, const fixed_wing::RewardConfig<T_ENV>& c){
    // L2F recomputes the stored rewards after each curriculum change (p. 6)
    constexpr TI N = ENVIRONMENT::ObservationPrivileged::DIM;
    T_ENV o[N], o2[N], a[ENVIRONMENT::ACTION_DIM];
    for(TI i = 0; i < filled(rb); i++){
        for(TI j = 0; j < N; j++){
            o[j] = rlt::get(rb.observations_privileged, i, j);
            o2[j] = rlt::get(rb.next_observations_privileged, i, j);
        }
        for(TI j = 0; j < ENVIRONMENT::ACTION_DIM; j++) a[j] = rlt::get(rb.actions, i, j);
        bool terminated = rlt::get(rb.terminated, i, 0) > T(0.5);
        rlt::set(rb.rewards, i, 0, T(fixed_wing::reward_from_critic_observations(o, o2, a, terminated, c)));
    }
}

// what the curriculum gate and progress.csv follow: course error [deg], or distance to the reference [m] (tracking)
constexpr const char* TASK_ERROR_NAME = FW_TRACKING ? "position error" : "course error";
constexpr const char* TASK_ERROR_UNIT = FW_TRACKING ? "m" : "deg";
constexpr double TASK_ERROR_TO_UNIT = FW_TRACKING ? 1.0 : 180 / fixed_wing::PI;

template <typename BUFFER>
double task_error(const BUFFER& rb, TI i){
    // of a stored step, from its next observation: course error [rad], or distance to the reference [m]
    if constexpr(FW_TRACKING){
        double sum = 0;
        for(TI j = 0; j < 3; j++){ double v = 20 * (double)rlt::get(rb.next_observations, i, j); sum += v * v; }
        return std::sqrt(sum);
    }
    else{
        return std::abs(std::atan2((double)rlt::get(rb.next_observations, i, 2), (double)rlt::get(rb.next_observations, i, 3)));
    }
}

struct Window{
    // statistics of the last n transitions of the replay buffer
    double task_error = 0, episode_length = 0, episode_return = 0;
};

template <typename BUFFER>
Window window(const BUFFER& rb, TI n){
    // task error over the last n transitions; length and return of the episodes that ended inside them
    Window w;
    TI available = std::min(n, filled(rb));
    double error = 0, episode_steps = 0, episode_rewards = 0, lengths = 0, returns = 0, episodes = 0;
    bool started = false;  // the first episode end seen opens the first complete episode
    for(TI k = available; k-- > 0;){  // oldest first
        TI i = (rb.position + BUFFER::CAPACITY - 1 - k) % BUFFER::CAPACITY;
        error += task_error(rb, i);
        episode_steps += 1;
        episode_rewards += rlt::get(rb.rewards, i, 0);
        if(rlt::get(rb.truncated, i, 0) > T(0.5)){
            if(started){ lengths += episode_steps; returns += episode_rewards; episodes += 1; }
            started = true;
            episode_steps = 0;
            episode_rewards = 0;
        }
    }
    if(available > 0) w.task_error = error / available;
    if(episodes > 0){
        w.episode_length = lengths / episodes;
        w.episode_return = returns / episodes;
    }
    return w;
}

// ----- evaluation on the Python test scenarios (environment.evaluate) --------------------------------------------------
struct Scenario{
    fixed_wing::Body<T_ENV> body;
    T_ENV airspeed, altitude, course;
};

std::vector<Scenario> load_scenarios(const std::string& path){
    std::ifstream in(path);
    if(!in) throw std::runtime_error("cannot read " + path);
    int n;
    in >> n;
    std::vector<Scenario> scenarios(n);
    for(auto& s: scenarios){
        for(auto& v: s.body.position) in >> v;
        for(auto& v: s.body.orientation) in >> v;
        for(auto& v: s.body.linear_velocity) in >> v;
        for(auto& v: s.body.angular_velocity) in >> v;
        for(auto& v: s.body.actuators) in >> v;
        in >> s.airspeed >> s.altitude >> s.course;
    }
    return scenarios;
}

constexpr TI N_SCENARIOS = 100;

std::vector<ENVIRONMENT::State> start_states(const std::vector<Scenario>& scenarios){
    // the Python scenarios as start states of the C++ task
    ENVIRONMENT env;
    ENVIRONMENT::Parameters p;
    fixed_wing::task_parameters(env.config, p);
    std::vector<ENVIRONMENT::State> states(scenarios.size());
    for(size_t k = 0; k < scenarios.size(); k++){
        const Scenario& s = scenarios[k];
        fixed_wing::start_state(p, s.body, s.airspeed, s.altitude, s.course, states[k]);
    }
    return states;
}

std::vector<ENVIRONMENT::State> sampled_starts(DEVICE& device, TI seed, fixed_wing::StartMode start){
    // test set sampled from the task itself: a fixed seed, so that every run is evaluated on the same flights
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config.start = start;
    ENVIRONMENT::Parameters p;
    fixed_wing::task_parameters(env.config, p);
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    std::vector<ENVIRONMENT::State> states(N_SCENARIOS);
    for(auto& s: states) rlt::sample_initial_state(device, env, p, s, rng);
    rlt::free(device, rng);
    return states;
}

constexpr TI N_RECORDED = 10;  // test flights written to the flights CSV, for plots

template <typename ACTOR>
std::string evaluate(DEVICE& device, const ACTOR& actor, const std::vector<ENVIRONMENT::State>& starts, RNG& rng,
                     const std::filesystem::path& flights_csv){
    using EVAL_ACTOR = typename ACTOR::template CHANGE_BATCH_SIZE<TI, N_SCENARIOS>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    typename EVAL_ACTOR::template State<true> actor_state;
    typename EVAL_ACTOR::template Buffer<true> actor_buffer;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_SCENARIOS, ENVIRONMENT::Observation::DIM>> observations;
    rlt::Matrix<rlt::matrix::Specification<T, TI, N_SCENARIOS, ENVIRONMENT::ACTION_DIM>> actions;
    rlt::malloc(device, actor_state);
    rlt::malloc(device, actor_buffer);
    rlt::malloc(device, observations);
    rlt::malloc(device, actions);
    rlt::reset(device, actor, actor_state, rng);

    ENVIRONMENT env;  // initial reward weights, the ones of the PID baseline
    ENVIRONMENT::Parameters p;
    fixed_wing::task_parameters(env.config, p);
    std::vector<ENVIRONMENT::State> states(N_SCENARIOS);
    std::vector<int> reason(N_SCENARIOS, 0);
    std::vector<double> returns(N_SCENARIOS, 0), change_sum(N_SCENARIOS, 0);
    // per step: airspeed, altitude and course errors, then distances to the reference position and velocity (tracking)
    std::vector<std::vector<std::array<double, 5>>> errors(N_SCENARIOS);
    std::vector<std::array<T_ENV, 4>> previous(N_SCENARIOS);
    std::vector<double> max_bank(N_SCENARIOS, 0);  // deg, over the flight
    for(TI k = 0; k < N_SCENARIOS; k++){
        states[k] = starts[k];
        for(int i = 0; i < 4; i++) previous[k][i] = states[k].action_history[fixed_wing::ACTION_HISTORY - 1][i];
    }
    std::ofstream flights(flights_csv);
    flights << "scenario,t,x,y,altitude,roll_deg,pitch_deg,heading_deg,airspeed,alpha_deg,beta_deg,course_deg,"
               "setpoint_airspeed,setpoint_altitude,setpoint_course_deg,reference_x,reference_y,reference_altitude,"
               "aileron,elevator,rudder,throttle,termination\n";
    auto record = [&](TI k, T_ENV t, const ENVIRONMENT::State& s, const T_ENV a[4], int termination){
        const T_ENV deg = 180 / fixed_wing::PI;
        auto m = fixed_wing::measurements(p, s.body);
        auto air = fixed_wing::air_data(p, s.body);
        flights << k << "," << t << "," << s.body.position[0] << "," << s.body.position[1] << "," << s.body.position[2]
                << "," << std::atan2(m.R[2][1], m.R[2][2]) * deg << "," << std::asin(fixed_wing::clip(m.R[2][0], T_ENV(-1), T_ENV(1))) * deg
                << "," << std::atan2(m.R[1][0], m.R[0][0]) * deg << "," << air.airspeed << "," << air.alpha * deg
                << "," << air.beta * deg << "," << m.course * deg << "," << s.setpoint_airspeed << "," << s.setpoint_altitude
                << "," << s.setpoint_course * deg;
        for(int i = 0; i < 3; i++) flights << "," << s.reference_position[i];  // 0 outside the tracking task
        for(int i = 0; i < 4; i++) flights << "," << a[i];
        flights << "," << termination << "\n";
    };
    for(TI step = 0; step < ENVIRONMENT::EPISODE_STEP_LIMIT; step++){
        for(TI k = 0; k < N_SCENARIOS; k++){
            fixed_wing::observe_actor(p, states[k], [&](int j, T_ENV v){ rlt::set(observations, k, j, T(v)); });
        }
        auto input = rlt::to_tensor(device, observations);
        auto output = rlt::to_tensor(device, actions);
        rlt::evaluate_step(device, actor, input, actor_state, output, actor_buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for(TI k = 0; k < N_SCENARIOS; k++){
            if(reason[k] != 0) continue;
            T_ENV a[4];
            for(int i = 0; i < 4; i++) a[i] = fixed_wing::clip(T_ENV(rlt::get(actions, k, i)), T_ENV(-1), T_ENV(1));
            ENVIRONMENT::State next;
            fixed_wing::simulate_task(p, states[k], a, next);
            returns[k] += fixed_wing::reward(p, states[k], a, next, env.config);
            reason[k] = fixed_wing::termination(p, next, env.config);
            auto m = fixed_wing::measurements(p, next.body);
            T_ENV ea, eh, ec, e_position[3], e_velocity[3];
            fixed_wing::errors(m, next, ea, eh, ec);
            fixed_wing::tracking_errors(next, e_position, e_velocity);
            errors[k].push_back({ea, eh, ec, fixed_wing::norm3(e_position), fixed_wing::norm3(e_velocity)});
            max_bank[k] = std::max(max_bank[k], std::abs(std::atan2(m.R[2][1], m.R[2][2])) * 180 / fixed_wing::PI);
            double change = 0;
            for(int i = 0; i < 4; i++){ change += std::abs(a[i] - previous[k][i]); previous[k][i] = a[i]; }
            change_sum[k] += change / 4;
            if(k < N_RECORDED) record(k, (step + 1) * p.dt, next, a, reason[k]);  // state after the step, commands of it
            states[k] = next;
        }
    }
    const char* names[] = {"", "diverged", "low_altitude", "alpha", "airspeed", "altitude", "position"};
    std::map<std::string, int> failures;
    double rms[5] = {0, 0, 0, 0, 0}, smoothness = 0, mean_return = 0, failed_bank = 0;
    int complete = 0, early = 0;
    for(TI k = 0; k < N_SCENARIOS; k++){
        mean_return += returns[k] / N_SCENARIOS;
        if(reason[k] != 0){
            failures[names[reason[k]]]++;
            early += errors[k].size() <= 250 ? 1 : 0;  // within the first 5 s
            failed_bank += max_bank[k];
            continue;
        }
        complete++;
        size_t half = errors[k].size() / 2;
        for(int j = 0; j < 5; j++){
            double sum = 0;
            for(size_t t = half; t < errors[k].size(); t++) sum += errors[k][t][j] * errors[k][t][j];
            rms[j] += std::sqrt(sum / (errors[k].size() - half));
        }
        smoothness += change_sum[k] / errors[k].size();
    }
    std::ostringstream json;
    json << std::setprecision(10) << "{\n  \"episodes\": " << N_SCENARIOS << ",\n  \"success_rate\": " << complete / double(N_SCENARIOS)
         << ",\n  \"mean_return\": " << mean_return << ",\n  \"max_return\": " << ENVIRONMENT::EPISODE_STEP_LIMIT << ",\n  \"failures\": {";
    bool first = true;
    for(auto& [name, count]: failures){ json << (first ? "" : ", ") << "\"" << name << "\": " << count; first = false; }
    json << "},\n  \"airspeed_rmse\": " << rms[0] / std::max(complete, 1) << ",\n  \"altitude_rmse\": " << rms[1] / std::max(complete, 1)
         << ",\n  \"course_rmse_deg\": " << rms[2] / std::max(complete, 1) * 180 / fixed_wing::PI;
    if constexpr(FW_TRACKING){
        json << ",\n  \"position_rmse\": " << rms[3] / std::max(complete, 1) << ",\n  \"velocity_rmse\": " << rms[4] / std::max(complete, 1);
    }
    json << ",\n  \"mean_action_change\": " << smoothness / std::max(complete, 1)
         << ",\n  \"failures_within_5_s\": " << early
         << ",\n  \"failed_flights_mean_max_bank_deg\": " << failed_bank / std::max(int(N_SCENARIOS) - complete, 1) << "\n}\n";
    rlt::free(device, actor_state);
    rlt::free(device, actor_buffer);
    rlt::free(device, observations);
    rlt::free(device, actions);
    return json.str();
}

template <typename ACTOR>
void save_actor(DEVICE& device, const ACTOR& actor, const std::filesystem::path& path){
    // the trained actor as C++ code with its weights, to fly it again (plots, deployment) without retraining
    using FORWARD = typename ACTOR::template CHANGE_BATCH_SIZE<TI, 1>::template CHANGE_CAPABILITY<rlt::nn::capability::Forward<true>>;
    FORWARD forward;
    rlt::malloc(device, forward);
    rlt::copy(device, device, actor, forward);
    std::ofstream(path) << rlt::save_code(device, forward, "fixed_wing::actor", true);
    rlt::free(device, forward);
}

int run(int argc, char** argv){
    if(argc < 3){
        std::cerr << "usage: train_sac <scenarios.txt> <out_dir> [seed] [steps] [--alpha-weight W] [--start trimmed|neutral|random] [--no-trim-reward] [--huber D] [--position-scale S] [--position-linear W]" << std::endl;
        return 1;
    }
    std::vector<Scenario> scenarios = load_scenarios(argv[1]);
    std::filesystem::path out_dir = argv[2];
    std::filesystem::create_directories(out_dir);
    TI seed = argc > 3 ? std::stoul(argv[3]) : 0;
    TI steps = argc > 4 ? std::stoul(argv[4]) : 1000000;
    if(steps > LOOP_CORE_PARAMETERS::STEP_LIMIT) throw std::runtime_error("steps above STEP_LIMIT_MAX, recompile with a larger one");
    T_ENV alpha_weight = 0;
    std::string start_name = "trimmed";
    bool trim_reward = true;
    T_ENV huber_delta = 0;
    fixed_wing::RewardConfig<T_ENV> defaults;
    T_ENV position_scale = defaults.position_scale, position_linear = defaults.position_linear_weight;
    for(int i = 5; i < argc; i++){
        std::string option = argv[i];
        if(option == "--alpha-weight" && i + 1 < argc) alpha_weight = std::stod(argv[++i]);
        else if(option == "--position-scale" && i + 1 < argc) position_scale = std::stod(argv[++i]);
        else if(option == "--position-linear" && i + 1 < argc) position_linear = std::stod(argv[++i]);
        else if(option == "--huber" && i + 1 < argc) huber_delta = std::stod(argv[++i]);
        else if(option == "--start" && i + 1 < argc) start_name = argv[++i];
        else if(option == "--no-trim-reward") trim_reward = false;
        else throw std::runtime_error("unknown option " + option);
    }
    if(position_linear > 0.3) throw std::runtime_error("--position-linear above 0.3: a step could get a negative reward");
    const std::map<std::string, fixed_wing::StartMode> start_modes = {
        {"trimmed", fixed_wing::TRIMMED}, {"neutral", fixed_wing::NEUTRAL}, {"random", fixed_wing::RANDOM}};
    if(!start_modes.count(start_name)) throw std::runtime_error("unknown start " + start_name);
    std::ofstream(out_dir / "config.json") << "{\n  \"seed\": " << seed << ",\n  \"steps\": " << steps
        << ",\n  \"critic_hidden\": " << FW_CRITIC_HIDDEN << ",\n  \"batch\": " << FW_BATCH
        << ",\n  \"alpha_margin_weight\": " << alpha_weight << ",\n  \"start\": \"" << start_name
        << "\",\n  \"trim_reward\": " << (trim_reward ? "true" : "false") << ",\n  \"huber_delta\": " << huber_delta
        << ",\n  \"error_integral\": " << (FW_ERROR_INTEGRAL ? "true" : "false")
        << ",\n  \"task\": \"" << (FW_TRACKING ? "tracking" : "course") << "\",\n  \"position_scale\": " << position_scale
        << ",\n  \"position_linear_weight\": " << position_linear << "\n}\n";

    DEVICE device;
    rlt::init(device);
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    Curriculum curriculum;
    curriculum.initial.alpha_margin_weight = alpha_weight;  // constant: not part of the curriculum
    curriculum.target.alpha_margin_weight = alpha_weight;
    curriculum.initial.huber_delta = huber_delta;
    curriculum.target.huber_delta = huber_delta;
    for(auto* c: {&curriculum.initial, &curriculum.target}){  // tracking task: constant over the curriculum
        c->position_scale = position_scale;
        c->position_linear_weight = position_linear;
    }
    if(!trim_reward){
        curriculum.initial.trim_weight = 0;
        curriculum.target.trim_weight = 0;
    }
    for(int s = 0; s <= curriculum.n_stages; s++){
        auto c = curriculum.at_stage(s);
        for(T_ENV w: {c.airspeed_weight, c.altitude_weight, c.course_weight, c.position_weight, c.velocity_weight,
                      c.sideslip_weight, c.action_change_weight, c.trim_weight, c.alpha_margin_weight}){
            if(!std::isfinite(w)) throw std::runtime_error("reward weight not finite at curriculum stage " + std::to_string(s));
        }
    }
    int stage = 0;
    auto set_reward = [&](const fixed_wing::RewardConfig<T_ENV>& c){
        for(TI i = 0; i < LOOP_CORE_PARAMETERS::N_ENVIRONMENTS; i++) rlt::get(ts.off_policy_runner.envs, 0, i).config.reward = c;
    };
    for(TI i = 0; i < LOOP_CORE_PARAMETERS::N_ENVIRONMENTS; i++) rlt::get(ts.off_policy_runner.envs, 0, i).config.start = start_modes.at(start_name);
    set_reward(curriculum.at_stage(0));

    std::ofstream progress(out_dir / "progress.csv");
    progress << "time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,"
             << (FW_TRACKING ? "train/position_error_m" : "train/course_error_deg") << ",curriculum/stage,time/time_elapsed\n";
    auto start = std::chrono::steady_clock::now();
    auto& rb = rlt::get(ts.off_policy_runner.replay_buffers, 0, 0);
    TI next_check = curriculum.period;
    for(TI step = 1; step <= steps; step++){
        rlt::step(device, ts);
        if(step % 10000 == 0){
            Window w = window(rb, 50000);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            progress << step << "," << w.episode_length << "," << w.episode_return << "," << w.task_error * TASK_ERROR_TO_UNIT
                     << "," << stage << "," << elapsed << std::endl;
            std::cout << "step " << step << "  episode length " << std::setprecision(4) << w.episode_length << "  return "
                      << w.episode_return << "  " << TASK_ERROR_NAME << " " << w.task_error * TASK_ERROR_TO_UNIT << " "
                      << TASK_ERROR_UNIT << "  stage " << stage << "  " << std::setprecision(3) << step / elapsed << " steps/s" << std::endl;
        }
        if(step >= next_check){
            // a stage is taken only if the setpoint was tracked over the last period
            if(window(rb, curriculum.period).task_error < curriculum.advance_error && stage < curriculum.n_stages){
                stage++;
                set_reward(curriculum.at_stage(stage));
                recompute_rewards(device, rb, curriculum.at_stage(stage));
            }
            next_check += curriculum.period;
        }
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    save_actor(device, rlt::get_actor(ts), out_dir / "actor.h");
    // test A: the Python scenarios, or for tracking normal flights (neutral start); test B: random starts
    std::vector<ENVIRONMENT::State> test_a = FW_TRACKING ? sampled_starts(device, 12345, fixed_wing::NEUTRAL) : start_states(scenarios);
    std::string result = evaluate(device, rlt::get_actor(ts), test_a, ts.rng, out_dir / "flights.csv");
    std::ofstream(out_dir / "evaluation.json") << result;
    std::string result_random = evaluate(device, rlt::get_actor(ts), sampled_starts(device, 12345, fixed_wing::RANDOM), ts.rng,
                                         out_dir / "flights_random.csv");
    std::ofstream(out_dir / "evaluation_random.json") << result_random;
    std::cout << result << "random starts:\n" << result_random << "training seconds: " << seconds << std::endl;
    rlt::free(device, ts);
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
