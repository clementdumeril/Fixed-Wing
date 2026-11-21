// SAC on the fixed-wing tasks with RLtools: asymmetric actor-critic, gated reward curriculum with replay-buffer
// recomputation, best checkpoint on validation flights, then the tests.
//
// train_sac <scenarios.txt> <out_dir> [seed] [steps] [options]
//   scenarios.txt: tools/export_scenarios.py, the 100 test scenarios of the Python environment (first task only).
//   out_dir gets config.json, progress.csv, the tests (evaluation*.json, the first 10 flights of each step by step in
//   flights*.csv) and actor.h (the trained actor as C++ code). steps counts airplane steps.
// Targets (CMakeLists.txt):
//   train_sac: the first task of python/examples/train_sac.py (airspeed, altitude, course) on the Aerosonde, with its
//   network sizes and hyperparameters. Test A: the Python scenarios; test B: 100 random starts (evaluation_random.json)
//   train_sac_*_tracking_*: reach and follow a point that flies straight (FW_TRACKING). Test A: 100 normal flights
//   (neutral start); test B: 100 random starts; evaluation_ahead.json: test B with the point within 45 deg ahead.
//   _px4plane: PX4's advanced_plane (FW_AIRPLANE=1); _eratix: the lab's E-flite UMX Eratix (FW_AIRPLANE=2,
//   docs/eratix_model.md); _i10a20: the critics updated every 10 steps and the actor every 20, as the L2F paper;
//   _fine: the position error also observed on a 2 m scale (FW_FINE_POSITION); _wind: wind, turbulence, gusts,
//   thermals and randomized airplanes (FW_DISTURBANCES), tested on 100 disturbed and 100 calm flights of
//   --eval-duration seconds (evaluation_disturbed.json, evaluation_calm.json)
// Options:
//   --start trimmed|neutral|random: how the training episodes start (TaskConfig::start)
//   --no-trim-reward: no trim term in the reward, so that the policy finds trim on its own
//   --alpha-weight W: weight of the reward term that keeps a margin to the stall (RewardConfig::alpha_margin_weight)
//   --position-scale S, --position-linear W: tracking task, scale of the position cost and weight of the linear distance
//   cost of the tanh reward (RewardConfig::position_scale, position_linear_weight)
//   --velocity-guidance K: tracking task, the velocity cost is on the error to the reference velocity plus K x the
//   position error toward the reference, at most 5 m/s (RewardConfig::velocity_guidance_gain)
//   --curriculum-period P: airplane steps between two checks of the curriculum gate (default 100000, as L2F)
//   --checkpoint-interval N: every N airplane steps, fly the actor on 100 validation flights (not the test flights) and
//   keep the best one: best_actor.h, its tests in evaluation_best*.json and checkpoints.csv. The score is the RMS error
//   of the completed flights (position, or course for the first task) + 10 per % of failed flights. The validation
//   flights have their own random numbers, so training is the same as without the option
//   --checkpoint-precise P N: the best checkpoint is the most precise one among those with a validation success of at
//   least P (fraction) after N airplane steps; if there is none, the default score among the others
//   --reward tanh|l2f-norm|simple: the reward form. tanh (default): survival minus a capped cost; l2f-norm: the form and
//   schedule of Learning to Fly in Seconds with the norms of the errors, as the L2F reward of RLtools at b32d9985
//   (survival 2 minus weighted norms, clipped at -2, a termination penalty of 2 / (1 - gamma), weights grown every
//   --curriculum-period; RewardConfig::l2f_form); simple: the reward of RLtools' L2F environment in meters (tracking
//   task, RewardConfig::simple_form)
//   --position-bonus W: with l2f-norm, a bonus W exp(-(d / s)^2) for being close to the point; its scale s starts at
//   20 m and follows half the mean training distance to the point, never growing and never below 1 m
//   --near-start F D: tracking task, a share F of the training starts puts the point within D m (horizontally and
//   vertically) instead of 50 m; the airplane state is drawn as usual. Validation and test flights are unchanged
//   --energy-weight W, --energy-ramp START END, --simple-constant C: --reward simple, the weight of the energy term
//   ramped from 0 to W between the two steps (default 1M and 3M), and the constant of the reward (default 35)
//   --distance-max D: tracking task, the flight ends more than D m from the point; --reference-heading-max DEG: spread of
//   the point's heading (tracking) or of the course setpoint (first task) around the airplane's course (default 90)
//   --eval-duration S, --calm, --wind-max V, --turbulence-max K, --gust-rate R, --thermal-fraction F, --turn-fraction F:
//   the _wind target, test duration (default 120 s) and the disturbances of the training, validation and test flights
//   (TaskConfig; --calm removes the wind, turbulence, gusts and thermals)
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
#include <cstdint>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <map>
#include <random>
#include <sstream>
#include <string>
#include <type_traits>
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
// discount factor: the critic sees about 1 / (1 - gamma) steps ahead, 100 steps = 2 s for 0.99
#ifndef FW_GAMMA
#define FW_GAMMA 0.99
#endif
// loop steps between two network updates (critics, actor and entropy coefficient, critic targets): fewer updates per
// step of experience, as the L2F configurations of RLtools (src/rl/zoo/l2f: every 5 to 16 steps) and the L2F paper
// (learning-to-fly src/config/config.h: the critics every 10 steps, the actor every 20)
#ifndef FW_TRAINING_INTERVAL
#define FW_TRAINING_INTERVAL 1
#endif
// the actor (and entropy coefficient) every FW_ACTOR_INTERVAL_FACTOR critic updates: 2 in the L2F paper, as in TD3
#ifndef FW_ACTOR_INTERVAL_FACTOR
#define FW_ACTOR_INTERVAL_FACTOR 1
#endif

struct LOOP_CORE_PARAMETERS: rlt::rl::algorithms::sac::loop::core::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT>{
    struct SAC_PARAMETERS: rlt::rl::algorithms::sac::DefaultParameters<TYPE_POLICY, TI, ENVIRONMENT::ACTION_DIM>{
        static constexpr T GAMMA = FW_GAMMA;
        static constexpr TI ACTOR_BATCH_SIZE = FW_BATCH;
        static constexpr TI CRITIC_BATCH_SIZE = FW_BATCH;
        static constexpr T ALPHA = 1.0;  // initial entropy coefficient, as stable-baselines3 "auto"
        static constexpr TI CRITIC_TRAINING_INTERVAL = FW_TRAINING_INTERVAL;
        static constexpr TI ACTOR_TRAINING_INTERVAL = FW_TRAINING_INTERVAL * FW_ACTOR_INTERVAL_FACTOR;
        static constexpr TI CRITIC_TARGET_UPDATE_INTERVAL = FW_TRAINING_INTERVAL;
    };
    static constexpr TI N_ENVIRONMENTS = 1;
    static constexpr TI N_WARMUP_STEPS = 10000;  // uniform random actions, as learning_starts
    static constexpr TI N_WARMUP_STEPS_CRITIC = N_WARMUP_STEPS;
    static constexpr TI N_WARMUP_STEPS_ACTOR = N_WARMUP_STEPS;
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
static_assert(!FW_DISTURBANCES || (FW_TRACKING && FW_AIRPLANE == 1), "the train_sac_*_wind targets fly the tracking task with the advanced_plane");

// ----- reward curriculum (environment.RewardCurriculum, gated as asymmetric_sac.CurriculumCallback) ---------------------
struct Curriculum{
    fixed_wing::RewardConfig<T_ENV> initial, target;
    TI period = 100000;
    int n_stages = 5;
    // distance to the reference [m], or course [rad]; the mean distance includes the approach from up to 50 m away
    // (Eratix: 20 m, so the gate is scaled with it)
    T_ENV advance_error = FW_TRACKING ? (FW_AIRPLANE == 2 ? 15 : 40) : 15 * fixed_wing::PI / 180;
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
    // --reward l2f-norm: the schedule of L2F (learning-to-fly src/steps/curriculum.h) instead of the gate: every period,
    // unconditionally, the position weight x1.2 and the velocity and action weights x1.4, each up to a cap, and the
    // stored rewards recomputed. L2F: position 2.5 to 20, velocity and action 0.005 to 0.5 (its errors are in meters
    // within a 0.6 m box); here the same factors on weights sized for our scaled errors (the start values keep the
    // reward of a start 50 m away near 0, as L2F's start costs stay below its survival reward)
    bool l2f = false;
    static constexpr T_ENV L2F_POSITION[2] = {0.3, 2.4}, L2F_VELOCITY[2] = {0.1, 1.0}, L2F_ACTION[2] = {0.05, 0.5};
    // the scale of the closeness bonus (--position-bonus): at every curriculum check it becomes bonus_adaptive times
    // the mean distance to the point over the last period of training flights (never larger than before, never below
    // bonus_scale_min), so that the bonus always reaches where the policy flies. bonus_scale is its current value
    T_ENV bonus_scale_min = 1, bonus_adaptive = 0.5, bonus_scale = 20;
    bool simple = false;
    fixed_wing::RewardConfig<T_ENV> at_stage(int stage) const {
        if(simple) return initial;
        if(l2f){
            fixed_wing::RewardConfig<T_ENV> c = initial;
            c.position_weight = std::fmin(L2F_POSITION[0] * std::pow(T_ENV(1.2), stage), L2F_POSITION[1]);
            c.velocity_weight = std::fmin(L2F_VELOCITY[0] * std::pow(T_ENV(1.4), stage), L2F_VELOCITY[1]);
            c.action_change_weight = std::fmin(L2F_ACTION[0] * std::pow(T_ENV(1.4), stage), L2F_ACTION[1]);
            c.position_bonus_scale = bonus_scale;
            return c;
        }
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

template <typename RUNNER>
void recompute_rewards(DEVICE& device, RUNNER& runner, const fixed_wing::RewardConfig<T_ENV>& c){
    // L2F recomputes the stored rewards after each curriculum change (p. 6)
    constexpr TI N = ENVIRONMENT::ObservationPrivileged::DIM;
    T_ENV o[N], o2[N], a[ENVIRONMENT::ACTION_DIM];
    auto& rb = rlt::get(runner.replay_buffers, 0, 0);
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

template <typename RUNNER>
double reward_recomputation_error(const RUNNER& runner, const fixed_wing::RewardConfig<T_ENV>& c){
    // largest difference between the stored rewards and the same rewards recomputed from the critic observations,
    // as the curriculum does: it must be ~0 (float) for the recomputation to be the reward of the environment
    constexpr TI N = ENVIRONMENT::ObservationPrivileged::DIM;
    T_ENV o[N], o2[N], a[ENVIRONMENT::ACTION_DIM];
    double worst = 0;
    const auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    for(TI i = 0; i < filled(rb); i++){
        for(TI j = 0; j < N; j++){
            o[j] = rlt::get(rb.observations_privileged, i, j);
            o2[j] = rlt::get(rb.next_observations_privileged, i, j);
        }
        for(TI j = 0; j < ENVIRONMENT::ACTION_DIM; j++) a[j] = rlt::get(rb.actions, i, j);
        bool terminated = rlt::get(rb.terminated, i, 0) > T(0.5);
        double r = fixed_wing::reward_from_critic_observations(o, o2, a, terminated, c);
        worst = std::max(worst, std::abs(r - double(rlt::get(rb.rewards, i, 0))));
    }
    return worst;
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
    double terminated_fraction = 0;
};

template <typename RUNNER>
Window window(const RUNNER& runner, TI n){
    // task error over the last n transitions; length and return of the episodes that ended inside them
    Window w;
    double error = 0, lengths = 0, returns = 0, episodes = 0, terminated = 0;
    const auto& rb = rlt::get(runner.replay_buffers, 0, 0);
    using BUFFER = std::decay_t<decltype(rb)>;
    TI available = std::min(n, filled(rb));
    double episode_steps = 0, episode_rewards = 0;
    bool started = false;  // the first episode end seen opens the first complete episode
    for(TI k = available; k-- > 0;){  // oldest first
        TI i = (rb.position + BUFFER::CAPACITY - 1 - k) % BUFFER::CAPACITY;
        error += task_error(rb, i);
        episode_steps += 1;
        episode_rewards += rlt::get(rb.rewards, i, 0);
        if(rlt::get(rb.truncated, i, 0) > T(0.5)){
            if(started){
                lengths += episode_steps;
                returns += episode_rewards;
                episodes += 1;
                terminated += rlt::get(rb.terminated, i, 0) > T(0.5) ? 1 : 0;
            }
            started = true;
            episode_steps = 0;
            episode_rewards = 0;
        }
    }
    if(available > 0) w.task_error = error / available;
    if(episodes > 0){
        w.episode_length = lengths / episodes;
        w.episode_return = returns / episodes;
        w.terminated_fraction = terminated / episodes;
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

std::vector<ENVIRONMENT::State> sampled_starts(DEVICE& device, TI seed, fixed_wing::StartMode start, T_ENV bearing_max = fixed_wing::PI){
    // test set sampled from the task itself: a fixed seed, so that every run is evaluated on the same flights
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config.start = start;
    env.config.reference_bearing_max = bearing_max;
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

struct EvaluationSummary{ double success_rate = 0, error = 0; };  // error: RMS position (tracking) or course (deg) error

template <typename ACTOR>
std::string evaluate(DEVICE& device, const ACTOR& actor, const std::vector<ENVIRONMENT::State>& starts, RNG& rng,
                     const std::filesystem::path& flights_csv, EvaluationSummary* summary = nullptr){
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
    std::ofstream flights;  // not opened without a path: the writes then do nothing
    if(!flights_csv.empty()) flights.open(flights_csv);
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
    if(summary){
        summary->success_rate = complete / double(N_SCENARIOS);
        // no completed flight: no error to compare, and no checkpoint to keep
        summary->error = complete > 0 ? (FW_TRACKING ? rms[3] : rms[2] * 180 / fixed_wing::PI) / complete
                                      : std::numeric_limits<double>::infinity();
    }
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

struct Flight{
    ENVIRONMENT::Parameters parameters;
    ENVIRONMENT::State start;
    std::uint64_t turbulence_seed;
};

std::vector<Flight> sample_flights(DEVICE& device, const fixed_wing::TaskConfig<T_ENV>& config, TI seed){
    ENVIRONMENT env;
    rlt::init(device, env);
    env.config = config;
    RNG rng;
    rlt::malloc(device, rng);
    rlt::init(device, rng, seed);
    std::vector<Flight> flights(N_SCENARIOS);
    for(TI k = 0; k < N_SCENARIOS; k++){
        rlt::sample_initial_parameters(device, env, flights[k].parameters, rng);
        rlt::sample_initial_state(device, env, flights[k].parameters, flights[k].start, rng);
        flights[k].turbulence_seed = std::uint64_t(seed) * 1000 + k;
    }
    rlt::free(device, rng);
    return flights;
}

fixed_wing::TaskConfig<T_ENV> calm_config(fixed_wing::TaskConfig<T_ENV> config){
    config.wind_speed_max = 0;
    config.turbulence_intensity_max = 0;
    config.gust_rate = 0;
    config.thermal_fraction = 0;
    return config;
}

struct FlightGroup{
    int flights = 0, completed = 0;
    double rmse_sum = 0, energy = 0, power_sum = 0, change_sum = 0;
    std::map<std::string, int> failures;
};

template <typename ACTOR>
std::string evaluate_flights(DEVICE& device, const ACTOR& actor, const std::vector<Flight>& flights,
                             const fixed_wing::TaskConfig<T_ENV>& config, TI n_steps, RNG& rng,
                             const std::filesystem::path& flights_csv, EvaluationSummary* summary = nullptr){
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
    std::vector<ENVIRONMENT::State> states(N_SCENARIOS);
    std::vector<int> reason(N_SCENARIOS, 0);
    std::vector<TI> steps_flown(N_SCENARIOS, 0);
    std::vector<double> squared_error(N_SCENARIOS, 0), energy(N_SCENARIOS, 0), change(N_SCENARIOS, 0);
    std::vector<std::mt19937_64> generators(N_SCENARIOS);
    for(TI k = 0; k < N_SCENARIOS; k++){
        states[k] = flights[k].start;
        generators[k].seed(flights[k].turbulence_seed);
    }
    std::vector<std::normal_distribution<T_ENV>> standard_normals(N_SCENARIOS, std::normal_distribution<T_ENV>(0, 1));
    std::ofstream csv;
    if(!flights_csv.empty()) csv.open(flights_csv);
    csv << "flight,t,x,y,altitude,reference_x,reference_y,reference_altitude,distance,airspeed,alpha_deg,beta_deg,roll_deg,"
           "wind_x,wind_y,wind_z,aileron,elevator,rudder,throttle,power_w,thermals,termination\n";
    const T_ENV deg = 180 / fixed_wing::PI;
    for(TI step = 0; step < n_steps; step++){
        for(TI k = 0; k < N_SCENARIOS; k++){
            fixed_wing::observe_actor(flights[k].parameters, states[k], [&](int j, T_ENV v){ rlt::set(observations, k, j, T(v)); });
        }
        auto input = rlt::to_tensor(device, observations);
        auto output = rlt::to_tensor(device, actions);
        rlt::evaluate_step(device, actor, input, actor_state, output, actor_buffer, rng, rlt::Mode<rlt::mode::Evaluation<>>{});
        for(TI k = 0; k < N_SCENARIOS; k++){
            if(reason[k] != 0) continue;
            const auto& p = flights[k].parameters;
            T_ENV a[4], normals[fixed_wing::N_TURBULENCE] = {};
            for(int i = 0; i < 4; i++) a[i] = fixed_wing::clip(T_ENV(rlt::get(actions, k, i)), T_ENV(-1), T_ENV(1));
            if(fixed_wing::turbulent(p)){
                for(auto& n: normals) n = standard_normals[k](generators[k]);
            }
            ENVIRONMENT::State next;
            fixed_wing::simulate_moving_task(p, states[k], a, normals, next);
            reason[k] = fixed_wing::termination(p, next, config);
            steps_flown[k]++;
            T_ENV e_position[3], e_velocity[3];
            fixed_wing::tracking_errors(next, e_position, e_velocity);
            T_ENV distance = fixed_wing::norm3(e_position);
            squared_error[k] += distance * distance;
            T_ENV power = fixed_wing::advanced_propeller_power(p, next.body);
            energy[k] += power * p.dt;
            double action_change = 0;
            for(int i = 0; i < 4; i++) action_change += std::abs(a[i] - states[k].action_history[fixed_wing::ACTION_HISTORY - 1][i]);
            change[k] += action_change / 4;
            if(k < N_RECORDED){
                auto m = fixed_wing::measurements(p, next.body);
                auto air = fixed_wing::air_data(p, next.body);
                T_ENV wind[3];
                fixed_wing::local_wind(p, next.body, m.R, wind);
                csv << k << "," << (step + 1) * p.dt << "," << next.body.position[0] << "," << next.body.position[1] << ","
                    << next.body.position[2] << "," << next.reference_position[0] << "," << next.reference_position[1] << ","
                    << next.reference_position[2] << "," << distance << "," << air.airspeed << "," << air.alpha * deg << ","
                    << air.beta * deg << "," << std::atan2(m.R[2][1], m.R[2][2]) * deg;
                for(int i = 0; i < 3; i++) csv << "," << wind[i];
                for(int i = 0; i < 4; i++) csv << "," << a[i];
                csv << "," << power << "," << (p.thermals.count > 0 ? 1 : 0) << "," << reason[k] << "\n";
            }
            states[k] = next;
        }
    }
    const char* names[] = {"", "diverged", "low_altitude", "alpha", "airspeed", "altitude", "position"};
    FlightGroup groups[3];
    for(TI k = 0; k < N_SCENARIOS; k++){
        bool thermals = flights[k].parameters.thermals.count > 0;
        for(int g: {0, thermals ? 1 : 2}){
            FlightGroup& group = groups[g];
            group.flights++;
            group.power_sum += energy[k] / (steps_flown[k] * flights[k].parameters.dt);
            if(reason[k] != 0){
                group.failures[names[reason[k]]]++;
                continue;
            }
            group.completed++;
            group.rmse_sum += std::sqrt(squared_error[k] / steps_flown[k]);
            group.energy += energy[k];
            group.change_sum += change[k] / steps_flown[k];
        }
    }
    auto group_json = [&](const FlightGroup& g){
        std::ostringstream out;
        auto over_completed = [&](double sum, double scale){
            std::ostringstream value;
            value << std::setprecision(10);
            if(g.completed > 0) value << sum / g.completed * scale;
            else value << "null";
            return value.str();
        };
        out << std::setprecision(10) << "{\"flights\": " << g.flights << ", \"success_rate\": " << g.completed / double(std::max(g.flights, 1))
            << ", \"position_rmse\": " << over_completed(g.rmse_sum, 1)
            << ", \"energy_kj\": " << over_completed(g.energy, 0.001)
            << ", \"mean_power_w\": " << g.power_sum / std::max(g.flights, 1)
            << ", \"mean_action_change\": " << over_completed(g.change_sum, 1) << ", \"failures\": {";
        bool first = true;
        for(auto& [name, count]: g.failures){ out << (first ? "" : ", ") << "\"" << name << "\": " << count; first = false; }
        out << "}}";
        return out.str();
    };
    std::ostringstream json;
    json << std::setprecision(10) << "{\n  \"duration_s\": " << n_steps * flights[0].parameters.dt
         << ",\n  \"all\": " << group_json(groups[0]) << ",\n  \"with_thermals\": " << group_json(groups[1])
         << ",\n  \"without_thermals\": " << group_json(groups[2]) << "\n}\n";
    if(summary){
        summary->success_rate = groups[0].completed / double(N_SCENARIOS);
        summary->error = groups[0].completed > 0 ? groups[0].rmse_sum / groups[0].completed : std::numeric_limits<double>::infinity();
    }
    rlt::free(device, actor_state);
    rlt::free(device, actor_buffer);
    rlt::free(device, observations);
    rlt::free(device, actions);
    return json.str();
}

int run(int argc, char** argv){
    if(argc < 3){
        std::cerr << "usage: train_sac <scenarios.txt> <out_dir> [seed] [steps] [--start trimmed|neutral|random] [--no-trim-reward] [--alpha-weight W] [--position-scale S] [--position-linear W] [--velocity-guidance K] [--curriculum-period P] [--checkpoint-interval N] [--checkpoint-precise P N] [--reward tanh|l2f-norm|simple] [--position-bonus W] [--near-start F D] [--energy-weight W] [--energy-ramp START END] [--simple-constant C] [--distance-max D] [--reference-heading-max DEG] [--eval-duration S] [--calm] [--wind-max V] [--turbulence-max K] [--gust-rate R] [--thermal-fraction F] [--turn-fraction F]" << std::endl;
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
    fixed_wing::RewardConfig<T_ENV> defaults;
    T_ENV position_scale = defaults.position_scale, position_linear = defaults.position_linear_weight;
    T_ENV guidance = defaults.velocity_guidance_gain;
    T_ENV near_fraction = 0, near_offset = 5;  // --near-start F D
    double precise_min_success = 0;  // --checkpoint-precise P N
    TI precise_after = 0;
    TI curriculum_period = Curriculum{}.period;
    TI checkpoint_interval = 0;  // 0: no validation flights, only the final actor
    T_ENV position_bonus = 0;  // --position-bonus W: weight of the closeness bonus (l2f-norm only)
    std::string reward_form = "tanh";  // tanh (ours), l2f-norm (L2F form with norms), simple (L2F environment, meters)
    T_ENV energy_weight = 0;
    T_ENV energy_ramp_start = 1000000, energy_ramp_end = 3000000;
    T_ENV simple_constant = defaults.simple_constant;
    T_ENV distance_max = 0;
    T_ENV heading_max_deg = fixed_wing::TaskConfig<T_ENV>{}.course_change_max * 180 / fixed_wing::PI;
    T_ENV eval_duration = 120;
    const bool disturbed = FW_DISTURBANCES;
    T_ENV wind_max = disturbed ? 6 : 0, turbulence_max = disturbed ? 1 : 0, gust_rate = disturbed ? 1 : 0;
    T_ENV thermal_fraction = disturbed ? T_ENV(0.5) : 0, turn_fraction = disturbed ? T_ENV(0.5) : 0;
    T_ENV randomization_mass = disturbed ? T_ENV(0.1) : 0, randomization_thrust = disturbed ? T_ENV(0.15) : 0;
    T_ENV randomization_aerodynamics = disturbed ? T_ENV(0.1) : 0;
    int latency_max = disturbed ? 2 : -1;
    for(int i = 5; i < argc; i++){
        std::string option = argv[i];
        if(option == "--alpha-weight" && i + 1 < argc) alpha_weight = std::stod(argv[++i]);
        else if(option == "--position-scale" && i + 1 < argc) position_scale = std::stod(argv[++i]);
        else if(option == "--position-linear" && i + 1 < argc) position_linear = std::stod(argv[++i]);
        else if(option == "--velocity-guidance" && i + 1 < argc) guidance = std::stod(argv[++i]);
        else if(option == "--checkpoint-precise" && i + 2 < argc){ precise_min_success = std::stod(argv[++i]); precise_after = std::stoul(argv[++i]); }
        else if(option == "--near-start" && i + 2 < argc){ near_fraction = std::stod(argv[++i]); near_offset = std::stod(argv[++i]); }
        else if(option == "--start" && i + 1 < argc) start_name = argv[++i];
        else if(option == "--curriculum-period" && i + 1 < argc) curriculum_period = std::stoul(argv[++i]);
        else if(option == "--checkpoint-interval" && i + 1 < argc) checkpoint_interval = std::stoul(argv[++i]);
        else if(option == "--reward" && i + 1 < argc) reward_form = argv[++i];
        else if(option == "--energy-weight" && i + 1 < argc) energy_weight = std::stod(argv[++i]);
        else if(option == "--energy-ramp" && i + 2 < argc){ energy_ramp_start = std::stod(argv[++i]); energy_ramp_end = std::stod(argv[++i]); }
        else if(option == "--simple-constant" && i + 1 < argc) simple_constant = std::stod(argv[++i]);
        else if(option == "--distance-max" && i + 1 < argc) distance_max = std::stod(argv[++i]);
        else if(option == "--reference-heading-max" && i + 1 < argc) heading_max_deg = std::stod(argv[++i]);
        else if(option == "--eval-duration" && i + 1 < argc) eval_duration = std::stod(argv[++i]);
        else if(option == "--calm"){ wind_max = 0; turbulence_max = 0; gust_rate = 0; thermal_fraction = 0; }
        else if(option == "--wind-max" && i + 1 < argc) wind_max = std::stod(argv[++i]);
        else if(option == "--turbulence-max" && i + 1 < argc) turbulence_max = std::stod(argv[++i]);
        else if(option == "--gust-rate" && i + 1 < argc) gust_rate = std::stod(argv[++i]);
        else if(option == "--thermal-fraction" && i + 1 < argc) thermal_fraction = std::stod(argv[++i]);
        else if(option == "--turn-fraction" && i + 1 < argc) turn_fraction = std::stod(argv[++i]);
        else if(option == "--position-bonus" && i + 1 < argc) position_bonus = std::stod(argv[++i]);
        else if(option == "--no-trim-reward") trim_reward = false;
        else throw std::runtime_error("unknown option " + option);
    }
    // the lowest reward of a flying step: survival - capped cost - far terms - position terms; it must stay >= 0,
    // or ending the flight would pay. The tracking margin to the stall is left out: it costs only within 4 deg of
    // the limit, a state the flight leaves (dynamics.h)
    T_ENV lowest = defaults.survival - defaults.cost_cap * defaults.survival - defaults.far_weight * 2 - position_linear;
    if(lowest < -1e-9) throw std::runtime_error("--position-linear too large: a step could get a negative reward");
    const std::map<std::string, fixed_wing::StartMode> start_modes = {
        {"trimmed", fixed_wing::TRIMMED}, {"neutral", fixed_wing::NEUTRAL}, {"random", fixed_wing::RANDOM}};
    if(!start_modes.count(start_name)) throw std::runtime_error("unknown start " + start_name);
    if(curriculum_period == 0 || curriculum_period % 10000 != 0) throw std::runtime_error("--curriculum-period must be a positive multiple of 10000");
    if(reward_form != "tanh" && reward_form != "l2f-norm" && reward_form != "simple") throw std::runtime_error("--reward must be tanh, l2f-norm or simple");
    const bool simple = reward_form == "simple";
    if(reward_form != "tanh" && !FW_TRACKING) throw std::runtime_error("--reward l2f-norm and simple are for the tracking task");
    if(!simple && (energy_weight != 0 || simple_constant != defaults.simple_constant)) throw std::runtime_error("--energy-weight and --simple-constant need --reward simple");
    if(energy_weight < 0 || energy_ramp_start < 0 || energy_ramp_end < energy_ramp_start) throw std::runtime_error("--energy-weight needs W >= 0, --energy-ramp 0 <= START <= END");
    if(distance_max < 0 || heading_max_deg < 0 || heading_max_deg > 180) throw std::runtime_error("--distance-max needs D >= 0, --reference-heading-max 0 <= DEG <= 180");
    if(eval_duration <= 0) throw std::runtime_error("--eval-duration must be positive");
    if(wind_max < 0 || turbulence_max < 0 || gust_rate < 0 || thermal_fraction < 0 || thermal_fraction > 1 || turn_fraction < 0 || turn_fraction > 1) throw std::runtime_error("disturbance options out of range");
    if(!FW_DISTURBANCES && (turbulence_max > 0 || gust_rate > 0 || thermal_fraction > 0)) throw std::runtime_error("--turbulence-max, --gust-rate and --thermal-fraction need a train_sac_*_wind target");
    if(turn_fraction > 0 && !FW_TRACKING) throw std::runtime_error("--turn-fraction is for the tracking task");
    if(precise_min_success < 0 || precise_min_success > 1) throw std::runtime_error("--checkpoint-precise needs 0 <= P <= 1");
    if(precise_min_success > 0 && checkpoint_interval == 0) throw std::runtime_error("--checkpoint-precise needs --checkpoint-interval");
    if(near_fraction < 0 || near_fraction > 1 || near_offset <= 0) throw std::runtime_error("--near-start needs 0 <= F <= 1 and D > 0");
    if(position_bonus != 0 && reward_form != "l2f-norm") throw std::runtime_error("--position-bonus needs --reward l2f-norm");
    std::ofstream(out_dir / "config.json") << "{\n  \"seed\": " << seed << ",\n  \"algorithm\": \"sac\",\n  \"steps\": " << steps
        << ",\n  \"critic_hidden\": " << FW_CRITIC_HIDDEN << ",\n  \"batch\": " << FW_BATCH << ",\n  \"gamma\": " << FW_GAMMA
        << ",\n  \"alpha_margin_weight\": " << alpha_weight << ",\n  \"start\": \"" << start_name
        << "\",\n  \"trim_reward\": " << (trim_reward ? "true" : "false")
        << ",\n  \"fine_position_scale\": " << (FW_FINE_POSITION ? fixed_wing::TaskConfig<T_ENV>{}.fine_position_scale : 0)
        << ",\n  \"task\": \"" << (FW_TRACKING ? "tracking" : "course") << "\",\n  \"airplane\": \""
        << (FW_AIRPLANE == 1 ? "px4_advanced_plane" : FW_AIRPLANE == 2 ? "eratix" : "aerosonde") << "\",\n  \"position_scale\": " << position_scale
        << ",\n  \"position_linear_weight\": " << position_linear << ",\n  \"velocity_guidance_gain\": " << guidance
        << ",\n  \"training_interval\": " << FW_TRAINING_INTERVAL
        << ",\n  \"actor_training_interval\": " << FW_TRAINING_INTERVAL * FW_ACTOR_INTERVAL_FACTOR
        << ",\n  \"reward\": \"" << reward_form << "\",\n  \"position_bonus_weight\": " << position_bonus
        << ",\n  \"near_start_fraction\": " << near_fraction << ",\n  \"near_start_offset\": " << near_offset
        << ",\n  \"curriculum_period\": " << curriculum_period << ",\n  \"checkpoint_interval\": " << checkpoint_interval
        << ",\n  \"checkpoint_precise\": [" << precise_min_success << ", " << precise_after << "]"
        << ",\n  \"energy_weight\": " << energy_weight << ",\n  \"energy_ramp\": [" << energy_ramp_start << ", " << energy_ramp_end << "]"
        << ",\n  \"simple_constant\": " << simple_constant << ",\n  \"distance_max\": " << distance_max
        << ",\n  \"reference_heading_max_deg\": " << heading_max_deg << ",\n  \"eval_duration\": " << eval_duration
        << ",\n  \"disturbances\": " << (FW_DISTURBANCES ? "true" : "false") << ",\n  \"wind_max\": " << wind_max
        << ",\n  \"turbulence_max\": " << turbulence_max << ",\n  \"gust_rate_per_minute\": " << gust_rate
        << ",\n  \"thermal_fraction\": " << thermal_fraction << ",\n  \"turn_fraction\": " << turn_fraction
        << ",\n  \"randomization\": {\"mass\": " << randomization_mass << ", \"thrust\": " << randomization_thrust
        << ", \"aerodynamics\": " << randomization_aerodynamics << ", \"latency_max\": " << latency_max << "}\n}\n";

    DEVICE device;
    rlt::init(device);
    LOOP_STATE ts;
    rlt::malloc(device, ts);
    rlt::init(device, ts, seed);
    Curriculum curriculum;
    curriculum.period = curriculum_period;
    curriculum.initial.alpha_margin_weight = alpha_weight;  // constant: not part of the curriculum
    curriculum.target.alpha_margin_weight = alpha_weight;
    for(auto* c: {&curriculum.initial, &curriculum.target}){  // tracking task: constant over the curriculum
        c->position_scale = position_scale;
        c->position_linear_weight = position_linear;
        c->velocity_guidance_gain = guidance;
    }
    if(!trim_reward){
        curriculum.initial.trim_weight = 0;
        curriculum.target.trim_weight = 0;
    }
    auto energy_weight_at = [&](TI step){
        if(energy_ramp_end <= energy_ramp_start) return T_ENV(step) >= energy_ramp_start ? energy_weight : T_ENV(0);
        T_ENV progress = (T_ENV(step) - energy_ramp_start) / (energy_ramp_end - energy_ramp_start);
        return energy_weight * fixed_wing::clip(progress, T_ENV(0), T_ENV(1));
    };
    if(simple){
        curriculum.simple = true;
        curriculum.n_stages = 0;
        curriculum.initial.simple_form = true;
        curriculum.initial.simple_constant = simple_constant;
        curriculum.initial.energy_weight = energy_weight_at(0);
    }
    if(reward_form == "l2f-norm"){
        // the L2F reward form and schedule (RewardConfig::l2f_form, Curriculum::l2f); the tracking scales
        // (--position-scale, velocity_scale), the velocity guidance and the stall margin stay as given
        curriculum.l2f = true;
        curriculum.n_stages = 20;  // the caps are reached after 12 (position) and 7 (velocity, commands) stages
        for(auto* c: {&curriculum.initial, &curriculum.target}){
            c->l2f_form = true;
            c->position_bonus_weight = position_bonus;  // constant over the curriculum
            c->sideslip_weight = T_ENV(0.3);  // constant: L2F has no sideslip term, an airplane needs one
            c->l2f_floor = -c->l2f_survival;
            // ending a flight is never better than staying at the floor for the rest of it
            c->l2f_termination_penalty = c->l2f_survival / (1 - T_ENV(FW_GAMMA));
        }
    }
    for(int s = 0; s <= curriculum.n_stages; s++){
        auto c = curriculum.at_stage(s);
        for(T_ENV w: {c.airspeed_weight, c.altitude_weight, c.course_weight, c.position_weight, c.velocity_weight,
                      c.sideslip_weight, c.action_change_weight, c.trim_weight, c.alpha_margin_weight}){
            if(!std::isfinite(w)) throw std::runtime_error("reward weight not finite at curriculum stage " + std::to_string(s));
        }
    }
    int stage = 0;
    auto& config = rlt::get(ts.off_policy_runner.envs, 0, 0).config;
    config.start = start_modes.at(start_name);
    config.reference_near_fraction = near_fraction;
    config.reference_near_offset_max = near_offset;
    config.course_change_max = heading_max_deg * fixed_wing::PI / 180;
    config.reference_distance_max = distance_max;
    config.wind_speed_max = wind_max;
    config.turbulence_intensity_max = turbulence_max;
    config.gust_rate = gust_rate;
    config.thermal_fraction = thermal_fraction;
    config.reference_turn_fraction = turn_fraction;
    config.randomization_mass = randomization_mass;
    config.randomization_thrust = randomization_thrust;
    config.randomization_aerodynamics = randomization_aerodynamics;
    config.latency_steps_random_max = latency_max;
    config.reward = curriculum.at_stage(stage);
    const fixed_wing::TaskConfig<T_ENV> training_config = config;

    std::ofstream progress(out_dir / "progress.csv");
    progress << "time/total_timesteps,rollout/ep_len_mean,rollout/ep_rew_mean,"
             << (FW_TRACKING ? "train/position_error_m" : "train/course_error_deg") << ",curriculum/stage,time/time_elapsed,curriculum/bonus_scale"
             << (simple ? ",curriculum/energy_weight,rollout/terminated_fraction" : "") << "\n";
    auto start = std::chrono::steady_clock::now();
    auto& runner = ts.off_policy_runner;
    TI next_check = curriculum.period;
    // best checkpoint on the validation flights (--checkpoint-interval)
    using ACTOR = std::remove_cv_t<std::remove_reference_t<decltype(rlt::get_actor(ts))>>;
    ACTOR best_actor;
    RNG validation_rng;
    std::vector<ENVIRONMENT::State> validation;
    std::vector<Flight> validation_flights;
    std::ofstream checkpoints;
    double best_score = std::numeric_limits<double>::infinity();
    TI best_step = 0;
    if(checkpoint_interval > 0){
        rlt::malloc(device, best_actor);
        rlt::malloc(device, validation_rng);
        rlt::init(device, validation_rng, seed + 1);
        if constexpr(FW_DISTURBANCES) validation_flights = sample_flights(device, training_config, 54321);
        else validation = sampled_starts(device, 54321, fixed_wing::RANDOM);  // other flights than the tests (12345)
        checkpoints.open(out_dir / "checkpoints.csv");
        checkpoints << "step,success_rate," << (FW_TRACKING ? "position_rmse" : "course_rmse_deg") << ",score\n";
    }
    for(TI step = 1; step <= steps; step++){
        rlt::step(device, ts);
        if(step % 10000 == 0){
            Window w = window(runner, 50000);
            double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
            progress << step << "," << w.episode_length << "," << w.episode_return << "," << w.task_error * TASK_ERROR_TO_UNIT
                     << "," << stage << "," << elapsed << "," << curriculum.at_stage(stage).position_bonus_scale;
            if(simple) progress << "," << curriculum.initial.energy_weight << "," << w.terminated_fraction;
            progress << std::endl;
            std::cout << "step " << step << "  episode length " << std::setprecision(4) << w.episode_length << "  return "
                      << w.episode_return << "  " << TASK_ERROR_NAME << " " << w.task_error * TASK_ERROR_TO_UNIT << " "
                      << TASK_ERROR_UNIT << "  stage " << stage << "  " << std::setprecision(3) << step / elapsed << " steps/s";
            if(simple) std::cout << "  terminated " << w.terminated_fraction << "  energy weight " << curriculum.initial.energy_weight;
            std::cout << std::endl;
        }
        if(step == 20000){  // before the first curriculum change
            double error = reward_recomputation_error(runner, curriculum.at_stage(stage));
            std::ofstream(out_dir / "reward_check.txt") << "max |stored - recomputed reward| over the replay buffer at step 20000: " << error << "\n";
            std::cout << "reward recomputation check: " << error << std::endl;
        }
        if(step >= next_check){
            // a stage is taken only if the setpoint was tracked over the last period (l2f-norm: unconditionally)
            bool changed = false;
            if((curriculum.l2f || window(runner, curriculum.period).task_error < curriculum.advance_error) && stage < curriculum.n_stages){
                stage++;
                changed = true;
            }
            if(curriculum.l2f && position_bonus > 0){
                T_ENV scale = std::fmax(std::fmin(curriculum.bonus_scale,
                                                  curriculum.bonus_adaptive * T_ENV(window(runner, curriculum.period).task_error)),
                                        curriculum.bonus_scale_min);
                changed = changed || scale != curriculum.bonus_scale;
                curriculum.bonus_scale = scale;
            }
            if(curriculum.simple){
                T_ENV weight = energy_weight_at(step);
                if(weight != curriculum.initial.energy_weight){
                    curriculum.initial.energy_weight = weight;
                    changed = true;
                }
            }
            if(changed){
                config.reward = curriculum.at_stage(stage);
                recompute_rewards(device, runner, curriculum.at_stage(stage));
            }
            next_check += curriculum.period;
        }
        if(checkpoint_interval > 0 && step % checkpoint_interval == 0){
            EvaluationSummary s;
            if constexpr(FW_DISTURBANCES){
                evaluate_flights(device, rlt::get_actor(ts), validation_flights, training_config, ENVIRONMENT::EPISODE_STEP_LIMIT, validation_rng, {}, &s);
            }
            else{
                evaluate(device, rlt::get_actor(ts), validation, validation_rng, {}, &s);
            }
            double score = s.error + 1000 * (1 - s.success_rate);  // 10 per % of failed flights
            if(precise_min_success > 0){
                // --checkpoint-precise: the error alone for the eligible checkpoints; the others after all of them,
                // in the order of the default score
                bool eligible = s.success_rate >= precise_min_success - 1e-9 && step >= precise_after;
                score = eligible ? s.error : 1e6 + score;
            }
            checkpoints << step << "," << s.success_rate << "," << s.error << "," << score << std::endl;
            if(score < best_score){
                best_score = score;
                best_step = step;
                rlt::copy(device, device, rlt::get_actor(ts), best_actor);
            }
        }
    }
    double seconds = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    // test A: the Python scenarios, or for tracking normal flights (neutral start); test B: random starts
    std::vector<ENVIRONMENT::State> test_a;
    if constexpr(!FW_DISTURBANCES) test_a = FW_TRACKING ? sampled_starts(device, 12345, fixed_wing::NEUTRAL) : start_states(scenarios);
    fixed_wing::TaskConfig<T_ENV> test_config = training_config;
    test_config.episode_duration = eval_duration;
    const TI test_steps = TI(std::lround(eval_duration * test_config.frequency));
    std::vector<Flight> test_disturbed, test_calm;
    if constexpr(FW_DISTURBANCES){
        test_disturbed = sample_flights(device, test_config, 12345);
        test_calm = sample_flights(device, calm_config(test_config), 12345);
    }
    auto test = [&](const ACTOR& actor, const std::string& prefix){  // prefix "" for the final actor, "best" for the best
        std::string tag = prefix.empty() ? "" : "_" + prefix;
        save_actor(device, actor, out_dir / (prefix + (prefix.empty() ? "" : "_") + "actor.h"));
        if constexpr(FW_DISTURBANCES){
            std::string disturbed_result = evaluate_flights(device, actor, test_disturbed, test_config, test_steps, ts.rng,
                                                            out_dir / ("flights" + tag + "_disturbed.csv"));
            std::ofstream(out_dir / ("evaluation" + tag + "_disturbed.json")) << disturbed_result;
            std::string calm_result = evaluate_flights(device, actor, test_calm, calm_config(test_config), test_steps, ts.rng,
                                                       out_dir / ("flights" + tag + "_calm.csv"));
            std::ofstream(out_dir / ("evaluation" + tag + "_calm.json")) << calm_result;
            std::cout << (prefix.empty() ? "" : prefix + " actor:\n") << "disturbed:\n" << disturbed_result << "calm:\n" << calm_result;
            return;
        }
        std::string result = evaluate(device, actor, test_a, ts.rng, out_dir / ("flights" + tag + ".csv"));
        std::ofstream(out_dir / ("evaluation" + tag + ".json")) << result;
        std::string result_random = evaluate(device, actor, sampled_starts(device, 12345, fixed_wing::RANDOM), ts.rng,
                                             out_dir / ("flights" + tag + "_random.csv"));
        std::ofstream(out_dir / ("evaluation" + tag + "_random.json")) << result_random;
        if constexpr(FW_TRACKING){  // test B with the point within 45 deg ahead, as a trajectory planner would place it
            std::string result_ahead = evaluate(device, actor, sampled_starts(device, 12345, fixed_wing::RANDOM,
                                                45 * fixed_wing::PI / 180), ts.rng, out_dir / ("flights" + tag + "_ahead.csv"));
            std::ofstream(out_dir / ("evaluation" + tag + "_ahead.json")) << result_ahead;
        }
        std::cout << (prefix.empty() ? "" : prefix + " actor:\n") << result << "random starts:\n" << result_random;
    };
    test(rlt::get_actor(ts), "");
    if(checkpoint_interval > 0){
        if(best_step > 0){  // at least one checkpoint
            std::cout << "best checkpoint: step " << best_step << ", validation score " << best_score << std::endl;
            test(best_actor, "best");
        }
        rlt::free(device, best_actor);
        rlt::free(device, validation_rng);
    }
    std::cout << "training seconds: " << seconds << std::endl;
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
