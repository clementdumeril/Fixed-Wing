// Compares a flight of the policy in Gazebo (PX4 SITL, deploy/ros2/fly_sitl.sh) with this simulator.
//
// compare_sitl <flight.csv> [out.csv] [start, s after the switch to offboard, default 10]
//   1. closed loop: this simulator flies the same policy (deploy/ros2/fw_policy/policy/policy_weights.h) from the logged
//      state at start (after the hand-over, which can be outside the training range), toward the same point, until the
//      end of the log; out.csv gets its flight
//   2. step by step: from every logged state, the logged command applied for one logged step with this simulator's
//      model; prediction minus Gazebo for the acceleration (body frame) and the angular acceleration, mean and RMS.
//      The surface and throttle positions are not logged: they follow the logged commands with the model's lag, with
//      0 to 3 control steps of latency (the one with the smallest error is the latency of the chain).
#include <rl_tools/operations/cpu.h>

#include <fixed_wing/dynamics.h>
#include "policy_weights.h"

#include <array>
#include <cmath>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

using T = double;
namespace fw = fixed_wing;

struct Row{
    T t, position[3], velocity[3], q[4], rates[3], reference[3], action[4];
    int nav_state;
};

std::vector<Row> load(const std::string& path){
    std::ifstream file(path);
    if(!file) throw std::runtime_error("cannot read " + path);
    std::string line;
    std::getline(file, line);
    std::vector<std::string> names;
    {
        std::stringstream header(line);
        std::string name;
        while(std::getline(header, name, ',')) names.push_back(name);
    }
    std::vector<Row> rows;
    while(std::getline(file, line)){
        std::stringstream cells(line);
        std::string cell;
        std::map<std::string, T> v;
        for(const auto& name: names){ std::getline(cells, cell, ','); v[name] = std::stod(cell); }
        Row r{v["t"], {v["x"], v["y"], v["altitude"]}, {v["vx"], v["vy"], v["vz"]}, {v["qw"], v["qx"], v["qy"], v["qz"]},
              {v["p"], v["q"], v["r"]}, {v["reference_x"], v["reference_y"], v["reference_altitude"]},
              {v["aileron"], v["elevator"], v["rudder"], v["throttle"]}, int(v["nav_state"])};
        rows.push_back(r);
    }
    return rows;
}

fw::Body<T> body_of(const Row& r){
    fw::Body<T> b{};
    for(int i = 0; i < 3; i++){ b.position[i] = r.position[i]; b.linear_velocity[i] = r.velocity[i]; b.angular_velocity[i] = r.rates[i]; }
    for(int i = 0; i < 4; i++) b.orientation[i] = r.q[i];
    return b;
}

T distance(const T a[3], const T b[3]){
    return std::sqrt((a[0] - b[0]) * (a[0] - b[0]) + (a[1] - b[1]) * (a[1] - b[1]) + (a[2] - b[2]) * (a[2] - b[2]));
}

int main(int argc, char** argv){
    if(argc < 2){ std::fprintf(stderr, "usage: compare_sitl <flight.csv> [out.csv]\n"); return 1; }
    std::vector<Row> all = load(argv[1]);
    std::vector<Row> rows;  // from the first step in offboard (nav_state 14) on
    for(const auto& r: all) if(!rows.empty() || r.nav_state == 14) rows.push_back(r);
    if(rows.size() < 100) throw std::runtime_error("fewer than 100 offboard steps");
    fw::TaskConfig<T> config;
    fw::Parameters<T> p;
    fw::task_parameters(config, p);
    T t0 = rows.front().t, duration = rows.back().t - t0;
    T start_time = argc > 3 ? std::stod(argv[3]) : 10;
    // reference velocity from the logged point (it moves in a straight line)
    T reference_velocity[3];
    for(int i = 0; i < 3; i++) reference_velocity[i] = (rows.back().reference[i] - rows.front().reference[i]) / duration;
    std::printf("%zu offboard steps, %.1f s, mean step %.1f ms; point at %.1f m/s\n", rows.size(), duration,
                1000 * duration / (rows.size() - 1), std::hypot(reference_velocity[0], reference_velocity[1]));

    // ----- 1. closed loop from the same start --------------------------------------------------------------------
    size_t first = 0;
    while(first + 1 < rows.size() && rows[first].t - t0 < start_time) first++;
    // the logged commands before the start fill the action history, the last one sets the surfaces
    fw::State<T> s{}, next{};
    fw::Body<T> start = body_of(rows[first]);
    T trim_throttle[fw::N_ACTUATORS] = {0, 0, 0, T(-0.5)};  // what the node assumes PX4 commanded at the hand-over
    fw::scale_action(p, rows[first > 0 ? first - 1 : 0].action, start.actuators);
    fw::start_state(p, start, T(15), start.position[2], T(0), s);
    for(int i = 0; i < 3; i++){ s.reference_position[i] = rows[first].reference[i]; s.reference_velocity[i] = reference_velocity[i]; }
    for(int h = 0; h < fw::ACTION_HISTORY; h++){
        size_t k = first >= size_t(fw::ACTION_HISTORY - h) ? first - (fw::ACTION_HISTORY - h) : 0;
        for(int i = 0; i < 4; i++) s.action_history[h][i] = rows[k].action[i];
    }
    std::ofstream out(argc > 2 ? argv[2] : "sim_flight.csv");
    out << "t,x,y,altitude,reference_x,reference_y,reference_altitude,aileron,elevator,rudder,throttle,termination\n";
    struct Stats{ double d2 = 0, dz = 0, a[4] = {0, 0, 0, 0}, speed = 0; int n = 0; } sim_stats, gz_stats;
    int steps = int(std::round((duration - (rows[first].t - t0)) / p.dt)), termination = 0;
    T sim_t0 = rows[first].t - t0;
    for(int k = 0; k <= steps; k++){
        float observation[fw::policy::INPUT], action[fw::policy::ACTIONS];
        fw::observe_actor(p, s, [&](int j, T v){ observation[j] = float(v); });
        fw::policy::evaluate(observation, action);
        T a[4] = {action[0], action[1], action[2], action[3]};
        T t = sim_t0 + k * p.dt;
        out << t << "," << s.body.position[0] << "," << s.body.position[1] << "," << s.body.position[2];
        for(int i = 0; i < 3; i++) out << "," << s.reference_position[i];
        for(int i = 0; i < 4; i++) out << "," << a[i];
        termination = fw::termination(p, s, config);
        out << "," << termination << "\n";
        if(termination != fw::FLYING) break;
        if(t >= duration - 30){
            sim_stats.d2 += std::pow(distance(s.body.position, s.reference_position), 2);
            sim_stats.dz += s.body.position[2] - s.reference_position[2];
            for(int i = 0; i < 4; i++) sim_stats.a[i] += a[i];
            sim_stats.speed += fw::norm3(s.body.linear_velocity);
            sim_stats.n++;
        }
        fw::simulate_task(p, s, a, next);
        s = next;
    }
    for(const auto& r: rows){
        if(r.t - t0 < duration - 30) continue;
        gz_stats.d2 += std::pow(distance(r.position, r.reference), 2);
        gz_stats.dz += r.position[2] - r.reference[2];
        for(int i = 0; i < 4; i++) gz_stats.a[i] += r.action[i];
        gz_stats.speed += fw::norm3(r.velocity);
        gz_stats.n++;
    }
    std::printf("\nclosed loop, last 30 s     distance RMSE   height vs point   speed   aileron elevator rudder throttle\n");
    for(auto [name, st]: {std::pair<const char*, Stats>{"Gazebo + PX4", gz_stats}, {"this simulator", sim_stats}}){
        if(st.n == 0){ std::printf("%-26s flight ended early (termination %d)\n", name, termination); continue; }
        std::printf("%-26s %10.1f m %13.1f m %8.1f   %7.2f %8.2f %6.2f %8.2f\n", name, std::sqrt(st.d2 / st.n), st.dz / st.n,
                    st.speed / st.n, st.a[0] / st.n, st.a[1] / st.n, st.a[2] / st.n, st.a[3] / st.n);
    }

    // ----- 2. step by step: model prediction against Gazebo --------------------------------------------------------
    std::printf("\nstep by step, prediction - Gazebo (mean +- RMS), body frame x forward, y left, z up\n");
    std::ofstream residuals("residuals.csv");
    residuals << "t,airspeed,alpha,beta,p,q,r,aileron,elevator,rudder,throttle,ax,ay,az,wx,wy,wz\n";
    for(int latency = 0; latency <= 3; latency++){
        T actuators[fw::N_ACTUATORS];
        T commanded[fw::N_ACTUATORS];
        fw::scale_action(p, trim_throttle, actuators);
        std::array<double, 6> mean{}, square{};
        int n = 0;
        for(size_t k = 0; k + 1 < rows.size(); k++){
            T dt = rows[k + 1].t - rows[k].t;
            // a stalled node (several steps without a command) or an odometry that was not updated: no comparison
            bool usable = dt > T(0.005) && dt < T(0.05) && rows[k + 1].velocity[0] != rows[k].velocity[0];
            if(!usable){
                fw::scale_action(p, rows[k].action, actuators);
                continue;
            }
            const Row& source = rows[k >= size_t(latency) ? k - latency : 0];
            fw::scale_action(p, source.action, commanded);
            fw::Body<T> x = body_of(rows[k]), y;
            for(int i = 0; i < fw::N_ACTUATORS; i++) x.actuators[i] = actuators[i];
            fw::rk4(p, x, commanded, dt, y);
            for(int i = 0; i < fw::N_ACTUATORS; i++) actuators[i] = fw::clip(y.actuators[i], p.actuator_limits[i][0], p.actuator_limits[i][1]);
            if(rows[k].t - t0 < 5) continue;  // the hand-over: unknown surface positions
            // velocity change in the body frame at k, and angular velocity change
            T R[3][3], dv_world[3], dv_body[3];
            fw::rotation_matrix(rows[k].q, R);
            for(int i = 0; i < 3; i++) dv_world[i] = y.linear_velocity[i] - rows[k + 1].velocity[i];
            fw::rotate_inverse(R, dv_world, dv_body);
            for(int i = 0; i < 3; i++){
                double e[2] = {dv_body[i] / dt, (y.angular_velocity[i] - rows[k + 1].rates[i]) / dt};
                for(int j = 0; j < 2; j++){ mean[i + 3 * j] += e[j]; square[i + 3 * j] += e[j] * e[j]; }
            }
            if(latency == 1){
                fw::AirData<T> air = fw::air_data(p, x);
                residuals << rows[k].t - t0 << "," << air.airspeed << "," << air.alpha << "," << air.beta;
                for(int i = 0; i < 3; i++) residuals << "," << x.angular_velocity[i];
                for(int i = 0; i < fw::N_ACTUATORS; i++) residuals << "," << x.actuators[i];
                for(int i = 0; i < 3; i++) residuals << "," << dv_body[i] / dt;
                for(int i = 0; i < 3; i++) residuals << "," << (y.angular_velocity[i] - rows[k + 1].rates[i]) / dt;
                residuals << "\n";
            }
            n++;
        }
        std::printf("latency %d step(s): accel x %+.2f+-%.2f y %+.2f+-%.2f z %+.2f+-%.2f m/s2 | angular accel x %+.2f+-%.2f y %+.2f+-%.2f z %+.2f+-%.2f rad/s2\n",
                    latency, mean[0] / n, std::sqrt(square[0] / n), mean[1] / n, std::sqrt(square[1] / n), mean[2] / n, std::sqrt(square[2] / n),
                    mean[3] / n, std::sqrt(square[3] / n), mean[4] / n, std::sqrt(square[4] / n), mean[5] / n, std::sqrt(square[5] / n));
    }
    return 0;
}
