// Replays flights recorded by the Python environment and prints what the C++ port computes at every step.
//
// check <cases.txt> <out.txt>; cases.txt is written by tools/compare_with_python.py, which also reads out.txt.
// cases.txt: trim airspeeds, then for each case the start state, the setpoint, the trim commands and the actions.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/dynamics.h>

using T = double;
using namespace fixed_wing;

int run(int argc, char** argv){
    if(argc != 3){
        std::cerr << "usage: check <cases.txt> <out.txt>" << std::endl;
        return 1;
    }
    std::ifstream in(argv[1]);
    std::ofstream out(argv[2]);
    out << std::setprecision(17);
    TaskConfig<T> config;
    Parameters<T> p;
    task_parameters(config, p);

    int n_trim;
    in >> n_trim;
    for(int k = 0; k < n_trim; k++){
        T airspeed;
        in >> airspeed;
        Body<T> x = trim(p, airspeed);
        T pitch = 2 * std::atan2(-x.orientation[2], x.orientation[0]);
        out << pitch << " " << x.actuators[ELEVATOR] << " " << x.actuators[THROTTLE] << "\n";
    }

    int n_cases;
    in >> n_cases;
    for(int c = 0; c < n_cases; c++){
        Body<T> body;
        for(T& v: body.position) in >> v;
        for(T& v: body.orientation) in >> v;
        for(T& v: body.linear_velocity) in >> v;
        for(T& v: body.angular_velocity) in >> v;
        for(T& v: body.actuators) in >> v;
        T airspeed, altitude, course;
        in >> airspeed >> altitude >> course;
        State<T> s;
        start_state(p, body, airspeed, altitude, course, s);
        T trim_action[N_ACTUATORS];
        for(T& v: trim_action) in >> v;
        for(int i = 0; i < N_ACTUATORS; i++) out << s.trim_action[i] - trim_action[i] << (i + 1 < N_ACTUATORS ? " " : "\n");
        int n_steps;
        in >> n_steps;
        for(int k = 0; k < n_steps; k++){
            T action[N_ACTUATORS];
            for(T& v: action) in >> v;
            State<T> next;
            simulate(p, s, action, next);
            T r = reward(p, s, action, next, config);
            Termination reason = termination(p, next, config);
            const Body<T>& b = next.body;
            for(T v: b.position) out << v << " ";
            for(T v: b.orientation) out << v << " ";
            for(T v: b.linear_velocity) out << v << " ";
            for(T v: b.angular_velocity) out << v << " ";
            for(T v: b.actuators) out << v << " ";
            out << r << " " << int(reason);
            std::vector<T> actor(ActorObservation<int>::DIM), critic(CriticObservation<int>::DIM);
            observe_actor(p, next, [&](int i, T v){ actor[i] = v; });
            observe_critic(p, next, [&](int i, T v){ critic[i] = v; });
            for(T v: actor) out << " " << v;
            for(T v: critic) out << " " << v;
            out << "\n";
            s = next;
            if(reason != FLYING) break;
        }
    }
    std::cout << "wrote " << argv[2] << std::endl;
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
