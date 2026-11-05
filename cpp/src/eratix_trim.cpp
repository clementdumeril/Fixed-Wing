// Checks of the Eratix model (FW_AIRPLANE=2) in the simulator itself, for docs/eratix_model.md: thrust-to-weight and
// thrust against airspeed, level-flight trim over the speed range, stall speed, the linear modes at cruise, and a
// hands-off flight from trim with the trimmed commands held.
//
// eratix_trim [out_dir]: prints everything; with out_dir, also writes trim.csv, thrust.csv and hands_off.csv there
#include <rl_tools/operations/cpu_mux.h>

#include <fixed_wing/operations_generic.h>

#include <cmath>
#include <complex>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>

static_assert(FW_AIRPLANE == 2, "eratix_trim checks the Eratix: build with FW_AIRPLANE=2");

using T = double;
namespace fw = fixed_wing;
constexpr T DEG = 180 / fw::PI;

// the variables of the linear modes: body velocity (FLU), body rates, roll and pitch. Heading and position do not
// change the forces (constant density, no wind)
constexpr int N = 8;
enum{U, V, W, P, Q, R, ROLL, PITCH};

fw::Body<T> body_from(const fw::Body<T>& trimmed, const T z[N]){
    fw::Body<T> x = trimmed;
    fw::euler_quaternion(z[ROLL], z[PITCH], T(0), x.orientation);
    T Rm[3][3], v[3] = {z[U], z[V], z[W]};
    fw::rotation_matrix(x.orientation, Rm);
    fw::rotate(Rm, v, x.linear_velocity);
    for(int i = 0; i < 3; i++) x.angular_velocity[i] = z[P + i];
    return x;
}

void state_of(const fw::Body<T>& x, T z[N]){
    T Rm[3][3], v[3];
    fw::rotation_matrix(x.orientation, Rm);
    fw::rotate_inverse(Rm, x.linear_velocity, v);
    for(int i = 0; i < 3; i++){ z[U + i] = v[i]; z[P + i] = x.angular_velocity[i]; }
    z[ROLL] = std::atan2(Rm[2][1], Rm[2][2]);
    z[PITCH] = std::asin(fw::clip(Rm[2][0], T(-1), T(1)));
}

void rates_of(const fw::Parameters<T>& p, const fw::Body<T>& trimmed, const T z[N], T dz[N]){
    // time derivative of z with the trimmed commands held: a tiny Euler step of derivative(), mapped back to z
    fw::Body<T> x = body_from(trimmed, z), dx, y;
    fw::derivative(p, x, trimmed.actuators, dx);
    const T h = 1e-7;
    fw::add_scaled(x, dx, h, y);
    T z1[N];
    state_of(y, z1);
    for(int i = 0; i < N; i++) dz[i] = (z1[i] - z[i]) / h;
}

std::vector<std::complex<T>> eigenvalues(const std::vector<std::vector<T>>& A){
    // characteristic polynomial (Faddeev-LeVerrier), then its roots (Durand-Kerner): enough for 4x4 and 8x8
    int n = int(A.size());
    std::vector<T> c(n + 1, 0);  // lambda^n + c[1] lambda^(n-1) + ... + c[n]
    c[0] = 1;
    std::vector<std::vector<T>> M(n, std::vector<T>(n, 0)), AM(n, std::vector<T>(n, 0));
    for(int k = 1; k <= n; k++){
        // M_k = A M_{k-1} + c_{k-1} I, c_k = -trace(A M_k) / k
        for(int i = 0; i < n; i++) for(int j = 0; j < n; j++){
            T s = 0;
            for(int l = 0; l < n; l++) s += A[i][l] * M[l][j];
            AM[i][j] = s;
        }
        for(int i = 0; i < n; i++) for(int j = 0; j < n; j++) M[i][j] = AM[i][j] + (i == j ? c[k - 1] : 0);
        T trace = 0;
        for(int i = 0; i < n; i++) for(int l = 0; l < n; l++) trace += A[i][l] * M[l][i];
        c[k] = -trace / k;
    }
    std::vector<std::complex<T>> r(n);
    for(int i = 0; i < n; i++) r[i] = std::pow(std::complex<T>(0.4, 0.9), i) * T(10);
    for(int iteration = 0; iteration < 2000; iteration++){
        for(int i = 0; i < n; i++){
            std::complex<T> value = 1, product = 1;
            for(int k = 1; k <= n; k++) value = value * r[i] + c[k];
            for(int j = 0; j < n; j++) if(j != i) product *= r[i] - r[j];
            r[i] -= value / product;
        }
    }
    return r;
}

void print_modes(const std::string& name, const std::vector<std::complex<T>>& lambda){
    std::cout << name << ":";
    for(auto& l: lambda){
        std::cout << "  " << std::setprecision(3) << l.real();
        if(std::abs(l.imag()) > 1e-6) std::cout << (l.imag() > 0 ? "+" : "") << l.imag() << "i";
    }
    std::cout << "  (1/s)" << std::endl;
}

int main(int argc, char** argv){
    std::filesystem::path out = argc > 1 ? argv[1] : "";
    if(!out.empty()) std::filesystem::create_directories(out);
    fw::TaskConfig<T> config;
    fw::Parameters<T> p;
    fw::task_parameters(config, p);
    namespace e = fw::eratix_coefficients;
    T weight = p.mass * -p.gravity[2];
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "Eratix: mass " << p.mass << " kg, S " << p.wing_area << " m2, b " << p.wing_span << " m, MAC "
              << p.mean_chord << " m, AR " << p.wing_span * p.wing_span / p.wing_area << ", wing loading "
              << p.mass * 1000 / (p.wing_area * 100) << " g/dm2" << std::endl;

    // ----- propulsion: static thrust and its fall with airspeed at full throttle -----------------------------------
    T thrust, torque, n;
    fw::eratix_propeller(p.air_density, T(1), T(0), thrust, torque, n);
    T current = (e::battery_voltage - 60 / (2 * fw::PI * e::kv) * 2 * fw::PI * n) / e::resistance;
    std::cout << "static, full throttle: thrust " << thrust << " N (" << thrust / 9.81 * 1000 << " g), T/W "
              << thrust / weight << ", " << n * 60 << " rpm, torque " << torque << " N m, current " << current
              << " A, electric power " << e::battery_voltage * current << " W" << std::endl;
    std::ofstream thrust_csv;
    if(!out.empty()) thrust_csv.open(out / "thrust.csv");
    thrust_csv << "airspeed,throttle,thrust,rpm,torque\n";
    std::cout << "full-throttle thrust against airspeed:";
    for(T v = 0; v <= 20 + 1e-9; v += 1){
        fw::eratix_propeller(p.air_density, T(1), v, thrust, torque, n);
        if(int(v) % 2 == 0) std::cout << "  " << std::setprecision(0) << v << " m/s " << std::setprecision(2) << thrust << " N";
        for(T d: {0.25, 0.5, 0.75, 1.0}){
            T t2, q2, n2;
            fw::eratix_propeller(p.air_density, d, v, t2, q2, n2);
            thrust_csv << v << "," << d << "," << t2 << "," << n2 * 60 << "," << q2 << "\n";
        }
    }
    std::cout << std::endl << std::setprecision(4);

    // ----- level-flight trim over the speed range ----------------------------------------------------------------
    std::ofstream trim_csv;
    if(!out.empty()) trim_csv.open(out / "trim.csv");
    trim_csv << "airspeed,pitch_deg,alpha_deg,elevator_deg,throttle,thrust,rpm,CL,L_over_D\n";
    std::cout << "level trim:  V [m/s]  pitch [deg]  alpha [deg]  elevator [deg]  throttle  thrust [N]  rpm    CL     L/D" << std::endl;
    T lowest = -1, highest = -1;
    for(T v = 4.0; v <= 20 + 1e-9; v += 0.5){
        fw::Body<T> x;
        try{ x = fw::trim(p, v); }
        catch(const std::exception&){ continue; }
        T throttle = x.actuators[fw::THROTTLE], elevator = x.actuators[fw::ELEVATOR];
        if(throttle < 0 || throttle > 1 || std::abs(elevator) > p.actuator_limits[fw::ELEVATOR][1]) continue;
        auto a = fw::air_data(p, x);
        fw::eratix_propeller(p.air_density, throttle, a.v_frd[0], thrust, torque, n);
        T pitch = fw::wrap(2 * std::atan2(-x.orientation[2], x.orientation[0]));
        T CL = weight / (0.5 * p.air_density * v * v * p.wing_area);  // level flight, thrust along the body axis neglected
        T drag = thrust * std::cos(a.alpha);  // along the flight path
        if(a.alpha < e::alpha_stall){  // the range counts attached flow only; the solutions past the stall are shown
            if(lowest < 0) lowest = v;
            highest = v;
        }
        std::cout << std::setprecision(1) << "             " << std::setw(6) << v << std::setw(13) << pitch * DEG
                  << std::setw(13) << a.alpha * DEG << std::setw(16) << elevator * DEG << std::setprecision(3)
                  << std::setw(10) << throttle << std::setw(12) << thrust << std::setprecision(0) << std::setw(7) << n * 60
                  << std::setprecision(3) << std::setw(7) << CL << std::setprecision(1) << std::setw(7) << weight / drag
                  << (a.alpha > e::alpha_stall ? "  past the stall" : "") << std::endl;
        trim_csv << v << "," << pitch * DEG << "," << a.alpha * DEG << "," << elevator * DEG << "," << throttle << ","
                 << thrust << "," << n * 60 << "," << CL << "," << weight / drag << "\n";
    }
    std::cout << std::setprecision(2) << "trimmed level flight in attached flow from " << lowest << " to " << highest
              << " m/s (throttle within [0, 1], elevator within its limits)" << std::endl;
    // stall speed: the lift coefficient of the model at the stall angle, elevator at 0
    {
        fw::Body<T> x{};
        x.orientation[0] = 1;
        T alpha = e::alpha_stall, v = 10;
        x.linear_velocity[0] = v * std::cos(alpha);
        x.linear_velocity[2] = -v * std::sin(alpha);  // air from below: positive angle of attack
        T force[3], moment[3];
        fw::aerodynamics(p, x, force, moment);
        T lift = force[0] * std::sin(alpha) + force[2] * std::cos(alpha);  // FLU, perpendicular to the air velocity
        T CL_max = lift / (0.5 * p.air_density * v * v * p.wing_area);
        std::cout << std::setprecision(3) << "at the stall angle (" << e::alpha_stall * DEG << " deg): C_L " << CL_max
                  << ", 1 g stall speed " << std::sqrt(2 * weight / (p.air_density * p.wing_area * CL_max)) << " m/s" << std::endl;
    }

    // ----- linear modes at cruise --------------------------------------------------------------------------------
    for(T v: {7.0, 8.0, 10.0}){
        fw::Body<T> trimmed = fw::trim(p, v);
        T z0[N];
        state_of(trimmed, z0);
        std::vector<std::vector<T>> A(N, std::vector<T>(N, 0));
        for(int j = 0; j < N; j++){
            T zp[N], zm[N], dp[N], dm[N];
            for(int i = 0; i < N; i++){ zp[i] = z0[i]; zm[i] = z0[i]; }
            T epsilon = 1e-5;
            zp[j] += epsilon; zm[j] -= epsilon;
            rates_of(p, trimmed, zp, dp);
            rates_of(p, trimmed, zm, dm);
            for(int i = 0; i < N; i++) A[i][j] = (dp[i] - dm[i]) / (2 * epsilon);
        }
        auto block = [&](std::initializer_list<int> index){
            std::vector<std::vector<T>> B;
            for(int i: index){ B.emplace_back(); for(int j: index) B.back().push_back(A[i][j]); }
            return B;
        };
        std::cout << std::setprecision(1) << "linear modes at " << v << " m/s" << std::endl;
        print_modes("  longitudinal (u, w, q, pitch)", eigenvalues(block({U, W, Q, PITCH})));
        print_modes("  lateral (v, p, r, roll)", eigenvalues(block({V, P, R, ROLL})));
        print_modes("  all 8 (coupled by the propeller torque and gyroscopic moment)", eigenvalues(A));
    }

    // ----- hands-off flight from trim ----------------------------------------------------------------------------
    // the commands of level trim at 8 m/s held for 10 s, in the task's simulation (50 Hz, 2 RK4 substeps). trim()
    // solves pitch, elevator and throttle only: aileron and rudder are set here so that the roll and yaw accelerations
    // are 0 too (the propeller torque), as a pilot trims them
    std::ofstream hands_off;
    if(!out.empty()) hands_off.open(out / "hands_off.csv");
    hands_off << "t,airspeed,altitude_change,pitch_deg,roll_deg,alpha_deg,beta_deg\n";
    fw::Body<T> trimmed = fw::trim(p, T(8));
    for(int iteration = 0; iteration < 20; iteration++){
        auto residual = [&](const fw::Body<T>& x, T r[2]){
            fw::Body<T> dx;
            fw::derivative(p, x, x.actuators, dx);
            r[0] = dx.angular_velocity[0]; r[1] = dx.angular_velocity[2];
        };
        T r[2], jacobian[2][2];
        residual(trimmed, r);
        for(int j = 0; j < 2; j++){
            fw::Body<T> x = trimmed;
            x.actuators[j == 0 ? fw::AILERON : fw::RUDDER] += 1e-6;
            T rp[2];
            residual(x, rp);
            for(int i = 0; i < 2; i++) jacobian[i][j] = (rp[i] - r[i]) / 1e-6;
        }
        T det = jacobian[0][0] * jacobian[1][1] - jacobian[0][1] * jacobian[1][0];
        trimmed.actuators[fw::AILERON] -= (jacobian[1][1] * r[0] - jacobian[0][1] * r[1]) / det;
        trimmed.actuators[fw::RUDDER] -= (-jacobian[1][0] * r[0] + jacobian[0][0] * r[1]) / det;
    }
    std::cout << std::setprecision(2) << "roll and yaw trim at 8 m/s: aileron " << trimmed.actuators[fw::AILERON] * DEG
              << " deg, rudder " << trimmed.actuators[fw::RUDDER] * DEG << " deg" << std::endl;
    fw::State<T> s{}, next;
    fw::start_state(p, trimmed, T(8), T(0), T(0), s);
    T action[fw::N_ACTUATORS];
    fw::normalize_action(p, trimmed.actuators, action);
    std::cout << "hands-off from level trim at 8 m/s (trimmed commands held):" << std::endl;
    for(int k = 0; k <= int(10 / p.dt); k++){
        T t = k * p.dt;
        T Rm[3][3];
        fw::rotation_matrix(s.body.orientation, Rm);
        auto a = fw::air_data(p, s.body);
        T roll = std::atan2(Rm[2][1], Rm[2][2]) * DEG, pitch = std::asin(fw::clip(Rm[2][0], T(-1), T(1))) * DEG;
        hands_off << t << "," << a.airspeed << "," << s.body.position[2] << "," << pitch << "," << roll << ","
                  << a.alpha * DEG << "," << a.beta * DEG << "\n";
        if(k % int(1 / p.dt) == 0){
            std::cout << std::setprecision(1) << "  t " << std::setw(4) << t << " s  airspeed " << std::setprecision(2)
                      << a.airspeed << " m/s  altitude change " << s.body.position[2] << " m  pitch " << pitch
                      << " deg  roll " << roll << " deg  alpha " << a.alpha * DEG << " deg" << std::endl;
        }
        fw::simulate(p, s, action, next);
        s = next;
    }
    return 0;
}
