// Airplane physics, independent of RLtools: port of operations.py, trim.py and the task functions of environment.py.
#pragma once

#include <cmath>
#include <stdexcept>

#include "fixed_wing.h"

namespace fixed_wing{
    constexpr double PI = 3.14159265358979323846;
    constexpr double MIN_AIRSPEED = 1e-3;  // below this the aerodynamic angles are undefined, no aerodynamic forces

    // ----- small linear algebra --------------------------------------------------------------------------------------
    template <typename T>
    void quaternion_multiply(const T a[4], const T b[4], T out[4]){
        T w = a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3];
        T x = a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2];
        T y = a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1];
        T z = a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0];
        out[0] = w; out[1] = x; out[2] = y; out[3] = z;
    }

    template <typename T>
    void rotation_matrix(const T q[4], T R[3][3]){
        T w = q[0], x = q[1], y = q[2], z = q[3];
        R[0][0] = 1 - 2 * (y * y + z * z); R[0][1] = 2 * (x * y - w * z); R[0][2] = 2 * (x * z + w * y);
        R[1][0] = 2 * (x * y + w * z); R[1][1] = 1 - 2 * (x * x + z * z); R[1][2] = 2 * (y * z - w * x);
        R[2][0] = 2 * (x * z - w * y); R[2][1] = 2 * (y * z + w * x); R[2][2] = 1 - 2 * (x * x + y * y);
    }

    template <typename T>
    void rotate(const T R[3][3], const T v[3], T out[3]){  // body to world
        for(int i = 0; i < 3; i++) out[i] = R[i][0] * v[0] + R[i][1] * v[1] + R[i][2] * v[2];
    }

    template <typename T>
    void rotate_inverse(const T R[3][3], const T v[3], T out[3]){  // world to body
        for(int i = 0; i < 3; i++) out[i] = R[0][i] * v[0] + R[1][i] * v[1] + R[2][i] * v[2];
    }

    template <typename T>
    void euler_quaternion(T roll, T pitch, T yaw, T q[4]){
        // roll > 0 right wing down, pitch > 0 nose up, yaw > 0 nose to the left (body y points left)
        T qz[4] = {std::cos(yaw / 2), 0, 0, std::sin(yaw / 2)};
        T qy[4] = {std::cos(pitch / 2), 0, -std::sin(pitch / 2), 0};
        T qx[4] = {std::cos(roll / 2), std::sin(roll / 2), 0, 0};
        T qzy[4];
        quaternion_multiply(qz, qy, qzy);
        quaternion_multiply(qzy, qx, q);
    }

    template <typename T>
    bool invert3(const T A[3][3], T out[3][3]){
        T det = A[0][0] * (A[1][1] * A[2][2] - A[1][2] * A[2][1])
              - A[0][1] * (A[1][0] * A[2][2] - A[1][2] * A[2][0])
              + A[0][2] * (A[1][0] * A[2][1] - A[1][1] * A[2][0]);
        if(det == 0) return false;
        out[0][0] = (A[1][1] * A[2][2] - A[1][2] * A[2][1]) / det;
        out[0][1] = (A[0][2] * A[2][1] - A[0][1] * A[2][2]) / det;
        out[0][2] = (A[0][1] * A[1][2] - A[0][2] * A[1][1]) / det;
        out[1][0] = (A[1][2] * A[2][0] - A[1][0] * A[2][2]) / det;
        out[1][1] = (A[0][0] * A[2][2] - A[0][2] * A[2][0]) / det;
        out[1][2] = (A[0][2] * A[1][0] - A[0][0] * A[1][2]) / det;
        out[2][0] = (A[1][0] * A[2][1] - A[1][1] * A[2][0]) / det;
        out[2][1] = (A[0][1] * A[2][0] - A[0][0] * A[2][1]) / det;
        out[2][2] = (A[0][0] * A[1][1] - A[0][1] * A[1][0]) / det;
        return true;
    }

    template <typename T>
    T wrap(T angle){
        T two_pi = T(2 * PI);
        T x = std::fmod(angle + T(PI), two_pi);
        if(x < 0) x += two_pi;  // Python's % keeps the sign of the divisor
        return x - T(PI);
    }

    template <typename T>
    T clip(T x, T lo, T hi){ return x < lo ? lo : (x > hi ? hi : x); }

    // ----- airplane --------------------------------------------------------------------------------------------------
    template <typename T>
    void aerosonde(Parameters<T>& p){
        // Placeholder until the lab's airplane: Aerosonde UAV, Beard & McLain, "Small Unmanned Aircraft", appendix E.
        // thrust_max and the actuator time constants are guesses (same as parameters.py).
        p.mass = T(13.5);
        p.gravity[0] = 0; p.gravity[1] = 0; p.gravity[2] = T(-9.81);
        T Jx = T(0.8244), Jy = T(1.135), Jz = T(1.759), Jxz = T(0.1204);
        // FRD inertia [[Jx, 0, -Jxz], [0, Jy, 0], [-Jxz, 0, Jz]] seen in FLU: the xz product changes sign twice
        T J[3][3] = {{Jx, 0, Jxz}, {0, Jy, 0}, {Jxz, 0, Jz}};
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) p.J[i][j] = J[i][j];
        invert3(p.J, p.J_inv);
        p.air_density = T(1.2682);
        p.wing_area = T(0.55);
        p.wing_span = T(2.8956);
        p.mean_chord = T(0.18994);
        const T c[N_COEFFICIENTS] = {
            T(0.28), T(3.45), T(0.0), T(-0.36),
            T(0.0437), T(0.9),
            T(0.0), T(-0.98), T(0.0), T(0.0), T(0.0), T(-0.17),
            T(0.0), T(-0.12), T(-0.26), T(0.14), T(0.08), T(0.105),
            T(-0.02338), T(-0.38), T(-3.6), T(-0.5),
            T(0.0), T(0.25), T(0.022), T(-0.35), T(0.06), T(-0.032),
        };
        for(int i = 0; i < N_COEFFICIENTS; i++) p.aerodynamics[i] = c[i];
        p.thrust_max = 40;
        p.throttle_time_constant_down = 0;
        T tau[N_ACTUATORS] = {T(0.05), T(0.05), T(0.05), T(0.2)};
        T surface = T(30 * PI / 180);
        for(int i = 0; i < N_ACTUATORS; i++){
            p.actuator_time_constants[i] = tau[i];
            p.actuator_limits[i][0] = i == THROTTLE ? T(0) : -surface;
            p.actuator_limits[i][1] = i == THROTTLE ? T(1) : surface;
        }
        p.dt = T(0.01);
        p.n_substeps = 1;
        p.latency_steps = 0;
        for(int i = 0; i < 3; i++) p.wind[i] = 0;
        for(int i = 0; i < 4 + N_COEFFICIENTS; i++) p.randomization[i] = 1;
    }

    // PX4 advanced_plane (PX4-gazebo-models, models/advanced_plane/model.sdf), as Gazebo simulates it: the gz-sim
    // AdvancedLiftDrag system with the coefficients of the model, its MulticopterMotorModel puller propeller, and PX4's
    // SITL output mapping (SIM_GZ_EC_MIN1 10, SIM_GZ_EC_MAX1 1600 rad/s; servos +-1 -> +-45 deg). Servos 0 and 1 are the
    // left and right elevons, moved in opposition as ailerons; servo 2 the elevator, servo 3 the rudder
    namespace advanced{
        constexpr double CL0 = 0.15188, CLa = 5.015, CD0 = 0.029, AR = 6.5, eff = 0.97;
        constexpr double Cem0 = 0.075, Cema = -0.463966, CYb = -0.258244, Cellb = -0.039250, Cenb = 0.100826;
        constexpr double CDp = 0.0, CYp = 0.065861, CLp = 0.0, Cellp = -0.487407, Cemp = 0.0, Cenp = -0.040416;
        constexpr double CDq = 0.055166, CYq = 0.0, CLq = 7.971792, Cellq = 0.0, Cemq = -12.140140, Cenq = 0.0;
        constexpr double CDr = 0.0, CYr = 0.230299, CLr = 0.0, Cellr = 0.078165, Cemr = 0.0, Cenr = -0.089947;
        constexpr double alpha_stall = 0.3391428111, Cema_stall = 0.0;
        constexpr double M = 15, CD_fp_k1 = -0.224, CD_fp_k2 = -0.115;  // AdvancedLiftDrag defaults
        constexpr double cp = -0.12;  // m, center of pressure on body x
        constexpr double area = 0.34, mac = 0.22, density = 1.2041;
        // per degree of joint angle, times the joint's direction: servo_0, servo_1, servo_2, servo_3
        constexpr double CD_ctrl[4] = {-0.000059, -0.000059, -0.000274, 0.0};
        constexpr double CY_ctrl[4] = {0.000171, -0.000171, 0.0, -0.003913};
        constexpr double CL_ctrl[4] = {-0.011940, -0.011940, -0.010696, 0.0};
        constexpr double Cell_ctrl[4] = {-0.003331, 0.003331, 0.0, -0.000257};
        constexpr double Cem_ctrl[4] = {0.001498, 0.001498, 0.025798, 0.0};
        constexpr double Cen_ctrl[4] = {-0.000057, 0.000057, 0.0, 0.001613};
        // propeller 0.3 m ahead of the center of mass, thrust along body x
        constexpr double rotor_x = 0.3, rotor_speed_min = 10, rotor_speed_max = 1600, motor_constant = 8.54858e-06;
        constexpr double moment_constant = 0.01, rotor_drag = 8.06428e-05, rolling_moment = 1e-06;
    }

    template <typename T>
    void advanced_plane(Parameters<T>& p){
        // base link 1 kg + propeller 5 g at 0.3 m (parallel axis); Gazebo's default world: g = 9.8
        p.mass = T(1.005);
        p.gravity[0] = 0; p.gravity[1] = 0; p.gravity[2] = T(-9.8);
        T rotor = T(0.005 * advanced::rotor_x * advanced::rotor_x);
        T J[3][3] = {{T(0.197563 + 9.75e-07), 0, 0}, {0, T(0.1458929 + 0.000166704) + rotor, 0}, {0, 0, T(0.1477 + 0.000167604) + rotor}};
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) p.J[i][j] = J[i][j];
        invert3(p.J, p.J_inv);
        p.air_density = T(advanced::density);
        p.wing_area = T(advanced::area);
        p.wing_span = std::sqrt(T(advanced::area * advanced::AR));
        p.mean_chord = T(advanced::mac);
        for(int i = 0; i < N_COEFFICIENTS; i++) p.aerodynamics[i] = 0;  // the coefficients are in advanced::
        p.thrust_max = T(advanced::motor_constant * advanced::rotor_speed_max * advanced::rotor_speed_max);  // 21.9 N
        // servos: Gazebo's joint position controller (gain 10) against the joint damping (1): first order, 0.1 s.
        // Rotor speed: 12.5 ms up, 25 ms down (timeConstantUp, timeConstantDown)
        T tau[N_ACTUATORS] = {T(0.1), T(0.1), T(0.1), T(0.0125)};
        p.throttle_time_constant_down = T(0.025);
        T surface = T(45 * PI / 180);
        for(int i = 0; i < N_ACTUATORS; i++){
            p.actuator_time_constants[i] = tau[i];
            p.actuator_limits[i][0] = i == THROTTLE ? T(0) : -surface;
            p.actuator_limits[i][1] = i == THROTTLE ? T(1) : surface;
        }
        p.dt = T(0.01);
        p.n_substeps = 1;
        p.latency_steps = 0;
        for(int i = 0; i < 3; i++) p.wind[i] = 0;
        for(int i = 0; i < 4 + N_COEFFICIENTS; i++) p.randomization[i] = 1;
    }

    // E-flite UMX Eratix 3D FF 450 mm (EFLU02050), the lab's airplane: a first model from its geometry and textbook
    // aerodynamics, before any identification. Every value is derived in docs/eratix_model.md (sources, photo scaling,
    // formulas, +- ranges); the comments here only say where each one comes from. [estimate] marks a value of ours.
    // Flat-plate foam airframe: symmetric, no camber, no incidence, no dihedral. Reference point: the manual's center of
    // gravity, 76 mm behind the (straight) wing leading edge, 63 % of the mean chord
    namespace eratix_coefficients{
        constexpr double mass = 0.0835;  // kg, manual, with the recommended 2S 300 mAh battery
        // [estimate] component mass breakdown, docs section 3: FLU = FRD for a diagonal tensor (products neglected)
        constexpr double Jx = 3.9e-4, Jy = 9.3e-4, Jz = 1.27e-3;  // kg m2
        constexpr double density = 1.204;  // kg/m3, sea level at 20 C, as the advanced_plane
        // span from the manual; area and mean chord [estimate] from the top-view product photo scaled on the span
        constexpr double span = 0.450, area = 0.0531, mac = 0.121;  // m, m2, m: aspect ratio 3.8
        constexpr double oswald = 0.8;  // [estimate] low-AR flat plate with a profile fuselage
        // attached flow, in the order of the Coefficient enum (per rad, rates per b/2V or c/2V), stability derivatives
        // about the center of gravity, signs as Beard & McLain: aileron > 0 rolls right, elevator > 0 (trailing edge
        // down) pitches down, rudder > 0 (trailing edge left) yaws left. All [estimate], docs section 4
        constexpr double linear[N_COEFFICIENTS] = {
            0.0, 4.2, 2.9, 0.63,  // C_L: 0 (flat, no incidence), alpha (Helmbold + tail), q (tail + downwash lag), elevator
            0.065, oswald,  // C_D_p: friction and bluntness of the foam, landing gear, motor; Oswald efficiency
            0.0, -0.85, -0.05, 0.6, 0.0, 0.36,  // C_Y: 0, beta (fin, profile fuselage, side force generators), p, r, -, rudder
            0.0, -0.03, -0.40, 0.12, 0.35, 0.02,  // C_l: 0, beta (fin above the CG), p (lifting line), r, aileron, rudder
            0.0, 0.65, -11.0, -1.13,  // C_m: 0, alpha (> 0: neutral point ahead of the manual CG), q (+ alpha dot), elevator
            0.0, 0.27, -0.02, -0.42, -0.03, -0.22,  // C_n: 0, beta, p, r, aileron (adverse yaw), rudder
        };
        // stall: the blend of the AdvancedLiftDrag system, sharper than its default (M 15) so that the post-stall model
        // weighs about 1 % at alpha = 0, as for the advanced_plane (it would be 8 % with M 15 and this stall angle)
        constexpr double alpha_stall = 0.20943951023931956;  // 12 deg [estimate]: C_L 0.88, flat plates of AR 2-4
        constexpr double M = 25;
        // after the stall the force on a flat plate is normal to it (C_D / C_L = tan alpha, Ortiz et al. 2015), of
        // C_N90 sqrt(|sin alpha|): 0.84 at 30 deg, C_N90 at 90 deg (ESDU 70015 for AR 3.8). The root is a fit of ours to
        // the plateau that low-AR plates show between the stall and 45 deg [estimate]; smoothed near 0
        constexpr double CN90 = 1.19;
        constexpr double CN_smoothing = 0.05;  // sin alpha below which the root becomes linear
        constexpr double Cm_post = -0.42;  // [estimate] C_m after the stall, times the same shape: the tail keeps pitching down
        // drag of the deflected surfaces, per rad2 [estimate]: a flat flap of that area at the deflection angle
        constexpr double CD_aileron = 0.4, CD_elevator = 0.2, CD_rudder = 0.15;
        // surface limits [estimate]: the high-rate throws of the 860 mm Eratix (mm at the trailing edge, its manual) over
        // the surface chords of this one scaled by 1.92; the 450 mm manual gives percentages only
        constexpr double aileron_max = 0.6981317007977318, elevator_max = 0.6981317007977318;  // 40 deg
        constexpr double rudder_max = 0.8726646259971648;  // 50 deg
        // servos: 0.13-0.14 s per 60 deg for the Spektrum 2.3-2.9 g linear servos (servo chart, 4.2 V); 95 % of a step in
        // that time is 3 time constants. The A202's own speed is not published
        constexpr double servo_time_constant = 0.045;  // s
        // propeller 148 mm x 35 mm (Horizon specs), 1404 3200 Kv motor (manual), 2S LiPo: the rotor speed is the one at
        // which the motor torque at the commanded duty balances the propeller torque at the current airspeed, so that
        // the thrust falls with the airspeed as the propeller unloads. C_T = C_T0 (1 - J / J0), C_P = C_P0 (1 - kP J),
        // J = V / (n D). All [estimate] except D and Kv, docs section 5
        constexpr double prop_diameter = 0.148;  // m
        constexpr double CT0 = 0.06, CP0 = 0.020, J0 = 0.35, kP = 2.0;
        constexpr double kv = 3200;  // rpm/V
        constexpr double battery_voltage = 7.6, resistance = 0.25, no_load_current = 0.5;  // V (2S, half charge), ohm, A
        constexpr double rotor_inertia = 3e-6;  // kg m2: propeller and motor bell
        // spin-up: rotor inertia over the slopes of the motor and propeller torques (about 0.05 s); spin-down slower,
        // the propeller alone braking
        constexpr double motor_time_constant_up = 0.05, motor_time_constant_down = 0.08;  // s
        // propeller turning clockwise seen from behind (a standard tractor propeller, [assumption]): the reaction
        // torque rolls the airplane left, the rotor's angular momentum points forward
        constexpr double rotation = 1;
    }

    template <typename T>
    void eratix_propeller(T air_density, T duty, T v_axial, T& thrust, T& torque, T& n){
        // rotor speed n [rev/s] where the motor torque kt ((duty U - kt omega) / R - I0) equals the propeller torque
        // rho D^5 C_P n^2 / (2 pi): a quadratic in n. No positive root: the motor cannot turn the propeller, no thrust
        using namespace eratix_coefficients;
        T D = T(prop_diameter), v = std::fmax(v_axial, T(0));  // flying backward: the static values
        T kt = T(60 / (2 * PI * kv));  // N m / A = V s / rad
        T a = air_density * D * D * D * D * D * T(CP0) / T(2 * PI);
        T b = -air_density * D * D * D * D * T(CP0 * kP) * v / T(2 * PI) + T(2 * PI) * kt * kt / T(resistance);
        T c = kt * (T(no_load_current) - duty * T(battery_voltage) / T(resistance));
        T discriminant = b * b - 4 * a * c;
        n = c < 0 && discriminant > 0 ? (-b + std::sqrt(discriminant)) / (2 * a) : T(0);
        // ideal speed controller and propeller, n D the speed of the blade tips per revolution
        thrust = air_density * D * D * D * D * T(CT0) * (n * n - n * v / (T(J0) * D));
        torque = air_density * D * D * D * D * D * T(CP0) * (n * n - T(kP) * n * v / D) / T(2 * PI);
    }

    template <typename T>
    void eratix(Parameters<T>& p){
        using namespace eratix_coefficients;
        p.mass = T(mass);
        p.gravity[0] = 0; p.gravity[1] = 0; p.gravity[2] = T(-9.81);
        T J[3][3] = {{T(Jx), 0, 0}, {0, T(Jy), 0}, {0, 0, T(Jz)}};
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) p.J[i][j] = J[i][j];
        invert3(p.J, p.J_inv);
        p.air_density = T(density);
        p.wing_area = T(area);
        p.wing_span = T(span);
        p.mean_chord = T(mac);
        // in p.aerodynamics, so that the randomization factors apply to them (the advanced_plane keeps its own)
        for(int i = 0; i < N_COEFFICIENTS; i++) p.aerodynamics[i] = T(linear[i]);
        // static thrust at full throttle (about 3.0 N, 3.6 times the weight): the propeller model is scaled by
        // thrust_max over this value, so that the thrust randomization factor applies to it
        T n, torque;
        eratix_propeller(p.air_density, T(1), T(0), p.thrust_max, torque, n);
        T tau[N_ACTUATORS] = {T(servo_time_constant), T(servo_time_constant), T(servo_time_constant), T(motor_time_constant_up)};
        p.throttle_time_constant_down = T(motor_time_constant_down);
        T limit[N_ACTUATORS] = {T(aileron_max), T(elevator_max), T(rudder_max), T(1)};
        for(int i = 0; i < N_ACTUATORS; i++){
            p.actuator_time_constants[i] = tau[i];
            p.actuator_limits[i][0] = i == THROTTLE ? T(0) : -limit[i];
            p.actuator_limits[i][1] = limit[i];
        }
        p.dt = T(0.01);
        p.n_substeps = 1;
        p.latency_steps = 0;
        for(int i = 0; i < 3; i++) p.wind[i] = 0;
        for(int i = 0; i < 4 + N_COEFFICIENTS; i++) p.randomization[i] = 1;
    }

    template <typename T>
    struct AirData{
        T v_frd[3];  // air-relative velocity, aerospace body frame (x forward, y right, z down)
        T airspeed;
        T alpha;
        T beta;
    };

    template <typename T>
    AirData<T> air_data(const Parameters<T>& p, const Body<T>& x){
        T R[3][3];
        rotation_matrix(x.orientation, R);
        T air[3], body[3];
        for(int i = 0; i < 3; i++) air[i] = x.linear_velocity[i] - p.wind[i];
        rotate_inverse(R, air, body);
        AirData<T> a;
        a.v_frd[0] = body[0]; a.v_frd[1] = -body[1]; a.v_frd[2] = -body[2];
        a.airspeed = std::sqrt(a.v_frd[0] * a.v_frd[0] + a.v_frd[1] * a.v_frd[1] + a.v_frd[2] * a.v_frd[2]);
        if(a.airspeed < T(MIN_AIRSPEED)){
            a.alpha = 0; a.beta = 0;
            return a;
        }
        a.alpha = std::atan2(a.v_frd[2], a.v_frd[0]);
        a.beta = std::asin(clip(a.v_frd[1] / a.airspeed, T(-1), T(1)));
        return a;
    }

    template <typename T>
    void advanced_aerodynamics(const Parameters<T>& p, const Body<T>& x, T force[3], T moment[3]){
        // AdvancedLiftDrag::Update and MulticopterMotorModel (velocity motor), written in the body frame
        using namespace advanced;
        T R[3][3];
        rotation_matrix(x.orientation, R);
        T air[3], body[3];
        for(int i = 0; i < 3; i++) air[i] = x.linear_velocity[i] - p.wind[i];
        rotate_inverse(R, air, body);
        const T* w = x.angular_velocity;  // FLU
        // air velocity at the center of pressure (-cp, 0, 0 in FLU), in FRD
        T u = body[0], v = -(body[1] + w[2] * T(cp)), wz = -(body[2] - w[1] * T(cp));
        T f[3] = {0, 0, 0}, m[3] = {0, 0, 0};  // FRD
        T V = std::sqrt(u * u + wz * wz);  // speed in the lift-drag plane: the sideslip part does not count
        if(V > T(1e-9)){
            T S = T(area), b = p.wing_span, c = T(mac);
            T alpha = std::atan2(wz, u), beta = std::atan2(v, u);
            T sa = std::sin(alpha), ca = std::cos(alpha);
            T P = w[0], Q = -w[1], Rr = -w[2];  // FRD rates
            T dyn = T(0.5) * p.air_density * V * V, hrv = T(0.5) * p.air_density * V;
            T up = std::exp(-T(M) * (alpha - T(alpha_stall))), down = std::exp(T(M) * (alpha + T(alpha_stall)));
            T sigma = (1 + up + down) / ((1 + up) * (1 + down));  // 0 before the stall, 1 after it
            // joint angles in degrees: servo_0 = aileron, servo_1 = -aileron, servo_2 = elevator, servo_3 = rudder
            T deg = T(180 / PI);
            T angle[4] = {x.actuators[AILERON] * deg, -x.actuators[AILERON] * deg, x.actuators[ELEVATOR] * deg, x.actuators[RUDDER] * deg};
            T CL_c = 0, CD_c = 0, CY_c = 0, Cell_c = 0, Cem_c = 0, Cen_c = 0;
            for(int i = 0; i < 4; i++){
                CL_c += angle[i] * T(CL_ctrl[i]); CD_c += angle[i] * T(CD_ctrl[i]); CY_c += angle[i] * T(CY_ctrl[i]);
                Cell_c += angle[i] * T(Cell_ctrl[i]); Cem_c += angle[i] * T(Cem_ctrl[i]); Cen_c += angle[i] * T(Cen_ctrl[i]);
            }
            T sign = alpha >= 0 ? T(1) : T(-1);
            T CL = (1 - sigma) * (T(CL0) + T(CLa) * alpha) + sigma * (2 * sign * sa * sa * ca) + CL_c;
            T lift = (CL * dyn + T(CLp) * (P * b / 2) * hrv + T(CLq) * (Q * c / 2) * hrv + T(CLr) * (Rr * b / 2) * hrv) * S;
            T CD_fp = 2 / (1 + std::exp(T(CD_fp_k1) + T(CD_fp_k2) * std::fmax(T(AR), 1 / T(AR))));
            T CD = (1 - sigma) * (T(CD0) + CL * CL / (T(PI) * T(AR) * T(eff))) + sigma * std::abs(CD_fp * (T(0.5) - T(0.5) * std::cos(2 * alpha))) + CD_c;
            T drag = (CD * dyn + T(CDp) * (P * b / 2) * hrv + T(CDq) * (Q * c / 2) * hrv + T(CDr) * (Rr * b / 2) * hrv) * S;
            T CY = T(CYb) * beta + CY_c;
            T side = (CY * dyn + T(CYp) * (P * b / 2) * hrv + T(CYq) * (Q * c / 2) * hrv + T(CYr) * (Rr * b / 2) * hrv) * S;
            T Cem = alpha > T(alpha_stall) ? T(Cem0) + T(Cema) * T(alpha_stall) + T(Cema_stall) * (alpha - T(alpha_stall))
                  : alpha < -T(alpha_stall) ? T(Cem0) - T(Cema) * T(alpha_stall) + T(Cema_stall) * (alpha + T(alpha_stall))
                  : T(Cem0) + T(Cema) * alpha;
            Cem += Cem_c;
            T Cell = T(Cellb) * beta + Cell_c, Cen = T(Cenb) * beta + Cen_c;
            // lift along -z of the stability axes, (sin alpha, 0, -cos alpha); drag along -(cos alpha, 0, sin alpha)
            f[0] = lift * sa - drag * ca;
            f[1] = side;
            f[2] = -lift * ca - drag * sa;
            m[0] = (Cell * dyn + T(Cellp) * (P * b / 2) * hrv + T(Cellq) * (Q * c / 2) * hrv + T(Cellr) * (Rr * b / 2) * hrv) * S * b;
            m[1] = (Cem * dyn + T(Cemp) * (P * b / 2) * hrv + T(Cemq) * (Q * c / 2) * hrv + T(Cemr) * (Rr * b / 2) * hrv) * S * c;
            m[2] = (Cen * dyn + T(Cenp) * (P * b / 2) * hrv + T(Cenq) * (Q * c / 2) * hrv + T(Cenr) * (Rr * b / 2) * hrv) * S * b;
            // the force acts at the center of pressure (cp, 0, 0): cp x f
            m[1] += -T(cp) * f[2];
            m[2] += T(cp) * f[1];
        }
        force[0] = f[0]; force[1] = -f[1]; force[2] = -f[2];  // FRD -> FLU
        moment[0] = m[0]; moment[1] = -m[1]; moment[2] = -m[2];
        // propeller, in FLU: thrust along x, reaction torque about x, rotor drag against the air velocity across the axis
        T omega = T(rotor_speed_min) + (T(rotor_speed_max) - T(rotor_speed_min)) * x.actuators[THROTTLE];
        T thrust = p.thrust_max * (omega / T(rotor_speed_max)) * (omega / T(rotor_speed_max));
        force[0] += thrust;
        moment[0] += T(moment_constant) * thrust;
        T across[3] = {0, body[1] + w[2] * T(rotor_x), body[2] - w[1] * T(rotor_x)};  // air velocity of the rotor, minus x
        T rotor_force[3] = {0, -omega * T(rotor_drag) * across[1], -omega * T(rotor_drag) * across[2]};
        for(int i = 1; i < 3; i++){
            force[i] += rotor_force[i];
            moment[i] += -omega * T(rolling_moment) * across[i];
        }
        moment[1] += -T(rotor_x) * rotor_force[2];  // (rotor_x, 0, 0) x rotor_force
        moment[2] += T(rotor_x) * rotor_force[1];
    }

    template <typename T>
    void eratix_aerodynamics(const Parameters<T>& p, const Body<T>& x, T force[3], T moment[3]){
        // the structure of advanced_aerodynamics (attached-flow coefficients blended into a flat-plate model past the
        // stall, rate terms in 0.5 rho V, so that they stay finite in slow flight), with the linear coefficients of
        // p.aerodynamics (randomized), the full airspeed and sideslip as aerosonde, the pitching moment blended too
        // (this airplane flies past the stall), the drag of the deflected surfaces and a motor-propeller model.
        // A separate function so that the advanced_plane keeps exactly its results. Not modelled: the propeller's
        // slipstream over the wing and tail (hover, torque roll), its normal force, ground effect
        using namespace eratix_coefficients;
        const T* c = p.aerodynamics;
        AirData<T> a = air_data(p, x);
        const T* w = x.angular_velocity;  // FLU
        T f[3] = {0, 0, 0}, m[3] = {0, 0, 0};  // FRD
        T delta_a = x.actuators[AILERON], delta_e = x.actuators[ELEVATOR], delta_r = x.actuators[RUDDER];
        if(a.airspeed >= T(MIN_AIRSPEED)){
            T S = p.wing_area, b = p.wing_span, cbar = p.mean_chord, V = a.airspeed;
            T alpha = a.alpha, beta = a.beta, sa = std::sin(alpha), ca = std::cos(alpha);
            T P = w[0], Q = -w[1], Rr = -w[2];  // FRD rates
            T dyn = T(0.5) * p.air_density * V * V, hrv = T(0.5) * p.air_density * V;
            T up = std::exp(-T(M) * (alpha - T(alpha_stall))), down = std::exp(T(M) * (alpha + T(alpha_stall)));
            T sigma = (1 + up + down) / ((1 + up) * (1 + down));  // 0 before the stall, 1 after it
            // flat plate past the stall: normal force C_N90 sqrt(|sin alpha|), with the sign of sin alpha
            T shape = sa / std::sqrt(std::sqrt(sa * sa + T(CN_smoothing * CN_smoothing)));
            T CN = T(CN90) * shape;
            T aspect_ratio = b * b / S;
            T CL_attached = c[C_L_0] + c[C_L_alpha] * alpha + c[C_L_delta_e] * delta_e;
            T CL = (1 - sigma) * (c[C_L_0] + c[C_L_alpha] * alpha) + sigma * CN * ca + c[C_L_delta_e] * delta_e;
            T CD = (1 - sigma) * (c[C_D_p] + CL_attached * CL_attached / (T(PI) * c[oswald_efficiency] * aspect_ratio))
                 + sigma * (c[C_D_p] + CN * sa)
                 + T(CD_aileron) * delta_a * delta_a + T(CD_elevator) * delta_e * delta_e + T(CD_rudder) * delta_r * delta_r;
            T lift = (CL * dyn + c[C_L_q] * (Q * cbar / 2) * hrv) * S;
            T drag = CD * dyn * S;
            T CY = c[C_Y_0] + c[C_Y_beta] * beta + c[C_Y_delta_a] * delta_a + c[C_Y_delta_r] * delta_r;
            T side = (CY * dyn + c[C_Y_p] * (P * b / 2) * hrv + c[C_Y_r] * (Rr * b / 2) * hrv) * S;
            T Cl = c[C_l_0] + c[C_l_beta] * beta + c[C_l_delta_a] * delta_a + c[C_l_delta_r] * delta_r;
            T Cm = (1 - sigma) * (c[C_m_0] + c[C_m_alpha] * alpha) + sigma * T(Cm_post) * shape + c[C_m_delta_e] * delta_e;
            T Cn = c[C_n_0] + c[C_n_beta] * beta + c[C_n_delta_a] * delta_a + c[C_n_delta_r] * delta_r;
            // lift and drag in the stability axes, as aerosonde
            f[0] = lift * sa - drag * ca;
            f[1] = side;
            f[2] = -lift * ca - drag * sa;
            m[0] = (Cl * dyn + c[C_l_p] * (P * b / 2) * hrv + c[C_l_r] * (Rr * b / 2) * hrv) * S * b;
            m[1] = (Cm * dyn + c[C_m_q] * (Q * cbar / 2) * hrv) * S * cbar;
            m[2] = (Cn * dyn + c[C_n_p] * (P * b / 2) * hrv + c[C_n_r] * (Rr * b / 2) * hrv) * S * b;
        }
        force[0] = f[0]; force[1] = -f[1]; force[2] = -f[2];  // FRD -> FLU
        moment[0] = m[0]; moment[1] = -m[1]; moment[2] = -m[2];
        // propeller on the thrust line through the center of gravity, in FLU; the thrust randomization factor scales
        // thrust and torque together
        T thrust, torque, n, static_thrust, static_torque, static_n;
        T v_axial = a.v_frd[0];
        eratix_propeller(p.air_density, x.actuators[THROTTLE], v_axial, thrust, torque, n);
        eratix_propeller(p.air_density, T(1), T(0), static_thrust, static_torque, static_n);
        T scale = p.thrust_max / static_thrust;
        force[0] += scale * thrust;
        moment[0] += -T(rotation) * scale * torque;  // reaction of the propeller's drag torque
        // the spinning rotor's angular momentum H along x: the airframe feels -w x H (gyroscopic moment)
        T H = T(rotation) * T(rotor_inertia) * T(2 * PI) * n;
        moment[1] += -w[2] * H;
        moment[2] += w[1] * H;
    }

    template <typename T>
    void aerodynamics(const Parameters<T>& p, const Body<T>& x, T force[3], T moment[3]){
        // aerodynamic + propulsion force and moment in the body frame (x forward, y left, z up)
        if constexpr(FW_AIRPLANE == 1){
            advanced_aerodynamics(p, x, force, moment);
            return;
        }
        if constexpr(FW_AIRPLANE == 2){
            eratix_aerodynamics(p, x, force, moment);
            return;
        }
        const T* c = p.aerodynamics;
        T delta_a = x.actuators[AILERON], delta_e = x.actuators[ELEVATOR], delta_r = x.actuators[RUDDER];
        T f[3] = {x.actuators[THROTTLE] * p.thrust_max, 0, 0};  // FRD
        T m[3] = {0, 0, 0};
        AirData<T> a = air_data(p, x);
        if(a.airspeed >= T(MIN_AIRSPEED)){
            T P = x.angular_velocity[0], Q = -x.angular_velocity[1], Rr = -x.angular_velocity[2];  // FRD rates
            T p_hat = P * p.wing_span / (2 * a.airspeed);
            T q_hat = Q * p.mean_chord / (2 * a.airspeed);
            T r_hat = Rr * p.wing_span / (2 * a.airspeed);
            T C_L = c[C_L_0] + c[C_L_alpha] * a.alpha + c[C_L_q] * q_hat + c[C_L_delta_e] * delta_e;
            T aspect_ratio = p.wing_span * p.wing_span / p.wing_area;
            T C_D = c[C_D_p] + C_L * C_L / (T(PI) * c[oswald_efficiency] * aspect_ratio);
            T C_Y = c[C_Y_0] + c[C_Y_beta] * a.beta + c[C_Y_p] * p_hat + c[C_Y_r] * r_hat + c[C_Y_delta_a] * delta_a + c[C_Y_delta_r] * delta_r;
            T C_l = c[C_l_0] + c[C_l_beta] * a.beta + c[C_l_p] * p_hat + c[C_l_r] * r_hat + c[C_l_delta_a] * delta_a + c[C_l_delta_r] * delta_r;
            T C_m = c[C_m_0] + c[C_m_alpha] * a.alpha + c[C_m_q] * q_hat + c[C_m_delta_e] * delta_e;
            T C_n = c[C_n_0] + c[C_n_beta] * a.beta + c[C_n_p] * p_hat + c[C_n_r] * r_hat + c[C_n_delta_a] * delta_a + c[C_n_delta_r] * delta_r;
            T qS = T(0.5) * p.air_density * a.airspeed * a.airspeed * p.wing_area;
            T sa = std::sin(a.alpha), ca = std::cos(a.alpha);
            f[0] += qS * (-C_D * ca + C_L * sa);
            f[1] += qS * C_Y;
            f[2] += qS * (-C_D * sa - C_L * ca);
            m[0] += qS * p.wing_span * C_l;
            m[1] += qS * p.mean_chord * C_m;
            m[2] += qS * p.wing_span * C_n;
        }
        force[0] = f[0]; force[1] = -f[1]; force[2] = -f[2];  // FRD -> FLU
        moment[0] = m[0]; moment[1] = -m[1]; moment[2] = -m[2];
    }

    template <typename T>
    void derivative(const Parameters<T>& p, const Body<T>& x, const T command[N_ACTUATORS], Body<T>& dx){
        // command: actuator positions held over the substep, same units as Body::actuators
        T force[3], moment[3];
        aerodynamics(p, x, force, moment);
        T R[3][3];
        rotation_matrix(x.orientation, R);
        T force_world[3];
        rotate(R, force, force_world);
        for(int i = 0; i < 3; i++){
            dx.position[i] = x.linear_velocity[i];
            dx.linear_velocity[i] = force_world[i] / p.mass + p.gravity[i];
        }
        T omega_quaternion[4] = {0, x.angular_velocity[0], x.angular_velocity[1], x.angular_velocity[2]};
        quaternion_multiply(x.orientation, omega_quaternion, dx.orientation);
        for(int i = 0; i < 4; i++) dx.orientation[i] *= T(0.5);
        const T* w = x.angular_velocity;
        T Jw[3];
        for(int i = 0; i < 3; i++) Jw[i] = p.J[i][0] * w[0] + p.J[i][1] * w[1] + p.J[i][2] * w[2];
        T rhs[3] = {
            moment[0] - (w[1] * Jw[2] - w[2] * Jw[1]),
            moment[1] - (w[2] * Jw[0] - w[0] * Jw[2]),
            moment[2] - (w[0] * Jw[1] - w[1] * Jw[0]),
        };
        for(int i = 0; i < 3; i++) dx.angular_velocity[i] = p.J_inv[i][0] * rhs[0] + p.J_inv[i][1] * rhs[1] + p.J_inv[i][2] * rhs[2];
        for(int i = 0; i < N_ACTUATORS; i++){
            T tau = p.actuator_time_constants[i];
            if(i == THROTTLE && p.throttle_time_constant_down > 0 && command[i] < x.actuators[i]) tau = p.throttle_time_constant_down;
            dx.actuators[i] = (command[i] - x.actuators[i]) / tau;
        }
    }

    template <typename T>
    void add_scaled(const Body<T>& x, const Body<T>& dx, T scale, Body<T>& out){
        for(int i = 0; i < 3; i++){
            out.position[i] = x.position[i] + scale * dx.position[i];
            out.linear_velocity[i] = x.linear_velocity[i] + scale * dx.linear_velocity[i];
            out.angular_velocity[i] = x.angular_velocity[i] + scale * dx.angular_velocity[i];
        }
        for(int i = 0; i < 4; i++) out.orientation[i] = x.orientation[i] + scale * dx.orientation[i];
        for(int i = 0; i < N_ACTUATORS; i++) out.actuators[i] = x.actuators[i] + scale * dx.actuators[i];
    }

    template <typename T>
    void rk4(const Parameters<T>& p, const Body<T>& x, const T command[N_ACTUATORS], T dt, Body<T>& out){
        Body<T> k1, k2, k3, k4, tmp;
        derivative(p, x, command, k1);
        add_scaled(x, k1, dt / 2, tmp);
        derivative(p, tmp, command, k2);
        add_scaled(x, k2, dt / 2, tmp);
        derivative(p, tmp, command, k3);
        add_scaled(x, k3, dt, tmp);
        derivative(p, tmp, command, k4);
        add_scaled(x, k1, dt / 6, out);
        add_scaled(out, k2, dt / 3, out);
        add_scaled(out, k3, dt / 3, out);
        add_scaled(out, k4, dt / 6, out);
    }

    template <typename T>
    void scale_action(const Parameters<T>& p, const T action[N_ACTUATORS], T out[N_ACTUATORS]){
        // [-1, 1] -> [actuator_limits min, max]
        for(int i = 0; i < N_ACTUATORS; i++){
            T lo = p.actuator_limits[i][0], hi = p.actuator_limits[i][1];
            out[i] = lo + (clip(action[i], T(-1), T(1)) + 1) / 2 * (hi - lo);
        }
    }

    template <typename T>
    void normalize_action(const Parameters<T>& p, const T scaled[N_ACTUATORS], T out[N_ACTUATORS]){
        for(int i = 0; i < N_ACTUATORS; i++){
            T lo = p.actuator_limits[i][0], hi = p.actuator_limits[i][1];
            out[i] = (scaled[i] - lo) / (hi - lo) * 2 - 1;
        }
    }

    template <typename T>
    void post_integration(const Parameters<T>& p, Body<T>& x){
        T n = std::sqrt(x.orientation[0] * x.orientation[0] + x.orientation[1] * x.orientation[1]
                      + x.orientation[2] * x.orientation[2] + x.orientation[3] * x.orientation[3]);
        for(int i = 0; i < 4; i++) x.orientation[i] /= n;
        for(int i = 0; i < N_ACTUATORS; i++) x.actuators[i] = clip(x.actuators[i], p.actuator_limits[i][0], p.actuator_limits[i][1]);
    }

    template <typename T>
    void simulate(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], State<T>& next){
        // one control period: latency buffer, n_substeps RK4 steps, post-integration once (as in L2F), action history
        next = state;
        T clipped[N_ACTUATORS], command[N_ACTUATORS], applied[N_ACTUATORS];
        for(int i = 0; i < N_ACTUATORS; i++) clipped[i] = clip(action[i], T(-1), T(1));
        scale_action(p, clipped, command);
        int n = p.latency_steps;
        if(n == 0){
            for(int i = 0; i < N_ACTUATORS; i++) applied[i] = command[i];
        }
        else{
            for(int i = 0; i < N_ACTUATORS; i++) applied[i] = state.action_buffer[0][i];
            for(int k = 0; k + 1 < n; k++) for(int i = 0; i < N_ACTUATORS; i++) next.action_buffer[k][i] = state.action_buffer[k + 1][i];
            for(int i = 0; i < N_ACTUATORS; i++) next.action_buffer[n - 1][i] = command[i];
        }
        T h = p.dt / p.n_substeps;
        Body<T> x = state.body;
        for(int k = 0; k < p.n_substeps; k++){
            Body<T> y;
            rk4(p, x, applied, h, y);
            x = y;
        }
        post_integration(p, x);
        next.body = x;
        for(int k = 0; k + 1 < ACTION_HISTORY; k++) for(int i = 0; i < N_ACTUATORS; i++) next.action_history[k][i] = state.action_history[k + 1][i];
        for(int i = 0; i < N_ACTUATORS; i++) next.action_history[ACTION_HISTORY - 1][i] = clipped[i];
    }

    // ----- trim ------------------------------------------------------------------------------------------------------
    template <typename T>
    Body<T> trim_body(const Parameters<T>& p, T airspeed, T pitch, T elevator, T throttle){
        Body<T> x{};
        x.orientation[0] = std::cos(pitch / 2); x.orientation[2] = -std::sin(pitch / 2);  // nose up: negative about y
        x.linear_velocity[0] = airspeed + p.wind[0];  // level flight relative to the air mass, along world x
        x.linear_velocity[1] = p.wind[1];
        x.linear_velocity[2] = p.wind[2];
        x.actuators[ELEVATOR] = elevator;
        x.actuators[THROTTLE] = throttle;
        return x;
    }

    template <typename T>
    void trim_residual(const Parameters<T>& p, T airspeed, const T z[3], T r[3]){
        Body<T> x = trim_body(p, airspeed, z[0], z[1], z[2]), dx;
        derivative(p, x, x.actuators, dx);
        r[0] = dx.linear_velocity[0]; r[1] = dx.linear_velocity[2]; r[2] = dx.angular_velocity[1];
    }

    template <typename T>
    Body<T> trim(const Parameters<T>& p, T airspeed){
        // steady level flight: Newton on pitch, elevator, throttle (trim.py, default mode)
        T z[3] = {T(0.05), T(0), T(0.5)}, r[3];
        T tolerance = sizeof(T) == sizeof(double) ? T(1e-10) : T(1e-4);
        for(int iteration = 0; iteration < 50; iteration++){
            trim_residual(p, airspeed, z, r);
            if(std::abs(r[0]) < tolerance && std::abs(r[1]) < tolerance && std::abs(r[2]) < tolerance){
                return trim_body(p, airspeed, z[0], z[1], z[2]);
            }
            T jacobian[3][3], inverse[3][3];
            T epsilon = sizeof(T) == sizeof(double) ? T(1e-6) : T(1e-3);
            for(int j = 0; j < 3; j++){
                T zp[3] = {z[0], z[1], z[2]}, rp[3];
                zp[j] += epsilon;
                trim_residual(p, airspeed, zp, rp);
                for(int i = 0; i < 3; i++) jacobian[i][j] = (rp[i] - r[i]) / epsilon;
            }
            if(!invert3(jacobian, inverse)) break;
            for(int i = 0; i < 3; i++) z[i] -= inverse[i][0] * r[0] + inverse[i][1] * r[1] + inverse[i][2] * r[2];
        }
        throw std::runtime_error("trim did not converge");
    }

    // ----- task ------------------------------------------------------------------------------------------------------
    template <typename T>
    struct Measurements{
        T airspeed, altitude, course;
        T R[3][3];
        T rates[3];
        T velocity_body[3];
    };

    template <typename T>
    Measurements<T> measurements(const Parameters<T>& p, const Body<T>& x){
        // what PX4 estimates: shared by the actor observation and the reward
        Measurements<T> m;
        rotation_matrix(x.orientation, m.R);
        m.airspeed = air_data(p, x).airspeed;
        m.altitude = x.position[2];
        m.course = std::atan2(x.linear_velocity[1], x.linear_velocity[0]);
        for(int i = 0; i < 3; i++) m.rates[i] = x.angular_velocity[i];
        rotate_inverse(m.R, x.linear_velocity, m.velocity_body);
        return m;
    }

    template <typename T>
    void errors(const Measurements<T>& m, const State<T>& s, T& e_airspeed, T& e_altitude, T& e_course){
        e_airspeed = m.airspeed - s.setpoint_airspeed;
        e_altitude = m.altitude - s.setpoint_altitude;
        e_course = wrap(m.course - s.setpoint_course);
    }

    template <typename T>
    void tracking_errors(const State<T>& s, T e_position[3], T e_velocity[3]){
        // tracking task: position and ground velocity minus the reference's, world frame
        for(int i = 0; i < 3; i++){
            e_position[i] = s.body.position[i] - s.reference_position[i];
            e_velocity[i] = s.body.linear_velocity[i] - s.reference_velocity[i];
        }
    }

    template <typename T>
    T norm3(const T v[3]){ return std::sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]); }

    template <typename T, typename WRITE>
    void observe_actor(const Parameters<T>& p, const State<T>& s, WRITE write){
        // same layout as environment.observe_actor; noise-free (observation noise is not ported yet)
        Measurements<T> m = measurements(p, s.body);
        int k = 0;
        if constexpr(FW_TRACKING){
            // in the body frame, so that the policy does not depend on the heading; position clipped at 200 m
            T e_position[3], e_velocity[3], body[3];
            tracking_errors(s, e_position, e_velocity);
            rotate_inverse(m.R, e_position, body);
            for(int i = 0; i < 3; i++) write(k++, clip(body[i] / T(20), T(-10), T(10)));
            rotate_inverse(m.R, e_velocity, body);
            for(int i = 0; i < 3; i++) write(k++, body[i] / 5);
        }
        else{
            T e_airspeed, e_altitude, e_course;
            errors(m, s, e_airspeed, e_altitude, e_course);
            write(k++, e_airspeed / 5);
            write(k++, e_altitude / 20);
            write(k++, std::sin(e_course));
            write(k++, std::cos(e_course));
        }
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) write(k++, m.R[i][j]);
        for(int i = 0; i < 3; i++) write(k++, m.rates[i] / T(PI));
        for(int i = 0; i < 3; i++) write(k++, m.velocity_body[i] / 25);
        write(k++, m.airspeed / 25);
        if constexpr(N_INTEGRALS > 0){
            if constexpr(FW_TRACKING){
                T body[3];
                rotate_inverse(m.R, s.error_integral, body);
                for(int i = 0; i < 3; i++) write(k++, clip(body[i] / p.position_integral_scale, T(-10), T(10)));
            }
            else{
                for(int i = 0; i < 3; i++) write(k++, s.error_integral[i] / p.error_integral_limits[i]);
            }
        }
        for(int h = 0; h < ACTION_HISTORY; h++) for(int i = 0; i < N_ACTUATORS; i++) write(k++, s.action_history[h][i]);
    }

    template <typename T, typename WRITE>
    void observe_critic(const Parameters<T>& p, const State<T>& s, WRITE write){
        // actor observation + air data, true wind, actuator positions, randomization factors, trim commands (last)
        observe_actor(p, s, write);
        int k = ActorObservation<int>::DIM;
        AirData<T> a = air_data(p, s.body);
        write(k++, a.alpha / T(0.3));
        write(k++, a.beta / T(0.3));
        write(k++, a.airspeed / 25);
        for(int i = 0; i < 3; i++) write(k++, p.wind[i] / 10);
        T normalized[N_ACTUATORS];
        normalize_action(p, s.body.actuators, normalized);
        for(int i = 0; i < N_ACTUATORS; i++) write(k++, normalized[i]);
        for(int i = 0; i < 4 + N_COEFFICIENTS; i++) write(k++, (p.randomization[i] - 1) / T(0.3));
        for(int i = 0; i < N_ACTUATORS; i++) write(k++, s.trim_action[i]);
    }

    template <typename T>
    T alpha_excess(T alpha, const RewardConfig<T>& c){
        // how far the angle of attack is inside the margin before the stall limits (0 away from them)
        return std::fmax(alpha - c.alpha_soft_max, T(0)) + std::fmax(c.alpha_soft_min - alpha, T(0));
    }

    template <typename T>
    T tracking_cost(T x, const RewardConfig<T>& c){
        // scaled tracking error: square, or Huber with the same value near 1
        if(c.huber_delta <= 0) return x * x;
        return std::abs(x) < c.huber_delta ? x * x / (2 * c.huber_delta) : std::abs(x) - c.huber_delta / 2;
    }

    template <typename T>
    T reward_from_cost(T tracking, T far, T beta, T alpha, T action_change_squared, T trim_deviation_squared, const RewardConfig<T>& c){
        // the terms shared by both tasks around the tracking cost and the far penalty of each
        auto sq = [](T x){ return x * x; };
        T cost = tracking
               + c.sideslip_weight * sq(beta / c.sideslip_scale)
               + c.alpha_margin_weight * sq(alpha_excess(alpha, c) / c.alpha_margin_scale)
               + c.action_change_weight * action_change_squared
               + c.trim_weight * trim_deviation_squared;
        T cap = c.cost_cap * c.survival;
        return c.survival - cap * std::tanh(cost / cap) - c.far_weight * far;
    }

    template <typename T>
    T reward_from_errors(T e_airspeed, T e_altitude, T e_course, T beta, T alpha, T action_change_squared, T trim_deviation_squared, const RewardConfig<T>& c){
        T tracking = c.airspeed_weight * tracking_cost(e_airspeed / c.airspeed_scale, c)
                   + c.altitude_weight * tracking_cost(e_altitude / c.altitude_scale, c)
                   + c.course_weight * tracking_cost(e_course / c.course_scale, c);
        T far = std::fmin(std::abs(e_airspeed) / c.far_airspeed_scale, T(1))
              + std::fmin(std::abs(e_altitude) / c.far_altitude_scale, T(1))
              + std::fmin(std::abs(e_course) / c.far_course_scale, T(1));
        return reward_from_cost(tracking, far, beta, alpha, action_change_squared, trim_deviation_squared, c);
    }

    template <typename T>
    T reward_from_tracking_errors(const T e_position[3], const T e_velocity[3], T beta, T alpha, T action_change_squared, T trim_deviation_squared, const RewardConfig<T>& c){
        // tracking task: errors to the reference position [m] and velocity [m/s], in any frame (only norms are used)
        T position_error = norm3(e_position);
        // error to the commanded velocity: reference velocity - gain x position error, the correction clipped
        T correction = c.velocity_guidance_gain * position_error;
        T scale = correction > c.velocity_guidance_max ? c.velocity_guidance_max / correction : T(1);
        T e_guided[3];
        for(int i = 0; i < 3; i++) e_guided[i] = e_velocity[i] + scale * c.velocity_guidance_gain * e_position[i];
        T velocity_error = norm3(e_guided);
        T tracking = c.position_weight * tracking_cost(position_error / c.position_scale, c)
                   + c.velocity_weight * tracking_cost(velocity_error / c.velocity_scale, c);
        T far = std::fmin(position_error / c.far_position_scale, T(1)) + std::fmin(velocity_error / c.far_velocity_scale, T(1));
        // the margin to the stall is outside the cap here: tens of meters from the point the position cost saturates
        // the tanh, and a cost added inside it changed the reward by almost nothing, while the early stalls happen
        // right there. At most the weight, at the stall limit; the reward of a step can then go below 0 near the
        // limit only, and the flight earns about 0.5 per step again once away from it, so stalling never pays
        RewardConfig<T> inside = c;
        inside.alpha_margin_weight = 0;
        T margin = std::fmin(alpha_excess(alpha, c) / c.alpha_margin_scale, T(1));
        return reward_from_cost(tracking, far, beta, alpha, action_change_squared, trim_deviation_squared, inside)
             - c.position_linear_weight * std::fmin(position_error / c.far_position_scale, T(1))
             - c.alpha_margin_weight * margin * margin;
    }

    // why a flight ends: 0 = still flying
    enum Termination{FLYING = 0, DIVERGED, LOW_ALTITUDE, ALPHA, AIRSPEED, ALTITUDE, POSITION};

    template <typename T>
    Termination termination(const Parameters<T>& p, const State<T>& s, const TaskConfig<T>& config){
        for(int i = 0; i < 3; i++) if(!std::isfinite(s.body.linear_velocity[i])) return DIVERGED;
        if(s.body.position[2] < config.altitude_floor) return LOW_ALTITUDE;
        AirData<T> a = air_data(p, s.body);
        if(!(config.alpha_range[0] <= a.alpha && a.alpha <= config.alpha_range[1])) return ALPHA;
        if(!(config.airspeed_limits[0] <= a.airspeed && a.airspeed <= config.airspeed_limits[1])) return AIRSPEED;
        if(std::abs(s.body.position[2] - s.setpoint_altitude) > config.altitude_error_max) return ALTITUDE;
        if constexpr(FW_TRACKING){
            T dx = s.body.position[0] - s.reference_position[0], dy = s.body.position[1] - s.reference_position[1];
            if(std::sqrt(dx * dx + dy * dy) > config.position_error_max) return POSITION;
        }
        return FLYING;
    }

    template <typename T>
    T reward(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], const State<T>& next, const TaskConfig<T>& config){
        // on the state after the step; zero when the flight ends, as in FixedWingEnv.step
        if(termination(p, next, config) != FLYING) return 0;
        AirData<T> air = air_data(p, next.body);
        T change = 0, deviation = 0;
        for(int i = 0; i < N_ACTUATORS; i++){
            T a = clip(action[i], T(-1), T(1));
            change += (a - state.action_history[ACTION_HISTORY - 1][i]) * (a - state.action_history[ACTION_HISTORY - 1][i]);
            deviation += (a - state.trim_action[i]) * (a - state.trim_action[i]);
        }
        if constexpr(FW_TRACKING){
            T e_position[3], e_velocity[3];
            tracking_errors(next, e_position, e_velocity);
            return reward_from_tracking_errors(e_position, e_velocity, air.beta, air.alpha, change, deviation, config.reward);
        }
        else{
            Measurements<T> m = measurements(p, next.body);
            T e_airspeed, e_altitude, e_course;
            errors(m, next, e_airspeed, e_altitude, e_course);
            return reward_from_errors(e_airspeed, e_altitude, e_course, air.beta, air.alpha, change, deviation, config.reward);
        }
    }

    template <typename T>
    T reward_from_critic_observations(const T* observation, const T* next_observation, const T* action, bool terminated, const RewardConfig<T>& c){
        // the reward of a stored transition, with new weights (curriculum): same as reward_from_observations in Python
        if(terminated) return 0;
        const int n = ActorObservation<int>::DIM;
        const int critic = CriticObservation<int>::DIM;
        T alpha = next_observation[n] * T(0.3);
        T beta = next_observation[n + 1] * T(0.3);
        T change = 0, deviation = 0;
        for(int i = 0; i < N_ACTUATORS; i++){
            T a = action[i];
            T previous = observation[n - N_ACTUATORS + i];
            T trim_action = observation[critic - N_ACTUATORS + i];
            change += (a - previous) * (a - previous);
            deviation += (a - trim_action) * (a - trim_action);
        }
        if constexpr(FW_TRACKING){
            // body frame: the reward does not depend on the frame; the position is clipped at 200 m in the observation
            T position[3] = {next_observation[0] * 20, next_observation[1] * 20, next_observation[2] * 20};
            T velocity[3] = {next_observation[3] * 5, next_observation[4] * 5, next_observation[5] * 5};
            return reward_from_tracking_errors(position, velocity, beta, alpha, change, deviation, c);
        }
        else{
            T e_airspeed = next_observation[0] * 5;
            T e_altitude = next_observation[1] * 20;
            T e_course = std::atan2(next_observation[2], next_observation[3]);
            return reward_from_errors(e_airspeed, e_altitude, e_course, beta, alpha, change, deviation, c);
        }
    }

    template <typename T>
    void task_parameters(const TaskConfig<T>& config, Parameters<T>& p){
        if constexpr(FW_AIRPLANE == 1) advanced_plane(p);
        else if constexpr(FW_AIRPLANE == 2) eratix(p);
        else aerosonde(p);
        p.dt = 1 / config.frequency;
        p.n_substeps = config.n_substeps;
        p.latency_steps = config.latency_steps;
        if(p.latency_steps > MAX_LATENCY) throw std::runtime_error("latency_steps > MAX_LATENCY");
        for(int i = 0; i < 3; i++) p.error_integral_limits[i] = config.error_integral_limits[i];
        p.position_integral_time = config.position_integral_time;
        p.position_integral_scale = config.position_integral_scale;
    }

    template <typename T>
    void simulate_task(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], State<T>& next){
        // simulate, then the error integrals: they need the setpoint, so they belong to the task, not to the physics
        simulate(p, state, action, next);
        if constexpr(FW_TRACKING){
            // the reference moves on; the altitude setpoint follows it (termination on the altitude error)
            for(int i = 0; i < 3; i++) next.reference_position[i] = state.reference_position[i] + state.reference_velocity[i] * p.dt;
            next.setpoint_altitude = next.reference_position[2];
            // integral of the position error with forgetting: a steady offset builds up, the approach fades out
            T e_position[3], e_velocity[3];
            tracking_errors(next, e_position, e_velocity);
            T keep = std::exp(-p.dt / p.position_integral_time);
            for(int i = 0; i < 3; i++) next.error_integral[i] = keep * state.error_integral[i] + e_position[i] * p.dt;
        }
        else{
            Measurements<T> m = measurements(p, next.body);
            T e[3];
            errors(m, next, e[0], e[1], e[2]);
            for(int i = 0; i < 3; i++){
                T limit = p.error_integral_limits[i];
                next.error_integral[i] = clip(state.error_integral[i] + e[i] * p.dt, -limit, limit);
            }
        }
    }

    template <typename T>
    void apply_randomization(Parameters<T>& p){
        // p.randomization holds the factors; the nominal values are in p
        p.mass *= p.randomization[0];
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) p.J[i][j] *= p.randomization[1];
        invert3(p.J, p.J_inv);
        p.thrust_max *= p.randomization[2];
        for(int i = 0; i < N_ACTUATORS; i++) p.actuator_time_constants[i] *= p.randomization[3];
        for(int i = 0; i < N_COEFFICIENTS; i++) p.aerodynamics[i] *= p.randomization[4 + i];
    }

    template <typename T>
    void start_state(const Parameters<T>& p, const Body<T>& body, T setpoint_airspeed, T setpoint_altitude, T setpoint_course, State<T>& s){
        // fills the latency buffer, the trim commands and the action history around a sampled body and setpoint
        s.body = body;
        for(int k = 0; k < MAX_LATENCY; k++) for(int i = 0; i < N_ACTUATORS; i++) s.action_buffer[k][i] = body.actuators[i];
        s.setpoint_airspeed = setpoint_airspeed;
        s.setpoint_altitude = setpoint_altitude;
        s.setpoint_course = setpoint_course;
        Body<T> trimmed = trim(p, setpoint_airspeed);
        normalize_action(p, trimmed.actuators, s.trim_action);
        T previous[N_ACTUATORS];
        normalize_action(p, body.actuators, previous);
        for(int h = 0; h < ACTION_HISTORY; h++) for(int i = 0; i < N_ACTUATORS; i++) s.action_history[h][i] = previous[i];
        for(int i = 0; i < 3; i++){
            s.error_integral[i] = 0;
            s.reference_position[i] = 0;  // tracking task: set by sample_initial_state
            s.reference_velocity[i] = 0;
        }
    }
}
