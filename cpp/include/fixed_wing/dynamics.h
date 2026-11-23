// Airplane physics, independent of RLtools: port of operations.py, trim.py and the task functions of environment.py.
#pragma once

#include <cmath>
#include <limits>
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

    constexpr double ALLEN_R1_R2[7] = {0.14, 0.25, 0.36, 0.47, 0.58, 0.69, 0.80};
    constexpr double ALLEN_SHAPES[7][4] = {
        {1.5352, 2.5826, -0.0113, -0.1950},
        {1.5265, 3.6054, -0.0176, -0.1265},
        {1.4866, 4.8356, -0.0320, -0.0818},
        {1.2042, 7.7904, 0.0848, -0.0445},
        {0.8816, 13.9720, 0.3404, -0.0216},
        {0.7067, 23.9940, 0.5689, -0.0099},
        {0.6189, 42.7965, 0.7157, -0.0033},
    };

    template <typename T>
    T allen_mean_radius(T height, T z_i){
        T zzi = height / z_i;
        return T(0.102) * std::pow(zzi, T(1) / T(3)) * (1 - T(0.25) * zzi) * z_i;
    }

    template <typename T>
    T allen_mean_updraft(T height, T z_i, T w_star){
        T zzi = height / z_i;
        return std::pow(zzi, T(1) / T(3)) * (1 - T(1.1) * zzi) * w_star;
    }

    template <typename T>
    T allen_core_factor(T r2){
        T r1 = (r2 < 600 ? T(0.0011) * r2 + T(0.14) : T(0.8)) * r2;
        return 3 * (std::pow(r2, T(3)) - std::pow(r2, T(2)) * r1) / (std::pow(r2, T(3)) - std::pow(r1, T(3)));
    }

    template <typename T>
    T thermal_updraft(const Thermals<T>& t, T x, T y, T height){
        if(t.count == 0 || height <= 0) return 0;
        T zzi = height / t.z_i;
        T r_bar = allen_mean_radius(height, t.z_i);
        T w_bar = allen_mean_updraft(height, t.z_i, t.w_star);
        int nearest = 0;
        T r = 0;
        for(int k = 0; k < t.count; k++){
            T dx = x - t.centers[k][0], dy = y - t.centers[k][1];
            T distance = std::sqrt(dx * dx + dy * dy);
            if(k == 0 || distance < r){
                r = distance;
                nearest = k;
            }
        }
        T r2 = std::fmax(r_bar * t.r_gain[nearest], T(10));
        T r1_r2 = r2 < 600 ? T(0.0011) * r2 + T(0.14) : T(0.8);
        T r1 = r1_r2 * r2;
        T w_t = w_bar * t.w_gain[nearest];
        T w_c = 3 * w_t * (std::pow(r2, T(3)) - std::pow(r2, T(2)) * r1) / (std::pow(r2, T(3)) - std::pow(r1, T(3)));
        T rr2 = r / r2;
        T w_s = 0;
        if(height < t.z_i){
            int row = 0;
            for(int i = 0; i + 1 < 7; i++) if((ALLEN_R1_R2[i + 1] + ALLEN_R1_R2[i]) / 2 <= r1_r2) row = i + 1;
            const double* k = ALLEN_SHAPES[row];
            w_s = std::fmax(1 / (1 + std::pow(T(k[0]) * std::abs(rr2 + T(k[2])), T(k[1]))) + T(k[3]) * rr2, T(0));
        }
        T w_1 = r > r1 && rr2 < 2 ? T(PI) / 6 * std::sin(T(PI) * rr2) : T(0);
        T s_wd = std::fmax(T(0.5) < zzi && zzi <= T(0.9) ? T(2.5) * (zzi - T(0.5)) : T(0), t.downdraft_ring);
        T w_d = std::fmin(s_wd * w_1, T(0));
        T w = w_s * w_c + w_d * w_t;
        if(t.sink){
            T updraft_area = t.count * T(PI) * (r_bar * r_bar);
            if(updraft_area < t.area){
                T w_e = std::fmin(-updraft_area * w_bar * (1 - s_wd) / (t.area - updraft_area), T(0));
                if(r > r1 && w_c != 0) w = w * (1 - w_e / w_c) + w_e;
            }
        }
        return w;
    }

    template <typename T>
    void turbulence_velocity(const Parameters<T>& p, const Body<T>& x, T out[3]){
        const T* n = x.turbulence;
        T half_root_3 = std::sqrt(T(3)) / 2;
        out[0] = p.turbulence_sigma[0] * n[0];
        out[1] = p.turbulence_sigma[1] * (T(0.5) * n[1] + half_root_3 * n[2]);
        out[2] = p.turbulence_sigma[2] * (T(0.5) * n[3] + half_root_3 * n[4]);
    }

    template <typename T>
    void local_wind(const Parameters<T>& p, const Body<T>& x, const T R[3][3], T out[3]){
        for(int i = 0; i < 3; i++) out[i] = p.wind[i];
        if constexpr(FW_DISTURBANCES){
            out[2] += thermal_updraft(p.thermals, x.position[0] - p.wind[0] * x.time, x.position[1] - p.wind[1] * x.time, x.position[2]);
            for(int g = 0; g < p.gust_count; g++){
                const Gust<T>& gust = p.gusts[g];
                T s = (x.time - gust.start) / gust.duration;
                if(s >= 0 && s < 1){
                    for(int i = 0; i < 3; i++) out[i] += gust.peak[i] * T(0.5) * (1 - std::cos(2 * T(PI) * s));
                }
            }
            T body[3], world[3];
            turbulence_velocity(p, x, body);
            rotate(R, body, world);
            for(int i = 0; i < 3; i++) out[i] += world[i];
        }
    }

    template <typename T>
    void symmetric_eigen_2x2(T a, T b, T c, T& rt1, T& rt2, T& cs1, T& sn1){
        T sm = a + c, df = a - c, adf = std::abs(df), tb = b + b, ab = std::abs(tb);
        T acmx = std::abs(a) > std::abs(c) ? a : c;
        T acmn = std::abs(a) > std::abs(c) ? c : a;
        T rt;
        if(adf > ab) rt = adf * std::sqrt(1 + (ab / adf) * (ab / adf));
        else if(adf < ab) rt = ab * std::sqrt(1 + (adf / ab) * (adf / ab));
        else rt = ab * std::sqrt(T(2));
        int sgn1;
        if(sm < 0){
            rt1 = T(0.5) * (sm - rt);
            sgn1 = -1;
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        }
        else if(sm > 0){
            rt1 = T(0.5) * (sm + rt);
            sgn1 = 1;
            rt2 = (acmx / rt1) * acmn - (b / rt1) * b;
        }
        else{
            rt1 = T(0.5) * rt;
            rt2 = T(-0.5) * rt;
            sgn1 = 1;
        }
        int sgn2;
        T cs;
        if(df >= 0){
            cs = df + rt;
            sgn2 = 1;
        }
        else{
            cs = df - rt;
            sgn2 = -1;
        }
        if(std::abs(cs) > ab){
            T ct = -tb / cs;
            sn1 = 1 / std::sqrt(1 + ct * ct);
            cs1 = ct * sn1;
        }
        else if(ab == 0){
            cs1 = 1;
            sn1 = 0;
        }
        else{
            T tn = -cs / tb;
            cs1 = 1 / std::sqrt(1 + tn * tn);
            sn1 = tn * cs1;
        }
        if(sgn1 == sgn2){
            T tn = cs1;
            cs1 = -sn1;
            sn1 = tn;
        }
    }

    template <typename T>
    void covariance_square_root(const T C[2][2], T G[2][2]){
        T d[2] = {C[0][0], C[1][1]}, e = C[1][0];
        T eps = std::numeric_limits<T>::epsilon() / 2;
        T safe_min = std::numeric_limits<T>::min();
        bool diagonal = e == 0 || std::abs(e) <= std::sqrt(std::abs(d[0])) * std::sqrt(std::abs(d[1])) * eps
                     || e * e <= (eps * eps * std::abs(d[0])) * std::abs(d[1]) + safe_min;
        T values[2], vectors[2][2];
        if(diagonal){
            values[0] = d[0];
            values[1] = d[1];
            vectors[0][0] = 1; vectors[0][1] = 0;
            vectors[1][0] = 0; vectors[1][1] = 1;
        }
        else{
            T cs, sn;
            symmetric_eigen_2x2(d[0], e, d[1], values[0], values[1], cs, sn);
            vectors[0][0] = cs; vectors[0][1] = -sn;
            vectors[1][0] = sn; vectors[1][1] = cs;
        }
        if(values[1] < values[0]){
            T v = values[0]; values[0] = values[1]; values[1] = v;
            for(int i = 0; i < 2; i++){ T w = vectors[i][0]; vectors[i][0] = vectors[i][1]; vectors[i][1] = w; }
        }
        for(int i = 0; i < 2; i++) for(int j = 0; j < 2; j++) G[i][j] = vectors[i][j] * std::sqrt(std::fmax(values[j], T(0)));
    }

    template <typename T>
    void advance_turbulence_states(const T length_scale[3], const T in[N_TURBULENCE], T distance, const T normals[N_TURBULENCE], T out[N_TURBULENCE]){
        T s[3], e[3];
        for(int i = 0; i < 3; i++){
            s[i] = distance / length_scale[i];
            e[i] = std::exp(-s[i]);
        }
        out[0] = e[0] * in[0] + std::sqrt(1 - e[0] * e[0]) * normals[0];
        for(int axis = 1; axis < 3; axis++){
            T si = s[axis], ei = e[axis];
            T M[2][2] = {{ei * (1 + si), ei * si}, {ei * -si, ei * (1 - si)}};
            T C[2][2], G[2][2];
            for(int a = 0; a < 2; a++) for(int b = 0; b < 2; b++) C[a][b] = (a == b ? T(1) : T(0)) - (M[a][0] * M[b][0] + M[a][1] * M[b][1]);
            covariance_square_root(C, G);
            const T* x = in + 2 * axis - 1;
            const T* n = normals + 2 * axis - 1;
            for(int a = 0; a < 2; a++) out[2 * axis - 1 + a] = (M[a][0] * x[0] + M[a][1] * x[1]) + (G[a][0] * n[0] + G[a][1] * n[1]);
        }
    }

    template <typename T>
    bool turbulent(const Parameters<T>& p){
        return p.turbulence_sigma[0] != 0 || p.turbulence_sigma[1] != 0 || p.turbulence_sigma[2] != 0;
    }

    template <typename T>
    void clear_disturbances(Parameters<T>& p){
        for(int i = 0; i < 3; i++){
            p.turbulence_sigma[i] = 0;
            p.turbulence_length_scale[i] = 1;
        }
        p.gust_count = 0;
        p.thermals.count = 0;
        p.thermals.w_star = T(2.56);
        p.thermals.z_i = T(1401);
        p.thermals.area = T(1e6);
        p.thermals.sink = true;
        p.thermals.downdraft_ring = 0;
    }

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
        clear_disturbances(p);
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
        clear_disturbances(p);
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
        clear_disturbances(p);
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
        T air[3], body[3], wind[3];
        local_wind(p, x, R, wind);
        for(int i = 0; i < 3; i++) air[i] = x.linear_velocity[i] - wind[i];
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
        T air[3], body[3], wind[3];
        local_wind(p, x, R, wind);
        for(int i = 0; i < 3; i++) air[i] = x.linear_velocity[i] - wind[i];
        rotate_inverse(R, air, body);
        const T* k = p.randomization + 4;
        const T CL0r = T(CL0) * k[C_L_0], CLar = T(CLa) * k[C_L_alpha], CD0r = T(CD0) * k[C_D_p], effr = T(eff) * k[oswald_efficiency];
        const T Cem0r = T(Cem0) * k[C_m_0], Cemar = T(Cema) * k[C_m_alpha];
        const T CYbr = T(CYb) * k[C_Y_beta], Cellbr = T(Cellb) * k[C_l_beta], Cenbr = T(Cenb) * k[C_n_beta];
        const T CDpr = T(CDp) * k[C_D_p], CYpr = T(CYp) * k[C_Y_p], Cellpr = T(Cellp) * k[C_l_p], Cenpr = T(Cenp) * k[C_n_p];
        const T CDqr = T(CDq) * k[C_D_p], CLqr = T(CLq) * k[C_L_q], Cemqr = T(Cemq) * k[C_m_q];
        const T CYrr = T(CYr) * k[C_Y_r], Cellrr = T(Cellr) * k[C_l_r], Cenrr = T(Cenr) * k[C_n_r];
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
                bool rudder = i == 3;
                CL_c += angle[i] * T(CL_ctrl[i]) * k[C_L_delta_e]; CD_c += angle[i] * T(CD_ctrl[i]) * k[C_D_p];
                CY_c += angle[i] * T(CY_ctrl[i]) * k[rudder ? C_Y_delta_r : C_Y_delta_a];
                Cell_c += angle[i] * T(Cell_ctrl[i]) * k[rudder ? C_l_delta_r : C_l_delta_a];
                Cem_c += angle[i] * T(Cem_ctrl[i]) * k[C_m_delta_e];
                Cen_c += angle[i] * T(Cen_ctrl[i]) * k[rudder ? C_n_delta_r : C_n_delta_a];
            }
            T sign = alpha >= 0 ? T(1) : T(-1);
            T CL = (1 - sigma) * (CL0r + CLar * alpha) + sigma * (2 * sign * sa * sa * ca) + CL_c;
            T lift = (CL * dyn + T(CLp) * (P * b / 2) * hrv + CLqr * (Q * c / 2) * hrv + T(CLr) * (Rr * b / 2) * hrv) * S;
            T CD_fp = 2 / (1 + std::exp(T(CD_fp_k1) + T(CD_fp_k2) * std::fmax(T(AR), 1 / T(AR))));
            T CD = (1 - sigma) * (CD0r + CL * CL / (T(PI) * T(AR) * effr)) + sigma * std::abs(CD_fp * (T(0.5) - T(0.5) * std::cos(2 * alpha))) + CD_c;
            T drag = (CD * dyn + CDpr * (P * b / 2) * hrv + CDqr * (Q * c / 2) * hrv + T(CDr) * (Rr * b / 2) * hrv) * S;
            T CY = CYbr * beta + CY_c;
            T side = (CY * dyn + CYpr * (P * b / 2) * hrv + T(CYq) * (Q * c / 2) * hrv + CYrr * (Rr * b / 2) * hrv) * S;
            T Cem = alpha > T(alpha_stall) ? Cem0r + Cemar * T(alpha_stall) + T(Cema_stall) * (alpha - T(alpha_stall))
                  : alpha < -T(alpha_stall) ? Cem0r - Cemar * T(alpha_stall) + T(Cema_stall) * (alpha + T(alpha_stall))
                  : Cem0r + Cemar * alpha;
            Cem += Cem_c;
            T Cell = Cellbr * beta + Cell_c, Cen = Cenbr * beta + Cen_c;
            // lift along -z of the stability axes, (sin alpha, 0, -cos alpha); drag along -(cos alpha, 0, sin alpha)
            f[0] = lift * sa - drag * ca;
            f[1] = side;
            f[2] = -lift * ca - drag * sa;
            m[0] = (Cell * dyn + Cellpr * (P * b / 2) * hrv + T(Cellq) * (Q * c / 2) * hrv + Cellrr * (Rr * b / 2) * hrv) * S * b;
            m[1] = (Cem * dyn + T(Cemp) * (P * b / 2) * hrv + Cemqr * (Q * c / 2) * hrv + T(Cemr) * (Rr * b / 2) * hrv) * S * c;
            m[2] = (Cen * dyn + Cenpr * (P * b / 2) * hrv + T(Cenq) * (Q * c / 2) * hrv + Cenrr * (Rr * b / 2) * hrv) * S * b;
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
        dx.time = 1;
        for(int i = 0; i < N_TURBULENCE; i++) dx.turbulence[i] = 0;
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
        out.time = x.time + scale * dx.time;
        for(int i = 0; i < N_TURBULENCE; i++) out.turbulence[i] = x.turbulence[i] + scale * dx.turbulence[i];
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
        if constexpr(N_FINE > 0 && FW_TRACKING){
            // the position error again, body frame, on a fine scale: slope 1 / fine_position_scale near the point
            T e_position[3], e_velocity[3], body[3];
            tracking_errors(s, e_position, e_velocity);
            rotate_inverse(m.R, e_position, body);
            for(int i = 0; i < 3; i++) write(k++, std::tanh(body[i] / p.fine_position_scale));
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
        T R[3][3], wind[3];
        rotation_matrix(s.body.orientation, R);
        local_wind(p, s.body, R, wind);
        for(int i = 0; i < 3; i++) write(k++, wind[i] / 10);
        T normalized[N_ACTUATORS];
        normalize_action(p, s.body.actuators, normalized);
        for(int i = 0; i < N_ACTUATORS; i++) write(k++, normalized[i]);
        for(int i = 0; i < 4 + N_COEFFICIENTS; i++) write(k++, (p.randomization[i] - 1) / T(0.3));
        if constexpr(N_DISTURBANCE_PRIVILEGED > 0){
            write(k++, T(p.latency_steps) / 2);
            T acceleration_body[3];
            rotate_inverse(R, s.reference_acceleration, acceleration_body);
            for(int i = 0; i < 3; i++) write(k++, acceleration_body[i] / T(9.8));
        }
        for(int i = 0; i < N_ACTUATORS; i++) write(k++, s.trim_action[i]);
    }

    constexpr int CRITIC_ACTUATORS_OFFSET = 6;
    constexpr int ACTOR_RATES_OFFSET = N_TASK_ERRORS + 9;

    template <typename T>
    T rotor_speed_fraction(T throttle){
        return (T(advanced::rotor_speed_min) + (T(advanced::rotor_speed_max) - T(advanced::rotor_speed_min)) * throttle) / T(advanced::rotor_speed_max);
    }

    template <typename T>
    T thrust_fraction(T throttle){
        if constexpr(FW_AIRPLANE == 1){
            T speed = rotor_speed_fraction(throttle);
            return speed * speed;
        }
        else{
            return throttle;
        }
    }

    template <typename T>
    T advanced_propeller_power(const Parameters<T>& p, const Body<T>& x){
        T omega = T(advanced::rotor_speed_min) + (T(advanced::rotor_speed_max) - T(advanced::rotor_speed_min)) * x.actuators[THROTTLE];
        T thrust = p.thrust_max * (omega / T(advanced::rotor_speed_max)) * (omega / T(advanced::rotor_speed_max));
        return T(advanced::moment_constant) * thrust * omega;
    }

    template <typename T>
    T simple_reward(T distance, const T action_change[N_ACTUATORS], T thrust_ratio, T rate, T velocity_error, T sideslip, const RewardConfig<T>& c){
        T change = 0;
        for(int i = 0; i < N_ACTUATORS; i++) change += action_change[i] * action_change[i];
        T energy = std::pow(std::fmax(thrust_ratio, T(0)), T(1.5));
        T cost = distance + c.simple_action_weight * std::sqrt(change) + c.energy_weight * energy;
        if(c.simple_rate_weight != 0) cost += c.simple_rate_weight * rate;
        if(c.simple_velocity_weight != 0) cost += c.simple_velocity_weight * velocity_error;
        if(c.simple_sideslip_weight != 0) cost += c.simple_sideslip_weight * std::abs(sideslip);
        return c.simple_constant - cost;
    }

    template <typename T>
    void advance_turbulence(const Parameters<T>& p, const State<T>& state, const T normals[N_TURBULENCE], State<T>& next){
        if(!turbulent(p)) return;
        T distance = air_data(p, state.body).airspeed * p.dt;
        advance_turbulence_states(p.turbulence_length_scale, state.body.turbulence, distance, normals, next.body.turbulence);
    }

    template <typename T>
    T alpha_excess(T alpha, const RewardConfig<T>& c){
        // how far the angle of attack is inside the margin before the stall limits (0 away from them)
        return std::fmax(alpha - c.alpha_soft_max, T(0)) + std::fmax(c.alpha_soft_min - alpha, T(0));
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
        auto sq = [](T x){ return x * x; };
        T tracking = c.airspeed_weight * sq(e_airspeed / c.airspeed_scale)
                   + c.altitude_weight * sq(e_altitude / c.altitude_scale)
                   + c.course_weight * sq(e_course / c.course_scale);
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
        auto sq = [](T x){ return x * x; };
        if(c.l2f_form){
            // each term is the norm itself (distance, speed error, |sideslip|, command change), as in the squared
            // reward of RLtools' L2F environment at b32d9985 (l2f/parameters/reward_functions/squared: despite its
            // name, sqrt(x^2 + y^2 + z^2) for the position and velocity, |action change|)
            T margin = std::fmin(alpha_excess(alpha, c) / c.alpha_margin_scale, T(1));
            T cost = c.position_weight * std::abs(position_error / c.position_scale)
                   + c.velocity_weight * std::abs(velocity_error / c.velocity_scale)
                   + c.sideslip_weight * std::abs(beta / c.sideslip_scale)
                   + c.action_change_weight * std::sqrt(action_change_squared)
                   + c.alpha_margin_weight * std::abs(margin);
            T bonus = c.position_bonus_weight * std::exp(-sq(position_error / c.position_bonus_scale));
            return std::fmax(c.l2f_survival - cost + bonus, c.l2f_floor);
        }
        T tracking = c.position_weight * sq(position_error / c.position_scale)
                   + c.velocity_weight * sq(velocity_error / c.velocity_scale);
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
            if(config.reference_distance_max > 0){
                T e_position[3], e_velocity[3];
                tracking_errors(s, e_position, e_velocity);
                if(norm3(e_position) > config.reference_distance_max) return POSITION;
            }
        }
        return FLYING;
    }

    template <typename T>
    T reward(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], const State<T>& next, const TaskConfig<T>& config){
        // on the state after the step; zero when the flight ends, as in FixedWingEnv.step
        if(config.reward.simple_form){
            if(termination(p, next, config) != FLYING) return -config.reward.simple_termination_penalty;
            T e_position[3], e_velocity[3], change[N_ACTUATORS];
            tracking_errors(next, e_position, e_velocity);
            for(int i = 0; i < N_ACTUATORS; i++) change[i] = clip(action[i], T(-1), T(1)) - state.action_history[ACTION_HISTORY - 1][i];
            return simple_reward(norm3(e_position), change, thrust_fraction(next.body.actuators[THROTTLE]),
                                 norm3(next.body.angular_velocity), norm3(e_velocity), air_data(p, next.body).beta, config.reward);
        }
        if(termination(p, next, config) != FLYING) return config.reward.l2f_form ? -config.reward.l2f_termination_penalty : 0;
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
            if(config.reward.l2f_form){
                // the errors as the critic observation stores them (body frame, position clipped at 200 m per axis), so
                // that the curriculum recomputes the same reward: the linear cost still grows beyond 200 m, where the
                // squared one is already at the floor
                Measurements<T> m = measurements(p, next.body);
                T body_position[3], body_velocity[3];
                rotate_inverse(m.R, e_position, body_position);
                rotate_inverse(m.R, e_velocity, body_velocity);
                for(int i = 0; i < 3; i++) body_position[i] = clip(body_position[i], T(-200), T(200));
                return reward_from_tracking_errors(body_position, body_velocity, air.beta, air.alpha, change, deviation, config.reward);
            }
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
        const int n = ActorObservation<int>::DIM;
        const int critic = CriticObservation<int>::DIM;
        if(c.simple_form){
            if(terminated) return -c.simple_termination_penalty;
            T position[3] = {next_observation[0] * 20, next_observation[1] * 20, next_observation[2] * 20};
            T change[N_ACTUATORS];
            for(int i = 0; i < N_ACTUATORS; i++) change[i] = action[i] - observation[n - N_ACTUATORS + i];
            T throttle = (next_observation[n + CRITIC_ACTUATORS_OFFSET + THROTTLE] + 1) / 2;
            T rates[3], velocity[3];
            for(int i = 0; i < 3; i++){
                rates[i] = next_observation[ACTOR_RATES_OFFSET + i] * T(PI);
                velocity[i] = next_observation[3 + i] * 5;
            }
            return simple_reward(norm3(position), change, thrust_fraction(throttle), norm3(rates), norm3(velocity), next_observation[n + 1] * T(0.3), c);
        }
        if(terminated) return c.l2f_form ? -c.l2f_termination_penalty : 0;
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
        p.fine_position_scale = config.fine_position_scale;
    }

    template <typename T>
    void simulate_task(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], State<T>& next){
        // simulate; in the tracking task the reference then moves on in a straight line at its constant velocity
        // (training and the tests), and the altitude setpoint follows it (termination on the altitude error)
        simulate(p, state, action, next);
        if constexpr(FW_TRACKING){
            for(int i = 0; i < 3; i++) next.reference_position[i] = state.reference_position[i] + state.reference_velocity[i] * p.dt;
            next.setpoint_altitude = next.reference_position[2];
        }
    }

    template <typename T>
    void simulate_task(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS],
                       const T reference_position[3], const T reference_velocity[3], State<T>& next){
        // tracking task with a reference that does not fly straight (reference_motion.h, evaluation only): the point is
        // given after the step, with its velocity, which the actor observes. The setpoints follow it, as at the start
        simulate(p, state, action, next);
        for(int i = 0; i < 3; i++){
            next.reference_position[i] = reference_position[i];
            next.reference_velocity[i] = reference_velocity[i];
        }
        next.setpoint_airspeed = norm3(reference_velocity);
        next.setpoint_course = std::atan2(reference_velocity[1], reference_velocity[0]);
        next.setpoint_altitude = next.reference_position[2];
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
            s.reference_position[i] = 0;  // tracking task: set by sample_initial_state
            s.reference_velocity[i] = 0;
            s.reference_acceleration[i] = 0;
        }
        s.reference_motion = ReferenceMotion<T>{};
    }
}
