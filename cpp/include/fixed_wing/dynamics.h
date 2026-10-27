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
    void aerodynamics(const Parameters<T>& p, const Body<T>& x, T force[3], T moment[3]){
        // aerodynamic + propulsion force and moment in the body frame (x forward, y left, z up)
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
        for(int i = 0; i < N_ACTUATORS; i++) dx.actuators[i] = (command[i] - x.actuators[i]) / p.actuator_time_constants[i];
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

    template <typename T, typename WRITE>
    void observe_actor(const Parameters<T>& p, const State<T>& s, WRITE write){
        // same layout as environment.observe_actor; noise-free (observation noise is not ported yet)
        Measurements<T> m = measurements(p, s.body);
        T e_airspeed, e_altitude, e_course;
        errors(m, s, e_airspeed, e_altitude, e_course);
        int k = 0;
        write(k++, e_airspeed / 5);
        write(k++, e_altitude / 20);
        write(k++, std::sin(e_course));
        write(k++, std::cos(e_course));
        for(int i = 0; i < 3; i++) for(int j = 0; j < 3; j++) write(k++, m.R[i][j]);
        for(int i = 0; i < 3; i++) write(k++, m.rates[i] / T(PI));
        for(int i = 0; i < 3; i++) write(k++, m.velocity_body[i] / 25);
        write(k++, m.airspeed / 25);
        if constexpr(N_INTEGRALS > 0){
            for(int i = 0; i < 3; i++) write(k++, s.error_integral[i] / p.error_integral_limits[i]);
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
    T reward_from_errors(T e_airspeed, T e_altitude, T e_course, T beta, T alpha, T action_change_squared, T trim_deviation_squared, const RewardConfig<T>& c){
        auto sq = [](T x){ return x * x; };
        auto tracking = [&](T x){  // scaled tracking error: square, or Huber with the same value near 1
            if(c.huber_delta <= 0) return sq(x);
            return std::abs(x) < c.huber_delta ? sq(x) / (2 * c.huber_delta) : std::abs(x) - c.huber_delta / 2;
        };
        T cost = c.airspeed_weight * tracking(e_airspeed / c.airspeed_scale)
               + c.altitude_weight * tracking(e_altitude / c.altitude_scale)
               + c.course_weight * tracking(e_course / c.course_scale)
               + c.sideslip_weight * sq(beta / c.sideslip_scale)
               + c.alpha_margin_weight * sq(alpha_excess(alpha, c) / c.alpha_margin_scale)
               + c.action_change_weight * action_change_squared
               + c.trim_weight * trim_deviation_squared;
        T far = c.far_weight * (std::fmin(std::abs(e_airspeed) / c.far_airspeed_scale, T(1))
                              + std::fmin(std::abs(e_altitude) / c.far_altitude_scale, T(1))
                              + std::fmin(std::abs(e_course) / c.far_course_scale, T(1)));
        T cap = c.cost_cap * c.survival;
        return c.survival - cap * std::tanh(cost / cap) - far;
    }

    // why a flight ends: 0 = still flying
    enum Termination{FLYING = 0, DIVERGED, LOW_ALTITUDE, ALPHA, AIRSPEED, ALTITUDE};

    template <typename T>
    Termination termination(const Parameters<T>& p, const State<T>& s, const TaskConfig<T>& config){
        for(int i = 0; i < 3; i++) if(!std::isfinite(s.body.linear_velocity[i])) return DIVERGED;
        if(s.body.position[2] < config.altitude_floor) return LOW_ALTITUDE;
        AirData<T> a = air_data(p, s.body);
        if(!(config.alpha_range[0] <= a.alpha && a.alpha <= config.alpha_range[1])) return ALPHA;
        if(!(config.airspeed_limits[0] <= a.airspeed && a.airspeed <= config.airspeed_limits[1])) return AIRSPEED;
        if(std::abs(s.body.position[2] - s.setpoint_altitude) > config.altitude_error_max) return ALTITUDE;
        return FLYING;
    }

    template <typename T>
    T reward(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], const State<T>& next, const TaskConfig<T>& config){
        // on the state after the step; zero when the flight ends, as in FixedWingEnv.step
        if(termination(p, next, config) != FLYING) return 0;
        Measurements<T> m = measurements(p, next.body);
        T e_airspeed, e_altitude, e_course;
        errors(m, next, e_airspeed, e_altitude, e_course);
        AirData<T> air = air_data(p, next.body);
        T change = 0, deviation = 0;
        for(int i = 0; i < N_ACTUATORS; i++){
            T a = clip(action[i], T(-1), T(1));
            change += (a - state.action_history[ACTION_HISTORY - 1][i]) * (a - state.action_history[ACTION_HISTORY - 1][i]);
            deviation += (a - state.trim_action[i]) * (a - state.trim_action[i]);
        }
        return reward_from_errors(e_airspeed, e_altitude, e_course, air.beta, air.alpha, change, deviation, config.reward);
    }

    template <typename T>
    T reward_from_critic_observations(const T* observation, const T* next_observation, const T* action, bool terminated, const RewardConfig<T>& c){
        // the reward of a stored transition, with new weights (curriculum): same as reward_from_observations in Python
        if(terminated) return 0;
        const int n = ActorObservation<int>::DIM;
        const int critic = CriticObservation<int>::DIM;
        T e_airspeed = next_observation[0] * 5;
        T e_altitude = next_observation[1] * 20;
        T e_course = std::atan2(next_observation[2], next_observation[3]);
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
        return reward_from_errors(e_airspeed, e_altitude, e_course, beta, alpha, change, deviation, c);
    }

    template <typename T>
    void task_parameters(const TaskConfig<T>& config, Parameters<T>& p){
        aerosonde(p);
        p.dt = 1 / config.frequency;
        p.n_substeps = config.n_substeps;
        p.latency_steps = config.latency_steps;
        if(p.latency_steps > MAX_LATENCY) throw std::runtime_error("latency_steps > MAX_LATENCY");
        for(int i = 0; i < 3; i++) p.error_integral_limits[i] = config.error_integral_limits[i];
    }

    template <typename T>
    void simulate_task(const Parameters<T>& p, const State<T>& state, const T action[N_ACTUATORS], State<T>& next){
        // simulate, then the error integrals: they need the setpoint, so they belong to the task, not to the physics
        simulate(p, state, action, next);
        Measurements<T> m = measurements(p, next.body);
        T e[3];
        errors(m, next, e[0], e[1], e[2]);
        for(int i = 0; i < 3; i++){
            T limit = p.error_integral_limits[i];
            next.error_integral[i] = clip(state.error_integral[i] + e[i] * p.dt, -limit, limit);
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
        for(int i = 0; i < 3; i++) s.error_integral[i] = 0;
    }
}
