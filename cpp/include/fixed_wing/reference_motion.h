// Reference motions other than the straight line of the tracking task, to see how a tracking policy copes when the
// point accelerates and where it breaks (src/evaluate_reference.cpp). Training and the tests still move the point in a
// straight line at constant velocity (simulate_task in dynamics.h); nothing here is used by them.
//
// Each motion starts where and as the sampled start puts the point (position, direction of its velocity), so that the
// starts are the ones of the tests and only the path changes. Its difficulty is one acceleration:
//   CIRCLE: horizontal circle at constant speed, centripetal acceleration a = V^2 / R (equivalent bank atan(a / g))
//   HELIX: the same circle while climbing or descending at a constant rate
//   FIGURE_EIGHT: the Lissajous figure of Learning to Fly in Seconds (Fig. 5), x = A sin(wt), y = A/2 sin(2wt), at
//     constant altitude, with A w set by the mean speed and w by the peak lateral acceleration
//   SPEED_OSCILLATION: straight line, the speed oscillates sinusoidally across the speed range of the task, the
//     acceleration along the path peaking at a
// World frame, z up, angles counterclockwise from world x, as in dynamics.h.
#pragma once

#include <cmath>

#include "dynamics.h"

namespace fixed_wing{
    enum MotionKind{STRAIGHT = 0, CIRCLE, HELIX, FIGURE_EIGHT, SPEED_OSCILLATION};

    template <typename T>
    struct ReferenceMotion{
        MotionKind kind = STRAIGHT;
        T origin[3] = {0, 0, 0};  // the point at t = 0
        T heading = 0;  // rad, direction of its horizontal velocity at t = 0
        T speed = 0;  // m/s, horizontal: constant (straight, circle, helix), mean over the eight, at t = 0 (oscillation)
        T climb = 0;  // m/s, constant
        T turn = 1;  // +1 turns left (counterclockwise seen from above), -1 right; oscillation: +1 speeds up first
        T acceleration = 0;  // m/s^2: centripetal (circle, helix), peak lateral (eight), peak along the path (oscillation)
        T phase = 0;  // figure-eight: where on the figure the point starts (0: the crossing, at its highest speed)
        T speed_mid = 15, speed_amplitude = 3;  // oscillation: m/s, the speed stays within mid +- amplitude
    };

    // shape factors of the figure-eight for A = w = 1, by sampling one period: mean speed, peak lateral acceleration
    // (normal to the velocity, what banking provides), lowest and highest speed, peak acceleration along the path
    struct FigureEightFactors{ double mean_speed, peak_lateral, min_speed, max_speed, peak_tangential, peak_total; };

    inline FigureEightFactors figure_eight_factors(){
        FigureEightFactors f{0, 0, 1e9, 0, 0, 0};
        constexpr int N = 100000;
        for(int i = 0; i < N; i++){
            double th = 2 * PI * i / N;
            double vx = std::cos(th), vy = std::cos(2 * th), ax = -std::sin(th), ay = -2 * std::sin(2 * th);
            double v = std::hypot(vx, vy);
            f.mean_speed += v / N;
            f.peak_lateral = std::fmax(f.peak_lateral, std::abs(vx * ay - vy * ax) / v);
            f.peak_tangential = std::fmax(f.peak_tangential, std::abs(vx * ax + vy * ay) / v);
            f.peak_total = std::fmax(f.peak_total, std::hypot(ax, ay));
            f.min_speed = std::fmin(f.min_speed, v);
            f.max_speed = std::fmax(f.max_speed, v);
        }
        return f;
    }

    template <typename T>
    T figure_eight_frequency(const ReferenceMotion<T>& m, const FigureEightFactors& f){
        // w [rad/s] that gives the mean speed and the peak lateral acceleration: A w = speed / mean factor, and the peak
        // lateral acceleration is A w^2 x the lateral factor
        return m.acceleration * T(f.mean_speed) / (m.speed * T(f.peak_lateral));
    }

    template <typename T>
    void reference_at(const ReferenceMotion<T>& m, T t, T position[3], T velocity[3], T acceleration[3]){
        // the point at time t [s] after the start: position [m], velocity [m/s], acceleration [m/s^2], world frame
        T c = std::cos(m.heading), s = std::sin(m.heading);
        position[2] = m.origin[2] + m.climb * t;
        velocity[2] = m.climb;
        acceleration[2] = 0;
        if(m.kind == CIRCLE || m.kind == HELIX){
            T w = m.acceleration / m.speed, R = m.speed * m.speed / m.acceleration;
            T start = m.heading - m.turn * T(PI / 2);  // direction of the point from the centre at t = 0
            T angle = start + m.turn * w * t;
            position[0] = m.origin[0] + R * (std::cos(angle) - std::cos(start));
            position[1] = m.origin[1] + R * (std::sin(angle) - std::sin(start));
            velocity[0] = -m.turn * m.speed * std::sin(angle);
            velocity[1] = m.turn * m.speed * std::cos(angle);
            acceleration[0] = -m.acceleration * std::cos(angle);
            acceleration[1] = -m.acceleration * std::sin(angle);
        }
        else if(m.kind == FIGURE_EIGHT){
            static const FigureEightFactors f = figure_eight_factors();
            T w = figure_eight_frequency(m, f), A = m.speed / (w * T(f.mean_speed));
            T th = m.phase + w * t;
            // frame of the figure, turned so that the velocity at t = 0 points along the heading; turn mirrors it
            T rotation = m.heading - std::atan2(m.turn * std::cos(2 * m.phase), std::cos(m.phase));
            T rc = std::cos(rotation), rs = std::sin(rotation);
            T local[3][2] = {
                {A * (std::sin(th) - std::sin(m.phase)), m.turn * A / 2 * (std::sin(2 * th) - std::sin(2 * m.phase))},
                {A * w * std::cos(th), m.turn * A * w * std::cos(2 * th)},
                {-A * w * w * std::sin(th), -m.turn * 2 * A * w * w * std::sin(2 * th)}};
            T* out[3] = {position, velocity, acceleration};
            for(int k = 0; k < 3; k++){
                T x = rc * local[k][0] - rs * local[k][1], y = rs * local[k][0] + rc * local[k][1];
                out[k][0] = (k == 0 ? m.origin[0] : T(0)) + x;
                out[k][1] = (k == 0 ? m.origin[1] : T(0)) + y;
            }
        }
        else if(m.kind == SPEED_OSCILLATION){
            // speed mid + amplitude sin(W t + start), W = peak acceleration / amplitude, the start set by the speed at t = 0
            T W = m.acceleration / m.speed_amplitude;
            T sine = clip((m.speed - m.speed_mid) / m.speed_amplitude, T(-1), T(1));
            T start = m.turn > 0 ? std::asin(sine) : T(PI) - std::asin(sine);
            T distance = m.speed_mid * t - m.speed_amplitude / W * (std::cos(W * t + start) - std::cos(start));
            T speed = m.speed_mid + m.speed_amplitude * std::sin(W * t + start);
            T along = m.speed_amplitude * W * std::cos(W * t + start);
            position[0] = m.origin[0] + distance * c;
            position[1] = m.origin[1] + distance * s;
            velocity[0] = speed * c;
            velocity[1] = speed * s;
            acceleration[0] = along * c;
            acceleration[1] = along * s;
        }
        else{
            position[0] = m.origin[0] + m.speed * c * t;
            position[1] = m.origin[1] + m.speed * s * t;
            velocity[0] = m.speed * c;
            velocity[1] = m.speed * s;
            acceleration[0] = 0;
            acceleration[1] = 0;
        }
    }
}
