// Flies a trained tracking policy on PX4: PX4 SITL with Gazebo's advanced_plane for now, the same chain as an onboard
// computer talking to the autopilot over uXRCE-DDS. At 50 Hz, as in training: PX4's odometry -> the training
// observation (fixed_wing::observe_actor, so that it cannot drift from the simulator) -> the exported actor
// (policy_weights.h, tools/export_policy.py) -> servo and motor commands, offboard with direct actuator control.
//
// Sequence: PX4 takes off and loiters on its own (commander takeoff). Once armed and above start_altitude, the node
// places a reference point as in training (start_distance away, start_bearing_deg from the course, flying straight at
// reference_speed along the course) and switches PX4 to offboard. It hands back to PX4 (loiter) after duration s, or
// when the airplane goes below min_altitude or more than 300 m from the point. Every step is written to log_path, with
// PX4's time (Gazebo's simulation time in SITL) to align it with other Gazebo logs.
// identify:=true: the policy flies, and doublets replace one command at a time (aileron, elevator, rudder, then a
// throttle step), twice with growing amplitudes, to measure the servo and motor response and the control effectiveness.
#include <array>
#include <chrono>
#include <cmath>
#include <fstream>
#include <memory>
#include <string>

#include <rclcpp/rclcpp.hpp>
#include <px4_msgs/msg/actuator_motors.hpp>
#include <px4_msgs/msg/actuator_servos.hpp>
#include <px4_msgs/msg/offboard_control_mode.hpp>
#include <px4_msgs/msg/vehicle_command.hpp>
#include <px4_msgs/msg/vehicle_odometry.hpp>
#include <px4_msgs/msg/vehicle_status.hpp>

#include <rl_tools/operations/cpu.h>  // RLtools' base definitions, before the simulator headers

#include <fixed_wing/dynamics.h>
#include "policy_weights.h"

using namespace std::chrono_literals;
using T = double;
static_assert(fixed_wing::ActorObservation<int>::DIM == fixed_wing::policy::INPUT, "policy and observation differ");

namespace{
    // PX4: NED world, FRD body. Simulator: world z up (x east, y north here), FLU body
    void ned_to_enu(const float in[3], T out[3]){ out[0] = in[1]; out[1] = in[0]; out[2] = -in[2]; }

    void quaternion_ned_frd_to_enu_flu(const float q[4], T out[4]){
        // R_enu_flu = A R_ned_frd B, A swaps x and y and flips z, B = diag(1, -1, -1)
        T w = q[0], x = q[1], y = q[2], z = q[3];
        T R[3][3] = {{1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)},
                     {2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)},
                     {2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)}};
        T M[3][3];
        for(int j = 0; j < 3; j++){
            T s = j == 0 ? 1 : -1;
            M[0][j] = R[1][j] * s; M[1][j] = R[0][j] * s; M[2][j] = -R[2][j] * s;
        }
        // rotation matrix -> quaternion [w, x, y, z]
        T trace = M[0][0] + M[1][1] + M[2][2];
        if(trace > 0){
            T s = std::sqrt(trace + 1) * 2;
            out[0] = s / 4; out[1] = (M[2][1] - M[1][2]) / s; out[2] = (M[0][2] - M[2][0]) / s; out[3] = (M[1][0] - M[0][1]) / s;
        }
        else if(M[0][0] > M[1][1] && M[0][0] > M[2][2]){
            T s = std::sqrt(1 + M[0][0] - M[1][1] - M[2][2]) * 2;
            out[0] = (M[2][1] - M[1][2]) / s; out[1] = s / 4; out[2] = (M[0][1] + M[1][0]) / s; out[3] = (M[0][2] + M[2][0]) / s;
        }
        else if(M[1][1] > M[2][2]){
            T s = std::sqrt(1 + M[1][1] - M[0][0] - M[2][2]) * 2;
            out[0] = (M[0][2] - M[2][0]) / s; out[1] = (M[0][1] + M[1][0]) / s; out[2] = s / 4; out[3] = (M[1][2] + M[2][1]) / s;
        }
        else{
            T s = std::sqrt(1 + M[2][2] - M[0][0] - M[1][1]) * 2;
            out[0] = (M[1][0] - M[0][1]) / s; out[1] = (M[0][2] + M[2][0]) / s; out[2] = (M[1][2] + M[2][1]) / s; out[3] = s / 4;
        }
    }
}

class PolicyNode: public rclcpp::Node{
public:
    PolicyNode(): Node("fw_policy"){
        start_altitude_ = declare_parameter("start_altitude", 40.0);  // m above home
        min_altitude_ = declare_parameter("min_altitude", 15.0);
        start_distance_ = declare_parameter("start_distance", 40.0);  // m
        start_bearing_ = declare_parameter("start_bearing_deg", 30.0) * fixed_wing::PI / 180;
        reference_speed_ = declare_parameter("reference_speed", 15.0);  // m/s
        duration_ = declare_parameter("duration", 60.0);  // s
        identify_ = declare_parameter("identify", false);
        // take over only once PX4 loiters (level flight after its takeoff), not during its full-throttle climb
        wait_for_loiter_ = declare_parameter("wait_for_loiter", false);
        // and below the speeds of training's random starts (11 to 19 m/s): PX4 switches to loiter at its takeoff
        // altitude while still at full throttle (44 m/s, climbing at 20 m/s)
        max_start_speed_ = declare_parameter("max_start_speed", 20.0);  // m/s, ground speed (no wind in SITL)
        std::string log_path = declare_parameter("log_path", std::string("fw_policy_flight.csv"));
        auto topic = [&](const char* name, const char* fallback){ return declare_parameter(name, std::string(fallback)); };
        fixed_wing::TaskConfig<T> config;
        fixed_wing::task_parameters(config, p_);
        auto qos = rclcpp::QoS(rclcpp::KeepLast(1)).best_effort().durability_volatile();
        odometry_sub_ = create_subscription<px4_msgs::msg::VehicleOdometry>(topic("odometry_topic", "/fmu/out/vehicle_odometry"), qos,
            [this](const px4_msgs::msg::VehicleOdometry& m){ odometry_ = m; have_odometry_ = true; });
        status_sub_ = create_subscription<px4_msgs::msg::VehicleStatus>(topic("status_topic", "/fmu/out/vehicle_status_v5"), qos,
            [this](const px4_msgs::msg::VehicleStatus& m){ status_ = m; have_status_ = true; });
        mode_pub_ = create_publisher<px4_msgs::msg::OffboardControlMode>(topic("offboard_mode_topic", "/fmu/in/offboard_control_mode"), 10);
        command_pub_ = create_publisher<px4_msgs::msg::VehicleCommand>(topic("command_topic", "/fmu/in/vehicle_command_v1"), 10);
        motors_pub_ = create_publisher<px4_msgs::msg::ActuatorMotors>(topic("motors_topic", "/fmu/in/actuator_motors"), 10);
        servos_pub_ = create_publisher<px4_msgs::msg::ActuatorServos>(topic("servos_topic", "/fmu/in/actuator_servos_v1"), 10);
        log_.open(log_path);
        log_ << "t,x,y,altitude,vx,vy,vz,qw,qx,qy,qz,p,q,r,reference_x,reference_y,reference_altitude,"
                "aileron,elevator,rudder,throttle,nav_state,px4_time\n";
        timer_ = create_wall_timer(20ms, [this]{ step(); });  // 50 Hz, the control period of training
        RCLCPP_INFO(get_logger(), "waiting for PX4 to be armed above %.0f m", start_altitude_);
    }

private:
    enum Phase{WAITING, SWITCHING, FLYING, DONE};

    uint64_t now_us(){ return get_clock()->now().nanoseconds() / 1000; }

    // identification: from t, each doublet sets one command to first for half (s), then to second for half
    struct Doublet{ int axis; T t, first, second, half; };
    static constexpr Doublet doublets[8] = {
        {fixed_wing::AILERON, 10, 0.3, -0.3, 0.5}, {fixed_wing::ELEVATOR, 14, 0.2, -0.2, 0.5},
        {fixed_wing::RUDDER, 18, 0.3, -0.3, 0.5}, {fixed_wing::THROTTLE, 22, 0.6, -0.9, 1.0},
        {fixed_wing::AILERON, 30, 0.6, -0.6, 0.5}, {fixed_wing::ELEVATOR, 34, 0.4, -0.4, 0.5},
        {fixed_wing::RUDDER, 38, 0.6, -0.6, 0.5}, {fixed_wing::THROTTLE, 42, 1.0, -1.0, 1.0}};

    void override_for_identification(T t, float action[fixed_wing::policy::ACTIONS]){
        for(const auto& d: doublets){
            if(t >= d.t && t < d.t + d.half) action[d.axis] = float(d.first);
            else if(t >= d.t + d.half && t < d.t + 2 * d.half) action[d.axis] = float(d.second);
        }
    }

    void command(uint16_t id, float param1 = 0, float param2 = 0, float param3 = 0){
        px4_msgs::msg::VehicleCommand c{};
        c.timestamp = now_us();
        c.command = id;
        c.param1 = param1; c.param2 = param2; c.param3 = param3;
        c.target_system = 1; c.target_component = 1; c.source_system = 1; c.source_component = 1;
        c.from_external = true;
        command_pub_->publish(c);
    }

    void hand_back(const char* reason){
        // PX4 auto loiter: base mode custom, main mode auto (4), sub mode loiter (3)
        command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 4, 3);
        RCLCPP_INFO(get_logger(), "handing back to PX4 (loiter): %s", reason);
        phase_ = DONE;
    }

    void read_body(fixed_wing::Body<T>& body){
        ned_to_enu(odometry_.position.data(), body.position);
        ned_to_enu(odometry_.velocity.data(), body.linear_velocity);  // NED velocity frame (PX4 default)
        quaternion_ned_frd_to_enu_flu(odometry_.q.data(), body.orientation);
        body.angular_velocity[0] = odometry_.angular_velocity[0];  // FRD -> FLU
        body.angular_velocity[1] = -odometry_.angular_velocity[1];
        body.angular_velocity[2] = -odometry_.angular_velocity[2];
        for(int i = 0; i < fixed_wing::N_ACTUATORS; i++) body.actuators[i] = 0;  // the actor does not see them
    }

    void step(){
        // heartbeat: PX4 only accepts and keeps offboard while it is streamed
        px4_msgs::msg::OffboardControlMode mode{};
        mode.timestamp = now_us();
        mode.direct_actuator = true;
        mode_pub_->publish(mode);
        if(phase_ == DONE){
            if(++done_steps_ > 100) rclcpp::shutdown();  // 2 s for the loiter command, then the node ends
            return;
        }
        if(!have_odometry_ || !have_status_) return;
        fixed_wing::State<T> s{};
        read_body(s.body);
        T t = (now_us() - start_us_) * 1e-6;  // wall clock: only for the offboard switch timeout
        // flight time on PX4's clock: in SITL the simulation's, which stops when the PC freezes, unlike the wall clock
        // (the point would jump ahead). The clock synchronization's own jumps are left out
        if(last_px4_us_ > 0 && odometry_.timestamp > last_px4_us_){
            flight_time_ += std::min((odometry_.timestamp - last_px4_us_) * 1e-6, 0.1);
        }
        last_px4_us_ = odometry_.timestamp;
        if(phase_ == WAITING){
            heartbeats_++;
            bool armed = status_.arming_state == px4_msgs::msg::VehicleStatus::ARMING_STATE_ARMED;
            if(!armed || s.body.position[2] < start_altitude_ || heartbeats_ < 50) return;
            if(wait_for_loiter_ && status_.nav_state != px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_AUTO_LOITER) return;
            if(fixed_wing::norm3(s.body.linear_velocity) > max_start_speed_) return;
            // the reference point as in training, relative to the current course
            T course = std::atan2(s.body.linear_velocity[1], s.body.linear_velocity[0]);
            T bearing = course + start_bearing_;
            for(int i = 0; i < 2; i++) reference_position_[i] = s.body.position[i] + start_distance_ * (i == 0 ? std::cos(bearing) : std::sin(bearing));
            reference_position_[2] = s.body.position[2];
            reference_velocity_ = {reference_speed_ * std::cos(course), reference_speed_ * std::sin(course), 0};
            // action history: what PX4 was roughly commanding, surfaces centred and the trim throttle (25%)
            for(auto& row: history_) row = {0, 0, 0, T(-0.5)};
            command(px4_msgs::msg::VehicleCommand::VEHICLE_CMD_DO_SET_MODE, 1, 6);  // custom main mode 6: offboard
            start_us_ = now_us();
            t = 0;
            flight_time_ = 0;
            phase_ = SWITCHING;
            RCLCPP_INFO(get_logger(), "reference placed, switching to offboard");
        }
        bool offboard = status_.nav_state == px4_msgs::msg::VehicleStatus::NAVIGATION_STATE_OFFBOARD;
        if(phase_ == SWITCHING){
            // no command before PX4 is in offboard: its own controller still drives the same outputs
            if(!offboard){
                if(t > 1) hand_back("PX4 did not switch to offboard");
                return;
            }
            phase_ = FLYING;
            RCLCPP_INFO(get_logger(), "offboard: the policy flies");
        }
        else if(!offboard){ hand_back("PX4 left offboard"); return; }
        // the loop runs on a PC: a late step means the airplane flew on the previous command meanwhile
        uint64_t now = now_us();
        if(last_step_us_ > 0 && now - last_step_us_ > 100000){
            RCLCPP_WARN(get_logger(), "control step %.0f ms late", (now - last_step_us_) * 1e-3 - 20);
        }
        last_step_us_ = now;
        // the reference moves on, as in simulate_task
        for(int i = 0; i < 3; i++) s.reference_position[i] = reference_position_[i] + reference_velocity_[i] * flight_time_;
        for(int i = 0; i < 3; i++) s.reference_velocity[i] = reference_velocity_[i];
        s.setpoint_altitude = s.reference_position[2];
        for(int h = 0; h < fixed_wing::ACTION_HISTORY; h++) for(int i = 0; i < 4; i++) s.action_history[h][i] = history_[h][i];
        float observation[fixed_wing::policy::INPUT], action[fixed_wing::policy::ACTIONS];
        fixed_wing::observe_actor(p_, s, [&](int k, T v){ observation[k] = float(v); });
        fixed_wing::policy::evaluate(observation, action);
        if(identify_) override_for_identification(flight_time_, action);
        for(int h = 0; h + 1 < fixed_wing::ACTION_HISTORY; h++) history_[h] = history_[h + 1];
        for(int i = 0; i < 4; i++) history_[fixed_wing::ACTION_HISTORY - 1][i] = action[i];
        // servos 0 and 1: elevons in opposition (aileron), 2: elevator, 3: rudder; +-1 = +-45 deg, as in training
        px4_msgs::msg::ActuatorServos servos{};
        servos.timestamp = now_us();
        for(auto& c: servos.control) c = 0;
        servos.control[0] = action[fixed_wing::AILERON];
        servos.control[1] = -action[fixed_wing::AILERON];
        servos.control[2] = action[fixed_wing::ELEVATOR];
        servos.control[3] = action[fixed_wing::RUDDER];
        px4_msgs::msg::ActuatorMotors motors{};
        motors.timestamp = servos.timestamp;
        for(auto& c: motors.control) c = NAN;
        motors.control[0] = (action[fixed_wing::THROTTLE] + 1) / 2;  // [-1, 1] -> [0, 1]
        servos_pub_->publish(servos);
        motors_pub_->publish(motors);
        log_ << flight_time_;
        for(int i = 0; i < 3; i++) log_ << "," << s.body.position[i];
        for(int i = 0; i < 3; i++) log_ << "," << s.body.linear_velocity[i];
        for(int i = 0; i < 4; i++) log_ << "," << s.body.orientation[i];
        for(int i = 0; i < 3; i++) log_ << "," << s.body.angular_velocity[i];
        for(int i = 0; i < 3; i++) log_ << "," << s.reference_position[i];
        for(int i = 0; i < 4; i++) log_ << "," << action[i];
        log_ << "," << int(status_.nav_state) << "," << odometry_.timestamp << "\n";
        T e[3], e_velocity[3];
        fixed_wing::tracking_errors(s, e, e_velocity);
        if(s.body.position[2] < min_altitude_) hand_back("below min_altitude");
        else if(fixed_wing::norm3(e) > 300) hand_back("more than 300 m from the point");
        else if(flight_time_ > duration_) hand_back("duration reached");
    }

    fixed_wing::Parameters<T> p_;
    T start_altitude_, min_altitude_, start_distance_, start_bearing_, reference_speed_, duration_, max_start_speed_;
    bool identify_ = false, wait_for_loiter_ = false;
    uint64_t last_step_us_ = 0;
    std::array<T, 3> reference_position_{}, reference_velocity_{};
    std::array<std::array<T, 4>, fixed_wing::ACTION_HISTORY> history_{};
    px4_msgs::msg::VehicleOdometry odometry_{};
    px4_msgs::msg::VehicleStatus status_{};
    bool have_odometry_ = false, have_status_ = false;
    int heartbeats_ = 0, done_steps_ = 0;
    uint64_t start_us_ = 0, last_px4_us_ = 0;
    T flight_time_ = 0;
    Phase phase_ = WAITING;
    std::ofstream log_;
    rclcpp::Subscription<px4_msgs::msg::VehicleOdometry>::SharedPtr odometry_sub_;
    rclcpp::Subscription<px4_msgs::msg::VehicleStatus>::SharedPtr status_sub_;
    rclcpp::Publisher<px4_msgs::msg::OffboardControlMode>::SharedPtr mode_pub_;
    rclcpp::Publisher<px4_msgs::msg::VehicleCommand>::SharedPtr command_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorMotors>::SharedPtr motors_pub_;
    rclcpp::Publisher<px4_msgs::msg::ActuatorServos>::SharedPtr servos_pub_;
    rclcpp::TimerBase::SharedPtr timer_;
};

int main(int argc, char** argv){
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<PolicyNode>());
    rclcpp::shutdown();
    return 0;
}
