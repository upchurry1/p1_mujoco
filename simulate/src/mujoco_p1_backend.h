#pragma once

#include "p1_types.h"

#include <mujoco/mujoco.h>

#include <array>
#include <deque>
#include <random>
#include <string>

namespace p1_sim {

class MujocoP1Backend {
public:
    MujocoP1Backend(mjModel* model, mjData* data);

    bool initialize();
    void setMitTargets(const std::array<double, kPolicyDof>& q_motor_rad,
                       const std::array<double, kPolicyDof>& kp,
                       const std::array<double, kPolicyDof>& kd);
    void setJointPositionsFromMotorTarget(const std::array<double, kPolicyDof>& q_motor_rad);
    void setRootPose(const std::array<double, 3>& pos,
                     const std::array<double, 4>& quat_wxyz);
    void configureMotorDelay(bool enabled, double min_seconds, double max_seconds);
    void setMotorDelayActive(bool active);
    void resetMotorDelay();
    bool readState(P1StateSnapshot& state) const;
    void applyMitTorques();
    double controlOfMotor(int motor_index) const;

private:
    struct MitCommandCache {
        std::array<double, kPolicyDof> q{};
        std::array<double, kPolicyDof> dq{};
        std::array<double, kPolicyDof> kp{};
        std::array<double, kPolicyDof> kd{};
        std::array<double, kPolicyDof> tau_ff{};
    };

    struct DelayedTorqueCommand {
        double release_time = 0.0;
        std::array<double, kPolicyDof> tau{};
    };

    int sensorAddress(const std::string& name) const;
    double readMotorPosition(int motor_index) const;
    double readMotorVelocity(int motor_index) const;
    double readMotorForce(int motor_index) const;
    void sampleMotorDelay();
    void writeMotorTorques(const std::array<double, kPolicyDof>& tau);

    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    std::array<int, kPolicyDof> actuator_ids_{};
    std::array<int, kPolicyDof> joint_ids_{};
    std::array<int, kPolicyDof> q_sensor_adr_{};
    std::array<int, kPolicyDof> dq_sensor_adr_{};
    std::array<int, kPolicyDof> tau_sensor_adr_{};
    int imu_quat_adr_ = -1;
    int imu_gyro_adr_ = -1;
    int imu_acc_adr_ = -1;
    int root_joint_id_ = -1;
    MitCommandCache command_{};
    bool motor_delay_enabled_ = false;
    bool motor_delay_active_ = false;
    bool motor_delay_has_output_ = false;
    double motor_delay_min_seconds_ = 0.010;
    double motor_delay_max_seconds_ = 0.020;
    double active_motor_delay_seconds_ = 0.010;
    std::deque<DelayedTorqueCommand> delayed_torque_queue_;
    std::array<double, kPolicyDof> applied_tau_{};
    std::mt19937 delay_rng_{5489u};
};

}  // namespace p1_sim
