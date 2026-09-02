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
    void setJointZeroOffset(const std::array<double, kPolicyDof>& offset_rad);
    void setMujocoJointDirection(const std::array<int, kPolicyDof>& direction);
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
    bool applyMitTorques();
    double controlOfMotor(int motor_index) const;
    double torquePermilleOfMotor(int motor_index, double torque_nm) const;
    void setImuNoiseEnabled(bool enabled);

private:
    struct MitCommandCache {
        std::array<double, kPolicyDof> q{};
        std::array<double, kPolicyDof> dq{};
        std::array<double, kPolicyDof> kp{};
        std::array<double, kPolicyDof> kd{};
        std::array<double, kPolicyDof> tau_ff{};
    };

    struct DelayedMitCommand {
        double release_time = 0.0;
        MitCommandCache command{};
    };

    int sensorAddress(const std::string& name) const;
    double sensorNoiseStd(int sensor_id) const;
    double sampleImuNoise(double stddev) const;
    double readRawMotorPosition(int motor_index) const;
    double readMotorPosition(int motor_index) const;
    double readMotorVelocity(int motor_index) const;
    double readMotorForce(int motor_index) const;
    bool computeMitTorques(const MitCommandCache& command,
                           std::array<double, kPolicyDof>& requested_tau) const;
    void sampleMotorDelay();
    bool writeMotorTorques(const std::array<double, kPolicyDof>& tau);

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
    int imu_quat_sensor_id_ = -1;
    int imu_gyro_sensor_id_ = -1;
    int imu_acc_sensor_id_ = -1;
    double imu_quat_noise_std_ = 0.0;
    double imu_gyro_noise_std_ = 0.0;
    double imu_acc_noise_std_ = 0.0;
    bool imu_noise_enabled_ = false;
    int root_joint_id_ = -1;
    std::array<double, kPolicyDof> joint_zero_offset_rad_{};
    std::array<int, kPolicyDof> mujoco_joint_direction_{{
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    }};
    MitCommandCache command_{};
    MitCommandCache applied_command_{};
    bool motor_delay_enabled_ = false;
    bool motor_delay_active_ = false;
    bool motor_delay_has_command_ = false;
    double motor_delay_min_seconds_ = 0.010;
    double motor_delay_max_seconds_ = 0.020;
    double active_motor_delay_seconds_ = 0.010;
    std::deque<DelayedMitCommand> delayed_command_queue_;
    std::array<double, kPolicyDof> applied_tau_{};
    std::mt19937 delay_rng_{5489u};
    mutable std::mt19937 imu_noise_rng_{20260827u};
};

}  // namespace p1_sim
