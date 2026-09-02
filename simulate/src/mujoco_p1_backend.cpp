#include "mujoco_p1_backend.h"

#include "p1_config.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace p1_sim {
namespace {

constexpr std::array<double, kPolicyDof> kMotorRatedTorqueNm{{
    43.0, 43.0, 20.0, 20.0, 10.5, 10.5,
    43.0, 43.0, 20.0, 20.0, 10.5, 10.5,
}};

std::array<double, 3> projectedGravityFromQuat(const std::array<double, 4>& quat_wxyz)
{
    const double w = quat_wxyz[0];
    const double x = quat_wxyz[1];
    const double y = quat_wxyz[2];
    const double z = quat_wxyz[3];

    const double norm = std::sqrt(w * w + x * x + y * y + z * z);
    if (norm < 1e-9 || !std::isfinite(norm)) {
        return {0.0, 0.0, -1.0};
    }

    const double wn = w / norm;
    const double xn = x / norm;
    const double yn = y / norm;
    const double zn = z / norm;

    return {
        2.0 * (yn * wn - xn * zn),
       -2.0 * (yn * zn + xn * wn),
       -1.0 + 2.0 * (xn * xn + yn * yn)
    };
}

}  // namespace

MujocoP1Backend::MujocoP1Backend(mjModel* model, mjData* data)
    : model_(model), data_(data)
{
}

bool MujocoP1Backend::initialize()
{
    if (!model_ || !data_) {
        return false;
    }
    applied_tau_.fill(0.0);
    delayed_command_queue_.clear();
    applied_command_ = command_;
    motor_delay_has_command_ = false;
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const char* name = kMotorNamesP1RealOrder[static_cast<std::size_t>(motor_index)];
        const int actuator_id = mj_name2id(model_, mjOBJ_ACTUATOR, name);
        if (actuator_id < 0) {
            std::cerr << "[ERROR] Missing P1 actuator in XML: " << name << "\n";
            return false;
        }

        actuator_ids_[static_cast<std::size_t>(motor_index)] = actuator_id;
        q_sensor_adr_[static_cast<std::size_t>(motor_index)] =
            sensorAddress(std::string(name) + "_p");
        dq_sensor_adr_[static_cast<std::size_t>(motor_index)] =
            sensorAddress(std::string(name) + "_v");
        tau_sensor_adr_[static_cast<std::size_t>(motor_index)] =
            sensorAddress(std::string(name) + "_f");

        if (model_->actuator_trntype[actuator_id] != mjTRN_JOINT) {
            std::cerr << "[ERROR] Actuator " << name
                      << " is not a joint torque actuator.\n";
            return false;
        }
        joint_ids_[static_cast<std::size_t>(motor_index)] =
            model_->actuator_trnid[2 * actuator_id];
    }

    imu_quat_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_quat");
    imu_gyro_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_gyro");
    imu_acc_sensor_id_ = mj_name2id(model_, mjOBJ_SENSOR, "imu_acc");
    imu_quat_adr_ = sensorAddress("imu_quat");
    imu_gyro_adr_ = sensorAddress("imu_gyro");
    imu_acc_adr_ = sensorAddress("imu_acc");
    imu_quat_noise_std_ = sensorNoiseStd(imu_quat_sensor_id_);
    imu_gyro_noise_std_ = sensorNoiseStd(imu_gyro_sensor_id_);
    imu_acc_noise_std_ = sensorNoiseStd(imu_acc_sensor_id_);
    root_joint_id_ = mj_name2id(model_, mjOBJ_JOINT, "root");
    if (imu_quat_adr_ < 0) {
        std::cerr << "[WARN] imu_quat sensor not found; using root qpos quaternion fallback.\n";
    }
    if (imu_gyro_adr_ < 0) {
        std::cerr << "[WARN] imu_gyro sensor not found; base angular velocity observation is zero.\n";
    }
    if (imu_acc_adr_ < 0) {
        std::cerr << "[WARN] imu_acc sensor not found; base linear acceleration display is zero.\n";
    }
    std::cout << "[INFO] IMU noise stddev: enabled="
              << (imu_noise_enabled_ ? "true" : "false")
              << " quat=" << imu_quat_noise_std_
              << " gyro=" << imu_gyro_noise_std_
              << " accel=" << imu_acc_noise_std_ << "\n";
    return true;
}

void MujocoP1Backend::setImuNoiseEnabled(bool enabled)
{
    imu_noise_enabled_ = enabled;
}

void MujocoP1Backend::setJointZeroOffset(
    const std::array<double, kPolicyDof>& offset_rad)
{
    joint_zero_offset_rad_ = offset_rad;
}

void MujocoP1Backend::setMujocoJointDirection(
    const std::array<int, kPolicyDof>& direction)
{
    mujoco_joint_direction_ = direction;
}

void MujocoP1Backend::setMitTargets(const std::array<double, kPolicyDof>& q_motor_rad,
                                    const std::array<double, kPolicyDof>& kp,
                                    const std::array<double, kPolicyDof>& kd)
{
    command_.q = q_motor_rad;
    command_.dq.fill(0.0);
    command_.kp = kp;
    command_.kd = kd;
    command_.tau_ff.fill(0.0);
}

void MujocoP1Backend::setJointPositionsFromMotorTarget(
    const std::array<double, kPolicyDof>& q_motor_rad)
{
    if (!model_ || !data_) {
        return;
    }

    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const int joint_id = joint_ids_[static_cast<std::size_t>(motor_index)];
        if (joint_id < 0) {
            continue;
        }

        const int qpos_adr = model_->jnt_qposadr[joint_id];
        const int qvel_adr = model_->jnt_dofadr[joint_id];
        const auto idx = static_cast<std::size_t>(motor_index);
        const double direction =
            static_cast<double>(mujoco_joint_direction_[idx]);
        data_->qpos[qpos_adr] =
            direction * (q_motor_rad[idx] - joint_zero_offset_rad_[idx]);
        data_->qvel[qvel_adr] = 0.0;
    }
    mj_forward(model_, data_);
}

void MujocoP1Backend::setRootPose(const std::array<double, 3>& pos,
                                  const std::array<double, 4>& quat_wxyz)
{
    if (!model_ || !data_ || root_joint_id_ < 0 ||
        model_->jnt_type[root_joint_id_] != mjJNT_FREE) {
        return;
    }

    const int qpos_adr = model_->jnt_qposadr[root_joint_id_];
    const int qvel_adr = model_->jnt_dofadr[root_joint_id_];
    data_->qpos[qpos_adr + 0] = pos[0];
    data_->qpos[qpos_adr + 1] = pos[1];
    data_->qpos[qpos_adr + 2] = pos[2];

    double quat[4] = {quat_wxyz[0], quat_wxyz[1], quat_wxyz[2], quat_wxyz[3]};
    const double norm =
        std::sqrt(quat[0] * quat[0] + quat[1] * quat[1] +
                  quat[2] * quat[2] + quat[3] * quat[3]);
    if (norm > 1e-9 && std::isfinite(norm)) {
        for (double& value : quat) {
            value /= norm;
        }
    } else {
        quat[0] = 1.0;
        quat[1] = 0.0;
        quat[2] = 0.0;
        quat[3] = 0.0;
    }

    data_->qpos[qpos_adr + 3] = quat[0];
    data_->qpos[qpos_adr + 4] = quat[1];
    data_->qpos[qpos_adr + 5] = quat[2];
    data_->qpos[qpos_adr + 6] = quat[3];
    for (int i = 0; i < 6; ++i) {
        data_->qvel[qvel_adr + i] = 0.0;
    }
    mj_forward(model_, data_);
}

void MujocoP1Backend::configureMotorDelay(bool enabled,
                                          double min_seconds,
                                          double max_seconds)
{
    motor_delay_enabled_ = enabled;
    motor_delay_min_seconds_ = min_seconds;
    motor_delay_max_seconds_ = max_seconds;
    motor_delay_active_ = enabled;
    resetMotorDelay();
}

void MujocoP1Backend::setMotorDelayActive(bool active)
{
    const bool next_active = motor_delay_enabled_ && active;
    if (motor_delay_active_ == next_active) {
        return;
    }

    motor_delay_active_ = next_active;
    resetMotorDelay();
}

void MujocoP1Backend::resetMotorDelay()
{
    delayed_command_queue_.clear();
    applied_command_ = command_;
    applied_tau_.fill(0.0);
    motor_delay_has_command_ = true;
    if (model_ && data_) {
        for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
            const auto idx = static_cast<std::size_t>(motor_index);
            const int actuator_id = actuator_ids_[idx];
            if (actuator_id >= 0) {
                applied_tau_[idx] = data_->ctrl[actuator_id];
            }
        }
    }
    sampleMotorDelay();
}

bool MujocoP1Backend::readState(P1StateSnapshot& state) const
{
    if (!model_ || !data_) {
        return false;
    }

    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        state.q[static_cast<std::size_t>(motor_index)] = readMotorPosition(motor_index);
        state.dq[static_cast<std::size_t>(motor_index)] = readMotorVelocity(motor_index);
        state.tau[static_cast<std::size_t>(motor_index)] = readMotorForce(motor_index);
    }

    if (imu_quat_adr_ >= 0) {
        for (int i = 0; i < 4; ++i) {
            state.quat[static_cast<std::size_t>(i)] =
                data_->sensordata[imu_quat_adr_ + i] +
                sampleImuNoise(imu_quat_noise_std_);
        }
    } else if (root_joint_id_ >= 0) {
        const int qpos_adr = model_->jnt_qposadr[root_joint_id_];
        state.quat = {
            data_->qpos[qpos_adr + 3],
            data_->qpos[qpos_adr + 4],
            data_->qpos[qpos_adr + 5],
            data_->qpos[qpos_adr + 6]
        };
    }

    if (imu_gyro_adr_ >= 0) {
        for (int i = 0; i < 3; ++i) {
            state.gyro[static_cast<std::size_t>(i)] =
                data_->sensordata[imu_gyro_adr_ + i] +
                sampleImuNoise(imu_gyro_noise_std_);
        }
    } else {
        state.gyro.fill(0.0);
    }
    if (imu_acc_adr_ >= 0) {
        for (int i = 0; i < 3; ++i) {
            state.accel[static_cast<std::size_t>(i)] =
                data_->sensordata[imu_acc_adr_ + i] +
                sampleImuNoise(imu_acc_noise_std_);
        }
    } else {
        state.accel.fill(0.0);
    }
    state.projected_gravity = projectedGravityFromQuat(state.quat);
    return true;
}

bool MujocoP1Backend::applyMitTorques()
{
    if (!model_ || !data_) {
        return false;
    }

    const MitCommandCache* effective_command = &command_;
    if (!motor_delay_enabled_ || !motor_delay_active_ ||
        active_motor_delay_seconds_ <= 1e-12) {
        delayed_command_queue_.clear();
        applied_command_ = command_;
        motor_delay_has_command_ = true;
    } else {
        if (!motor_delay_has_command_) {
            applied_command_ = command_;
            motor_delay_has_command_ = true;
        }

        delayed_command_queue_.push_back(
            DelayedMitCommand{data_->time + active_motor_delay_seconds_,
                              command_});
        while (!delayed_command_queue_.empty() &&
               delayed_command_queue_.front().release_time <= data_->time + 1e-12) {
            applied_command_ = delayed_command_queue_.front().command;
            delayed_command_queue_.pop_front();
        }
        effective_command = &applied_command_;
    }

    if (!computeMitTorques(*effective_command, applied_tau_)) {
        return false;
    }
    return writeMotorTorques(applied_tau_);
}

bool MujocoP1Backend::computeMitTorques(
    const MitCommandCache& command,
    std::array<double, kPolicyDof>& requested_tau) const
{
    requested_tau.fill(0.0);
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const auto idx = static_cast<std::size_t>(motor_index);
        const int actuator_id = actuator_ids_[idx];
        const double q = readMotorPosition(motor_index);
        const double dq = readMotorVelocity(motor_index);
        const double target_q = command.q[idx];
        const double target_dq = command.dq[idx];
        const double kp = command.kp[idx];
        const double kd = command.kd[idx];
        const double tau_ff = command.tau_ff[idx];
        if (!std::isfinite(q) || !std::isfinite(dq) ||
            !std::isfinite(target_q) || !std::isfinite(target_dq) ||
            !std::isfinite(kp) || !std::isfinite(kd) ||
            !std::isfinite(tau_ff)) {
            const char* actuator_name =
                mj_id2name(model_, mjOBJ_ACTUATOR, actuator_id);
            std::cerr << "[ERROR] Non-finite MIT torque input at t="
                      << data_->time
                      << " motor_index=" << motor_index
                      << " actuator_id=" << actuator_id
                      << " actuator="
                      << (actuator_name ? actuator_name : "<unnamed>")
                      << " q=" << q
                      << " dq=" << dq
                      << " target_q=" << target_q
                      << " target_dq=" << target_dq
                      << " kp=" << kp
                      << " kd=" << kd
                      << " tau_ff=" << tau_ff << "\n";
            return false;
        }

        double tau = command.tau_ff[idx] +
                     command.kp[idx] * (command.q[idx] - q) +
                     command.kd[idx] * (command.dq[idx] - dq);
        if (!std::isfinite(tau)) {
            const char* actuator_name =
                mj_id2name(model_, mjOBJ_ACTUATOR, actuator_id);
            std::cerr << "[ERROR] Non-finite MIT torque at t="
                      << data_->time
                      << " motor_index=" << motor_index
                      << " actuator_id=" << actuator_id
                      << " actuator="
                      << (actuator_name ? actuator_name : "<unnamed>")
                      << " q=" << q
                      << " dq=" << dq
                      << " target_q=" << target_q
                      << " target_dq=" << target_dq
                      << " kp=" << kp
                      << " kd=" << kd
                      << " tau_ff=" << tau_ff
                      << " tau=" << tau << "\n";
            return false;
        }

        if (model_->actuator_ctrllimited[actuator_id]) {
            const double lo = model_->actuator_ctrlrange[2 * actuator_id + 0];
            const double hi = model_->actuator_ctrlrange[2 * actuator_id + 1];
            if (!std::isfinite(lo) || !std::isfinite(hi) || lo > hi) {
                const char* actuator_name =
                    mj_id2name(model_, mjOBJ_ACTUATOR, actuator_id);
                std::cerr << "[ERROR] Invalid actuator ctrlrange at t="
                          << data_->time
                          << " motor_index=" << motor_index
                          << " actuator_id=" << actuator_id
                          << " actuator="
                          << (actuator_name ? actuator_name : "<unnamed>")
                          << " lo=" << lo
                          << " hi=" << hi << "\n";
                return false;
            }
            tau = std::clamp(tau, lo, hi);
        }
        requested_tau[idx] = tau;
    }
    return true;
}

double MujocoP1Backend::controlOfMotor(int motor_index) const
{
    if (motor_index < 0 || motor_index >= kPolicyDof) {
        return 0.0;
    }
    const int actuator_id = actuator_ids_[static_cast<std::size_t>(motor_index)];
    return data_->ctrl[actuator_id];
}

double MujocoP1Backend::torquePermilleOfMotor(int motor_index, double torque_nm) const
{
    if (!model_ || motor_index < 0 || motor_index >= kPolicyDof ||
        !std::isfinite(torque_nm)) {
        return 0.0;
    }

    const double rated_torque_nm =
        kMotorRatedTorqueNm[static_cast<std::size_t>(motor_index)];
    if (!std::isfinite(rated_torque_nm) || rated_torque_nm <= 1e-9) {
        return 0.0;
    }
    return 1000.0 * torque_nm / rated_torque_nm;
}

int MujocoP1Backend::sensorAddress(const std::string& name) const
{
    const int id = mj_name2id(model_, mjOBJ_SENSOR, name.c_str());
    if (id < 0) {
        return -1;
    }
    return model_->sensor_adr[id];
}

double MujocoP1Backend::sensorNoiseStd(int sensor_id) const
{
    if (!model_ || sensor_id < 0 || sensor_id >= model_->nsensor) {
        return 0.0;
    }
    const double stddev = model_->sensor_noise[sensor_id];
    return std::isfinite(stddev) && stddev > 0.0 ? stddev : 0.0;
}

double MujocoP1Backend::sampleImuNoise(double stddev) const
{
    if (!imu_noise_enabled_ || stddev <= 0.0 || !std::isfinite(stddev)) {
        return 0.0;
    }
    std::normal_distribution<double> distribution(0.0, stddev);
    return distribution(imu_noise_rng_);
}

double MujocoP1Backend::readMotorPosition(int motor_index) const
{
    const auto idx = static_cast<std::size_t>(motor_index);
    const double direction =
        static_cast<double>(mujoco_joint_direction_[idx]);
    return direction * readRawMotorPosition(motor_index) +
           joint_zero_offset_rad_[idx];
}

double MujocoP1Backend::readRawMotorPosition(int motor_index) const
{
    const auto idx = static_cast<std::size_t>(motor_index);
    const int adr = q_sensor_adr_[idx];
    if (adr >= 0) {
        return data_->sensordata[adr];
    }

    const int joint_id = joint_ids_[idx];
    if (joint_id >= 0) {
        return data_->qpos[model_->jnt_qposadr[joint_id]];
    }
    return 0.0;
}

double MujocoP1Backend::readMotorVelocity(int motor_index) const
{
    const auto idx = static_cast<std::size_t>(motor_index);
    const double direction =
        static_cast<double>(mujoco_joint_direction_[idx]);
    const int adr = dq_sensor_adr_[idx];
    if (adr >= 0) {
        return direction * data_->sensordata[adr];
    }

    const int joint_id = joint_ids_[idx];
    if (joint_id >= 0) {
        return direction * data_->qvel[model_->jnt_dofadr[joint_id]];
    }
    return 0.0;
}

double MujocoP1Backend::readMotorForce(int motor_index) const
{
    const auto idx = static_cast<std::size_t>(motor_index);
    const double direction =
        static_cast<double>(mujoco_joint_direction_[idx]);
    const int adr = tau_sensor_adr_[idx];
    if (adr >= 0) {
        return direction * data_->sensordata[adr];
    }

    const int actuator_id = actuator_ids_[idx];
    if (actuator_id >= 0) {
        return direction * data_->actuator_force[actuator_id];
    }
    return 0.0;
}

void MujocoP1Backend::sampleMotorDelay()
{
    if (motor_delay_max_seconds_ <= motor_delay_min_seconds_) {
        active_motor_delay_seconds_ = motor_delay_min_seconds_;
        return;
    }

    std::uniform_real_distribution<double> distribution(
        motor_delay_min_seconds_, motor_delay_max_seconds_);
    active_motor_delay_seconds_ = distribution(delay_rng_);
}

bool MujocoP1Backend::writeMotorTorques(
    const std::array<double, kPolicyDof>& tau)
{
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const auto idx = static_cast<std::size_t>(motor_index);
        const int actuator_id = actuator_ids_[idx];
        if (!std::isfinite(tau[idx])) {
            const char* actuator_name =
                mj_id2name(model_, mjOBJ_ACTUATOR, actuator_id);
            std::cerr << "[ERROR] Refusing to write non-finite MuJoCo ctrl at t="
                      << data_->time
                      << " motor_index=" << motor_index
                      << " actuator_id=" << actuator_id
                      << " actuator="
                      << (actuator_name ? actuator_name : "<unnamed>")
                      << " ctrl=" << tau[idx] << "\n";
            return false;
        }
        data_->ctrl[actuator_id] =
            static_cast<double>(mujoco_joint_direction_[idx]) * tau[idx];
    }
    return true;
}

}  // namespace p1_sim
