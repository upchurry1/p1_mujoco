#include "mujoco_p1_backend.h"

#include "p1_config.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iostream>

namespace p1_sim {
namespace {

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
    delayed_torque_queue_.clear();
    motor_delay_has_output_ = false;
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

    imu_quat_adr_ = sensorAddress("imu_quat");
    imu_gyro_adr_ = sensorAddress("imu_gyro");
    imu_acc_adr_ = sensorAddress("imu_acc");
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
    return true;
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
        data_->qpos[qpos_adr] = q_motor_rad[static_cast<std::size_t>(motor_index)];
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
    delayed_torque_queue_.clear();
    applied_tau_.fill(0.0);
    motor_delay_has_output_ = false;
    if (model_ && data_) {
        for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
            const auto idx = static_cast<std::size_t>(motor_index);
            const int actuator_id = actuator_ids_[idx];
            if (actuator_id >= 0) {
                applied_tau_[idx] = data_->ctrl[actuator_id];
                motor_delay_has_output_ = true;
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
            state.quat[static_cast<std::size_t>(i)] = data_->sensordata[imu_quat_adr_ + i];
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
            state.gyro[static_cast<std::size_t>(i)] = data_->sensordata[imu_gyro_adr_ + i];
        }
    } else {
        state.gyro.fill(0.0);
    }
    if (imu_acc_adr_ >= 0) {
        for (int i = 0; i < 3; ++i) {
            state.accel[static_cast<std::size_t>(i)] = data_->sensordata[imu_acc_adr_ + i];
        }
    } else {
        state.accel.fill(0.0);
    }
    state.projected_gravity = projectedGravityFromQuat(state.quat);
    return true;
}

void MujocoP1Backend::applyMitTorques()
{
    if (!model_ || !data_) {
        return;
    }

    std::array<double, kPolicyDof> requested_tau{};
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const auto idx = static_cast<std::size_t>(motor_index);
        const int actuator_id = actuator_ids_[idx];
        const double q = readMotorPosition(motor_index);
        const double dq = readMotorVelocity(motor_index);
        double tau = command_.tau_ff[idx] +
                     command_.kp[idx] * (command_.q[idx] - q) +
                     command_.kd[idx] * (command_.dq[idx] - dq);

        if (model_->actuator_ctrllimited[actuator_id]) {
            const double lo = model_->actuator_ctrlrange[2 * actuator_id + 0];
            const double hi = model_->actuator_ctrlrange[2 * actuator_id + 1];
            tau = std::clamp(tau, lo, hi);
        }
        requested_tau[idx] = tau;
    }

    if (!motor_delay_enabled_ || !motor_delay_active_ ||
        active_motor_delay_seconds_ <= 1e-12) {
        delayed_torque_queue_.clear();
        applied_tau_ = requested_tau;
        motor_delay_has_output_ = true;
        writeMotorTorques(applied_tau_);
        return;
    }

    if (!motor_delay_has_output_) {
        applied_tau_ = requested_tau;
        motor_delay_has_output_ = true;
    }

    delayed_torque_queue_.push_back(
        DelayedTorqueCommand{data_->time + active_motor_delay_seconds_,
                             requested_tau});
    while (!delayed_torque_queue_.empty() &&
           delayed_torque_queue_.front().release_time <= data_->time + 1e-12) {
        applied_tau_ = delayed_torque_queue_.front().tau;
        delayed_torque_queue_.pop_front();
    }
    writeMotorTorques(applied_tau_);
}

double MujocoP1Backend::controlOfMotor(int motor_index) const
{
    if (motor_index < 0 || motor_index >= kPolicyDof) {
        return 0.0;
    }
    const int actuator_id = actuator_ids_[static_cast<std::size_t>(motor_index)];
    return data_->ctrl[actuator_id];
}

int MujocoP1Backend::sensorAddress(const std::string& name) const
{
    const int id = mj_name2id(model_, mjOBJ_SENSOR, name.c_str());
    if (id < 0) {
        return -1;
    }
    return model_->sensor_adr[id];
}

double MujocoP1Backend::readMotorPosition(int motor_index) const
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
    const int adr = dq_sensor_adr_[idx];
    if (adr >= 0) {
        return data_->sensordata[adr];
    }

    const int joint_id = joint_ids_[idx];
    if (joint_id >= 0) {
        return data_->qvel[model_->jnt_dofadr[joint_id]];
    }
    return 0.0;
}

double MujocoP1Backend::readMotorForce(int motor_index) const
{
    const auto idx = static_cast<std::size_t>(motor_index);
    const int adr = tau_sensor_adr_[idx];
    if (adr >= 0) {
        return data_->sensordata[adr];
    }

    const int actuator_id = actuator_ids_[idx];
    if (actuator_id >= 0) {
        return data_->actuator_force[actuator_id];
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

void MujocoP1Backend::writeMotorTorques(
    const std::array<double, kPolicyDof>& tau)
{
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const auto idx = static_cast<std::size_t>(motor_index);
        const int actuator_id = actuator_ids_[idx];
        data_->ctrl[actuator_id] = tau[idx];
    }
}

}  // namespace p1_sim
