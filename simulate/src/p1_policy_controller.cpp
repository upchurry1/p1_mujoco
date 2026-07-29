#include "p1_policy_controller.h"

#include <algorithm>
#include <cstddef>
#include <iostream>
#include <utility>

namespace p1_sim {
namespace {

static_assert(inference::TorchPolicyRunner::kInputSize == kPolicyObservationSize,
              "TorchPolicyRunner input size must match robot_deploy non-V3 observations.");
static_assert(inference::TorchPolicyRunner::kOutputSize == kPolicyDof,
              "TorchPolicyRunner output size must be 12.");

template <std::size_t TermSize, std::size_t ObservationSize>
void fillTermHistory(std::array<float, ObservationSize>& history,
                     std::size_t offset,
                     std::size_t frame_stack,
                     const std::array<float, TermSize>& current_term)
{
    for (std::size_t frame = 0; frame < frame_stack; ++frame) {
        std::copy(current_term.begin(),
                  current_term.end(),
                  history.begin() + offset + frame * TermSize);
    }
}

template <std::size_t TermSize, std::size_t ObservationSize>
void appendTermHistory(std::array<float, ObservationSize>& history,
                       std::size_t offset,
                       std::size_t frame_stack,
                       const std::array<float, TermSize>& current_term)
{
    const std::size_t term_history_size = frame_stack * TermSize;
    std::copy(history.begin() + offset + TermSize,
              history.begin() + offset + term_history_size,
              history.begin() + offset);
    std::copy(current_term.begin(),
              current_term.end(),
              history.begin() + offset + term_history_size - TermSize);
}

}  // namespace

P1RealDeployPolicy::P1RealDeployPolicy(PolicyConfig config)
    : config_(std::move(config))
{
    resetPolicyState();
}

bool P1RealDeployPolicy::load()
{
    mapConfiguredGains();
    if (!runner_.load(config_.policy_model_path)) {
        std::cerr << "[ERROR] Failed to load TorchScript policy: "
                  << runner_.last_error() << "\n";
        return false;
    }
    resetPolicyState();
    return true;
}

void P1RealDeployPolicy::resetPolicyState()
{
    last_action_raw_.fill(0.0F);
    observation_history_.fill(0.0F);
    observation_history_ready_ = false;
}

void P1RealDeployPolicy::buildStandMotorTarget(
    std::array<double, kPolicyDof>& target_motor) const
{
    buildMotorTargetFromModel(config_, config_.stand_pose_rad, target_motor);
}

bool P1RealDeployPolicy::step(const P1StateSnapshot& state,
                              double vx,
                              double vy,
                              double yaw_rate,
                              std::array<double, kPolicyDof>& target_motor_rad,
                              std::array<float, kPolicyDof>& raw_action)
{
    std::array<float, kPolicyObservationSize> observation{};
    if (!buildObservation(state, vx, vy, yaw_rate, observation)) {
        return false;
    }

    raw_action.fill(0.0F);
    if (!runner_.infer(observation, raw_action)) {
        std::cerr << "[ERROR] TorchScript inference failed: "
                  << runner_.last_error() << "\n";
        return false;
    }

    last_action_raw_ = raw_action;

    std::array<double, kPolicyDof> target_model_rad{};
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto i = static_cast<std::size_t>(model_index);
        const double scaled_action =
            static_cast<double>(raw_action[i]) * config_.action_scale[i];
        const double clipped_offset =
            std::clamp(scaled_action,
                       config_.action_clip[i][0],
                       config_.action_clip[i][1]);
        target_model_rad[i] = config_.stand_pose_rad[i] + clipped_offset;
    }

    buildMotorTargetFromModel(config_, target_model_rad, target_motor_rad);
    return true;
}

const std::array<double, kPolicyDof>& P1RealDeployPolicy::policyKpMotor() const
{
    return policy_mit_kp_motor_;
}

const std::array<double, kPolicyDof>& P1RealDeployPolicy::policyKdMotor() const
{
    return policy_mit_kd_motor_;
}

void P1RealDeployPolicy::mapConfiguredGains()
{
    mapControlArrayToMotor(config_, config_.policy_mit_kp_model, policy_mit_kp_motor_);
    mapControlArrayToMotor(config_, config_.policy_mit_kd_model, policy_mit_kd_motor_);
}

bool P1RealDeployPolicy::buildObservation(
    const P1StateSnapshot& state,
    double vx,
    double vy,
    double yaw_rate,
    std::array<float, kPolicyObservationSize>& observation)
{
    std::array<float, 3> base_ang_vel{};
    std::array<float, 3> projected_gravity{};
    std::array<float, 3> velocity_commands{
        static_cast<float>(vx * config_.command_scale[0]),
        static_cast<float>(vy * config_.command_scale[1]),
        static_cast<float>(yaw_rate * config_.command_scale[2])
    };
    std::array<float, kPolicyDof> joint_pos_rel{};
    std::array<float, kPolicyDof> joint_vel_rel{};

    for (int i = 0; i < 3; ++i) {
        base_ang_vel[static_cast<std::size_t>(i)] =
            static_cast<float>(state.gyro[static_cast<std::size_t>(i)] *
                               config_.body_ang_vel_scale[static_cast<std::size_t>(i)]);
        projected_gravity[static_cast<std::size_t>(i)] =
            static_cast<float>(state.projected_gravity[static_cast<std::size_t>(i)]);
    }

    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const int motor_index =
            config_.control_to_motor_index[static_cast<std::size_t>(model_index)];
        const double q_model =
            state.q[static_cast<std::size_t>(motor_index)];
        const double dq_model =
            state.dq[static_cast<std::size_t>(motor_index)];

        joint_pos_rel[static_cast<std::size_t>(model_index)] =
            static_cast<float>((q_model -
                config_.stand_pose_rad[static_cast<std::size_t>(model_index)]) *
                config_.dof_pos_scale[static_cast<std::size_t>(model_index)]);
        joint_vel_rel[static_cast<std::size_t>(model_index)] =
            static_cast<float>(dq_model *
                config_.dof_vel_scale[static_cast<std::size_t>(model_index)]);
    }

    constexpr std::size_t kBaseAngVelOffset = 0;
    constexpr std::size_t kProjectedGravityOffset = 15;
    constexpr std::size_t kVelocityCommandsOffset = 30;
    constexpr std::size_t kJointPosRelOffset = 45;
    constexpr std::size_t kJointVelRelOffset = 105;
    constexpr std::size_t kLastActionOffset = 165;

    if (!observation_history_ready_) {
        fillTermHistory(observation_history_, kBaseAngVelOffset,
                        kPolicyFrameStack, base_ang_vel);
        fillTermHistory(observation_history_, kProjectedGravityOffset,
                        kPolicyFrameStack, projected_gravity);
        fillTermHistory(observation_history_, kVelocityCommandsOffset,
                        kPolicyFrameStack, velocity_commands);
        fillTermHistory(observation_history_, kJointPosRelOffset,
                        kPolicyFrameStack, joint_pos_rel);
        fillTermHistory(observation_history_, kJointVelRelOffset,
                        kPolicyFrameStack, joint_vel_rel);
        fillTermHistory(observation_history_, kLastActionOffset,
                        kPolicyFrameStack, last_action_raw_);
        observation_history_ready_ = true;
    } else {
        appendTermHistory(observation_history_, kBaseAngVelOffset,
                          kPolicyFrameStack, base_ang_vel);
        appendTermHistory(observation_history_, kProjectedGravityOffset,
                          kPolicyFrameStack, projected_gravity);
        appendTermHistory(observation_history_, kVelocityCommandsOffset,
                          kPolicyFrameStack, velocity_commands);
        appendTermHistory(observation_history_, kJointPosRelOffset,
                          kPolicyFrameStack, joint_pos_rel);
        appendTermHistory(observation_history_, kJointVelRelOffset,
                          kPolicyFrameStack, joint_vel_rel);
        appendTermHistory(observation_history_, kLastActionOffset,
                          kPolicyFrameStack, last_action_raw_);
    }

    observation = observation_history_;
    return true;
}

void buildMotorTargetFromModel(const PolicyConfig& config,
                               const std::array<double, kPolicyDof>& target_q_model_rad,
                               std::array<double, kPolicyDof>& target_q_motor_rad)
{
    target_q_motor_rad.fill(0.0);
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto model_idx = static_cast<std::size_t>(model_index);
        const int motor_index = config.control_to_motor_index[model_idx];
        target_q_motor_rad[static_cast<std::size_t>(motor_index)] =
            target_q_model_rad[model_idx];
    }
}

}  // namespace p1_sim
