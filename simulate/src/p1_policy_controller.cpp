#include "p1_policy_controller.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <iomanip>
#include <iostream>
#include <limits>
#include <utility>
#include <vector>

namespace p1_sim {
namespace {

constexpr double kTwoPi = 6.28318530717958647692;
constexpr float kHugeRawActionAbs = 1.0e6F;

static_assert(inference::TorchPolicyRunner::kOutputSize == kPolicyDof,
              "TorchPolicyRunner output size must be 12.");

struct ObservationLayout {
    bool include_gait_phase = true;
    std::size_t single_size = kPolicySingleObservationSizeWithGait;
    std::size_t total_size = kPolicyObservationSizeWithGait;
    std::size_t base_ang_vel_offset = 0;
    std::size_t projected_gravity_offset = 15;
    std::size_t velocity_commands_offset = 30;
    std::size_t gait_phase_offset = 45;
    std::size_t joint_pos_rel_offset = 55;
    std::size_t joint_vel_rel_offset = 115;
    std::size_t last_action_offset = 175;
};

ObservationLayout makeObservationLayout(bool include_gait_phase)
{
    ObservationLayout layout;
    layout.include_gait_phase = include_gait_phase;
    layout.single_size = include_gait_phase
        ? kPolicySingleObservationSizeWithGait
        : kPolicySingleObservationSizeNoGait;
    layout.total_size = layout.single_size * kPolicyFrameStack;
    layout.base_ang_vel_offset = 0;
    layout.projected_gravity_offset = kPolicyFrameStack * 3;
    layout.velocity_commands_offset =
        layout.projected_gravity_offset + kPolicyFrameStack * 3;
    layout.gait_phase_offset =
        layout.velocity_commands_offset + kPolicyFrameStack * 3;
    layout.joint_pos_rel_offset = include_gait_phase
        ? layout.gait_phase_offset + kPolicyFrameStack * 2
        : layout.gait_phase_offset;
    layout.joint_vel_rel_offset =
        layout.joint_pos_rel_offset + kPolicyFrameStack * kPolicyDof;
    layout.last_action_offset =
        layout.joint_vel_rel_offset + kPolicyFrameStack * kPolicyDof;
    return layout;
}

bool policyIndexInRange(int index)
{
    return index >= 0 && index < kPolicyDof;
}

template <std::size_t TermSize, typename History>
void fillTermHistory(History& history,
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

template <std::size_t TermSize, typename History>
void appendTermHistory(History& history,
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

template <typename Values>
bool arrayIsFinite(const Values& values,
                   const char* label,
                   std::uint64_t policy_step_count)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (!std::isfinite(values[i])) {
            std::cerr << "[ERROR] Non-finite " << label
                      << " at policy_step=" << policy_step_count
                      << " index=" << i
                      << " value=" << values[i] << "\n";
            return false;
        }
    }
    return true;
}

template <std::size_t N>
bool arrayWithinAbsLimit(const std::array<float, N>& values,
                         float limit,
                         const char* label,
                         std::uint64_t policy_step_count)
{
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (std::abs(values[i]) > limit) {
            std::cerr << "[ERROR] Huge finite " << label
                      << " at policy_step=" << policy_step_count
                      << " index=" << i
                      << " value=" << values[i]
                      << " limit=" << limit << "\n";
            return false;
        }
    }
    return true;
}

double transformObservationValue(double value,
                                 double scale,
                                 bool clip_enabled,
                                 const std::array<double, 2>& clip,
                                 bool scale_first)
{
    if (!std::isfinite(value)) {
        return value;
    }

    if (scale_first) {
        value *= scale;
        if (clip_enabled) {
            value = std::clamp(value, clip[0], clip[1]);
        }
    } else {
        if (clip_enabled) {
            value = std::clamp(value, clip[0], clip[1]);
        }
        value *= scale;
    }
    return value;
}

template <std::size_t N>
std::array<float, N> transformObservationTerm(
    const std::array<double, N>& raw,
    const std::array<double, N>& scale,
    bool clip_enabled,
    const std::array<double, 2>& clip,
    bool scale_first)
{
    std::array<float, N> transformed{};
    for (std::size_t i = 0; i < N; ++i) {
        transformed[i] = static_cast<float>(
            transformObservationValue(raw[i],
                                      scale[i],
                                      clip_enabled,
                                      clip,
                                      scale_first));
    }
    return transformed;
}

double gaitPhaseGate(const std::array<float, 3>& velocity_commands,
                     const PolicyConfig& config)
{
    const double command_norm = std::sqrt(
        static_cast<double>(velocity_commands[0]) * velocity_commands[0] +
        static_cast<double>(velocity_commands[1]) * velocity_commands[1] +
        static_cast<double>(velocity_commands[2]) * velocity_commands[2]);

    const double denom =
        config.gait_phase_move_threshold - config.gait_phase_stand_threshold;
    double gate = 0.0;
    if (std::abs(denom) > 1e-12) {
        gate = std::clamp(
            (command_norm - config.gait_phase_stand_threshold) / denom,
            0.0,
            1.0);
    } else {
        gate = command_norm > config.gait_phase_move_threshold ? 1.0 : 0.0;
    }
    return gate * gate * (3.0 - 2.0 * gate);
}

template <std::size_t N>
void printFloatArray(const char* label, const std::array<float, N>& values)
{
    std::cerr << "[DEBUG] " << label << "=";
    for (std::size_t i = 0; i < values.size(); ++i) {
        if (i != 0) {
            std::cerr << ",";
        }
        std::cerr << values[i];
    }
    std::cerr << "\n";
}

template <std::size_t TermSize>
std::array<float, TermSize> latestObservationTerm(
    const std::vector<float>& observation,
    std::size_t offset)
{
    std::array<float, TermSize> values{};
    const std::size_t frame_offset = offset + (kPolicyFrameStack - 1) * TermSize;
    std::copy(observation.begin() + frame_offset,
              observation.begin() + frame_offset + TermSize,
              values.begin());
    return values;
}

template <std::size_t TermSize>
std::array<float, TermSize> maxAbsObservationTerm(
    const std::vector<float>& observation,
    std::size_t offset)
{
    std::array<float, TermSize> values{};
    values.fill(0.0F);
    for (std::size_t frame = 0; frame < kPolicyFrameStack; ++frame) {
        const std::size_t frame_offset = offset + frame * TermSize;
        for (std::size_t i = 0; i < TermSize; ++i) {
            const float value = observation[frame_offset + i];
            if (!std::isfinite(value)) {
                values[i] = value;
            } else if (std::isfinite(values[i])) {
                values[i] = std::max(values[i], std::abs(value));
            }
        }
    }
    return values;
}

void printActionDiagnostics(const PolicyConfig& config,
                            const std::array<float, kPolicyDof>& action,
                            std::uint64_t policy_step_count)
{
    std::cerr << "[DEBUG] action_model_order at policy_step="
              << policy_step_count << "\n";
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto idx = static_cast<std::size_t>(model_index);
        const int motor_index = config.control_to_motor_index[idx];
        const char* motor_name = policyIndexInRange(motor_index)
            ? kMotorNamesP1RealOrder[static_cast<std::size_t>(motor_index)]
            : "<invalid>";
        std::cerr << "[DEBUG] action[" << model_index << "]"
                  << " motor_index=" << motor_index
                  << " motor=" << motor_name
                  << " raw=" << action[idx]
                  << " raw_clip=" << config.raw_action_clip
                  << " scale=" << config.action_scale[idx]
                  << " clip=(" << config.action_clip[idx][0]
                  << "," << config.action_clip[idx][1] << ")"
                  << " stand_pose=" << config.stand_pose_rad[idx]
                  << "\n";
    }
}

void printPolicyFailureContext(
    const PolicyConfig& config,
    const P1StateSnapshot& state,
    double vx,
    double vy,
    double yaw_rate,
    const std::vector<float>& observation,
    const std::array<float, kPolicyDof>& last_action_raw,
    const std::array<float, kPolicyDof>& action,
    std::uint64_t policy_step_count)
{
    const ObservationLayout layout =
        makeObservationLayout(config.include_gait_phase_observation);
    std::cerr << std::setprecision(9);
    std::cerr << "[DEBUG] Policy failure context at policy_step="
              << policy_step_count
              << " command=(" << vx << "," << vy << "," << yaw_rate << ")"
              << " policy_dt=" << config.policy_step_dt_s
              << " gait_period=" << config.gait_phase_period_s
              << " include_gait_phase_observation="
              << (layout.include_gait_phase ? "true" : "false")
              << " observation_size=" << observation.size()
              << " observation_scale_first="
              << (config.observation_scale_first ? "true" : "false")
              << " last_action_clip_enabled="
              << (config.last_action_clip_enabled ? "true" : "false")
              << " last_action_clip=(" << config.last_action_clip[0]
              << "," << config.last_action_clip[1] << ")\n";

    printFloatArray("latest_obs.base_ang_vel_scaled",
                    latestObservationTerm<3>(observation,
                                             layout.base_ang_vel_offset));
    printFloatArray("latest_obs.projected_gravity",
                    latestObservationTerm<3>(observation,
                                             layout.projected_gravity_offset));
    printFloatArray("latest_obs.velocity_commands_scaled",
                    latestObservationTerm<3>(observation,
                                             layout.velocity_commands_offset));
    if (layout.include_gait_phase) {
        printFloatArray("latest_obs.gait_phase_sin_cos",
                        latestObservationTerm<2>(observation,
                                                 layout.gait_phase_offset));
    }
    printFloatArray("latest_obs.joint_pos_rel_model_order",
                    latestObservationTerm<kPolicyDof>(
                        observation, layout.joint_pos_rel_offset));
    printFloatArray("latest_obs.joint_vel_rel_model_order",
                    latestObservationTerm<kPolicyDof>(
                        observation, layout.joint_vel_rel_offset));
    printFloatArray("latest_obs.last_action_model_order",
                    latestObservationTerm<kPolicyDof>(
                        observation, layout.last_action_offset));

    printFloatArray("history_max_abs.base_ang_vel_scaled",
                    maxAbsObservationTerm<3>(observation,
                                             layout.base_ang_vel_offset));
    printFloatArray("history_max_abs.projected_gravity",
                    maxAbsObservationTerm<3>(observation,
                                             layout.projected_gravity_offset));
    printFloatArray("history_max_abs.velocity_commands_scaled",
                    maxAbsObservationTerm<3>(observation,
                                             layout.velocity_commands_offset));
    if (layout.include_gait_phase) {
        printFloatArray("history_max_abs.gait_phase_sin_cos",
                        maxAbsObservationTerm<2>(observation,
                                                 layout.gait_phase_offset));
    }
    printFloatArray("history_max_abs.joint_pos_rel_model_order",
                    maxAbsObservationTerm<kPolicyDof>(
                        observation, layout.joint_pos_rel_offset));
    printFloatArray("history_max_abs.joint_vel_rel_model_order",
                    maxAbsObservationTerm<kPolicyDof>(
                        observation, layout.joint_vel_rel_offset));
    printFloatArray("history_max_abs.last_action_model_order",
                    maxAbsObservationTerm<kPolicyDof>(
                        observation, layout.last_action_offset));
    printFloatArray("last_action_for_observation_model_order", last_action_raw);
    printFloatArray("current_action_raw_model_order", action);

    std::cerr << "[DEBUG] joint_state_model_order at policy_step="
              << policy_step_count << "\n";
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto idx = static_cast<std::size_t>(model_index);
        const int motor_index = config.control_to_motor_index[idx];
        const char* motor_name = policyIndexInRange(motor_index)
            ? kMotorNamesP1RealOrder[static_cast<std::size_t>(motor_index)]
            : "<invalid>";
        double q = std::numeric_limits<double>::quiet_NaN();
        double dq = std::numeric_limits<double>::quiet_NaN();
        if (policyIndexInRange(motor_index)) {
            q = state.q[static_cast<std::size_t>(motor_index)];
            dq = state.dq[static_cast<std::size_t>(motor_index)];
        }
        const double q_rel_raw = q - config.stand_pose_rad[idx];
        const double q_rel_obs =
            transformObservationValue(q_rel_raw,
                                      config.dof_pos_scale[idx],
                                      config.joint_pos_rel_clip_enabled,
                                      config.joint_pos_rel_clip,
                                      config.observation_scale_first);
        const double dq_obs =
            transformObservationValue(dq,
                                      config.dof_vel_scale[idx],
                                      config.joint_vel_rel_clip_enabled,
                                      config.joint_vel_rel_clip,
                                      config.observation_scale_first);
        std::cerr << "[DEBUG] model_index=" << model_index
                  << " motor_index=" << motor_index
                  << " motor=" << motor_name
                  << " q=" << q
                  << " dq=" << dq
                  << " q_rel_raw=" << q_rel_raw
                  << " q_rel_obs=" << q_rel_obs
                  << " dq_obs=" << dq_obs
                  << " stand_pose=" << config.stand_pose_rad[idx]
                  << " pos_scale=" << config.dof_pos_scale[idx]
                  << " vel_scale=" << config.dof_vel_scale[idx]
                  << "\n";
    }

    std::cerr << "[DEBUG] imu_raw gyro=("
              << state.gyro[0] << "," << state.gyro[1] << ","
              << state.gyro[2] << ") projected_gravity=("
              << state.projected_gravity[0] << ","
              << state.projected_gravity[1] << ","
              << state.projected_gravity[2] << ") quat=("
              << state.quat[0] << "," << state.quat[1] << ","
              << state.quat[2] << "," << state.quat[3] << ")\n";

    printActionDiagnostics(config, action, policy_step_count);
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
    if (!runner_.load(config_.policy_model_path, policyObservationSize())) {
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
    observation_history_.assign(policyObservationSize(), 0.0F);
    policy_step_count_ = 0;
    observation_history_ready_ = false;
}

std::size_t P1RealDeployPolicy::policySingleObservationSize() const
{
    return config_.include_gait_phase_observation
        ? kPolicySingleObservationSizeWithGait
        : kPolicySingleObservationSizeNoGait;
}

std::size_t P1RealDeployPolicy::policyObservationSize() const
{
    return policySingleObservationSize() * kPolicyFrameStack;
}

void P1RealDeployPolicy::printObservationLayout(std::ostream& stream) const
{
    const ObservationLayout layout =
        makeObservationLayout(config_.include_gait_phase_observation);
    stream << "[INFO] Observation layout: single_size=" << layout.single_size
           << " total_size=" << layout.total_size
           << " frame_stack=" << kPolicyFrameStack
           << " history_order=oldest->newest within each term block\n";

    auto print_range = [&stream](const char* name,
                                 std::size_t offset,
                                 std::size_t dim) {
        const std::size_t size = dim * kPolicyFrameStack;
        stream << "  [" << offset << ":" << (offset + size) << ") "
               << name << "_history"
               << " latest=[" << (offset + size - dim) << ":"
               << (offset + size) << ")\n";
    };

    print_range("base_ang_vel", layout.base_ang_vel_offset, 3);
    print_range("projected_gravity", layout.projected_gravity_offset, 3);
    print_range("velocity_commands", layout.velocity_commands_offset, 3);
    if (layout.include_gait_phase) {
        print_range("gait_phase_sin_cos", layout.gait_phase_offset, 2);
    }
    print_range("joint_pos_rel", layout.joint_pos_rel_offset, kPolicyDof);
    print_range("joint_vel_rel", layout.joint_vel_rel_offset, kPolicyDof);
    print_range("last_action", layout.last_action_offset, kPolicyDof);
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
                              std::array<double, kPolicyDof>& target_q_model_rad,
                              std::array<double, kPolicyDof>& target_motor_rad,
                              std::array<float, kPolicyDof>& raw_action)
{
    std::vector<float> observation;
    if (!buildObservation(state, vx, vy, yaw_rate, observation)) {
        return false;
    }
    if (!arrayIsFinite(observation, "policy observation", policy_step_count_)) {
        std::array<float, kPolicyDof> empty_action{};
        empty_action.fill(std::numeric_limits<float>::quiet_NaN());
        printPolicyFailureContext(config_,
                                  state,
                                  vx,
                                  vy,
                                  yaw_rate,
                                  observation,
                                  last_action_raw_,
                                  empty_action,
                                  policy_step_count_);
        return false;
    }

    raw_action.fill(0.0F);
    if (!inferObservation(observation, raw_action)) {
        return false;
    }
    if (!arrayIsFinite(raw_action, "TorchScript action", policy_step_count_)) {
        printPolicyFailureContext(config_,
                                  state,
                                  vx,
                                  vy,
                                  yaw_rate,
                                  observation,
                                  last_action_raw_,
                                  raw_action,
                                  policy_step_count_);
        return false;
    }
    if (!arrayWithinAbsLimit(raw_action,
                             kHugeRawActionAbs,
                             "TorchScript action",
                                  policy_step_count_)) {
        printPolicyFailureContext(config_,
                                  state,
                                  vx,
                                  vy,
                                  yaw_rate,
                                  observation,
                                  last_action_raw_,
                                  raw_action,
                                  policy_step_count_);
        return false;
    }

    advancePolicyStep(raw_action);

    P1ActionPostprocess postprocess;
    if (!postprocessAction(raw_action, postprocess)) {
        printPolicyFailureContext(config_,
                                  state,
                                  vx,
                                  vy,
                                  yaw_rate,
                                  observation,
                                  last_action_raw_,
                                  raw_action,
                                  policy_step_count_);
        return false;
    }

    target_q_model_rad = postprocess.target_q_model_rad;
    target_motor_rad = postprocess.target_motor_rad;
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
    std::vector<float>& observation)
{
    P1ObservationTerms terms;
    terms.velocity_commands = {vx, vy, yaw_rate};
    terms.last_action = last_action_raw_;
    for (int i = 0; i < 3; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        terms.base_ang_vel[idx] = state.gyro[idx];
        terms.projected_gravity[idx] = state.projected_gravity[idx];
    }
    terms.joint_pos = state.q;
    terms.joint_vel = state.dq;
    return buildObservationFromTerms(terms, observation);
}

bool P1RealDeployPolicy::buildObservationFromTerms(
    const P1ObservationTerms& terms,
    std::vector<float>& observation)
{
    const ObservationLayout layout =
        makeObservationLayout(config_.include_gait_phase_observation);
    if (observation_history_.size() != layout.total_size) {
        observation_history_.assign(layout.total_size, 0.0F);
        observation_history_ready_ = false;
    }

    std::array<double, 2> gait_phase_raw{};
    std::array<double, kPolicyDof> joint_pos_rel_raw{};
    std::array<double, kPolicyDof> joint_vel_rel_raw{};

    const double elapsed_s =
        static_cast<double>(policy_step_count_) * config_.policy_step_dt_s;
    double global_phase =
        std::fmod(elapsed_s, config_.gait_phase_period_s) /
        config_.gait_phase_period_s;
    if (global_phase < 0.0) {
        global_phase += 1.0;
    }
    gait_phase_raw[0] = std::sin(global_phase * kTwoPi);
    gait_phase_raw[1] = std::cos(global_phase * kTwoPi);

    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const int motor_index =
            config_.control_to_motor_index[static_cast<std::size_t>(model_index)];
        const double q_model =
            terms.joint_pos[static_cast<std::size_t>(motor_index)];
        const double dq_model =
            terms.joint_vel[static_cast<std::size_t>(motor_index)];

        joint_pos_rel_raw[static_cast<std::size_t>(model_index)] =
            q_model - config_.stand_pose_rad[static_cast<std::size_t>(model_index)];
        joint_vel_rel_raw[static_cast<std::size_t>(model_index)] = dq_model;
    }

    const std::array<float, 3> base_ang_vel =
        transformObservationTerm(terms.base_ang_vel,
                                 config_.body_ang_vel_scale,
                                 config_.base_ang_vel_clip_enabled,
                                 config_.base_ang_vel_clip,
                                 config_.observation_scale_first);
    const std::array<float, 3> projected_gravity =
        transformObservationTerm(terms.projected_gravity,
                                 config_.projected_gravity_scale,
                                 config_.projected_gravity_clip_enabled,
                                 config_.projected_gravity_clip,
                                 config_.observation_scale_first);
    const std::array<float, 3> velocity_commands =
        transformObservationTerm(terms.velocity_commands,
                                 config_.command_scale,
                                 config_.command_clip_enabled,
                                 config_.command_clip,
                                 config_.observation_scale_first);
    const double gait_gate = gaitPhaseGate(velocity_commands, config_);
    gait_phase_raw[0] *= gait_gate;
    gait_phase_raw[1] *= gait_gate;
    const std::array<float, 2> gait_phase =
        transformObservationTerm(gait_phase_raw,
                                 config_.gait_phase_scale,
                                 config_.gait_phase_clip_enabled,
                                 config_.gait_phase_clip,
                                 config_.observation_scale_first);
    const std::array<float, kPolicyDof> joint_pos_rel =
        transformObservationTerm(joint_pos_rel_raw,
                                 config_.dof_pos_scale,
                                 config_.joint_pos_rel_clip_enabled,
                                 config_.joint_pos_rel_clip,
                                 config_.observation_scale_first);
    const std::array<float, kPolicyDof> joint_vel_rel =
        transformObservationTerm(joint_vel_rel_raw,
                                 config_.dof_vel_scale,
                                 config_.joint_vel_rel_clip_enabled,
                                 config_.joint_vel_rel_clip,
                                 config_.observation_scale_first);

    if (!observation_history_ready_) {
        fillTermHistory(observation_history_, layout.base_ang_vel_offset,
                        kPolicyFrameStack, base_ang_vel);
        fillTermHistory(observation_history_, layout.projected_gravity_offset,
                        kPolicyFrameStack, projected_gravity);
        fillTermHistory(observation_history_, layout.velocity_commands_offset,
                        kPolicyFrameStack, velocity_commands);
        if (layout.include_gait_phase) {
            fillTermHistory(observation_history_, layout.gait_phase_offset,
                            kPolicyFrameStack, gait_phase);
        }
        fillTermHistory(observation_history_, layout.joint_pos_rel_offset,
                        kPolicyFrameStack, joint_pos_rel);
        fillTermHistory(observation_history_, layout.joint_vel_rel_offset,
                        kPolicyFrameStack, joint_vel_rel);
        fillTermHistory(observation_history_, layout.last_action_offset,
                        kPolicyFrameStack, terms.last_action);
        observation_history_ready_ = true;
    } else {
        appendTermHistory(observation_history_, layout.base_ang_vel_offset,
                          kPolicyFrameStack, base_ang_vel);
        appendTermHistory(observation_history_, layout.projected_gravity_offset,
                          kPolicyFrameStack, projected_gravity);
        appendTermHistory(observation_history_, layout.velocity_commands_offset,
                          kPolicyFrameStack, velocity_commands);
        if (layout.include_gait_phase) {
            appendTermHistory(observation_history_, layout.gait_phase_offset,
                              kPolicyFrameStack, gait_phase);
        }
        appendTermHistory(observation_history_, layout.joint_pos_rel_offset,
                          kPolicyFrameStack, joint_pos_rel);
        appendTermHistory(observation_history_, layout.joint_vel_rel_offset,
                          kPolicyFrameStack, joint_vel_rel);
        appendTermHistory(observation_history_, layout.last_action_offset,
                          kPolicyFrameStack, terms.last_action);
    }

    observation = observation_history_;
    return true;
}

bool P1RealDeployPolicy::inferObservation(
    const std::vector<float>& observation,
    std::array<float, kPolicyDof>& raw_action)
{
    raw_action.fill(0.0F);
    if (!runner_.infer(observation, raw_action)) {
        std::cerr << "[ERROR] TorchScript inference failed: "
                  << runner_.last_error() << "\n";
        return false;
    }
    return true;
}

bool P1RealDeployPolicy::postprocessAction(
    const std::array<float, kPolicyDof>& raw_action,
    P1ActionPostprocess& postprocess) const
{
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto i = static_cast<std::size_t>(model_index);
        const double clipped_raw_action =
            std::clamp(static_cast<double>(raw_action[i]),
                       -config_.raw_action_clip,
                       config_.raw_action_clip);
        postprocess.clipped_raw_action[i] =
            static_cast<float>(clipped_raw_action);
        postprocess.scaled_action[i] = clipped_raw_action * config_.action_scale[i];
        postprocess.clipped_scaled_action[i] =
            std::clamp(postprocess.scaled_action[i],
                       config_.action_clip[i][0],
                       config_.action_clip[i][1]);
        postprocess.target_q_model_rad[i] =
            config_.stand_pose_rad[i] + postprocess.clipped_scaled_action[i];
        if (!std::isfinite(postprocess.target_q_model_rad[i])) {
            std::cerr << "[ERROR] Non-finite policy target"
                      << " model_index=" << model_index
                      << " raw_action=" << raw_action[i]
                      << " raw_clip=" << config_.raw_action_clip
                      << " scale=" << config_.action_scale[i]
                      << " clip=(" << config_.action_clip[i][0]
                      << ", " << config_.action_clip[i][1] << ")"
                      << " stand_pose=" << config_.stand_pose_rad[i]
                      << " target=" << postprocess.target_q_model_rad[i] << "\n";
            return false;
        }
    }
    buildMotorTargetFromModel(config_,
                              postprocess.target_q_model_rad,
                              postprocess.target_motor_rad);
    return true;
}

void P1RealDeployPolicy::setLastActionForObservation(
    const std::array<float, kPolicyDof>& last_action)
{
    last_action_raw_ = last_action;
}

void P1RealDeployPolicy::advancePolicyStep(
    const std::array<float, kPolicyDof>& raw_action)
{
    ++policy_step_count_;
    last_action_raw_ = raw_action;
}

void buildMotorTargetFromModel(const PolicyConfig& config,
                               const std::array<double, kPolicyDof>& target_q_model_rad,
                               std::array<double, kPolicyDof>& target_q_motor_rad)
{
    target_q_motor_rad.fill(0.0);
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto model_idx = static_cast<std::size_t>(model_index);
        const int motor_index = config.control_to_motor_index[model_idx];
        const auto motor_idx = static_cast<std::size_t>(motor_index);
        target_q_motor_rad[motor_idx] =
            static_cast<double>(config.motor_to_model_direction[motor_idx]) *
            target_q_model_rad[model_idx];
    }
}

}  // namespace p1_sim
