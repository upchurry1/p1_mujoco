#pragma once

#include <array>
#include <cstddef>

namespace p1_sim {

constexpr int kPolicyDof = 12;
constexpr int kPolicySingleObservationSizeNoGait = 45;
constexpr int kPolicySingleObservationSizeWithGait = 47;
constexpr int kPolicySingleObservationSize = kPolicySingleObservationSizeWithGait;
constexpr int kPolicyFrameStack = 15;
constexpr int kPolicyObservationSizeNoGait =
    kPolicySingleObservationSizeNoGait * kPolicyFrameStack;
constexpr int kPolicyObservationSizeWithGait =
    kPolicySingleObservationSizeWithGait * kPolicyFrameStack;
constexpr int kPolicyObservationSize = kPolicySingleObservationSize * kPolicyFrameStack;

template <typename T, std::size_t N>
std::array<T, N> filledArray(T value)
{
    std::array<T, N> out{};
    out.fill(value);
    return out;
}

struct P1StateSnapshot {
    std::array<double, kPolicyDof> q{};
    std::array<double, kPolicyDof> dq{};
    std::array<double, kPolicyDof> tau{};
    std::array<double, 4> quat{1.0, 0.0, 0.0, 0.0};
    std::array<double, 3> gyro{};
    std::array<double, 3> accel{};
    std::array<double, 3> projected_gravity{0.0, 0.0, -1.0};
};

struct P1ObservationTerms {
    std::array<double, 3> base_ang_vel{};
    std::array<double, 3> projected_gravity{0.0, 0.0, -1.0};
    std::array<double, 3> velocity_commands{};
    std::array<double, kPolicyDof> joint_pos{};
    std::array<double, kPolicyDof> joint_vel{};
    std::array<float, kPolicyDof> last_action{};
};

struct P1ActionPostprocess {
    std::array<float, kPolicyDof> clipped_raw_action{};
    std::array<double, kPolicyDof> scaled_action{};
    std::array<double, kPolicyDof> clipped_scaled_action{};
    std::array<double, kPolicyDof> target_q_model_rad{};
    std::array<double, kPolicyDof> target_motor_rad{};
};

enum class ViewerOverlayPage {
    kSummary = 0,
    kJoints = 1,
    kImu = 2,
    kAll = 3,
};

enum class ViewerCurveSignal {
    kPosition = 0,
    kVelocity = 1,
    kTorque = 2,
    kControl = 3,
    kPositionVelocity = 4,
    kAll = 5,
};

enum class ObservationDelaySource {
    kImu = 0,
    kMotor = 1,
};

}  // namespace p1_sim
