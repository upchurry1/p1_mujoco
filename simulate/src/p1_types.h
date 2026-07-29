#pragma once

#include <array>
#include <cstddef>

namespace p1_sim {

constexpr int kPolicyDof = 12;
constexpr int kPolicySingleObservationSize = 45;
constexpr int kPolicyFrameStack = 5;
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

}  // namespace p1_sim
