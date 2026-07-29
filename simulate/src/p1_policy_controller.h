#pragma once

#include "p1_config.h"
#include "p1_types.h"

#include "torch_policy_runner.hpp"

#include <array>

namespace p1_sim {

class P1RealDeployPolicy {
public:
    explicit P1RealDeployPolicy(PolicyConfig config);

    bool load();
    void resetPolicyState();
    void buildStandMotorTarget(std::array<double, kPolicyDof>& target_motor) const;
    bool step(const P1StateSnapshot& state,
              double vx,
              double vy,
              double yaw_rate,
              std::array<double, kPolicyDof>& target_motor_rad,
              std::array<float, kPolicyDof>& raw_action);

    const std::array<double, kPolicyDof>& policyKpMotor() const;
    const std::array<double, kPolicyDof>& policyKdMotor() const;

private:
    void mapConfiguredGains();
    bool buildObservation(const P1StateSnapshot& state,
                          double vx,
                          double vy,
                          double yaw_rate,
                          std::array<float, kPolicyObservationSize>& observation);

    PolicyConfig config_;
    inference::TorchPolicyRunner runner_;
    std::array<double, kPolicyDof> policy_mit_kp_motor_{};
    std::array<double, kPolicyDof> policy_mit_kd_motor_{};
    std::array<float, kPolicyDof> last_action_raw_{};
    std::array<float, kPolicyObservationSize> observation_history_{};
    bool observation_history_ready_ = false;
};

void buildMotorTargetFromModel(const PolicyConfig& config,
                               const std::array<double, kPolicyDof>& target_q_model_rad,
                               std::array<double, kPolicyDof>& target_q_motor_rad);

}  // namespace p1_sim
