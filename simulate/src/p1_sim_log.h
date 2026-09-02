#pragma once

#include "p1_types.h"

#include <array>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <string>

namespace p1_sim {

class P1SimCsvLogger {
public:
    bool open(const std::string& path_or_dir,
              const std::string& fallback_dir);
    void close();

    bool isOpen() const;
    const std::string& path() const;

    bool writePolicyFrame(
        std::uint64_t frame_index,
        double sim_time_s,
        std::chrono::steady_clock::time_point state_timestamp,
        std::chrono::steady_clock::time_point inference_start,
        std::chrono::steady_clock::time_point inference_end,
        std::chrono::steady_clock::time_point command_timestamp,
        bool command_applied,
        const std::array<float, kPolicyDof>& raw_action,
        const std::array<double, kPolicyDof>& target_q_model_rad,
        const std::array<double, kPolicyDof>& target_motor_rad,
        const P1StateSnapshot& state,
        const std::array<double, kPolicyDof>& torque_permille);

private:
    std::ofstream stream_;
    std::string path_;
    bool failed_ = false;
};

}  // namespace p1_sim
