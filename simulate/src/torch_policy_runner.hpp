#pragma once

#include "p1_types.h"

#include <array>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

namespace inference {

class TorchPolicyRunner {
public:
    static constexpr std::size_t kDefaultInputSize =
        p1_sim::kPolicyObservationSizeWithGait;
    static constexpr std::size_t kOutputSize = p1_sim::kPolicyDof;

    TorchPolicyRunner();
    ~TorchPolicyRunner();

    TorchPolicyRunner(const TorchPolicyRunner&) = delete;
    TorchPolicyRunner& operator=(const TorchPolicyRunner&) = delete;

    bool load(const std::string& model_path, std::size_t input_size);
    void unload();
    bool is_loaded() const { return loaded_; }
    std::size_t input_size() const { return input_size_; }

    bool infer(const std::vector<float>& observation,
               std::array<float, kOutputSize>& action);

    const std::string& last_error() const { return last_error_; }

private:
    struct Impl;

    bool dry_run_and_validate_output();
    void set_error(const std::string& message);

    std::unique_ptr<Impl> impl_;
    std::size_t input_size_ = kDefaultInputSize;
    bool loaded_ = false;
    std::string last_error_;
};

}  // namespace inference
