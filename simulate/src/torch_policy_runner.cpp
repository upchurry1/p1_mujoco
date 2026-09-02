#include "torch_policy_runner.hpp"

#include <torch/script.h>

#include <algorithm>
#include <exception>
#include <memory>
#include <sstream>

namespace inference {
namespace {

bool tensorIsValidPolicyOutput(const torch::Tensor& tensor,
                               std::size_t expected_count,
                               std::string& error)
{
    if (tensor.scalar_type() != torch::kFloat32) {
        std::ostringstream oss;
        oss << "TorchScript policy output must be float32, got scalar_type="
            << tensor.scalar_type();
        error = oss.str();
        return false;
    }
    if (tensor.numel() != static_cast<int64_t>(expected_count)) {
        std::ostringstream oss;
        oss << "TorchScript policy output size mismatch, expected "
            << expected_count << " got " << tensor.numel();
        error = oss.str();
        return false;
    }
    return true;
}

}  // namespace

struct TorchPolicyRunner::Impl {
    std::unique_ptr<torch::jit::script::Module> module;
};

TorchPolicyRunner::TorchPolicyRunner()
    : impl_(std::make_unique<Impl>())
{
}

TorchPolicyRunner::~TorchPolicyRunner()
{
    unload();
}

bool TorchPolicyRunner::load(const std::string& model_path,
                             std::size_t input_size)
{
    unload();
    last_error_.clear();
    input_size_ = input_size;

    if (input_size_ == 0) {
        set_error("TorchScript policy input size must be > 0");
        return false;
    }

    try {
        impl_->module = std::make_unique<torch::jit::script::Module>(
            torch::jit::load(model_path, torch::kCPU));
        impl_->module->eval();
        loaded_ = true;

        if (!dry_run_and_validate_output()) {
            unload();
            return false;
        }
    } catch (const c10::Error& e) {
        set_error("failed to load TorchScript model: " + std::string(e.what()));
        unload();
        return false;
    } catch (const std::exception& e) {
        set_error("failed to load TorchScript model: " + std::string(e.what()));
        unload();
        return false;
    }

    return true;
}

bool TorchPolicyRunner::dry_run_and_validate_output()
{
    if (!loaded_ || !impl_->module) {
        set_error("TorchScript policy is not loaded");
        return false;
    }

    try {
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::zeros(
            {1, static_cast<int64_t>(input_size_)},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        torch::jit::IValue output_value = impl_->module->forward({input});
        if (!output_value.isTensor()) {
            set_error("TorchScript policy output must be a tensor");
            return false;
        }

        std::string output_error;
        if (!tensorIsValidPolicyOutput(output_value.toTensor(),
                                       kOutputSize,
                                       output_error)) {
            set_error(output_error);
            return false;
        }
    } catch (const c10::Error& e) {
        set_error("TorchScript dry-run failed for input size " +
                  std::to_string(input_size_) + ": " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        set_error("TorchScript dry-run failed for input size " +
                  std::to_string(input_size_) + ": " + std::string(e.what()));
        return false;
    }

    return true;
}

void TorchPolicyRunner::set_error(const std::string& message)
{
    last_error_ = message;
}

void TorchPolicyRunner::unload()
{
    if (impl_) {
        impl_->module.reset();
    }
    loaded_ = false;
}

bool TorchPolicyRunner::infer(const std::vector<float>& observation,
                              std::array<float, kOutputSize>& action)
{
    if (!loaded_ || !impl_->module) {
        set_error("TorchScript policy is not loaded");
        return false;
    }
    if (observation.size() != input_size_) {
        set_error("TorchScript observation size mismatch, expected " +
                  std::to_string(input_size_) + " got " +
                  std::to_string(observation.size()));
        return false;
    }

    try {
        torch::NoGradGuard no_grad;
        torch::Tensor input = torch::from_blob(
            const_cast<float*>(observation.data()),
            {1, static_cast<int64_t>(observation.size())},
            torch::TensorOptions().dtype(torch::kFloat32).device(torch::kCPU));

        torch::jit::IValue output_value = impl_->module->forward({input});
        if (!output_value.isTensor()) {
            set_error("TorchScript policy output must be a tensor");
            return false;
        }

        torch::Tensor output = output_value.toTensor();
        std::string output_error;
        if (!tensorIsValidPolicyOutput(output, action.size(), output_error)) {
            set_error(output_error);
            return false;
        }

        output = output.to(torch::kCPU).contiguous();
        const float* output_data = output.data_ptr<float>();
        std::copy(output_data, output_data + action.size(), action.begin());
    } catch (const c10::Error& e) {
        set_error("TorchScript inference failed: " + std::string(e.what()));
        return false;
    } catch (const std::exception& e) {
        set_error("TorchScript inference failed: " + std::string(e.what()));
        return false;
    }

    return true;
}

}  // namespace inference
