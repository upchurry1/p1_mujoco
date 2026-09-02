#include "p1_sim_log.h"

#include <cmath>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <ctime>

namespace p1_sim {
namespace {

std::uint64_t steadyClockNs(std::chrono::steady_clock::time_point time)
{
    return static_cast<std::uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            time.time_since_epoch()).count());
}

std::uint64_t simTimeUs(double sim_time_s)
{
    if (!std::isfinite(sim_time_s) || sim_time_s <= 0.0) {
        return 0;
    }
    return static_cast<std::uint64_t>(std::llround(sim_time_s * 1.0e6));
}

std::string localTimestampForFilename()
{
    const auto now = std::chrono::system_clock::now();
    const auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                        now.time_since_epoch()) %
                    1000;
    const std::time_t raw_time = std::chrono::system_clock::to_time_t(now);
    std::tm local_time{};
    localtime_r(&raw_time, &local_time);

    std::ostringstream out;
    out << std::put_time(&local_time, "%Y%m%d_%H%M%S")
        << '_' << std::setw(3) << std::setfill('0') << ms.count();
    return out.str();
}

std::filesystem::path resolveLogPath(const std::string& path_or_dir,
                                     const std::string& fallback_dir)
{
    std::filesystem::path path =
        path_or_dir.empty() ? std::filesystem::path(fallback_dir)
                            : std::filesystem::path(path_or_dir);
    if (path.empty()) {
        path = std::filesystem::current_path() / "log";
    }

    if (!path.has_extension()) {
        path /= "p1_mujoco_deploy_sim_" + localTimestampForFilename() + ".csv";
    }
    return path.lexically_normal();
}

void appendIndexedColumns(std::ostream& stream,
                          const char* prefix,
                          int count)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << i;
    }
}

void appendMotorColumns(std::ostream& stream,
                        const char* prefix,
                        int count)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << "_M" << i;
    }
}

void writeHeader(std::ostream& stream)
{
    stream << "frame_index,elapsed_us,state_timestamp_ns,"
           << "inference_start_ns,inference_end_ns,inference_duration_us,"
           << "command_timestamp_ns,command_applied";
    appendIndexedColumns(stream, "raw_action_", kPolicyDof);
    appendIndexedColumns(stream, "target_q_model_rad_", kPolicyDof);
    appendMotorColumns(stream, "target_pos_rad", kPolicyDof);
    appendMotorColumns(stream, "target_effort_permille", kPolicyDof);
    appendMotorColumns(stream, "rx_pos_rad", kPolicyDof);
    appendMotorColumns(stream, "rx_vel_rad_s", kPolicyDof);
    appendMotorColumns(stream, "torque_pct", kPolicyDof);
    appendMotorColumns(stream, "comm_ok", kPolicyDof);
    appendMotorColumns(stream, "enabled", kPolicyDof);
    appendMotorColumns(stream, "faulted", kPolicyDof);
    stream << '\n';
}

template <typename Values>
void writeValues(std::ostream& stream, const Values& values)
{
    for (const auto& value : values) {
        stream << ',' << value;
    }
}

void writeRepeated(std::ostream& stream, int count, int value)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << value;
    }
}

}  // namespace

bool P1SimCsvLogger::open(const std::string& path_or_dir,
                          const std::string& fallback_dir)
{
    close();
    failed_ = false;
    const std::filesystem::path log_path =
        resolveLogPath(path_or_dir, fallback_dir);

    try {
        const std::filesystem::path parent = log_path.parent_path();
        if (!parent.empty()) {
            std::filesystem::create_directories(parent);
        }
    } catch (const std::filesystem::filesystem_error& error) {
        std::cerr << "[ERROR] Failed to create sim log directory for "
                  << log_path << ": " << error.what() << "\n";
        failed_ = true;
        return false;
    }

    stream_.open(log_path, std::ios::out);
    if (!stream_) {
        std::cerr << "[ERROR] Failed to open sim log: " << log_path << "\n";
        failed_ = true;
        return false;
    }

    path_ = log_path.string();
    stream_ << std::setprecision(17);
    writeHeader(stream_);
    std::cout << "[INFO] sim tracking log: " << path_ << "\n";
    return true;
}

void P1SimCsvLogger::close()
{
    if (stream_.is_open()) {
        stream_.close();
    }
}

bool P1SimCsvLogger::isOpen() const
{
    return stream_.is_open() && !failed_;
}

const std::string& P1SimCsvLogger::path() const
{
    return path_;
}

bool P1SimCsvLogger::writePolicyFrame(
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
    const std::array<double, kPolicyDof>& torque_permille)
{
    if (!isOpen()) {
        return false;
    }

    const auto inference_duration =
        std::chrono::duration_cast<std::chrono::microseconds>(
            inference_end - inference_start).count();

    stream_ << frame_index
            << ',' << simTimeUs(sim_time_s)
            << ',' << steadyClockNs(state_timestamp)
            << ',' << steadyClockNs(inference_start)
            << ',' << steadyClockNs(inference_end)
            << ',' << inference_duration
            << ',' << steadyClockNs(command_timestamp)
            << ',' << (command_applied ? 1 : 0);
    writeValues(stream_, raw_action);
    writeValues(stream_, target_q_model_rad);
    writeValues(stream_, target_motor_rad);
    writeRepeated(stream_, kPolicyDof, 0);
    writeValues(stream_, state.q);
    writeValues(stream_, state.dq);
    writeValues(stream_, torque_permille);
    writeRepeated(stream_, kPolicyDof, 1);
    writeRepeated(stream_, kPolicyDof, 1);
    writeRepeated(stream_, kPolicyDof, 0);
    stream_ << '\n';
    stream_.flush();

    if (!stream_) {
        std::cerr << "[ERROR] Failed to write sim log: " << path_ << "\n";
        close();
        failed_ = true;
        return false;
    }
    return true;
}

}  // namespace p1_sim
