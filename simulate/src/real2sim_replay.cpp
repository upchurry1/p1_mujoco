#include "mujoco_p1_backend.h"
#include "ankle_motor_fk.hpp"
#include "p1_config.h"
#include "p1_policy_controller.h"
#include "p1_types.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

using namespace p1_sim;

enum class ReplayMode {
    kOffline,
    kMujoco,
};

enum class ObservationStage {
    kRaw,
    kPolicyInput,
};

struct ReplayArgs {
    std::string runner_config_path;
    std::string input_log_path;
    std::string output_path = "real2sim_result.csv";
    ReplayMode mode = ReplayMode::kOffline;
    ObservationStage observation_stage = ObservationStage::kRaw;
    bool use_log_timestamp = true;
    double fixed_dt = 0.02;
    bool fixed_dt_explicit = false;
    double start_time = 0.0;
    double duration = 0.0;
    bool has_start_time = false;
    bool has_duration = false;
    std::vector<std::string> sim_config_args;
};

struct CsvTable {
    std::vector<std::string> header;
    std::unordered_map<std::string, int> column;
    std::vector<std::vector<std::string>> rows;
};

struct ReplayColumns {
    int timestamp = -1;
    bool timestamp_is_us = false;
    std::array<int, 3> command{};
    std::array<int, 3> base_ang_vel{};
    std::array<int, 3> projected_gravity{};
    std::array<int, kPolicyDof> joint_pos{};
    std::array<int, kPolicyDof> joint_vel{};
    std::array<int, kPolicyDof> last_action{};
    std::array<int, kPolicyDof> logged_action{};
    std::array<int, kPolicyDof> real_q_target{};
    std::array<int, kPolicyDof> real_target_pos_rad{};
    std::array<int, 3> base_pos{};
    std::array<int, 4> base_quat{};
    std::vector<int> policy_input;
    bool have_explicit_last_action = false;
    bool have_logged_action = false;
    bool have_real_q_target = false;
    bool have_real_target_pos_rad = false;
    bool have_base_pos = false;
    bool have_base_quat = false;
};

struct ReplayFrame {
    double timestamp = 0.0;
    P1ObservationTerms terms;
    std::vector<float> policy_input;
    std::array<float, kPolicyDof> logged_action{};
    std::array<double, kPolicyDof> real_q_target{};
    std::array<double, kPolicyDof> real_target_pos_rad{};
    std::array<double, 3> base_pos{};
    std::array<double, 4> base_quat{1.0, 0.0, 0.0, 0.0};
    bool have_logged_action = false;
    bool have_real_q_target = false;
    bool have_real_target_pos_rad = false;
    bool have_base_pos = false;
    bool have_base_quat = false;
};

struct ActionMetrics {
    std::uint64_t scalar_count = 0;
    double squared_error_sum = 0.0;
    double absolute_error_sum = 0.0;
    double max_abs_error = 0.0;
    double dot_sum = 0.0;
    double pred_norm_sq_sum = 0.0;
    double logged_norm_sq_sum = 0.0;

    void add(const std::array<float, kPolicyDof>& predicted,
             const std::array<float, kPolicyDof>& logged)
    {
        for (int i = 0; i < kPolicyDof; ++i) {
            const double p = predicted[static_cast<std::size_t>(i)];
            const double l = logged[static_cast<std::size_t>(i)];
            const double e = p - l;
            squared_error_sum += e * e;
            absolute_error_sum += std::abs(e);
            max_abs_error = std::max(max_abs_error, std::abs(e));
            dot_sum += p * l;
            pred_norm_sq_sum += p * p;
            logged_norm_sq_sum += l * l;
            ++scalar_count;
        }
    }

    double mse() const
    {
        return scalar_count > 0 ? squared_error_sum / scalar_count
                                : std::numeric_limits<double>::quiet_NaN();
    }

    double mae() const
    {
        return scalar_count > 0 ? absolute_error_sum / scalar_count
                                : std::numeric_limits<double>::quiet_NaN();
    }

    double cosineSimilarity() const
    {
        const double denom = std::sqrt(pred_norm_sq_sum * logged_norm_sq_sum);
        return denom > 1e-12 ? dot_sum / denom
                             : std::numeric_limits<double>::quiet_NaN();
    }
};

std::string trim(const std::string& text)
{
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::vector<std::string> splitCsvLine(const std::string& line)
{
    std::vector<std::string> out;
    std::string cell;
    bool in_quote = false;
    for (char ch : line) {
        if (ch == '"') {
            in_quote = !in_quote;
        } else if (ch == ',' && !in_quote) {
            out.push_back(trim(cell));
            cell.clear();
        } else {
            cell.push_back(ch);
        }
    }
    out.push_back(trim(cell));
    return out;
}

double parseDoubleCell(const std::string& text, const std::string& source)
{
    const std::string value = trim(text);
    if (value.empty()) {
        throw std::runtime_error("empty numeric field: " + source);
    }
    try {
        std::size_t parsed = 0;
        const double out = std::stod(value, &parsed);
        if (parsed != value.size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (!std::isfinite(out)) {
            throw std::runtime_error("non-finite numeric field: " + source);
        }
        return out;
    } catch (const std::exception& error) {
        throw std::runtime_error("invalid numeric field " + source + "='" +
                                 value + "': " + error.what());
    }
}

double optionalDoubleCell(const std::string& text)
{
    const std::string value = trim(text);
    if (value.empty()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::size_t parsed = 0;
    const double out = std::stod(value, &parsed);
    if (parsed != value.size()) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    return out;
}

std::string requireValue(int& i, int argc, char** argv, const std::string& flag)
{
    if (i + 1 >= argc) {
        throw std::runtime_error(flag + " requires a value");
    }
    return argv[++i];
}

double parseFlagDouble(const std::string& text, const std::string& flag)
{
    try {
        std::size_t parsed = 0;
        const double value = std::stod(text, &parsed);
        if (parsed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        if (!std::isfinite(value)) {
            throw std::invalid_argument("not finite");
        }
        return value;
    } catch (const std::exception& error) {
        throw std::runtime_error("invalid value for " + flag + ": " + text +
                                 " (" + error.what() + ")");
    }
}

const char* modeName(ReplayMode mode)
{
    return mode == ReplayMode::kOffline ? "offline" : "mujoco";
}

const char* stageName(ObservationStage stage)
{
    return stage == ObservationStage::kRaw ? "raw" : "policy_input";
}

void printUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " --log real_walk.csv [options]\n\n"
        << "Options:\n"
        << "  --config PATH              Runner config, default uses simulate/p1_mujoco_deploy.yaml\n"
        << "  --model PATH               Override MuJoCo XML path\n"
        << "  --deploy-config PATH       Override deploy config path\n"
        << "  --policy PATH              Override TorchScript policy path\n"
        << "  --policy-hz VALUE          Override policy frequency\n"
        << "  --log PATH                 Input real robot CSV log\n"
        << "  --output PATH              Result CSV, default real2sim_result.csv\n"
        << "  --mode offline|mujoco      Replay mode, default offline\n"
        << "  --log-observation-stage raw|policy_input\n"
        << "  --use-log-timestamp        Step MuJoCo using log dt, default\n"
        << "  --fixed-dt [VALUE]         Step MuJoCo with fixed dt; VALUE defaults to policy dt\n"
        << "  --start-time VALUE         First timestamp to write\n"
        << "  --duration VALUE           Duration to write from start-time\n";
}

ReplayArgs parseReplayArgs(int argc, char** argv)
{
    ReplayArgs args;
    args.sim_config_args.push_back(argv[0]);
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            printUsage(argv[0]);
            std::exit(0);
        } else if (arg == "--config" || arg == "--runner-config") {
            args.runner_config_path = requireValue(i, argc, argv, arg);
            args.sim_config_args.push_back("--runner-config");
            args.sim_config_args.push_back(args.runner_config_path);
        } else if (arg == "--model" || arg == "--deploy-config" ||
                   arg == "--policy" || arg == "--policy-hz") {
            const std::string value = requireValue(i, argc, argv, arg);
            args.sim_config_args.push_back(arg);
            args.sim_config_args.push_back(value);
        } else if (arg == "--log" || arg == "--input-log") {
            args.input_log_path = requireValue(i, argc, argv, arg);
        } else if (arg == "--output") {
            args.output_path = requireValue(i, argc, argv, arg);
        } else if (arg == "--mode") {
            const std::string value = requireValue(i, argc, argv, arg);
            if (value == "offline") {
                args.mode = ReplayMode::kOffline;
            } else if (value == "mujoco" || value == "real_observation_mujoco") {
                args.mode = ReplayMode::kMujoco;
            } else {
                throw std::runtime_error("--mode must be offline or mujoco");
            }
        } else if (arg == "--log-observation-stage" ||
                   arg == "--observation-stage") {
            const std::string value = requireValue(i, argc, argv, arg);
            if (value == "raw") {
                args.observation_stage = ObservationStage::kRaw;
            } else if (value == "policy_input") {
                args.observation_stage = ObservationStage::kPolicyInput;
            } else {
                throw std::runtime_error(
                    "--log-observation-stage must be raw or policy_input");
            }
        } else if (arg == "--use-log-timestamp") {
            args.use_log_timestamp = true;
        } else if (arg == "--fixed-dt") {
            args.use_log_timestamp = false;
            args.fixed_dt_explicit = true;
            if (i + 1 < argc && argv[i + 1][0] != '-') {
                args.fixed_dt = parseFlagDouble(argv[++i], arg);
            }
        } else if (arg == "--start-time") {
            args.start_time = parseFlagDouble(requireValue(i, argc, argv, arg), arg);
            args.has_start_time = true;
        } else if (arg == "--duration") {
            args.duration = parseFlagDouble(requireValue(i, argc, argv, arg), arg);
            args.has_duration = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (args.input_log_path.empty()) {
        throw std::runtime_error("--log is required");
    }
    if (args.has_duration && args.duration < 0.0) {
        throw std::runtime_error("--duration must be >= 0");
    }
    if (args.has_start_time && args.start_time < 0.0) {
        throw std::runtime_error("--start-time must be >= 0");
    }
    if (args.fixed_dt <= 0.0) {
        throw std::runtime_error("--fixed-dt must be > 0");
    }
    return args;
}

CsvTable readCsv(const std::string& path)
{
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("failed to open log CSV: " + path);
    }
    CsvTable table;
    std::string line;
    if (!std::getline(stream, line)) {
        throw std::runtime_error("log CSV is empty: " + path);
    }
    table.header = splitCsvLine(line);
    for (std::size_t i = 0; i < table.header.size(); ++i) {
        table.column[table.header[i]] = static_cast<int>(i);
    }
    while (std::getline(stream, line)) {
        if (trim(line).empty()) {
            continue;
        }
        std::vector<std::string> row = splitCsvLine(line);
        if (row.size() < table.header.size()) {
            std::cerr << "[WARN] Skipping incomplete CSV row with "
                      << row.size() << " columns; expected "
                      << table.header.size() << ". This usually means the "
                      << "log was stopped while the last row was being written.\n";
            continue;
        }
        row.resize(table.header.size());
        table.rows.push_back(std::move(row));
    }
    if (table.rows.empty()) {
        throw std::runtime_error("log CSV has no data rows: " + path);
    }
    return table;
}

int findColumn(const CsvTable& table, const std::vector<std::string>& names)
{
    for (const std::string& name : names) {
        const auto it = table.column.find(name);
        if (it != table.column.end()) {
            return it->second;
        }
    }
    return -1;
}

int findIndexedColumn(const CsvTable& table,
                      const std::vector<std::string>& prefixes,
                      int index,
                      const std::vector<std::string>& suffixes = {})
{
    std::vector<std::string> names;
    for (const std::string& prefix : prefixes) {
        names.push_back(prefix + "_" + std::to_string(index));
        names.push_back(prefix + std::to_string(index));
        names.push_back(prefix + "_M" + std::to_string(index));
        names.push_back(prefix + "_m" + std::to_string(index));
        if (index < static_cast<int>(suffixes.size())) {
            names.push_back(prefix + "_" + suffixes[static_cast<std::size_t>(index)]);
        }
    }
    return findColumn(table, names);
}

template <std::size_t N>
bool findIndexedVector(const CsvTable& table,
                       const std::vector<std::string>& prefixes,
                       std::array<int, N>& columns,
                       const std::vector<std::string>& suffixes = {})
{
    bool ok = true;
    for (std::size_t i = 0; i < N; ++i) {
        columns[i] = findIndexedColumn(table,
                                       prefixes,
                                       static_cast<int>(i),
                                       suffixes);
        ok = ok && columns[i] >= 0;
    }
    return ok;
}

template <std::size_t N>
bool allColumnsPresent(const std::array<int, N>& columns)
{
    return std::all_of(columns.begin(), columns.end(), [](int col) {
        return col >= 0;
    });
}

template <std::size_t N, typename Value>
void readArray(const CsvTable& table,
               const std::vector<std::string>& row,
               const std::array<int, N>& columns,
               const std::string& label,
               std::array<Value, N>& out)
{
    for (std::size_t i = 0; i < N; ++i) {
        out[i] = static_cast<Value>(
            parseDoubleCell(row[static_cast<std::size_t>(columns[i])],
                            label + "_" + std::to_string(i)));
    }
    (void)table;
}

double readTimestamp(const std::vector<std::string>& row,
                     const ReplayColumns& columns)
{
    double timestamp =
        parseDoubleCell(row[static_cast<std::size_t>(columns.timestamp)],
                        "timestamp");
    if (columns.timestamp_is_us) {
        timestamp *= 1.0e-6;
    }
    return timestamp;
}

template <std::size_t N>
std::array<double, N> nanArray()
{
    std::array<double, N> out{};
    out.fill(std::numeric_limits<double>::quiet_NaN());
    return out;
}

ReplayColumns discoverColumns(const CsvTable& table,
                              ObservationStage stage,
                              std::size_t policy_observation_size)
{
    ReplayColumns columns;
    columns.command.fill(-1);
    columns.base_ang_vel.fill(-1);
    columns.projected_gravity.fill(-1);
    columns.joint_pos.fill(-1);
    columns.joint_vel.fill(-1);
    columns.last_action.fill(-1);
    columns.logged_action.fill(-1);
    columns.real_q_target.fill(-1);
    columns.real_target_pos_rad.fill(-1);
    columns.base_pos.fill(-1);
    columns.base_quat.fill(-1);

    columns.timestamp = findColumn(
        table, {"timestamp", "time", "time_s", "elapsed_s", "sim_time"});
    if (columns.timestamp < 0) {
        columns.timestamp = findColumn(table, {"elapsed_us"});
        columns.timestamp_is_us = columns.timestamp >= 0;
    }
    if (columns.timestamp < 0) {
        throw std::runtime_error(
            "missing timestamp column; expected timestamp/time_s/elapsed_s/elapsed_us");
    }

    if (stage == ObservationStage::kPolicyInput) {
        columns.policy_input.resize(policy_observation_size, -1);
        for (std::size_t i = 0; i < policy_observation_size; ++i) {
            columns.policy_input[i] =
                findIndexedColumn(table,
                                  {"policy_obs", "policy_input",
                                   "observation", "obs"},
                                  static_cast<int>(i));
            if (columns.policy_input[i] < 0) {
                throw std::runtime_error(
                    "policy_input stage requires flattened policy observation "
                    "columns policy_obs_0..policy_obs_" +
                    std::to_string(policy_observation_size - 1));
            }
        }
    } else {
        const std::vector<std::string> xyz{"x", "y", "z"};
        findIndexedVector(table, {"command", "real_command", "cmd"}, columns.command, xyz);
        if (!allColumnsPresent(columns.command)) {
            columns.command[0] = findColumn(table, {"vx", "lin_vel_x", "command_vx"});
            columns.command[1] = findColumn(table, {"vy", "lin_vel_y", "command_vy"});
            columns.command[2] = findColumn(table, {"yaw_rate", "ang_vel_z", "command_yaw_rate"});
        }
        findIndexedVector(table,
                          {"base_ang_vel", "real_base_ang_vel", "gyro", "imu_gyro"},
                          columns.base_ang_vel,
                          xyz);
        findIndexedVector(table,
                          {"projected_gravity", "real_projected_gravity", "proj_g", "gravity"},
                          columns.projected_gravity,
                          xyz);
        findIndexedVector(table,
                          {"joint_pos", "real_joint_pos", "q", "rx_pos_rad"},
                          columns.joint_pos);
        findIndexedVector(table,
                          {"joint_vel", "real_joint_vel", "dq", "rx_vel_rad_s"},
                          columns.joint_vel);
        columns.have_explicit_last_action =
            findIndexedVector(table,
                              {"last_action", "logged_last_action", "real_last_action"},
                              columns.last_action);

        if (!allColumnsPresent(columns.command)) {
            throw std::runtime_error("missing command columns, expected command_0..2 or vx/vy/yaw_rate");
        }
        if (!allColumnsPresent(columns.base_ang_vel)) {
            throw std::runtime_error("missing base_ang_vel columns, expected base_ang_vel_0..2 or gyro_0..2");
        }
        if (!allColumnsPresent(columns.projected_gravity)) {
            throw std::runtime_error("missing projected_gravity columns, expected projected_gravity_0..2");
        }
        if (!allColumnsPresent(columns.joint_pos)) {
            throw std::runtime_error("missing joint_pos columns, expected joint_pos_0..11 or rx_pos_rad_M0..M11");
        }
        if (!allColumnsPresent(columns.joint_vel)) {
            throw std::runtime_error("missing joint_vel columns, expected joint_vel_0..11 or rx_vel_rad_s_M0..M11");
        }
    }

    columns.have_logged_action =
        findIndexedVector(table,
                          {"logged_raw_action", "raw_action", "action"},
                          columns.logged_action);
    columns.have_real_q_target =
        findIndexedVector(table,
                          {"real_q_target", "target_q_model_rad", "q_target"},
                          columns.real_q_target);
    columns.have_real_target_pos_rad =
        findIndexedVector(table,
                          {"real_target_pos_rad", "target_pos_rad",
                           "target_motor_pos_rad", "motor_target_pos_rad"},
                          columns.real_target_pos_rad);
    columns.have_base_pos =
        findIndexedVector(table, {"base_pos", "root_pos"}, columns.base_pos, {"x", "y", "z"});
    columns.have_base_quat =
        findIndexedVector(table,
                          {"base_quat", "root_quat", "quat", "quat_wxyz"},
                          columns.base_quat,
                          {"w", "x", "y", "z"});
    return columns;
}

std::vector<ReplayFrame> buildFrames(const CsvTable& table,
                                     const ReplayColumns& columns,
                                     ObservationStage stage,
                                     std::size_t policy_observation_size)
{
    std::vector<ReplayFrame> frames;
    frames.reserve(table.rows.size());
    std::array<float, kPolicyDof> derived_last_action{};
    derived_last_action.fill(0.0F);

    for (std::size_t r = 0; r < table.rows.size(); ++r) {
        const auto& row = table.rows[r];
        ReplayFrame frame;
        frame.timestamp = readTimestamp(row, columns);
        if (stage == ObservationStage::kPolicyInput) {
            frame.terms.base_ang_vel = nanArray<3>();
            frame.terms.projected_gravity = nanArray<3>();
            frame.terms.velocity_commands = nanArray<3>();
            frame.terms.joint_pos = nanArray<kPolicyDof>();
            frame.terms.joint_vel = nanArray<kPolicyDof>();
            frame.terms.last_action.fill(std::numeric_limits<float>::quiet_NaN());
            frame.policy_input.resize(policy_observation_size);
            for (std::size_t i = 0; i < policy_observation_size; ++i) {
                frame.policy_input[i] = static_cast<float>(
                    parseDoubleCell(row[static_cast<std::size_t>(columns.policy_input[i])],
                                    "policy_obs_" + std::to_string(i)));
            }
        } else {
            readArray(table, row, columns.command, "command", frame.terms.velocity_commands);
            readArray(table, row, columns.base_ang_vel, "base_ang_vel", frame.terms.base_ang_vel);
            readArray(table, row, columns.projected_gravity, "projected_gravity", frame.terms.projected_gravity);
            readArray(table, row, columns.joint_pos, "joint_pos", frame.terms.joint_pos);
            readArray(table, row, columns.joint_vel, "joint_vel", frame.terms.joint_vel);
            if (columns.have_explicit_last_action) {
                readArray(table, row, columns.last_action, "last_action", frame.terms.last_action);
            } else {
                frame.terms.last_action = derived_last_action;
            }
        }

        if (columns.have_logged_action) {
            readArray(table, row, columns.logged_action, "logged_raw_action", frame.logged_action);
            frame.have_logged_action = true;
            derived_last_action = frame.logged_action;
        }
        if (columns.have_real_q_target) {
            readArray(table, row, columns.real_q_target, "real_q_target", frame.real_q_target);
            frame.have_real_q_target = true;
        }
        if (columns.have_real_target_pos_rad) {
            readArray(table,
                      row,
                      columns.real_target_pos_rad,
                      "real_target_pos_rad",
                      frame.real_target_pos_rad);
            frame.have_real_target_pos_rad = true;
        }
        if (columns.have_base_pos) {
            readArray(table, row, columns.base_pos, "base_pos", frame.base_pos);
            frame.have_base_pos = true;
        }
        if (columns.have_base_quat) {
            readArray(table, row, columns.base_quat, "base_quat", frame.base_quat);
            frame.have_base_quat = true;
        }
        frames.push_back(std::move(frame));
    }
    return frames;
}

void validateTimestamps(const std::vector<ReplayFrame>& frames)
{
    for (std::size_t i = 1; i < frames.size(); ++i) {
        if (!(frames[i].timestamp > frames[i - 1].timestamp)) {
            throw std::runtime_error(
                "timestamps must be strictly increasing; row " +
                std::to_string(i + 1) + " has timestamp " +
                std::to_string(frames[i].timestamp) + " after " +
                std::to_string(frames[i - 1].timestamp));
        }
    }
}

std::array<double, 4> dtStats(const std::vector<ReplayFrame>& frames)
{
    if (frames.size() < 2) {
        return {0.0, 0.0, 0.0, 0.0};
    }
    std::vector<double> dts;
    dts.reserve(frames.size() - 1);
    for (std::size_t i = 1; i < frames.size(); ++i) {
        dts.push_back(frames[i].timestamp - frames[i - 1].timestamp);
    }
    const double mean =
        std::accumulate(dts.begin(), dts.end(), 0.0) / dts.size();
    double variance = 0.0;
    for (double dt : dts) {
        const double diff = dt - mean;
        variance += diff * diff;
    }
    variance /= dts.size();
    const auto minmax = std::minmax_element(dts.begin(), dts.end());
    return {mean, std::sqrt(variance), *minmax.first, *minmax.second};
}

std::size_t firstOutputIndex(const std::vector<ReplayFrame>& frames,
                             const ReplayArgs& args)
{
    if (!args.has_start_time) {
        return 0;
    }
    const auto it = std::lower_bound(
        frames.begin(),
        frames.end(),
        args.start_time,
        [](const ReplayFrame& frame, double time) {
            return frame.timestamp < time;
        });
    return static_cast<std::size_t>(it - frames.begin());
}

std::size_t endOutputIndex(const std::vector<ReplayFrame>& frames,
                           const ReplayArgs& args,
                           std::size_t begin)
{
    if (!args.has_duration) {
        return frames.size();
    }
    const double end_time = (args.has_start_time ? args.start_time
                                                : frames[begin].timestamp) +
                            args.duration;
    const auto it = std::lower_bound(
        frames.begin(),
        frames.end(),
        end_time,
        [](const ReplayFrame& frame, double time) {
            return frame.timestamp < time;
        });
    return static_cast<std::size_t>(it - frames.begin());
}

void appendArrayHeader(std::ostream& stream, const std::string& prefix, int count)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << i;
    }
}

void appendMotorHeader(std::ostream& stream, const std::string& prefix, int count)
{
    for (int i = 0; i < count; ++i) {
        stream << ',' << prefix << "_M" << i;
    }
}

bool isAnkleModelDof(const PolicyConfig& config, int model_index)
{
    return model_index == config.left_ankle_parallel.model_pitch_dof ||
           model_index == config.left_ankle_parallel.model_roll_dof ||
           model_index == config.right_ankle_parallel.model_pitch_dof ||
           model_index == config.right_ankle_parallel.model_roll_dof;
}

bool motorIndexValid(int index)
{
    return index >= 0 && index < kPolicyDof;
}

void fillDirectModelTargetsFromMotor(
    const PolicyConfig& config,
    const std::array<double, kPolicyDof>& motor_target_rad,
    std::array<double, kPolicyDof>& model_target_rad)
{
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        if (isAnkleModelDof(config, model_index)) {
            continue;
        }
        const int motor_index =
            config.control_to_motor_index[static_cast<std::size_t>(model_index)];
        if (!motorIndexValid(motor_index)) {
            continue;
        }
        const auto motor_idx = static_cast<std::size_t>(motor_index);
        const double motor_value = motor_target_rad[motor_idx];
        if (std::isfinite(motor_value)) {
            model_target_rad[static_cast<std::size_t>(model_index)] =
                static_cast<double>(config.motor_to_model_direction[motor_idx]) *
                motor_value;
        }
    }
}

void fillAnkleModelTargetsFromMotor(
    const PolicyConfig& config,
    const AnkleParallelMap& ankle_map,
    const std::array<double, kPolicyDof>& motor_target_rad,
    ankle_motor_fk::Solver& solver,
    std::array<double, kPolicyDof>& model_target_rad)
{
    if (!motorIndexValid(ankle_map.upper_motor_index) ||
        !motorIndexValid(ankle_map.lower_motor_index) ||
        !motorIndexValid(ankle_map.model_pitch_dof) ||
        !motorIndexValid(ankle_map.model_roll_dof)) {
        return;
    }

    const auto upper_idx = static_cast<std::size_t>(ankle_map.upper_motor_index);
    const auto lower_idx = static_cast<std::size_t>(ankle_map.lower_motor_index);
    const double upper_motor =
        static_cast<double>(config.motor_to_model_direction[upper_idx]) *
        motor_target_rad[upper_idx];
    const double lower_motor =
        static_cast<double>(config.motor_to_model_direction[lower_idx]) *
        motor_target_rad[lower_idx];
    const ankle_motor_fk::FootAngles foot =
        solver.solve(upper_motor, lower_motor);
    if (!foot.reachable) {
        return;
    }

    model_target_rad[static_cast<std::size_t>(ankle_map.model_pitch_dof)] =
        foot.pitch;
    model_target_rad[static_cast<std::size_t>(ankle_map.model_roll_dof)] =
        foot.roll;
}

std::array<double, kPolicyDof> modelTargetFromMotorTarget(
    const PolicyConfig& config,
    const std::array<double, kPolicyDof>& motor_target_rad,
    ankle_motor_fk::Solver& left_ankle_solver,
    ankle_motor_fk::Solver& right_ankle_solver)
{
    std::array<double, kPolicyDof> model_target_rad = nanArray<kPolicyDof>();
    fillDirectModelTargetsFromMotor(config, motor_target_rad, model_target_rad);
    fillAnkleModelTargetsFromMotor(config,
                                   config.left_ankle_parallel,
                                   motor_target_rad,
                                   left_ankle_solver,
                                   model_target_rad);
    fillAnkleModelTargetsFromMotor(config,
                                   config.right_ankle_parallel,
                                   motor_target_rad,
                                   right_ankle_solver,
                                   model_target_rad);
    return model_target_rad;
}

template <typename Values>
void appendValues(std::ostream& stream, const Values& values)
{
    for (const auto& value : values) {
        stream << ',' << value;
    }
}

void writeResultHeader(std::ostream& stream)
{
    stream << "frame_index,timestamp";
    appendArrayHeader(stream, "real_base_ang_vel_", 3);
    appendArrayHeader(stream, "real_projected_gravity_", 3);
    appendArrayHeader(stream, "real_command_", 3);
    appendArrayHeader(stream, "predicted_raw_action_", kPolicyDof);
    appendArrayHeader(stream, "logged_raw_action_", kPolicyDof);
    appendArrayHeader(stream, "action_error_", kPolicyDof);
    appendArrayHeader(stream, "scaled_action_", kPolicyDof);
    appendArrayHeader(stream, "q_target_model_rad_", kPolicyDof);
    appendMotorHeader(stream, "q_target_motor_rad", kPolicyDof);
    appendMotorHeader(stream, "mujoco_q_rad", kPolicyDof);
    appendMotorHeader(stream, "mujoco_dq_rad_s", kPolicyDof);
    appendArrayHeader(stream, "real_q_target_", kPolicyDof);
    appendMotorHeader(stream, "real_target_pos_rad", kPolicyDof);
    appendArrayHeader(stream, "real_target_q_model_from_motor_rad_", kPolicyDof);
    stream << '\n';
}

std::string defaultSummaryPath(const std::string& output_path)
{
    std::filesystem::path path(output_path);
    if (path.has_extension()) {
        path.replace_extension(".summary.txt");
    } else {
        path += ".summary.txt";
    }
    return path.string();
}

void printFirstFrameDebug(const ReplayFrame& frame,
                          const std::vector<float>& observation)
{
    auto print3 = [](const char* label, const std::array<double, 3>& values) {
        std::cout << "[INFO] first " << label << "=("
                  << values[0] << ", " << values[1] << ", "
                  << values[2] << ")\n";
    };
    print3("base_ang_vel", frame.terms.base_ang_vel);
    print3("projected_gravity", frame.terms.projected_gravity);
    print3("command", frame.terms.velocity_commands);
    std::cout << "[INFO] first joint_pos[0:4]=("
              << frame.terms.joint_pos[0] << ", "
              << frame.terms.joint_pos[1] << ", "
              << frame.terms.joint_pos[2] << ", "
              << frame.terms.joint_pos[3] << ")\n";
    std::cout << "[INFO] first joint_vel[0:4]=("
              << frame.terms.joint_vel[0] << ", "
              << frame.terms.joint_vel[1] << ", "
              << frame.terms.joint_vel[2] << ", "
              << frame.terms.joint_vel[3] << ")\n";
    std::cout << "[INFO] first last_action[0:4]=("
              << frame.terms.last_action[0] << ", "
              << frame.terms.last_action[1] << ", "
              << frame.terms.last_action[2] << ", "
              << frame.terms.last_action[3] << ")\n";
    std::cout << "[INFO] first policy_tensor[0:20]=";
    const std::size_t count = std::min<std::size_t>(20, observation.size());
    for (std::size_t i = 0; i < count; ++i) {
        if (i != 0) {
            std::cout << ',';
        }
        std::cout << observation[i];
    }
    std::cout << "\n";
}

std::vector<char*> mutableArgv(std::vector<std::string>& args)
{
    std::vector<char*> out;
    out.reserve(args.size());
    for (std::string& arg : args) {
        out.push_back(arg.data());
    }
    return out;
}

void ensureParentDirectory(const std::string& path_text)
{
    const std::filesystem::path path(path_text);
    const std::filesystem::path parent = path.parent_path();
    if (!parent.empty()) {
        std::filesystem::create_directories(parent);
    }
}

void initializeMujocoState(MujocoP1Backend& backend,
                           const RunnerOptions& options,
                           const ReplayFrame& frame)
{
    std::array<double, kPolicyDof> initial_q = frame.terms.joint_pos;
    bool initialized_joints_from_log = true;
    for (double value : initial_q) {
        initialized_joints_from_log =
            initialized_joints_from_log && std::isfinite(value);
    }
    if (initialized_joints_from_log) {
        backend.setJointPositionsFromMotorTarget(initial_q);
    }

    const std::array<double, 3> base_pos =
        frame.have_base_pos ? frame.base_pos : options.initial_base_pos;
    const std::array<double, 4> base_quat =
        frame.have_base_quat ? frame.base_quat : options.initial_base_quat;
    backend.setRootPose(base_pos, base_quat);

    std::cout << "[INFO] MuJoCo init: joints="
              << (initialized_joints_from_log ? "log_first_frame" : "default")
              << " base_pose="
              << (frame.have_base_pos || frame.have_base_quat
                      ? "partial_log_pose"
                      : "default_config_pose")
              << ". Joint velocities and unobserved full state are not reconstructed.\n";
}

void stepMujocoForFrame(mjModel* model,
                        mjData* data,
                        MujocoP1Backend& backend,
                        double dt)
{
    double remaining = std::max(0.0, dt);
    const double sim_dt = model->opt.timestep;
    while (remaining > 1e-12) {
        if (!backend.applyMitTorques()) {
            throw std::runtime_error("MIT torque application failed during replay");
        }
        mj_step(model, data);
        remaining -= sim_dt;
    }
}

int run(int argc, char** argv)
{
    ReplayArgs args = parseReplayArgs(argc, argv);
    RunnerOptions options;
    PolicyConfig config;
    std::vector<std::string> sim_args = args.sim_config_args;
    std::vector<char*> sim_argv = mutableArgv(sim_args);
    if (!parseArgs(static_cast<int>(sim_argv.size()), sim_argv.data(), options, config)) {
        return 2;
    }
    if (!args.fixed_dt_explicit) {
        args.fixed_dt = config.policy_step_dt_s;
    }

    P1RealDeployPolicy policy(config);
    if (!policy.load()) {
        return 1;
    }

    const CsvTable table = readCsv(args.input_log_path);
    const ReplayColumns columns =
        discoverColumns(table, args.observation_stage, policy.policyObservationSize());
    std::vector<ReplayFrame> frames =
        buildFrames(table, columns, args.observation_stage, policy.policyObservationSize());
    validateTimestamps(frames);
    const auto stats = dtStats(frames);

    const std::size_t output_begin = firstOutputIndex(frames, args);
    const std::size_t output_end = endOutputIndex(frames, args, output_begin);
    if (output_begin >= output_end || output_begin >= frames.size()) {
        throw std::runtime_error("selected replay window contains no frames");
    }

    mjModel* model = nullptr;
    mjData* data = nullptr;
    std::unique_ptr<MujocoP1Backend> backend;
    if (args.mode == ReplayMode::kMujoco) {
        char load_error[1024] = "Could not load XML";
        model = mj_loadXML(options.model_xml_path.c_str(),
                           nullptr,
                           load_error,
                           sizeof(load_error));
        if (!model) {
            throw std::runtime_error(std::string("MuJoCo XML load failed: ") +
                                     load_error);
        }
        data = mj_makeData(model);
        if (!data) {
            mj_deleteModel(model);
            throw std::runtime_error("mj_makeData failed");
        }
        backend = std::make_unique<MujocoP1Backend>(model, data);
        backend->setImuNoiseEnabled(false);
        if (!backend->initialize()) {
            mj_deleteData(data);
            mj_deleteModel(model);
            return 1;
        }
        backend->setJointZeroOffset(options.joint_zero_offset_rad);
        backend->setMujocoJointDirection(options.mujoco_joint_direction);
        backend->configureMotorDelay(options.motor_delay_enabled,
                                     options.motor_delay_min_seconds,
                                     options.motor_delay_max_seconds);
        backend->setMotorDelayActive(true);
        initializeMujocoState(*backend, options, frames[output_begin]);
    }

    ensureParentDirectory(args.output_path);
    std::ofstream result(args.output_path);
    if (!result) {
        throw std::runtime_error("failed to open result CSV: " + args.output_path);
    }
    result << std::setprecision(17);
    writeResultHeader(result);

    std::cout << std::setprecision(9);
    std::cout << "[INFO] Policy path: " << config.policy_model_path << "\n";
    std::cout << "[INFO] Robot model: " << options.model_xml_path << "\n";
    std::cout << "[INFO] Observation dimension: "
              << policy.policyObservationSize() << "\n";
    std::cout << "[INFO] Observation history: " << kPolicyFrameStack << "\n";
    std::cout << "[INFO] Joint count: " << kPolicyDof << "\n";
    std::cout << "[INFO] Action dimension: " << kPolicyDof << "\n";
    std::cout << "[INFO] Control dt: " << config.policy_step_dt_s << "\n";
    std::cout << "[INFO] Log file: " << args.input_log_path << "\n";
    std::cout << "[INFO] Replay mode: " << modeName(args.mode) << "\n";
    std::cout << "[INFO] Observation stage: "
              << stageName(args.observation_stage) << "\n";
    std::cout << "[INFO] Log dt mean/std/min/max: "
              << stats[0] << " / " << stats[1] << " / "
              << stats[2] << " / " << stats[3] << "\n";
    std::cout << "[INFO] Output frames: [" << output_begin << ", "
              << output_end << "), warmup_prefix_frames=" << output_begin
              << "\n";
    policy.printObservationLayout(std::cout);

    ActionMetrics metrics;
    std::uint64_t written_frames = 0;
    bool printed_first_debug = false;
    ankle_motor_fk::Solver real_left_ankle_target_fk;
    ankle_motor_fk::Solver real_right_ankle_target_fk;

    const std::size_t process_begin =
        args.observation_stage == ObservationStage::kRaw ? 0 : output_begin;
    for (std::size_t i = process_begin; i < output_end; ++i) {
        ReplayFrame& frame = frames[i];
        std::vector<float> observation;
        if (args.observation_stage == ObservationStage::kRaw) {
            if (!policy.buildObservationFromTerms(frame.terms, observation)) {
                throw std::runtime_error("failed to build observation at row " +
                                         std::to_string(i + 2));
            }
        } else {
            observation = frame.policy_input;
        }

        std::array<float, kPolicyDof> predicted_action{};
        if (!policy.inferObservation(observation, predicted_action)) {
            throw std::runtime_error("policy inference failed at row " +
                                     std::to_string(i + 2));
        }

        P1ActionPostprocess postprocess;
        if (!policy.postprocessAction(predicted_action, postprocess)) {
            throw std::runtime_error("action postprocess failed at row " +
                                     std::to_string(i + 2));
        }
        policy.advancePolicyStep(predicted_action);

        if (i < output_begin) {
            continue;
        }

        if (!printed_first_debug) {
            printFirstFrameDebug(frame, observation);
            printed_first_debug = true;
        }

        std::array<double, kPolicyDof> mujoco_q = nanArray<kPolicyDof>();
        std::array<double, kPolicyDof> mujoco_dq = nanArray<kPolicyDof>();
        if (backend) {
            backend->setMitTargets(postprocess.target_motor_rad,
                                   policy.policyKpMotor(),
                                   policy.policyKdMotor());
            double step_dt = args.fixed_dt;
            if (args.use_log_timestamp && i + 1 < frames.size()) {
                step_dt = frames[i + 1].timestamp - frames[i].timestamp;
            }
            stepMujocoForFrame(model, data, *backend, step_dt);
            P1StateSnapshot sim_state;
            backend->readState(sim_state);
            mujoco_q = sim_state.q;
            mujoco_dq = sim_state.dq;
        }

        std::array<float, kPolicyDof> action_error{};
        action_error.fill(std::numeric_limits<float>::quiet_NaN());
        if (frame.have_logged_action) {
            for (int j = 0; j < kPolicyDof; ++j) {
                const auto idx = static_cast<std::size_t>(j);
                action_error[idx] = predicted_action[idx] - frame.logged_action[idx];
            }
            metrics.add(predicted_action, frame.logged_action);
        }
        const std::array<double, kPolicyDof> real_target_q_model_from_motor_rad =
            frame.have_real_target_pos_rad
                ? modelTargetFromMotorTarget(config,
                                             frame.real_target_pos_rad,
                                             real_left_ankle_target_fk,
                                             real_right_ankle_target_fk)
                : nanArray<kPolicyDof>();

        result << written_frames << ',' << frame.timestamp;
        appendValues(result, frame.terms.base_ang_vel);
        appendValues(result, frame.terms.projected_gravity);
        appendValues(result, frame.terms.velocity_commands);
        appendValues(result, predicted_action);
        appendValues(result,
                     frame.have_logged_action
                         ? frame.logged_action
                         : filledArray<float, kPolicyDof>(
                               std::numeric_limits<float>::quiet_NaN()));
        appendValues(result, action_error);
        appendValues(result, postprocess.scaled_action);
        appendValues(result, postprocess.target_q_model_rad);
        appendValues(result, postprocess.target_motor_rad);
        appendValues(result, mujoco_q);
        appendValues(result, mujoco_dq);
        appendValues(result,
                     frame.have_real_q_target
                         ? frame.real_q_target
                         : nanArray<kPolicyDof>());
        appendValues(result,
                     frame.have_real_target_pos_rad
                         ? frame.real_target_pos_rad
                         : nanArray<kPolicyDof>());
        appendValues(result, real_target_q_model_from_motor_rad);
        result << '\n';
        ++written_frames;
    }

    const std::string summary_path = defaultSummaryPath(args.output_path);
    ensureParentDirectory(summary_path);
    std::ofstream summary(summary_path);
    if (!summary) {
        throw std::runtime_error("failed to open summary: " + summary_path);
    }
    summary << std::setprecision(17)
            << "number_of_frames=" << written_frames << "\n"
            << "action_MSE=" << metrics.mse() << "\n"
            << "action_MAE=" << metrics.mae() << "\n"
            << "action_max_error=" << metrics.max_abs_error << "\n"
            << "action_cosine_similarity=" << metrics.cosineSimilarity() << "\n"
            << "mean_control_dt=" << stats[0] << "\n"
            << "std_control_dt=" << stats[1] << "\n"
            << "min_control_dt=" << stats[2] << "\n"
            << "max_control_dt=" << stats[3] << "\n";

    std::cout << "[INFO] Wrote result CSV: " << args.output_path << "\n";
    std::cout << "[INFO] Wrote summary: " << summary_path << "\n";
    std::cout << "[INFO] action_MSE=" << metrics.mse()
              << " action_MAE=" << metrics.mae()
              << " action_max_error=" << metrics.max_abs_error
              << " action_cosine_similarity=" << metrics.cosineSimilarity()
              << "\n";

    if (data) {
        mj_deleteData(data);
    }
    if (model) {
        mj_deleteModel(model);
    }
    return 0;
}

}  // namespace

int main(int argc, char** argv)
{
    try {
        return run(argc, argv);
    } catch (const std::exception& error) {
        std::cerr << "[ERROR] " << error.what() << "\n";
        return 1;
    }
}
