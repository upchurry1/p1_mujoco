#include "p1_config.h"

#include <yaml-cpp/yaml.h>

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <initializer_list>
#include <iostream>
#include <sstream>
#include <stdexcept>

#include <limits.h>
#include <unistd.h>

namespace p1_sim {
namespace {

constexpr double kDegToRad = 0.017453292519943295769;

std::filesystem::path executablePath()
{
    std::array<char, PATH_MAX> path{};
    const ssize_t length = readlink("/proc/self/exe", path.data(), path.size() - 1);
    if (length <= 0) {
        return {};
    }
    path[static_cast<std::size_t>(length)] = '\0';
    return std::filesystem::path(path.data());
}

std::string defaultModelPath()
{
    const std::filesystem::path exe = executablePath();
    if (!exe.empty()) {
        const auto repo_root = exe.parent_path().parent_path().parent_path();
        const auto candidate = repo_root / "unitree_robots" / "p1_803_nobattery" / "scene.xml";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    const auto cwd_candidate =
        std::filesystem::current_path() / "unitree_robots" / "p1_803_nobattery" / "scene.xml";
    if (std::filesystem::exists(cwd_candidate)) {
        return cwd_candidate.string();
    }

    return "../unitree_robots/p1_803_nobattery/scene.xml";
}

std::string defaultRunnerConfigPath()
{
    const std::filesystem::path exe = executablePath();
    if (!exe.empty()) {
        const auto repo_root = exe.parent_path().parent_path().parent_path();
        const auto candidate = repo_root / "simulate" / "p1_mujoco_deploy.yaml";
        if (std::filesystem::exists(candidate)) {
            return candidate.string();
        }
    }

    const auto cwd_candidate =
        std::filesystem::current_path() / "simulate" / "p1_mujoco_deploy.yaml";
    if (std::filesystem::exists(cwd_candidate)) {
        return cwd_candidate.string();
    }

    const auto local_candidate =
        std::filesystem::current_path() / "p1_mujoco_deploy.yaml";
    if (std::filesystem::exists(local_candidate)) {
        return local_candidate.string();
    }

    return {};
}

std::filesystem::path expandUserPath(const std::string& text)
{
    if (text == "~" || text.rfind("~/", 0) == 0) {
        const char* home = std::getenv("HOME");
        if (home && home[0] != '\0') {
            if (text.size() == 1) {
                return std::filesystem::path(home);
            }
            return std::filesystem::path(home) / text.substr(2);
        }
    }
    return std::filesystem::path(text);
}

std::string trimAscii(const std::string& text)
{
    const std::size_t begin = text.find_first_not_of(" \t\r\n");
    if (begin == std::string::npos) {
        return {};
    }
    const std::size_t end = text.find_last_not_of(" \t\r\n");
    return text.substr(begin, end - begin + 1);
}

std::string resolveConfigPath(const std::string& text,
                              const std::filesystem::path& config_path)
{
    if (text.empty()) {
        return {};
    }

    std::filesystem::path path = expandUserPath(text);
    if (path.is_relative() && !config_path.empty()) {
        path = config_path.parent_path() / path;
    }
    return path.lexically_normal().string();
}

YAML::Node optionalAnyNode(const YAML::Node& node,
                           std::initializer_list<const char*> keys)
{
    if (!node || !node.IsMap()) {
        return {};
    }

    for (const char* key : keys) {
        for (YAML::const_iterator it = node.begin(); it != node.end(); ++it) {
            if (it->first.IsScalar() && it->first.Scalar() == key) {
                return it->second;
            }
        }
    }
    return {};
}

bool nodePresent(const YAML::Node& node)
{
    return node.IsDefined() && !node.IsNull();
}

double parseDouble(const std::string& text, const std::string& flag)
{
    const std::string trimmed = trimAscii(text);
    try {
        std::size_t parsed = 0;
        const double value = std::stod(trimmed, &parsed);
        if (parsed != trimmed.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception& error) {
        throw std::runtime_error("invalid value for " + flag + ": " + text +
                                 " (" + error.what() + ")");
    }
}

template <std::size_t N>
void parseDoubleScalarOrCommaArray(const std::string& text,
                                   const std::string& flag,
                                   std::array<double, N>& values)
{
    if (text.find(',') == std::string::npos) {
        values.fill(parseDouble(text, flag));
        return;
    }

    std::istringstream stream(text);
    std::string token;
    std::size_t index = 0;
    while (std::getline(stream, token, ',')) {
        if (index >= N) {
            throw std::runtime_error(flag + " must contain exactly " +
                                     std::to_string(N) +
                                     " comma-separated values");
        }
        const std::string value = trimAscii(token);
        if (value.empty()) {
            throw std::runtime_error(flag + " contains an empty value");
        }
        values[index++] = parseDouble(value, flag);
    }
    if (index != N) {
        throw std::runtime_error(flag + " must contain exactly " +
                                 std::to_string(N) +
                                 " comma-separated values");
    }
}

int parseInt(const std::string& text, const std::string& flag);

template <std::size_t N>
void parseIntCommaArray(const std::string& text,
                        const std::string& flag,
                        std::array<int, N>& values)
{
    std::istringstream stream(text);
    std::string token;
    std::size_t index = 0;
    while (std::getline(stream, token, ',')) {
        if (index >= N) {
            throw std::runtime_error(flag + " must contain exactly " +
                                     std::to_string(N) +
                                     " comma-separated values");
        }
        const std::string value = trimAscii(token);
        if (value.empty()) {
            throw std::runtime_error(flag + " contains an empty value");
        }
        values[index++] = parseInt(value, flag);
    }
    if (index != N) {
        throw std::runtime_error(flag + " must contain exactly " +
                                 std::to_string(N) +
                                 " comma-separated values");
    }
}

int parseInt(const std::string& text, const std::string& flag)
{
    try {
        std::size_t parsed = 0;
        const int value = std::stoi(text, &parsed);
        if (parsed != text.size()) {
            throw std::invalid_argument("trailing characters");
        }
        return value;
    } catch (const std::exception& error) {
        throw std::runtime_error("invalid value for " + flag + ": " + text +
                                 " (" + error.what() + ")");
    }
}

std::string normalizedToken(std::string text)
{
    std::transform(text.begin(), text.end(), text.begin(), [](unsigned char ch) {
        if (ch == '-') {
            return static_cast<char>('_');
        }
        return static_cast<char>(std::tolower(ch));
    });
    return text;
}

std::string scalarText(const YAML::Node& node, const std::string& source)
{
    if (!node || !node.IsScalar()) {
        throw std::runtime_error(source + " must be a scalar");
    }
    return node.Scalar();
}

bool parseBoolScalar(const YAML::Node& node, const std::string& source)
{
    const std::string value = normalizedToken(scalarText(node, source));
    if (value == "true" || value == "1" || value == "on" || value == "yes") {
        return true;
    }
    if (value == "false" || value == "0" || value == "off" || value == "no") {
        return false;
    }
    throw std::runtime_error(source + " must be true or false");
}

ViewerOverlayPage parseViewerOverlayPage(const std::string& text,
                                         const std::string& source)
{
    const std::string value = normalizedToken(text);
    if (value == "0" || value == "summary" || value == "status") {
        return ViewerOverlayPage::kSummary;
    }
    if (value == "1" || value == "joints" || value == "joint") {
        return ViewerOverlayPage::kJoints;
    }
    if (value == "2" || value == "imu") {
        return ViewerOverlayPage::kImu;
    }
    if (value == "3" || value == "all" || value == "full") {
        return ViewerOverlayPage::kAll;
    }
    throw std::runtime_error(source +
                             " must be one of summary, joints, imu, all");
}

bool parseViewerOverlayUnitsUseDegrees(const std::string& text,
                                       const std::string& source)
{
    const std::string value = normalizedToken(text);
    if (value == "rad" || value == "radian" || value == "radians") {
        return false;
    }
    if (value == "deg" || value == "degree" || value == "degrees") {
        return true;
    }
    throw std::runtime_error(source + " must be rad or deg");
}

ViewerCurveSignal parseViewerCurveSignal(const std::string& text,
                                         const std::string& source)
{
    const std::string value = normalizedToken(text);
    if (value == "0" || value == "q" || value == "pos" ||
        value == "position") {
        return ViewerCurveSignal::kPosition;
    }
    if (value == "1" || value == "dq" || value == "vel" ||
        value == "velocity") {
        return ViewerCurveSignal::kVelocity;
    }
    if (value == "2" || value == "tau" || value == "torque") {
        return ViewerCurveSignal::kTorque;
    }
    if (value == "3" || value == "ctrl" || value == "control") {
        return ViewerCurveSignal::kControl;
    }
    if (value == "4" || value == "q_dq" || value == "q+dq" ||
        value == "position_velocity" || value == "positionvelocity") {
        return ViewerCurveSignal::kPositionVelocity;
    }
    if (value == "5" || value == "all" || value == "full") {
        return ViewerCurveSignal::kAll;
    }
    throw std::runtime_error(source +
                             " must be one of q, dq, tau, ctrl, q+dq, all");
}

ObservationDelaySource parseObservationDelaySource(const std::string& text,
                                                   const std::string& source)
{
    const std::string value = normalizedToken(text);
    if (value == "imu" || value == "base" || value == "inertial") {
        return ObservationDelaySource::kImu;
    }
    if (value == "motor" || value == "motors" || value == "joint" ||
        value == "joints" || value == "encoder" || value == "encoders") {
        return ObservationDelaySource::kMotor;
    }
    throw std::runtime_error(source + " must be imu or motor");
}

const char* observationDelaySourceName(ObservationDelaySource source)
{
    switch (source) {
    case ObservationDelaySource::kImu:
        return "imu";
    case ObservationDelaySource::kMotor:
        return "motor";
    }
    return "imu";
}

bool parseGaitPhaseObservationEnabled(const std::string& text,
                                      const std::string& source)
{
    const std::string value = normalizedToken(text);
    if (value == "true" || value == "1" || value == "on" ||
        value == "yes" || value == "gait_phase" ||
        value == "with_gait_phase" || value == "with_gait" ||
        value == "47" || value == "47x15" || value == "705" ||
        value == "47x5" || value == "235") {
        return true;
    }
    if (value == "false" || value == "0" || value == "off" ||
        value == "no" || value == "no_gait_phase" ||
        value == "without_gait_phase" || value == "without_gait" ||
        value == "legacy" || value == "45" || value == "45x15" ||
        value == "675" || value == "45x5" || value == "225") {
        return false;
    }
    throw std::runtime_error(source +
                             " must be true/false, with_gait_phase, "
                             "without_gait_phase, 705, or 675");
}

const char* viewerCurveSignalName(ViewerCurveSignal signal)
{
    switch (signal) {
    case ViewerCurveSignal::kPosition:
        return "q";
    case ViewerCurveSignal::kVelocity:
        return "dq";
    case ViewerCurveSignal::kTorque:
        return "tau";
    case ViewerCurveSignal::kControl:
        return "ctrl";
    case ViewerCurveSignal::kPositionVelocity:
        return "q+dq";
    case ViewerCurveSignal::kAll:
        return "all";
    }
    return "q+dq";
}

const char* viewerOverlayPageName(ViewerOverlayPage page)
{
    switch (page) {
    case ViewerOverlayPage::kSummary:
        return "summary";
    case ViewerOverlayPage::kJoints:
        return "joints";
    case ViewerOverlayPage::kImu:
        return "imu";
    case ViewerOverlayPage::kAll:
        return "all";
    }
    return "summary";
}

YAML::Node requireNode(const YAML::Node& node,
                       const std::string& key,
                       const std::string& path);

template <std::size_t N>
void loadDoubleArray(const YAML::Node& node,
                     const std::string& path,
                     std::array<double, N>& values)
{
    if (!node || !node.IsSequence() || node.size() != N) {
        throw std::runtime_error(path + " must be a " + std::to_string(N) +
                                 "-element numeric array");
    }
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = node[i].as<double>();
    }
}

template <std::size_t N>
void loadDoubleScalarOrArray(const YAML::Node& node,
                             const std::string& path,
                             std::array<double, N>& values)
{
    if (!node) {
        return;
    }
    if (node.IsScalar()) {
        parseDoubleScalarOrCommaArray(scalarText(node, path), path, values);
        return;
    }
    loadDoubleArray(node, path, values);
}

template <std::size_t N>
void loadDegreeScalarOrArrayAsRadians(const YAML::Node& node,
                                      const std::string& path,
                                      std::array<double, N>& values)
{
    std::array<double, N> values_deg{};
    loadDoubleScalarOrArray(node, path, values_deg);
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = values_deg[i] * kDegToRad;
    }
}

void loadObservationClip(const YAML::Node& term,
                         const std::string& path,
                         bool& enabled,
                         std::array<double, 2>& range)
{
    const YAML::Node clip = optionalAnyNode(term, {"clip", "range"});
    if (nodePresent(clip)) {
        loadDoubleArray(clip, path + ".clip", range);
        enabled = true;
    } else {
        enabled = false;
    }
}

template <std::size_t N>
void loadIntArray(const YAML::Node& node,
                  const std::string& path,
                  std::array<int, N>& values)
{
    if (!node || !node.IsSequence() || node.size() != N) {
        throw std::runtime_error(path + " must be a " + std::to_string(N) +
                                 "-element integer array");
    }
    std::array<bool, N> seen{};
    for (std::size_t i = 0; i < N; ++i) {
        const int value = node[i].as<int>();
        if (value < 0 || value >= static_cast<int>(N) || seen[static_cast<std::size_t>(value)]) {
            throw std::runtime_error(path + " must be a permutation of 0.." +
                                     std::to_string(N - 1));
        }
        values[i] = value;
        seen[static_cast<std::size_t>(value)] = true;
    }
}

template <std::size_t N>
void loadPlainIntArray(const YAML::Node& node,
                       const std::string& path,
                       std::array<int, N>& values)
{
    if (!node || !node.IsSequence() || node.size() != N) {
        throw std::runtime_error(path + " must be a " + std::to_string(N) +
                                 "-element integer array");
    }
    for (std::size_t i = 0; i < N; ++i) {
        values[i] = node[i].as<int>();
    }
}

void loadCompactIntArray(const YAML::Node& node,
                         const std::string& path,
                         std::array<int, kPolicyDof>& values,
                         int& count)
{
    if (!node || !node.IsSequence() || node.size() > kPolicyDof) {
        throw std::runtime_error(path + " must be an integer array with at most " +
                                 std::to_string(kPolicyDof) + " values");
    }

    values.fill(-1);
    count = static_cast<int>(node.size());
    for (std::size_t i = 0; i < node.size(); ++i) {
        values[i] = node[i].as<int>();
    }
}

bool indexInRange(int index, int count)
{
    return index >= 0 && index < count;
}

bool isParallelAnkleModelDof(const PolicyConfig& config, int model_index)
{
    return model_index == config.left_ankle_parallel.model_pitch_dof ||
           model_index == config.left_ankle_parallel.model_roll_dof ||
           model_index == config.right_ankle_parallel.model_pitch_dof ||
           model_index == config.right_ankle_parallel.model_roll_dof;
}

bool ankleParallelMapIndicesInRange(const AnkleParallelMap& ankle_map, int count)
{
    return indexInRange(ankle_map.model_pitch_dof, count) &&
           indexInRange(ankle_map.model_roll_dof, count) &&
           indexInRange(ankle_map.upper_motor_index, count) &&
           indexInRange(ankle_map.lower_motor_index, count);
}

void loadAnkleParallelMap(const YAML::Node& node,
                          const std::string& path,
                          AnkleParallelMap& ankle_map)
{
    if (!node || !node.IsMap()) {
        throw std::runtime_error(path + " must be a map");
    }
    ankle_map.model_pitch_dof =
        requireNode(node, "model_pitch_dof", path).as<int>();
    ankle_map.model_roll_dof =
        requireNode(node, "model_roll_dof", path).as<int>();
    ankle_map.upper_motor_index =
        requireNode(node, "upper_motor_index", path).as<int>();
    ankle_map.lower_motor_index =
        requireNode(node, "lower_motor_index", path).as<int>();
}

void deriveParallelMappingFromLegacyJointMap(PolicyConfig& config)
{
    config.left_ankle_parallel = {8, 10,
                                  config.control_to_motor_index[8],
                                  config.control_to_motor_index[10]};
    config.right_ankle_parallel = {9, 11,
                                   config.control_to_motor_index[9],
                                   config.control_to_motor_index[11]};

    config.model_to_motor_index.fill(-1);
    config.model_to_motor_count = 0;
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        if (isParallelAnkleModelDof(config, model_index)) {
            continue;
        }
        config.model_to_motor_index[static_cast<std::size_t>(
            config.model_to_motor_count++)] =
            config.control_to_motor_index[static_cast<std::size_t>(model_index)];
    }
}

void synthesizeLegacyJointMapFromParallelMapping(PolicyConfig& config)
{
    config.control_to_motor_index.fill(-1);
    int direct_slot = 0;
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        if (model_index == config.left_ankle_parallel.model_pitch_dof) {
            config.control_to_motor_index[static_cast<std::size_t>(model_index)] =
                config.left_ankle_parallel.upper_motor_index;
        } else if (model_index == config.left_ankle_parallel.model_roll_dof) {
            config.control_to_motor_index[static_cast<std::size_t>(model_index)] =
                config.left_ankle_parallel.lower_motor_index;
        } else if (model_index == config.right_ankle_parallel.model_pitch_dof) {
            config.control_to_motor_index[static_cast<std::size_t>(model_index)] =
                config.right_ankle_parallel.upper_motor_index;
        } else if (model_index == config.right_ankle_parallel.model_roll_dof) {
            config.control_to_motor_index[static_cast<std::size_t>(model_index)] =
                config.right_ankle_parallel.lower_motor_index;
        } else if (direct_slot < config.model_to_motor_count) {
            config.control_to_motor_index[static_cast<std::size_t>(model_index)] =
                config.model_to_motor_index[static_cast<std::size_t>(direct_slot++)];
        }
    }
}

void loadActionClip(const YAML::Node& node,
                    const std::string& path,
                    std::array<std::array<double, 2>, kPolicyDof>& clip)
{
    if (!node || !node.IsSequence() || node.size() != kPolicyDof) {
        throw std::runtime_error(path + " must contain 12 [lower, upper] ranges");
    }
    for (std::size_t i = 0; i < clip.size(); ++i) {
        const YAML::Node range = node[i];
        if (!range || !range.IsSequence() || range.size() != 2) {
            throw std::runtime_error(path + "[" + std::to_string(i) +
                                     "] must be [lower, upper]");
        }
        clip[i][0] = range[0].as<double>();
        clip[i][1] = range[1].as<double>();
    }
}

YAML::Node requireNode(const YAML::Node& node, const std::string& key, const std::string& path)
{
    const YAML::Node child = node[key];
    if (!child) {
        throw std::runtime_error("missing deploy config field: " + path + "." + key);
    }
    return child;
}

YAML::Node requireAnyNode(const YAML::Node& node,
                          std::initializer_list<const char*> keys,
                          const std::string& path)
{
    std::string choices;
    for (const char* key : keys) {
        const YAML::Node child = node[key];
        if (child) {
            return child;
        }
        if (!choices.empty()) {
            choices += ", ";
        }
        choices += key;
    }
    throw std::runtime_error("missing deploy config field: " + path +
                             ". expected one of [" + choices + "]");
}

void loadDeployConfig(const std::string& path,
                      RunnerOptions& options,
                      PolicyConfig& config)
{
    const YAML::Node root = YAML::LoadFile(path);

    const YAML::Node direct_model_to_motor = root["model_to_motor_index"];
    if (nodePresent(direct_model_to_motor)) {
        loadCompactIntArray(direct_model_to_motor,
                            "model_to_motor_index",
                            config.model_to_motor_index,
                            config.model_to_motor_count);
        loadAnkleParallelMap(requireNode(root, "left_ankle_parallel", "deploy"),
                             "left_ankle_parallel",
                             config.left_ankle_parallel);
        loadAnkleParallelMap(requireNode(root, "right_ankle_parallel", "deploy"),
                             "right_ankle_parallel",
                             config.right_ankle_parallel);
        synthesizeLegacyJointMapFromParallelMapping(config);
    } else {
        loadIntArray(requireNode(root, "joint_ids_map", "deploy"),
                     "joint_ids_map",
                     config.control_to_motor_index);
        deriveParallelMappingFromLegacyJointMap(config);
    }

    const YAML::Node motor_direction = root["motor_to_model_direction"];
    if (nodePresent(motor_direction)) {
        loadPlainIntArray(motor_direction,
                          "motor_to_model_direction",
                          config.motor_to_model_direction);
    }

    const double step_dt = requireNode(root, "step_dt", "deploy").as<double>();
    if (step_dt <= 0.0 || !std::isfinite(step_dt)) {
        throw std::runtime_error("step_dt must be finite and > 0");
    }
    config.policy_step_dt_s = step_dt;
    options.policy_hz = 1.0 / step_dt;

    loadDoubleArray(requireNode(root, "stiffness", "deploy"),
                    "stiffness",
                    config.policy_mit_kp_model);
    loadDoubleArray(requireNode(root, "damping", "deploy"),
                    "damping",
                    config.policy_mit_kd_model);
    loadDoubleArray(requireNode(root, "default_joint_pos", "deploy"),
                    "default_joint_pos",
                    config.stand_pose_rad);

    const YAML::Node actions = requireNode(root, "actions", "deploy");
    const YAML::Node joint_action =
        requireNode(actions, "JointPositionAction", "deploy.actions");
    loadActionClip(requireNode(joint_action, "clip", "deploy.actions.JointPositionAction"),
                   "actions.JointPositionAction.clip",
                   config.action_clip);
    loadDoubleArray(requireNode(joint_action, "scale", "deploy.actions.JointPositionAction"),
                    "actions.JointPositionAction.scale",
                    config.action_scale);
    const YAML::Node raw_action_clip =
        optionalAnyNode(joint_action, {"raw_action_clip", "raw_clip"});
    if (nodePresent(raw_action_clip)) {
        config.raw_action_clip =
            parseDouble(scalarText(raw_action_clip,
                                   "actions.JointPositionAction.raw_action_clip"),
                        "actions.JointPositionAction.raw_action_clip");
    }
    const YAML::Node root_raw_action_clip =
        optionalAnyNode(root, {"raw_action_clip", "policy_raw_action_clip"});
    if (nodePresent(root_raw_action_clip)) {
        config.raw_action_clip =
            parseDouble(scalarText(root_raw_action_clip, "raw_action_clip"),
                        "raw_action_clip");
    }
    const YAML::Node action_offset =
        optionalAnyNode(joint_action, {"offset", "default_joint_pos"});
    if (nodePresent(action_offset)) {
        loadDoubleArray(action_offset,
                        "actions.JointPositionAction.offset",
                        config.stand_pose_rad);
    }

    const YAML::Node joint_min =
        optionalAnyNode(root, {"joint_min_rad", "joint_min", "joint_lower_rad"});
    const YAML::Node joint_max =
        optionalAnyNode(root, {"joint_max_rad", "joint_max", "joint_upper_rad"});
    if (nodePresent(joint_min) && nodePresent(joint_max) &&
        joint_min.IsSequence() && joint_max.IsSequence() &&
        joint_min.size() == kPolicyDof && joint_max.size() == kPolicyDof) {
        loadDoubleArray(joint_min, "joint_min_rad", config.joint_min_rad);
        loadDoubleArray(joint_max, "joint_max_rad", config.joint_max_rad);
    } else {
        for (int i = 0; i < kPolicyDof; ++i) {
            const auto idx = static_cast<std::size_t>(i);
            config.joint_min_rad[idx] = config.action_clip[idx][0];
            config.joint_max_rad[idx] = config.action_clip[idx][1];
        }
    }

    const YAML::Node observations = requireNode(root, "observations", "deploy");
    const YAML::Node observation_scale_first = observations["scale_first"];
    if (nodePresent(observation_scale_first)) {
        config.observation_scale_first =
            parseBoolScalar(observation_scale_first, "observations.scale_first");
    }

    const YAML::Node base_ang_vel =
        requireNode(observations, "base_ang_vel", "deploy.observations");
    loadDoubleArray(requireNode(base_ang_vel,
                                "scale",
                                "deploy.observations.base_ang_vel"),
                    "observations.base_ang_vel.scale",
                    config.body_ang_vel_scale);
    loadObservationClip(base_ang_vel,
                        "observations.base_ang_vel",
                        config.base_ang_vel_clip_enabled,
                        config.base_ang_vel_clip);

    const YAML::Node projected_gravity =
        requireNode(observations, "projected_gravity", "deploy.observations");
    loadDoubleArray(requireNode(projected_gravity,
                                "scale",
                                "deploy.observations.projected_gravity"),
                    "observations.projected_gravity.scale",
                    config.projected_gravity_scale);
    loadObservationClip(projected_gravity,
                        "observations.projected_gravity",
                        config.projected_gravity_clip_enabled,
                        config.projected_gravity_clip);

    const YAML::Node velocity_commands =
        requireAnyNode(observations,
                       {"keyboard_velocity_commands", "velocity_commands"},
                       "deploy.observations");
    loadDoubleArray(requireNode(velocity_commands,
                                "scale",
                                "deploy.observations.velocity_commands"),
                    "observations.velocity_commands.scale",
                    config.command_scale);
    loadObservationClip(velocity_commands,
                        "observations.velocity_commands",
                        config.command_clip_enabled,
                        config.command_clip);

    const YAML::Node joint_pos_rel =
        requireNode(observations, "joint_pos_rel", "deploy.observations");
    loadDoubleArray(requireNode(joint_pos_rel,
                                "scale",
                                "deploy.observations.joint_pos_rel"),
                    "observations.joint_pos_rel.scale",
                    config.dof_pos_scale);
    loadObservationClip(joint_pos_rel,
                        "observations.joint_pos_rel",
                        config.joint_pos_rel_clip_enabled,
                        config.joint_pos_rel_clip);

    const YAML::Node joint_vel_rel =
        requireNode(observations, "joint_vel_rel", "deploy.observations");
    loadDoubleArray(requireNode(joint_vel_rel,
                                "scale",
                                "deploy.observations.joint_vel_rel"),
                    "observations.joint_vel_rel.scale",
                    config.dof_vel_scale);
    loadObservationClip(joint_vel_rel,
                        "observations.joint_vel_rel",
                        config.joint_vel_rel_clip_enabled,
                        config.joint_vel_rel_clip);

    const YAML::Node last_action =
        requireNode(observations, "last_action", "deploy.observations");
    loadDoubleArray(requireNode(last_action,
                                "scale",
                                "deploy.observations.last_action"),
                    "observations.last_action.scale",
                    config.last_action_scale);
    loadObservationClip(last_action,
                        "observations.last_action",
                        config.last_action_clip_enabled,
                        config.last_action_clip);

    const YAML::Node gait_phase =
        optionalAnyNode(observations, {"gait_phase"});
    if (nodePresent(gait_phase)) {
        const YAML::Node scale = gait_phase["scale"];
        if (nodePresent(scale)) {
            loadDoubleArray(scale,
                            "observations.gait_phase.scale",
                            config.gait_phase_scale);
        }
        loadObservationClip(gait_phase,
                            "observations.gait_phase",
                            config.gait_phase_clip_enabled,
                            config.gait_phase_clip);

        const YAML::Node params = gait_phase["params"];
        if (nodePresent(params)) {
            const YAML::Node period = params["period"];
            if (nodePresent(period)) {
                config.gait_phase_period_s = period.as<double>();
            }
            const YAML::Node stand_threshold =
                optionalAnyNode(params, {"stand_threshold",
                                         "gait_phase_stand_threshold",
                                         "stand_command_threshold"});
            if (nodePresent(stand_threshold)) {
                config.gait_phase_stand_threshold =
                    parseDouble(scalarText(stand_threshold,
                                           "observations.gait_phase.params.stand_threshold"),
                                "observations.gait_phase.params.stand_threshold");
            }
            const YAML::Node move_threshold =
                optionalAnyNode(params, {"move_threshold",
                                         "gait_phase_move_threshold",
                                         "move_command_threshold"});
            if (nodePresent(move_threshold)) {
                config.gait_phase_move_threshold =
                    parseDouble(scalarText(move_threshold,
                                           "observations.gait_phase.params.move_threshold"),
                                "observations.gait_phase.params.move_threshold");
            }
        }
    }
}

void loadViewerConfigDefaultsFromNode(const YAML::Node& node,
                                      const std::string& path,
                                      RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map");
    }

    const YAML::Node overlay =
        optionalAnyNode(node, {"overlay", "enabled", "show_overlay"});
    if (nodePresent(overlay)) {
        options.viewer_overlay_enabled = parseBoolScalar(overlay,
                                                         path + ".overlay");
    }

    const YAML::Node page =
        optionalAnyNode(node, {"page", "overlay_page"});
    if (nodePresent(page)) {
        options.viewer_overlay_page =
            parseViewerOverlayPage(scalarText(page, path + ".page"),
                                   path + ".page");
    }

    const YAML::Node units =
        optionalAnyNode(node, {"angle_units", "joint_units", "units"});
    if (nodePresent(units)) {
        options.viewer_overlay_degrees =
            parseViewerOverlayUnitsUseDegrees(scalarText(units,
                                                         path + ".angle_units"),
                                              path + ".angle_units");
    }

    const YAML::Node use_degrees =
        optionalAnyNode(node, {"use_degrees", "degrees"});
    if (nodePresent(use_degrees)) {
        options.viewer_overlay_degrees = parseBoolScalar(use_degrees,
                                                         path + ".use_degrees");
    }

    const YAML::Node curve =
        optionalAnyNode(node, {"curve", "curve_enabled", "show_curve"});
    if (nodePresent(curve)) {
        options.viewer_curve_enabled = parseBoolScalar(curve,
                                                       path + ".curve");
    }

    const YAML::Node curve_signal =
        optionalAnyNode(node, {"curve_signal", "signal"});
    if (nodePresent(curve_signal)) {
        options.viewer_curve_signal =
            parseViewerCurveSignal(scalarText(curve_signal,
                                              path + ".curve_signal"),
                                   path + ".curve_signal");
    }

    const YAML::Node curve_joint_index =
        optionalAnyNode(node, {"curve_joint_index", "curve_joint", "joint"});
    if (nodePresent(curve_joint_index)) {
        options.viewer_curve_joint_index =
            parseInt(scalarText(curve_joint_index,
                                path + ".curve_joint_index"),
                     path + ".curve_joint_index");
    }

    const YAML::Node curve_window =
        optionalAnyNode(node, {"curve_window_seconds", "curve_window", "window"});
    if (nodePresent(curve_window)) {
        options.viewer_curve_window_seconds =
            parseDouble(scalarText(curve_window,
                                   path + ".curve_window_seconds"),
                        path + ".curve_window_seconds");
    }
}

void loadMotorDelayConfigDefaultsFromNode(const YAML::Node& node,
                                          const std::string& path,
                                          RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (node.IsScalar()) {
        options.motor_delay_enabled = parseBoolScalar(node, path);
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map or boolean");
    }

    const YAML::Node enabled =
        optionalAnyNode(node, {"enabled", "enable", "use_delay"});
    if (nodePresent(enabled)) {
        options.motor_delay_enabled = parseBoolScalar(enabled,
                                                      path + ".enabled");
    }

    const YAML::Node range_ms =
        optionalAnyNode(node, {"range_ms", "delay_range_ms", "latency_range_ms"});
    if (nodePresent(range_ms)) {
        if (!range_ms.IsSequence() || range_ms.size() != 2) {
            throw std::runtime_error(path + ".range_ms must be [min_ms, max_ms]");
        }
        options.motor_delay_min_seconds = range_ms[0].as<double>() * 0.001;
        options.motor_delay_max_seconds = range_ms[1].as<double>() * 0.001;
    }

    const YAML::Node range_seconds =
        optionalAnyNode(node, {"range_seconds", "delay_range_seconds"});
    if (nodePresent(range_seconds)) {
        if (!range_seconds.IsSequence() || range_seconds.size() != 2) {
            throw std::runtime_error(path +
                                     ".range_seconds must be [min_seconds, max_seconds]");
        }
        options.motor_delay_min_seconds = range_seconds[0].as<double>();
        options.motor_delay_max_seconds = range_seconds[1].as<double>();
    }

    const YAML::Node min_ms =
        optionalAnyNode(node, {"min_ms", "delay_min_ms", "latency_min_ms"});
    if (nodePresent(min_ms)) {
        options.motor_delay_min_seconds =
            parseDouble(scalarText(min_ms, path + ".min_ms"),
                        path + ".min_ms") *
            0.001;
    }

    const YAML::Node max_ms =
        optionalAnyNode(node, {"max_ms", "delay_max_ms", "latency_max_ms"});
    if (nodePresent(max_ms)) {
        options.motor_delay_max_seconds =
            parseDouble(scalarText(max_ms, path + ".max_ms"),
                        path + ".max_ms") *
            0.001;
    }

    const YAML::Node min_seconds =
        optionalAnyNode(node, {"min_seconds", "delay_min_seconds",
                               "latency_min_seconds"});
    if (nodePresent(min_seconds)) {
        options.motor_delay_min_seconds =
            parseDouble(scalarText(min_seconds, path + ".min_seconds"),
                        path + ".min_seconds");
    }

    const YAML::Node max_seconds =
        optionalAnyNode(node, {"max_seconds", "delay_max_seconds",
                               "latency_max_seconds"});
    if (nodePresent(max_seconds)) {
        options.motor_delay_max_seconds =
            parseDouble(scalarText(max_seconds, path + ".max_seconds"),
                        path + ".max_seconds");
    }
}

void loadObservationDelayConfigDefaultsFromNode(const YAML::Node& node,
                                                const std::string& path,
                                                RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (node.IsScalar()) {
        options.observation_delay_enabled = parseBoolScalar(node, path);
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map or boolean");
    }

    const YAML::Node enabled =
        optionalAnyNode(node, {"enabled", "enable", "use_delay"});
    if (nodePresent(enabled)) {
        options.observation_delay_enabled =
            parseBoolScalar(enabled, path + ".enabled");
    }

    const YAML::Node source =
        optionalAnyNode(node, {"source", "delay_source", "delayed_source"});
    if (nodePresent(source)) {
        options.observation_delay_source =
            parseObservationDelaySource(scalarText(source, path + ".source"),
                                        path + ".source");
    }

    const YAML::Node delay_ms =
        optionalAnyNode(node, {"delay_ms", "latency_ms", "ms"});
    if (nodePresent(delay_ms)) {
        options.observation_delay_seconds =
            parseDouble(scalarText(delay_ms, path + ".delay_ms"),
                        path + ".delay_ms") *
            0.001;
    }

    const YAML::Node delay_seconds =
        optionalAnyNode(node, {"delay_seconds", "latency_seconds", "seconds"});
    if (nodePresent(delay_seconds)) {
        options.observation_delay_seconds =
            parseDouble(scalarText(delay_seconds, path + ".delay_seconds"),
                        path + ".delay_seconds");
    }
}

void loadJoystickConfigDefaultsFromNode(const YAML::Node& node,
                                        const std::string& path,
                                        RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (node.IsScalar()) {
        options.joystick_enabled = parseBoolScalar(node, path);
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map or boolean");
    }

    const YAML::Node enabled =
        optionalAnyNode(node, {"enabled", "enable"});
    if (nodePresent(enabled)) {
        options.joystick_enabled = parseBoolScalar(enabled,
                                                   path + ".enabled");
    }

    const YAML::Node type =
        optionalAnyNode(node, {"type", "controller", "controller_type"});
    if (nodePresent(type)) {
        options.joystick_type = normalizedToken(scalarText(type,
                                                           path + ".type"));
    }

    const YAML::Node device =
        optionalAnyNode(node, {"device", "path", "device_path"});
    if (nodePresent(device)) {
        options.joystick_device = scalarText(device, path + ".device");
    }

    const YAML::Node bits =
        optionalAnyNode(node, {"bits", "axis_bits"});
    if (nodePresent(bits)) {
        options.joystick_bits =
            parseInt(scalarText(bits, path + ".bits"), path + ".bits");
    }

    const YAML::Node deadzone =
        optionalAnyNode(node, {"deadzone", "dead_zone"});
    if (nodePresent(deadzone)) {
        options.joystick_deadzone =
            parseDouble(scalarText(deadzone, path + ".deadzone"),
                        path + ".deadzone");
    }

    const YAML::Node limits =
        optionalAnyNode(node, {"limits", "velocity_limits", "command_limits"});
    if (nodePresent(limits)) {
        loadDoubleArray(limits, path + ".limits", options.joystick_limits);
    }

    const YAML::Node signs =
        optionalAnyNode(node, {"signs", "axis_signs", "command_signs"});
    if (nodePresent(signs)) {
        loadDoubleArray(signs, path + ".signs", options.joystick_signs);
    }

    const YAML::Node vx_limit =
        optionalAnyNode(node, {"vx_limit", "lin_vel_x_limit"});
    if (nodePresent(vx_limit)) {
        options.joystick_limits[0] =
            parseDouble(scalarText(vx_limit, path + ".vx_limit"),
                        path + ".vx_limit");
    }
    const YAML::Node vy_limit =
        optionalAnyNode(node, {"vy_limit", "lin_vel_y_limit"});
    if (nodePresent(vy_limit)) {
        options.joystick_limits[1] =
            parseDouble(scalarText(vy_limit, path + ".vy_limit"),
                        path + ".vy_limit");
    }
    const YAML::Node yaw_rate_limit =
        optionalAnyNode(node, {"yaw_rate_limit", "ang_vel_z_limit"});
    if (nodePresent(yaw_rate_limit)) {
        options.joystick_limits[2] =
            parseDouble(scalarText(yaw_rate_limit, path + ".yaw_rate_limit"),
                        path + ".yaw_rate_limit");
    }
}

void loadLoggingConfigDefaultsFromNode(const YAML::Node& node,
                                       const std::string& path,
                                       const std::filesystem::path& config_path,
                                       RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (node.IsScalar()) {
        options.sim_log_enabled = parseBoolScalar(node, path);
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map or boolean");
    }

    const YAML::Node enabled =
        optionalAnyNode(node, {"enabled", "enable"});
    if (nodePresent(enabled)) {
        options.sim_log_enabled = parseBoolScalar(enabled,
                                                  path + ".enabled");
    }

    const YAML::Node log_path =
        optionalAnyNode(node, {"path", "csv", "csv_path", "file"});
    if (nodePresent(log_path)) {
        options.sim_log_path =
            resolveConfigPath(scalarText(log_path, path + ".path"),
                              config_path);
    }

    const YAML::Node log_dir =
        optionalAnyNode(node, {"dir", "directory", "log_dir", "folder"});
    if (nodePresent(log_dir)) {
        options.sim_log_path =
            resolveConfigPath(scalarText(log_dir, path + ".dir"),
                              config_path);
    }
}

void loadElasticRopeConfigDefaultsFromNode(const YAML::Node& node,
                                           const std::string& path,
                                           RunnerOptions& options)
{
    if (!nodePresent(node)) {
        return;
    }
    if (node.IsScalar()) {
        options.elastic_rope_enabled = parseBoolScalar(node, path);
        return;
    }
    if (!node.IsMap()) {
        throw std::runtime_error(path + " must be a map or boolean");
    }

    const YAML::Node enabled =
        optionalAnyNode(node, {"enabled", "enable", "use_rope"});
    if (nodePresent(enabled)) {
        options.elastic_rope_enabled = parseBoolScalar(enabled,
                                                       path + ".enabled");
    }
}

void loadRunnerConfigDefaults(const std::string& path,
                              RunnerOptions& options,
                              PolicyConfig& config)
{
    const YAML::Node root = YAML::LoadFile(path);
    const YAML::Node paths = root["paths"] ? root["paths"] : root;
    const auto config_path = std::filesystem::path(path);

    const YAML::Node model =
        optionalAnyNode(paths, {"model_xml_path", "model", "xml"});
    if (nodePresent(model) && model.IsScalar()) {
        options.model_xml_path =
            resolveConfigPath(model.as<std::string>(), config_path);
    }

    const YAML::Node deploy =
        optionalAnyNode(paths, {"deploy_config_path", "deploy_config", "deploy"});
    if (nodePresent(deploy) && deploy.IsScalar()) {
        options.deploy_config_path =
            resolveConfigPath(deploy.as<std::string>(), config_path);
    }

    const YAML::Node policy =
        optionalAnyNode(paths, {"policy_model_path", "policy_model", "policy"});
    if (nodePresent(policy) && policy.IsScalar()) {
        config.policy_model_path =
            resolveConfigPath(policy.as<std::string>(), config_path);
    }

    const YAML::Node real_observation_log =
        optionalAnyNode(paths, {"real_observation_log_path",
                                "real_observation_log",
                                "real_policy_observation_log"});
    if (nodePresent(real_observation_log) && real_observation_log.IsScalar()) {
        options.real_observation_log_path =
            resolveConfigPath(real_observation_log.as<std::string>(),
                              config_path);
    }

    const YAML::Node policy_observation =
        optionalAnyNode(root, {"policy_observation",
                               "policy_observations",
                               "observation_layout",
                               "policy_observation_layout"});
    if (nodePresent(policy_observation)) {
        if (policy_observation.IsScalar()) {
            config.include_gait_phase_observation =
                parseGaitPhaseObservationEnabled(
                    scalarText(policy_observation, "policy_observation"),
                    "policy_observation");
        } else if (policy_observation.IsMap()) {
            const YAML::Node gait_phase =
                optionalAnyNode(policy_observation,
                                {"gait_phase",
                                 "include_gait_phase",
                                 "with_gait_phase",
                                 "use_gait_phase"});
            const YAML::Node layout =
                optionalAnyNode(policy_observation, {"layout", "type", "mode"});
            if (nodePresent(gait_phase)) {
                config.include_gait_phase_observation =
                    parseGaitPhaseObservationEnabled(
                        scalarText(gait_phase, "policy_observation.gait_phase"),
                        "policy_observation.gait_phase");
            } else if (nodePresent(layout)) {
                config.include_gait_phase_observation =
                    parseGaitPhaseObservationEnabled(
                        scalarText(layout, "policy_observation.layout"),
                        "policy_observation.layout");
            }
        } else {
            throw std::runtime_error("policy_observation must be a scalar or map");
        }
    }

    const YAML::Node gait_phase_observation =
        optionalAnyNode(root, {"policy_observation_gait_phase",
                               "include_gait_phase_observation"});
    if (nodePresent(gait_phase_observation)) {
        config.include_gait_phase_observation =
            parseGaitPhaseObservationEnabled(
                scalarText(gait_phase_observation,
                           "policy_observation_gait_phase"),
                "policy_observation_gait_phase");
    }

    const YAML::Node raw_action_clip =
        optionalAnyNode(root, {"raw_action_clip", "policy_raw_action_clip"});
    if (nodePresent(raw_action_clip)) {
        config.raw_action_clip =
            parseDouble(scalarText(raw_action_clip, "raw_action_clip"),
                        "raw_action_clip");
    }

    const YAML::Node motor_direction_override =
        optionalAnyNode(root, {"mujoco_motor_to_model_direction",
                               "motor_to_model_direction_override",
                               "motor_to_model_direction"});
    if (nodePresent(motor_direction_override)) {
        loadPlainIntArray(motor_direction_override,
                          "mujoco_motor_to_model_direction",
                          options.motor_to_model_direction);
        options.motor_to_model_direction_override = true;
    }

    const YAML::Node mujoco_joint_direction =
        optionalAnyNode(root, {"mujoco_joint_direction",
                               "mujoco_joint_sign",
                               "mujoco_joint_directions"});
    if (nodePresent(mujoco_joint_direction)) {
        loadPlainIntArray(mujoco_joint_direction,
                          "mujoco_joint_direction",
                          options.mujoco_joint_direction);
    }

    const YAML::Node gait_phase_stand_threshold =
        optionalAnyNode(root, {"gait_phase_stand_threshold",
                               "policy_gait_phase_stand_threshold"});
    if (nodePresent(gait_phase_stand_threshold)) {
        config.gait_phase_stand_threshold =
            parseDouble(scalarText(gait_phase_stand_threshold,
                                   "gait_phase_stand_threshold"),
                        "gait_phase_stand_threshold");
    }
    const YAML::Node gait_phase_move_threshold =
        optionalAnyNode(root, {"gait_phase_move_threshold",
                               "policy_gait_phase_move_threshold"});
    if (nodePresent(gait_phase_move_threshold)) {
        config.gait_phase_move_threshold =
            parseDouble(scalarText(gait_phase_move_threshold,
                                   "gait_phase_move_threshold"),
                        "gait_phase_move_threshold");
    }

    const YAML::Node auto_start_policy =
        optionalAnyNode(root, {"auto_start_policy", "start_policy",
                               "policy_auto_start", "policy_autostart"});
    if (nodePresent(auto_start_policy)) {
        options.auto_start_policy =
            parseBoolScalar(auto_start_policy, "auto_start_policy");
    }

    const YAML::Node use_cli_command =
        optionalAnyNode(root, {"use_cli_command_on_policy_start",
                               "use_cli_command_on_start",
                               "start_with_cli_command",
                               "fixed_cli_command"});
    if (nodePresent(use_cli_command)) {
        options.use_cli_command_on_policy_start =
            parseBoolScalar(use_cli_command, "use_cli_command_on_policy_start");
    }

    const YAML::Node imu_noise =
        optionalAnyNode(root, {"imu_noise", "imu_sensor_noise", "sensor_noise"});
    if (nodePresent(imu_noise)) {
        if (imu_noise.IsScalar()) {
            options.imu_noise_enabled =
                parseBoolScalar(imu_noise, "imu_noise");
        } else if (imu_noise.IsMap()) {
            const YAML::Node enabled =
                optionalAnyNode(imu_noise, {"enabled", "enable", "active"});
            if (nodePresent(enabled)) {
                options.imu_noise_enabled =
                    parseBoolScalar(enabled, "imu_noise.enabled");
            }
        } else {
            throw std::runtime_error("imu_noise must be a scalar or map");
        }
    }

    loadViewerConfigDefaultsFromNode(root["viewer"], "viewer", options);
    loadMotorDelayConfigDefaultsFromNode(root["motor_delay"],
                                         "motor_delay",
                                         options);
    loadObservationDelayConfigDefaultsFromNode(
        optionalAnyNode(root, {"observation_delay",
                               "sensor_time_offset",
                               "state_observation_delay"}),
        "observation_delay",
        options);
    loadJoystickConfigDefaultsFromNode(root["joystick"],
                                       "joystick",
                                       options);
    loadLoggingConfigDefaultsFromNode(root["logging"],
                                      "logging",
                                      config_path,
                                      options);
    loadElasticRopeConfigDefaultsFromNode(
        optionalAnyNode(root, {"elastic_rope", "rope", "suspension_rope"}),
        "elastic_rope",
        options);

    const YAML::Node elastic_rope_enabled =
        optionalAnyNode(root, {"elastic_rope_enabled", "rope_enabled"});
    if (nodePresent(elastic_rope_enabled)) {
        options.elastic_rope_enabled =
            parseBoolScalar(elastic_rope_enabled, "elastic_rope_enabled");
    }

    const YAML::Node joint_zero_offset_rad =
        optionalAnyNode(root, {"joint_zero_offset_rad",
                               "joint_zero_offsets_rad",
                               "motor_zero_offset_rad",
                               "encoder_zero_offset_rad",
                               "encoder_zero_offsets_rad"});
    const YAML::Node joint_zero_offset_deg =
        optionalAnyNode(root, {"joint_zero_offset_deg",
                               "joint_zero_offsets_deg",
                               "motor_zero_offset_deg",
                               "encoder_zero_offset_deg",
                               "encoder_zero_offsets_deg"});
    if (nodePresent(joint_zero_offset_rad) &&
        nodePresent(joint_zero_offset_deg)) {
        throw std::runtime_error("set only one of joint_zero_offset_rad or "
                                 "joint_zero_offset_deg");
    }
    if (nodePresent(joint_zero_offset_rad)) {
        loadDoubleScalarOrArray(joint_zero_offset_rad,
                                "joint_zero_offset_rad",
                                options.joint_zero_offset_rad);
    }
    if (nodePresent(joint_zero_offset_deg)) {
        loadDegreeScalarOrArrayAsRadians(joint_zero_offset_deg,
                                         "joint_zero_offset_deg",
                                         options.joint_zero_offset_rad);
    }

    const YAML::Node viewer_overlay =
        optionalAnyNode(root, {"viewer_overlay", "viewer_overlay_enabled"});
    if (nodePresent(viewer_overlay)) {
        options.viewer_overlay_enabled =
            parseBoolScalar(viewer_overlay, "viewer_overlay");
    }
    const YAML::Node viewer_page =
        optionalAnyNode(root, {"viewer_overlay_page", "viewer_page"});
    if (nodePresent(viewer_page)) {
        options.viewer_overlay_page =
            parseViewerOverlayPage(scalarText(viewer_page,
                                              "viewer_overlay_page"),
                                   "viewer_overlay_page");
    }
    const YAML::Node viewer_units =
        optionalAnyNode(root, {"viewer_angle_units", "viewer_joint_units"});
    if (nodePresent(viewer_units)) {
        options.viewer_overlay_degrees =
            parseViewerOverlayUnitsUseDegrees(scalarText(viewer_units,
                                                         "viewer_angle_units"),
                                              "viewer_angle_units");
    }
    const YAML::Node viewer_degrees =
        optionalAnyNode(root, {"viewer_overlay_degrees", "viewer_use_degrees"});
    if (nodePresent(viewer_degrees)) {
        options.viewer_overlay_degrees =
            parseBoolScalar(viewer_degrees, "viewer_overlay_degrees");
    }

    const YAML::Node viewer_curve =
        optionalAnyNode(root, {"viewer_curve", "viewer_curve_enabled"});
    if (nodePresent(viewer_curve)) {
        options.viewer_curve_enabled =
            parseBoolScalar(viewer_curve, "viewer_curve");
    }
    const YAML::Node viewer_curve_signal =
        optionalAnyNode(root, {"viewer_curve_signal", "viewer_signal"});
    if (nodePresent(viewer_curve_signal)) {
        options.viewer_curve_signal =
            parseViewerCurveSignal(scalarText(viewer_curve_signal,
                                              "viewer_curve_signal"),
                                   "viewer_curve_signal");
    }
    const YAML::Node viewer_curve_joint =
        optionalAnyNode(root, {"viewer_curve_joint_index", "viewer_curve_joint"});
    if (nodePresent(viewer_curve_joint)) {
        options.viewer_curve_joint_index =
            parseInt(scalarText(viewer_curve_joint,
                                "viewer_curve_joint_index"),
                     "viewer_curve_joint_index");
    }
    const YAML::Node viewer_curve_window =
        optionalAnyNode(root, {"viewer_curve_window_seconds", "viewer_curve_window"});
    if (nodePresent(viewer_curve_window)) {
        options.viewer_curve_window_seconds =
            parseDouble(scalarText(viewer_curve_window,
                                   "viewer_curve_window_seconds"),
                        "viewer_curve_window_seconds");
    }

    const YAML::Node motor_delay_enabled =
        optionalAnyNode(root, {"motor_delay_enabled", "actuator_delay_enabled"});
    if (nodePresent(motor_delay_enabled)) {
        options.motor_delay_enabled =
            parseBoolScalar(motor_delay_enabled, "motor_delay_enabled");
    }
    const YAML::Node motor_delay_range_ms =
        optionalAnyNode(root, {"motor_delay_range_ms",
                               "actuator_delay_range_ms"});
    if (nodePresent(motor_delay_range_ms)) {
        if (!motor_delay_range_ms.IsSequence() ||
            motor_delay_range_ms.size() != 2) {
            throw std::runtime_error("motor_delay_range_ms must be [min_ms, max_ms]");
        }
        options.motor_delay_min_seconds =
            motor_delay_range_ms[0].as<double>() * 0.001;
        options.motor_delay_max_seconds =
            motor_delay_range_ms[1].as<double>() * 0.001;
    }
    const YAML::Node motor_delay_min_ms =
        optionalAnyNode(root, {"motor_delay_min_ms", "actuator_delay_min_ms"});
    if (nodePresent(motor_delay_min_ms)) {
        options.motor_delay_min_seconds =
            parseDouble(scalarText(motor_delay_min_ms, "motor_delay_min_ms"),
                        "motor_delay_min_ms") *
            0.001;
    }
    const YAML::Node motor_delay_max_ms =
        optionalAnyNode(root, {"motor_delay_max_ms", "actuator_delay_max_ms"});
    if (nodePresent(motor_delay_max_ms)) {
        options.motor_delay_max_seconds =
            parseDouble(scalarText(motor_delay_max_ms, "motor_delay_max_ms"),
                        "motor_delay_max_ms") *
            0.001;
    }
    const YAML::Node motor_delay_min_seconds =
        optionalAnyNode(root, {"motor_delay_min_seconds",
                               "actuator_delay_min_seconds"});
    if (nodePresent(motor_delay_min_seconds)) {
        options.motor_delay_min_seconds =
            parseDouble(scalarText(motor_delay_min_seconds,
                                   "motor_delay_min_seconds"),
                        "motor_delay_min_seconds");
    }
    const YAML::Node motor_delay_max_seconds =
        optionalAnyNode(root, {"motor_delay_max_seconds",
                               "actuator_delay_max_seconds"});
    if (nodePresent(motor_delay_max_seconds)) {
        options.motor_delay_max_seconds =
            parseDouble(scalarText(motor_delay_max_seconds,
                                   "motor_delay_max_seconds"),
                        "motor_delay_max_seconds");
    }

    const YAML::Node observation_delay_enabled =
        optionalAnyNode(root, {"observation_delay_enabled",
                               "sensor_time_offset_enabled",
                               "state_observation_delay_enabled"});
    if (nodePresent(observation_delay_enabled)) {
        options.observation_delay_enabled =
            parseBoolScalar(observation_delay_enabled,
                            "observation_delay_enabled");
    }
    const YAML::Node observation_delay_source =
        optionalAnyNode(root, {"observation_delay_source",
                               "sensor_time_offset_source",
                               "delayed_observation_source"});
    if (nodePresent(observation_delay_source)) {
        options.observation_delay_source =
            parseObservationDelaySource(
                scalarText(observation_delay_source,
                           "observation_delay_source"),
                "observation_delay_source");
    }
    const YAML::Node observation_delay_ms =
        optionalAnyNode(root, {"observation_delay_ms",
                               "sensor_time_offset_ms",
                               "state_observation_delay_ms"});
    if (nodePresent(observation_delay_ms)) {
        options.observation_delay_seconds =
            parseDouble(scalarText(observation_delay_ms,
                                   "observation_delay_ms"),
                        "observation_delay_ms") *
            0.001;
    }
    const YAML::Node observation_delay_seconds =
        optionalAnyNode(root, {"observation_delay_seconds",
                               "sensor_time_offset_seconds",
                               "state_observation_delay_seconds"});
    if (nodePresent(observation_delay_seconds)) {
        options.observation_delay_seconds =
            parseDouble(scalarText(observation_delay_seconds,
                                   "observation_delay_seconds"),
                        "observation_delay_seconds");
    }

    const YAML::Node joystick_enabled =
        optionalAnyNode(root, {"joystick_enabled", "xbox_enabled"});
    if (nodePresent(joystick_enabled)) {
        options.joystick_enabled =
            parseBoolScalar(joystick_enabled, "joystick_enabled");
    }
    const YAML::Node joystick_type =
        optionalAnyNode(root, {"joystick_type", "controller_type"});
    if (nodePresent(joystick_type)) {
        options.joystick_type =
            normalizedToken(scalarText(joystick_type, "joystick_type"));
    }
    const YAML::Node joystick_device =
        optionalAnyNode(root, {"joystick_device", "joystick_path"});
    if (nodePresent(joystick_device)) {
        options.joystick_device =
            scalarText(joystick_device, "joystick_device");
    }
    const YAML::Node joystick_deadzone =
        optionalAnyNode(root, {"joystick_deadzone", "joystick_dead_zone"});
    if (nodePresent(joystick_deadzone)) {
        options.joystick_deadzone =
            parseDouble(scalarText(joystick_deadzone, "joystick_deadzone"),
                        "joystick_deadzone");
    }
    const YAML::Node joystick_limits =
        optionalAnyNode(root, {"joystick_limits", "joystick_velocity_limits"});
    if (nodePresent(joystick_limits)) {
        loadDoubleArray(joystick_limits,
                        "joystick_limits",
                        options.joystick_limits);
    }
    const YAML::Node joystick_signs =
        optionalAnyNode(root, {"joystick_signs", "joystick_axis_signs"});
    if (nodePresent(joystick_signs)) {
        loadDoubleArray(joystick_signs,
                        "joystick_signs",
                        options.joystick_signs);
    }

    const YAML::Node sim_log_enabled =
        optionalAnyNode(root, {"sim_log_enabled", "tracking_log_enabled",
                               "csv_log_enabled"});
    if (nodePresent(sim_log_enabled)) {
        options.sim_log_enabled =
            parseBoolScalar(sim_log_enabled, "sim_log_enabled");
    }
    const YAML::Node sim_log_path =
        optionalAnyNode(root, {"sim_log_path", "tracking_log_path",
                               "csv_log_path"});
    if (nodePresent(sim_log_path)) {
        options.sim_log_path =
            resolveConfigPath(scalarText(sim_log_path, "sim_log_path"),
                              config_path);
    }
}

void validatePolicyConfig(const PolicyConfig& config)
{
    auto validate_clip = [](bool enabled,
                            const std::array<double, 2>& range,
                            const char* name) {
        if (!enabled) {
            return;
        }
        if (!std::isfinite(range[0]) ||
            !std::isfinite(range[1]) ||
            range[0] > range[1]) {
            throw std::runtime_error(std::string(name) +
                                     " must be finite [lower, upper]");
        }
    };

    auto validate_scale = [](const auto& values, const char* name) {
        for (double value : values) {
            if (!std::isfinite(value)) {
                throw std::runtime_error(std::string(name) +
                                         " must contain finite values");
            }
        }
    };

    if (config.policy_step_dt_s <= 0.0 ||
        !std::isfinite(config.policy_step_dt_s)) {
        throw std::runtime_error("policy step_dt must be finite and > 0");
    }
    if (config.gait_phase_period_s <= 0.0 ||
        !std::isfinite(config.gait_phase_period_s)) {
        throw std::runtime_error("observations.gait_phase.params.period must be finite and > 0");
    }
    if (config.raw_action_clip <= 0.0 ||
        !std::isfinite(config.raw_action_clip)) {
        throw std::runtime_error("raw_action_clip must be finite and > 0");
    }
    if (!std::isfinite(config.gait_phase_stand_threshold) ||
        !std::isfinite(config.gait_phase_move_threshold) ||
        config.gait_phase_move_threshold <= config.gait_phase_stand_threshold) {
        throw std::runtime_error("gait phase move threshold must be finite and > stand threshold");
    }
    validate_scale(config.body_ang_vel_scale, "observations.base_ang_vel.scale");
    validate_scale(config.projected_gravity_scale,
                   "observations.projected_gravity.scale");
    validate_scale(config.command_scale, "observations.velocity_commands.scale");
    validate_scale(config.gait_phase_scale, "observations.gait_phase.scale");
    validate_scale(config.dof_pos_scale, "observations.joint_pos_rel.scale");
    validate_scale(config.dof_vel_scale, "observations.joint_vel_rel.scale");
    validate_scale(config.last_action_scale, "observations.last_action.scale");

    validate_clip(config.base_ang_vel_clip_enabled,
                  config.base_ang_vel_clip,
                  "observations.base_ang_vel.clip");
    validate_clip(config.projected_gravity_clip_enabled,
                  config.projected_gravity_clip,
                  "observations.projected_gravity.clip");
    validate_clip(config.command_clip_enabled,
                  config.command_clip,
                  "observations.velocity_commands.clip");
    validate_clip(config.gait_phase_clip_enabled,
                  config.gait_phase_clip,
                  "observations.gait_phase.clip");
    validate_clip(config.joint_pos_rel_clip_enabled,
                  config.joint_pos_rel_clip,
                  "observations.joint_pos_rel.clip");
    validate_clip(config.joint_vel_rel_clip_enabled,
                  config.joint_vel_rel_clip,
                  "observations.joint_vel_rel.clip");
    validate_clip(config.last_action_clip_enabled,
                  config.last_action_clip,
                  "observations.last_action.clip");

    if (!ankleParallelMapIndicesInRange(config.left_ankle_parallel, kPolicyDof) ||
        !ankleParallelMapIndicesInRange(config.right_ankle_parallel, kPolicyDof)) {
        throw std::runtime_error("ankle parallel maps contain an out-of-range index");
    }

    std::array<bool, kPolicyDof> seen_ankle_model_dof{};
    std::array<bool, kPolicyDof> seen_motor{};

    auto mark_model_dof = [&](int model_dof) {
        if (seen_ankle_model_dof[static_cast<std::size_t>(model_dof)]) {
            return false;
        }
        seen_ankle_model_dof[static_cast<std::size_t>(model_dof)] = true;
        return true;
    };

    auto mark_motor = [&](int motor_index) {
        if (seen_motor[static_cast<std::size_t>(motor_index)]) {
            return false;
        }
        seen_motor[static_cast<std::size_t>(motor_index)] = true;
        return true;
    };

    const auto& left = config.left_ankle_parallel;
    const auto& right = config.right_ankle_parallel;
    if (!mark_model_dof(left.model_pitch_dof) ||
        !mark_model_dof(left.model_roll_dof) ||
        !mark_model_dof(right.model_pitch_dof) ||
        !mark_model_dof(right.model_roll_dof)) {
        throw std::runtime_error("ankle parallel model DOFs must be distinct");
    }

    if (!mark_motor(left.upper_motor_index) ||
        !mark_motor(left.lower_motor_index) ||
        !mark_motor(right.upper_motor_index) ||
        !mark_motor(right.lower_motor_index)) {
        throw std::runtime_error("ankle parallel motor indices must be distinct");
    }

    const int direct_model_dof_count = static_cast<int>(
        std::count(seen_ankle_model_dof.begin(),
                   seen_ankle_model_dof.end(),
                   false));
    if (config.model_to_motor_count != direct_model_dof_count) {
        std::ostringstream message;
        message << "model_to_motor_index must have "
                << direct_model_dof_count
                << " direct-DOF values";
        throw std::runtime_error(message.str());
    }

    for (int mapping_slot = 0; mapping_slot < config.model_to_motor_count;
         ++mapping_slot) {
        const int motor_index =
            config.model_to_motor_index[static_cast<std::size_t>(mapping_slot)];
        if (motor_index < 0 || motor_index >= kPolicyDof) {
            throw std::runtime_error("model_to_motor_index contains an "
                                     "out-of-range motor index");
        }
        if (!mark_motor(motor_index)) {
            throw std::runtime_error("direct and ankle motor mappings must be "
                                     "a permutation without duplicates");
        }
    }

    if (std::any_of(seen_motor.begin(), seen_motor.end(), [](bool is_seen) {
            return !is_seen;
        })) {
        throw std::runtime_error("direct and ankle motor mappings must cover all motors");
    }

    for (int i = 0; i < kPolicyDof; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        const int direction = config.motor_to_model_direction[idx];
        if (direction != 1 && direction != -1) {
            throw std::runtime_error("motor_to_model_direction values must be 1 or -1");
        }

        if (config.action_clip[idx][0] > config.action_clip[idx][1]) {
            throw std::runtime_error("actions.JointPositionAction.clip lower bound must be "
                                     "<= upper bound for every DOF");
        }
        if (!std::isfinite(config.action_clip[idx][0]) ||
            !std::isfinite(config.action_clip[idx][1]) ||
            !std::isfinite(config.action_scale[idx]) ||
            !std::isfinite(config.stand_pose_rad[idx]) ||
            !std::isfinite(config.joint_min_rad[idx]) ||
            !std::isfinite(config.joint_max_rad[idx]) ||
            !std::isfinite(config.policy_mit_kp_model[idx]) ||
            !std::isfinite(config.policy_mit_kd_model[idx])) {
            throw std::runtime_error("deploy-derived scalar config contains non-finite values");
        }
        if (config.action_scale[idx] <= 0.0) {
            throw std::runtime_error("actions.JointPositionAction.scale must be > 0");
        }
        if (config.joint_min_rad[idx] > config.joint_max_rad[idx]) {
            throw std::runtime_error("joint_min_rad must be <= joint_max_rad");
        }
        if (config.policy_mit_kp_model[idx] < 0.0 ||
            config.policy_mit_kd_model[idx] < 0.0) {
            throw std::runtime_error("stiffness/damping must be >= 0");
        }
    }
}

template <typename T, std::size_t N>
void printArrayLine(const char* name, const std::array<T, N>& values)
{
    std::cout << name << "=";
    for (std::size_t i = 0; i < N; ++i) {
        if (i != 0) {
            std::cout << ",";
        }
        std::cout << values[i];
    }
    std::cout << "\n";
}

}  // namespace

const std::array<const char*, kPolicyDof> kMotorNamesP1RealOrder{{
    "hip_roll_l_joint",
    "hip_pitch_l_joint",
    "hip_yaw_l_joint",
    "knee_pitch_l_joint",
    "ankle_pitch_l_joint",
    "ankle_roll_l_joint",
    "hip_roll_r_joint",
    "hip_pitch_r_joint",
    "hip_yaw_r_joint",
    "knee_pitch_r_joint",
    "ankle_pitch_r_joint",
    "ankle_roll_r_joint",
}};

void mapControlArrayToMotor(const PolicyConfig& config,
                            const std::array<double, kPolicyDof>& control_values,
                            std::array<double, kPolicyDof>& motor_values)
{
    motor_values.fill(0.0);
    int direct_slot = 0;
    for (int model_index = 0; model_index < kPolicyDof; ++model_index) {
        const auto model_idx = static_cast<std::size_t>(model_index);
        int motor_index = -1;
        if (model_index == config.left_ankle_parallel.model_pitch_dof) {
            motor_index = config.left_ankle_parallel.upper_motor_index;
        } else if (model_index == config.left_ankle_parallel.model_roll_dof) {
            motor_index = config.left_ankle_parallel.lower_motor_index;
        } else if (model_index == config.right_ankle_parallel.model_pitch_dof) {
            motor_index = config.right_ankle_parallel.upper_motor_index;
        } else if (model_index == config.right_ankle_parallel.model_roll_dof) {
            motor_index = config.right_ankle_parallel.lower_motor_index;
        } else if (direct_slot < config.model_to_motor_count) {
            motor_index =
                config.model_to_motor_index[static_cast<std::size_t>(direct_slot++)];
        }

        if (indexInRange(motor_index, kPolicyDof)) {
            motor_values[static_cast<std::size_t>(motor_index)] =
                control_values[model_idx];
        }
    }
}

bool fileReadable(const std::string& path)
{
    std::ifstream file(path, std::ios::binary);
    return file.good();
}

void printUsage(const char* program)
{
    std::cout
        << "Usage: " << program << " [options]\n\n"
        << "P1 single-process MuJoCo runner. It loads the P1 XML, runs the\n"
        << "robot_deploy TorchScript policy, computes MIT torques, and writes\n"
        << "directly to MuJoCo data->ctrl. No Unitree SDK, DDS, p1_ctrl, or g1_ctrl.\n\n"
        << "Options:\n"
        << "  --runner-config PATH  Runtime paths YAML, default simulate/p1_mujoco_deploy.yaml\n"
        << "  --model PATH          MuJoCo XML path, overrides runner config\n"
        << "  --deploy-config PATH  IsaacLab exported deploy.yaml path\n"
        << "  --policy PATH         TorchScript policy path\n"
        << "  --vx VALUE            Parsed for compatibility; runtime command starts at 0\n"
        << "  --vy VALUE            Parsed for compatibility; runtime command starts at 0\n"
        << "  --yaw-rate VALUE      Parsed for compatibility; runtime command starts at 0\n"
        << "  --real-observation-log PATH\n"
        << "                        Drive policy inputs from CSV policy_obs_0..N instead of MuJoCo sensors\n"
        << "  --use-cli-command     Use --vx/--vy/--yaw-rate after policy connects\n"
        << "  --joystick            Enable joystick velocity command input\n"
        << "  --no-joystick         Disable joystick velocity command input\n"
        << "  --joystick-type VALUE Joystick layout: xbox or switch, default xbox\n"
        << "  --joystick-device PATH\n"
        << "                        Linux joystick device, default /dev/input/js0\n"
        << "  --joystick-deadzone VALUE\n"
        << "                        Stick deadzone in 0..0.95, default 0.08\n"
        << "  --joystick-vx-limit VALUE\n"
        << "                        Xbox left-stick forward speed limit in m/s\n"
        << "  --joystick-vy-limit VALUE\n"
        << "                        Xbox left-stick lateral speed limit in m/s\n"
        << "  --joystick-yaw-rate-limit VALUE\n"
        << "                        Xbox right-stick yaw-rate limit in rad/s\n"
        << "  --policy-hz VALUE     Optional override after deploy.yaml step_dt\n"
        << "  --stand-seconds VALUE Ready prompt delay before policy connect, default 2\n"
        << "  --duration VALUE      Stop after seconds; 0 means run until window close/Ctrl+C\n"
        << "  --start-policy        Auto-connect policy once stand delay is reached\n"
        << "  --headless            Run without viewer\n"
        << "  --rope                Enable elastic-rope suspension\n"
        << "  --no-rope             Disable elastic-rope suspension\n"
        << "  --quiet               Disable per-second timing output\n"
        << "  --viewer-overlay      Enable the MuJoCo HUD overlay\n"
        << "  --no-viewer-overlay   Disable the MuJoCo HUD overlay\n"
        << "  --viewer-page VALUE   Default HUD page: summary, joints, imu, all\n"
        << "  --viewer-angle-units VALUE\n"
        << "                        HUD joint/IMU angle units: rad or deg\n"
        << "  --viewer-curve        Enable the joint curve figure\n"
        << "  --no-viewer-curve     Disable the joint curve figure\n"
        << "  --viewer-curve-signal VALUE\n"
        << "                        Curve signal: q, dq, tau, ctrl, q+dq, all\n"
        << "  --viewer-curve-joint VALUE\n"
        << "                        Initial curve joint index, 0..11\n"
        << "  --viewer-curve-window VALUE\n"
        << "                        Curve history window in seconds, default 5\n"
        << "  --motor-delay        Enable delayed MIT command response\n"
        << "  --no-motor-delay     Disable delayed MIT command response\n"
        << "  --motor-delay-min-ms VALUE\n"
        << "                        Minimum sampled motor delay in milliseconds\n"
        << "  --motor-delay-max-ms VALUE\n"
        << "                        Maximum sampled motor delay in milliseconds\n"
        << "  --observation-delay-ms VALUE\n"
        << "                        Delay one observation source before policy inference\n"
        << "  --observation-delay-source VALUE\n"
        << "                        Delayed source: imu or motor, default imu\n"
        << "  --motor-to-model-direction VALUES\n"
        << "                        12 comma values in motor order, each 1 or -1\n"
        << "  --mujoco-joint-direction VALUES\n"
        << "                        12 comma MuJoCo joint signs in motor order, each 1 or -1\n"
        << "  --no-observation-delay\n"
        << "                        Disable IMU/motor observation time offset\n"
        << "  --imu-noise          Enable IMU observation noise from XML sensor noise\n"
        << "  --no-imu-noise       Disable IMU observation noise\n"
        << "  --joint-zero-offset-deg VALUE_OR_LIST\n"
        << "                        Encoder zero bias in motor order; scalar or 12 comma values\n"
        << "  --joint-zero-offset-rad VALUE_OR_LIST\n"
        << "                        Same zero bias in radians\n"
        << "  --log-csv PATH       Write real-robot-style tracking CSV to PATH or directory\n"
        << "  --log-dir PATH       Write timestamped tracking CSV in directory PATH\n"
        << "  --no-log             Disable tracking CSV output\n"
        << "  --print-config        Print resolved config and exit\n"
        << "  --help                Show this help\n\n"
        << "Keyboard:\n"
        << "  7                     Hoist up by shortening the elastic rope\n"
        << "  8                     Lower down by lengthening the elastic rope\n"
        << "  R                     Toggle/cancel elastic rope suspension\n"
        << "  Enter                 Connect policy while keeping rope control active\n"
        << "  0                     Reset velocity command to 0,0,0\n"
        << "  H                     Debug: switch to stand MIT hold under elastic rope\n"
        << "  Space                 Pause/resume simulation\n"
        << "  Backspace             Reset to deploy initial pose\n"
        << "  Esc                   Exit\n\n"
        << "HUD:\n"
        << "  V                     Show/hide telemetry overlay\n"
        << "  Tab                   Cycle HUD page\n"
        << "  C / J / I / M         Summary / joints / IMU / all page\n"
        << "  U                     Toggle rad/deg display units\n\n"
        << "Curve:\n"
        << "  G                     Show/hide joint curve figure\n"
        << "  [ / ]                 Previous/next joint curve\n"
        << "  N                     Cycle curve signal\n"
        << "  X                     Clear curve history\n\n"
        << "Camera:\n"
        << "  1                     Fixed oblique world view\n"
        << "  2                     Fixed side world view\n"
        << "  3                     Fixed rear world view\n"
        << "  4                     Fixed top world view\n"
        << "  F                     Toggle pelvis-follow camera\n"
        << "\nXbox Joystick:\n"
        << "  Left stick Y          vx command after policy is connected\n"
        << "  Left stick X          vy command after policy is connected\n"
        << "  Right stick X         yaw-rate command after policy is connected\n";
}

bool parseArgs(int argc, char** argv, RunnerOptions& options, PolicyConfig& config)
{
    bool policy_hz_override = false;
    double policy_hz_override_value = options.policy_hz;
    bool runner_config_explicit = false;

    auto requireValue = [&](int& index, const std::string& flag) -> std::string {
        if (index + 1 >= argc) {
            throw std::runtime_error("missing value for " + flag);
        }
        ++index;
        return argv[index];
    };

    options.runner_config_path = defaultRunnerConfigPath();
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--runner-config" || arg == "--config") {
            options.runner_config_path = requireValue(i, arg);
            runner_config_explicit = true;
        }
    }

    if (!options.runner_config_path.empty()) {
        if (!fileReadable(options.runner_config_path)) {
            if (runner_config_explicit) {
                throw std::runtime_error("runner config is not readable: " +
                                         options.runner_config_path);
            }
            options.runner_config_path.clear();
        } else {
            loadRunnerConfigDefaults(options.runner_config_path, options, config);
        }
    }

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--help" || arg == "-h") {
            options.show_help = true;
            return true;
        } else if (arg == "--runner-config" || arg == "--config") {
            requireValue(i, arg);
        } else if (arg == "--model") {
            options.model_xml_path = requireValue(i, arg);
        } else if (arg == "--deploy-config") {
            options.deploy_config_path = requireValue(i, arg);
        } else if (arg == "--policy") {
            config.policy_model_path = requireValue(i, arg);
        } else if (arg == "--vx") {
            options.command_vx = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--vy") {
            options.command_vy = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--yaw-rate") {
            options.command_yaw_rate = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--real-observation-log" ||
                   arg == "--real-policy-observation-log") {
            options.real_observation_log_path = requireValue(i, arg);
        } else if (arg == "--use-cli-command" ||
                   arg == "--start-with-cli-command") {
            options.use_cli_command_on_policy_start = true;
        } else if (arg == "--no-use-cli-command" ||
                   arg == "--no-start-with-cli-command") {
            options.use_cli_command_on_policy_start = false;
        } else if (arg == "--joystick") {
            options.joystick_enabled = true;
        } else if (arg == "--no-joystick") {
            options.joystick_enabled = false;
        } else if (arg == "--joystick-type") {
            options.joystick_type = normalizedToken(requireValue(i, arg));
        } else if (arg == "--joystick-device") {
            options.joystick_device = requireValue(i, arg);
        } else if (arg == "--joystick-deadzone") {
            options.joystick_deadzone = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--joystick-vx-limit") {
            options.joystick_limits[0] = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--joystick-vy-limit") {
            options.joystick_limits[1] = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--joystick-yaw-rate-limit") {
            options.joystick_limits[2] = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--policy-hz") {
            options.policy_hz = parseDouble(requireValue(i, arg), arg);
            policy_hz_override = true;
            policy_hz_override_value = options.policy_hz;
        } else if (arg == "--stand-seconds") {
            options.stand_seconds = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--duration") {
            options.duration_seconds = parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--start-policy" || arg == "--auto-start-policy") {
            options.auto_start_policy = true;
        } else if (arg == "--no-start-policy" || arg == "--no-auto-start-policy") {
            options.auto_start_policy = false;
        } else if (arg == "--headless") {
            options.headless = true;
        } else if (arg == "--rope" || arg == "--elastic-rope") {
            options.elastic_rope_enabled = true;
        } else if (arg == "--no-rope" || arg == "--no-elastic-rope") {
            options.elastic_rope_enabled = false;
        } else if (arg == "--quiet") {
            options.print_timing = false;
        } else if (arg == "--viewer-overlay") {
            options.viewer_overlay_enabled = true;
        } else if (arg == "--no-viewer-overlay") {
            options.viewer_overlay_enabled = false;
        } else if (arg == "--viewer-page") {
            options.viewer_overlay_page =
                parseViewerOverlayPage(requireValue(i, arg), arg);
        } else if (arg == "--viewer-angle-units") {
            options.viewer_overlay_degrees =
                parseViewerOverlayUnitsUseDegrees(requireValue(i, arg), arg);
        } else if (arg == "--viewer-curve") {
            options.viewer_curve_enabled = true;
        } else if (arg == "--no-viewer-curve") {
            options.viewer_curve_enabled = false;
        } else if (arg == "--viewer-curve-signal") {
            options.viewer_curve_signal =
                parseViewerCurveSignal(requireValue(i, arg), arg);
        } else if (arg == "--viewer-curve-joint") {
            options.viewer_curve_joint_index = parseInt(requireValue(i, arg), arg);
        } else if (arg == "--viewer-curve-window") {
            options.viewer_curve_window_seconds =
                parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--motor-delay") {
            options.motor_delay_enabled = true;
        } else if (arg == "--no-motor-delay") {
            options.motor_delay_enabled = false;
        } else if (arg == "--motor-delay-min-ms") {
            options.motor_delay_min_seconds =
                parseDouble(requireValue(i, arg), arg) * 0.001;
        } else if (arg == "--motor-delay-max-ms") {
            options.motor_delay_max_seconds =
                parseDouble(requireValue(i, arg), arg) * 0.001;
        } else if (arg == "--observation-delay" ||
                   arg == "--sensor-time-offset") {
            options.observation_delay_enabled = true;
        } else if (arg == "--no-observation-delay" ||
                   arg == "--no-sensor-time-offset") {
            options.observation_delay_enabled = false;
        } else if (arg == "--observation-delay-ms" ||
                   arg == "--sensor-time-offset-ms") {
            options.observation_delay_enabled = true;
            options.observation_delay_seconds =
                parseDouble(requireValue(i, arg), arg) * 0.001;
        } else if (arg == "--observation-delay-seconds" ||
                   arg == "--sensor-time-offset-seconds") {
            options.observation_delay_enabled = true;
            options.observation_delay_seconds =
                parseDouble(requireValue(i, arg), arg);
        } else if (arg == "--observation-delay-source" ||
                   arg == "--delayed-observation-source") {
            options.observation_delay_source =
                parseObservationDelaySource(requireValue(i, arg), arg);
        } else if (arg == "--motor-to-model-direction" ||
                   arg == "--mujoco-motor-to-model-direction") {
            parseIntCommaArray(requireValue(i, arg),
                               arg,
                               options.motor_to_model_direction);
            options.motor_to_model_direction_override = true;
        } else if (arg == "--mujoco-joint-direction" ||
                   arg == "--mujoco-joint-sign") {
            parseIntCommaArray(requireValue(i, arg),
                               arg,
                               options.mujoco_joint_direction);
        } else if (arg == "--imu-noise") {
            options.imu_noise_enabled = true;
        } else if (arg == "--no-imu-noise") {
            options.imu_noise_enabled = false;
        } else if (arg == "--joint-zero-offset-deg" ||
                   arg == "--motor-zero-offset-deg" ||
                   arg == "--encoder-zero-offset-deg") {
            std::array<double, kPolicyDof> values_deg{};
            parseDoubleScalarOrCommaArray(requireValue(i, arg), arg, values_deg);
            for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
                const auto idx = static_cast<std::size_t>(motor_index);
                options.joint_zero_offset_rad[idx] = values_deg[idx] * kDegToRad;
            }
        } else if (arg == "--joint-zero-offset-rad" ||
                   arg == "--motor-zero-offset-rad" ||
                   arg == "--encoder-zero-offset-rad") {
            parseDoubleScalarOrCommaArray(requireValue(i, arg),
                                          arg,
                                          options.joint_zero_offset_rad);
        } else if (arg == "--log-csv" || arg == "--log-dir") {
            options.sim_log_enabled = true;
            options.sim_log_path = requireValue(i, arg);
        } else if (arg == "--log") {
            options.sim_log_enabled = true;
        } else if (arg == "--no-log") {
            options.sim_log_enabled = false;
        } else if (arg == "--print-config") {
            options.print_config = true;
        } else {
            throw std::runtime_error("unknown option: " + arg);
        }
    }

    if (options.model_xml_path.empty()) {
        options.model_xml_path = defaultModelPath();
    }
    bool deploy_config_loaded = false;
    if (!options.deploy_config_path.empty()) {
        if (!fileReadable(options.deploy_config_path)) {
            throw std::runtime_error("deploy config is not readable: " +
                                     options.deploy_config_path);
        }
        loadDeployConfig(options.deploy_config_path, options, config);
        deploy_config_loaded = true;
        if (options.motor_to_model_direction_override) {
            config.motor_to_model_direction =
                options.motor_to_model_direction;
        }
        if (policy_hz_override) {
            options.policy_hz = policy_hz_override_value;
        }
    }
#if !P1_ENABLE_BUILTIN_POLICY_DEFAULTS
    if (!deploy_config_loaded) {
        throw std::runtime_error("deploy config is required. Reconfigure with "
                                 "-DP1_ENABLE_BUILTIN_POLICY_DEFAULTS=ON only "
                                 "if hard-coded policy defaults are intended.");
    }
#endif
    if (options.policy_hz <= 0.0 || !std::isfinite(options.policy_hz)) {
        throw std::runtime_error("--policy-hz must be finite and > 0");
    }
    config.policy_step_dt_s = 1.0 / options.policy_hz;
    if (options.stand_seconds < 0.0 || !std::isfinite(options.stand_seconds)) {
        throw std::runtime_error("--stand-seconds must be finite and >= 0");
    }
    if (options.duration_seconds < 0.0 || !std::isfinite(options.duration_seconds)) {
        throw std::runtime_error("--duration must be finite and >= 0");
    }
    if (options.joystick_type != "xbox" && options.joystick_type != "switch") {
        throw std::runtime_error("--joystick-type must be xbox or switch");
    }
    if (options.joystick_device.empty()) {
        throw std::runtime_error("--joystick-device must not be empty");
    }
    if (options.joystick_bits < 2 || options.joystick_bits > 30) {
        throw std::runtime_error("joystick.bits must be in range 2..30");
    }
    if (options.joystick_deadzone < 0.0 ||
        options.joystick_deadzone > 0.95 ||
        !std::isfinite(options.joystick_deadzone)) {
        throw std::runtime_error("--joystick-deadzone must be finite and in range 0..0.95");
    }
    for (double limit : options.joystick_limits) {
        if (limit < 0.0 || !std::isfinite(limit)) {
            throw std::runtime_error("joystick velocity limits must be finite and >= 0");
        }
    }
    for (double sign : options.joystick_signs) {
        if (!std::isfinite(sign)) {
            throw std::runtime_error("joystick signs must be finite");
        }
    }
    if (options.viewer_curve_joint_index < 0 ||
        options.viewer_curve_joint_index >= kPolicyDof) {
        throw std::runtime_error("--viewer-curve-joint must be in range 0..11");
    }
    if (options.viewer_curve_window_seconds <= 0.0 ||
        !std::isfinite(options.viewer_curve_window_seconds)) {
        throw std::runtime_error("--viewer-curve-window must be finite and > 0");
    }
    if (options.motor_delay_min_seconds < 0.0 ||
        !std::isfinite(options.motor_delay_min_seconds)) {
        throw std::runtime_error("--motor-delay-min-ms must be finite and >= 0");
    }
    if (options.motor_delay_max_seconds < 0.0 ||
        !std::isfinite(options.motor_delay_max_seconds)) {
        throw std::runtime_error("--motor-delay-max-ms must be finite and >= 0");
    }
    if (options.motor_delay_min_seconds > options.motor_delay_max_seconds) {
        throw std::runtime_error("motor delay min must be <= max");
    }
    if (options.observation_delay_seconds < 0.0 ||
        !std::isfinite(options.observation_delay_seconds)) {
        throw std::runtime_error("--observation-delay-ms must be finite and >= 0");
    }
    for (double offset : options.joint_zero_offset_rad) {
        if (!std::isfinite(offset)) {
            throw std::runtime_error("joint zero offsets must be finite");
        }
    }
    for (int direction : options.mujoco_joint_direction) {
        if (direction != 1 && direction != -1) {
            throw std::runtime_error("mujoco_joint_direction values must be 1 or -1");
        }
    }
    validatePolicyConfig(config);
    return true;
}

void printResolvedConfig(const RunnerOptions& options, const PolicyConfig& config)
{
    const int policy_single_observation_size =
        config.include_gait_phase_observation
            ? kPolicySingleObservationSizeWithGait
            : kPolicySingleObservationSizeNoGait;
    const int policy_observation_size =
        policy_single_observation_size * kPolicyFrameStack;
    std::array<double, kPolicyDof> joint_zero_offset_deg{};
    for (int motor_index = 0; motor_index < kPolicyDof; ++motor_index) {
        const auto idx = static_cast<std::size_t>(motor_index);
        joint_zero_offset_deg[idx] = options.joint_zero_offset_rad[idx] / kDegToRad;
    }

    std::cout << "model_xml_path=" << options.model_xml_path << "\n"
              << "runner_config_path=" << options.runner_config_path << "\n"
              << "deploy_config_path=" << options.deploy_config_path << "\n"
              << "policy_model_path=" << config.policy_model_path << "\n"
              << "real_observation_log_path="
              << options.real_observation_log_path << "\n"
              << "builtin_policy_defaults="
              << (P1_ENABLE_BUILTIN_POLICY_DEFAULTS ? "enabled" : "disabled") << "\n"
              << "policy_hz=" << options.policy_hz << "\n"
              << "policy_step_dt_s=" << config.policy_step_dt_s << "\n"
              << "gait_phase_period_s=" << config.gait_phase_period_s << "\n"
              << "gait_phase_stand_threshold="
              << config.gait_phase_stand_threshold << "\n"
              << "gait_phase_move_threshold="
              << config.gait_phase_move_threshold << "\n"
              << "raw_action_clip=" << config.raw_action_clip << "\n"
              << "include_gait_phase_observation="
              << (config.include_gait_phase_observation ? "true" : "false") << "\n"
              << "policy_single_observation_size="
              << policy_single_observation_size << "\n"
              << "policy_frame_stack=" << kPolicyFrameStack << "\n"
              << "policy_observation_size=" << policy_observation_size << "\n"
              << "auto_start_policy="
              << (options.auto_start_policy ? "true" : "false") << "\n"
              << "use_cli_command_on_policy_start="
              << (options.use_cli_command_on_policy_start ? "true" : "false") << "\n"
              << "cli_command="
              << options.command_vx << ","
              << options.command_vy << ","
              << options.command_yaw_rate << "\n"
              << "joystick_enabled="
              << (options.joystick_enabled ? "true" : "false") << "\n"
              << "joystick_type=" << options.joystick_type << "\n"
              << "joystick_device=" << options.joystick_device << "\n"
              << "joystick_bits=" << options.joystick_bits << "\n"
              << "joystick_deadzone=" << options.joystick_deadzone << "\n"
              << "joystick_limits="
              << options.joystick_limits[0] << ","
              << options.joystick_limits[1] << ","
              << options.joystick_limits[2] << "\n"
              << "joystick_signs="
              << options.joystick_signs[0] << ","
              << options.joystick_signs[1] << ","
              << options.joystick_signs[2] << "\n"
              << "initial_joint_pose=deploy_action_offset\n"
              << "elastic_rope_enabled="
              << (options.elastic_rope_enabled ? "true" : "false") << "\n"
              << "override_initial_base_pos="
              << options.initial_base_pos[0] << ","
              << options.initial_base_pos[1] << ","
              << options.initial_base_pos[2] << "\n"
              << "rope_length_m=" << options.rope_length_m << "\n"
              << "rope_auto_length="
              << (options.rope_auto_length ? "true" : "false") << "\n"
              << "rope_support_ratio=" << options.rope_support_ratio << "\n"
              << "rope_length_step_m=" << options.rope_length_step_m << "\n"
              << "rope_length_range_m="
              << options.rope_min_length_m << ","
              << options.rope_max_length_m << "\n"
              << "rope_stiffness=" << options.rope_stiffness << "\n"
              << "rope_damping=" << options.rope_damping << "\n"
              << "rope_anchor_z=" << options.rope_anchor_z << "\n"
              << "viewer_overlay_enabled="
              << (options.viewer_overlay_enabled ? "true" : "false") << "\n"
              << "viewer_overlay_page="
              << viewerOverlayPageName(options.viewer_overlay_page) << "\n"
              << "viewer_angle_units="
              << (options.viewer_overlay_degrees ? "deg" : "rad") << "\n"
              << "viewer_curve_enabled="
              << (options.viewer_curve_enabled ? "true" : "false") << "\n"
              << "viewer_curve_signal="
              << viewerCurveSignalName(options.viewer_curve_signal) << "\n"
              << "viewer_curve_joint_index="
              << options.viewer_curve_joint_index << "\n"
              << "viewer_curve_window_seconds="
              << options.viewer_curve_window_seconds << "\n"
              << "motor_delay_enabled="
              << (options.motor_delay_enabled ? "true" : "false") << "\n"
              << "motor_delay_applies=mit_command\n"
              << "motor_delay_range_ms="
              << options.motor_delay_min_seconds * 1000.0 << ","
              << options.motor_delay_max_seconds * 1000.0 << "\n"
              << "observation_delay_enabled="
              << (options.observation_delay_enabled ? "true" : "false") << "\n"
              << "observation_delay_source="
              << observationDelaySourceName(options.observation_delay_source) << "\n"
              << "observation_delay_ms="
              << options.observation_delay_seconds * 1000.0 << "\n"
              << "imu_noise_enabled="
              << (options.imu_noise_enabled ? "true" : "false") << "\n"
              << "sim_log_enabled="
              << (options.sim_log_enabled ? "true" : "false") << "\n"
              << "sim_log_path=" << options.sim_log_path << "\n";
    printArrayLine("joint_zero_offset_rad", options.joint_zero_offset_rad);
    printArrayLine("joint_zero_offset_deg", joint_zero_offset_deg);
    printArrayLine("control_to_motor_index", config.control_to_motor_index);
    printArrayLine("model_to_motor_index_storage", config.model_to_motor_index);
    std::cout << "model_to_motor_count=" << config.model_to_motor_count << "\n"
              << "left_ankle_parallel="
              << config.left_ankle_parallel.model_pitch_dof << ","
              << config.left_ankle_parallel.model_roll_dof << ","
              << config.left_ankle_parallel.upper_motor_index << ","
              << config.left_ankle_parallel.lower_motor_index << "\n"
              << "right_ankle_parallel="
              << config.right_ankle_parallel.model_pitch_dof << ","
              << config.right_ankle_parallel.model_roll_dof << ","
              << config.right_ankle_parallel.upper_motor_index << ","
              << config.right_ankle_parallel.lower_motor_index << "\n";
    printArrayLine("motor_to_model_direction", config.motor_to_model_direction);
    printArrayLine("mujoco_joint_direction", options.mujoco_joint_direction);
    printArrayLine("stand_pose_rad", config.stand_pose_rad);
    printArrayLine("joint_min_rad", config.joint_min_rad);
    printArrayLine("joint_max_rad", config.joint_max_rad);
    printArrayLine("action_scale", config.action_scale);
    std::cout << "observation_scale_first="
              << (config.observation_scale_first ? "true" : "false") << "\n";
    printArrayLine("projected_gravity_scale", config.projected_gravity_scale);
    printArrayLine("gait_phase_scale", config.gait_phase_scale);
    printArrayLine("last_action_scale", config.last_action_scale);
    std::cout << "base_ang_vel_clip_enabled="
              << (config.base_ang_vel_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("base_ang_vel_clip", config.base_ang_vel_clip);
    std::cout << "projected_gravity_clip_enabled="
              << (config.projected_gravity_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("projected_gravity_clip", config.projected_gravity_clip);
    std::cout << "command_clip_enabled="
              << (config.command_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("command_clip", config.command_clip);
    std::cout << "gait_phase_clip_enabled="
              << (config.gait_phase_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("gait_phase_clip", config.gait_phase_clip);
    std::cout << "joint_pos_rel_clip_enabled="
              << (config.joint_pos_rel_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("joint_pos_rel_clip", config.joint_pos_rel_clip);
    std::cout << "joint_vel_rel_clip_enabled="
              << (config.joint_vel_rel_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("joint_vel_rel_clip", config.joint_vel_rel_clip);
    std::cout << "last_action_clip_enabled="
              << (config.last_action_clip_enabled ? "true" : "false") << "\n";
    printArrayLine("last_action_clip", config.last_action_clip);
    printArrayLine("policy_mit_kp_model", config.policy_mit_kp_model);
    printArrayLine("policy_mit_kd_model", config.policy_mit_kd_model);
}

}  // namespace p1_sim
