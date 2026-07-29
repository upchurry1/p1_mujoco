#include "elastic_rope.h"
#include "mujoco_p1_backend.h"
#include "mujoco_viewer.h"
#include "p1_config.h"
#include "p1_policy_controller.h"
#include "joystick/p1_joystick_velocity.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <cmath>
#include <csignal>
#include <cstdint>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>

namespace {

using namespace p1_sim;

std::atomic<bool> g_stop_requested{false};
std::atomic<bool> g_reset_requested{false};
std::atomic<bool> g_paused{false};
std::atomic<bool> g_start_policy_requested{false};
std::atomic<bool> g_stand_suspend_requested{false};
std::atomic<int> g_rope_length_delta_ticks{0};
std::atomic<bool> g_toggle_rope_requested{false};
std::atomic<bool> g_zero_command_requested{false};

void signalHandler(int)
{
    g_stop_requested.store(true);
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

void keyboard(GLFWwindow*, int key, int, int action, int)
{
    if (action == GLFW_PRESS || action == GLFW_REPEAT) {
        if (key == GLFW_KEY_7 || key == GLFW_KEY_KP_7) {
            g_rope_length_delta_ticks.fetch_add(-1);
            return;
        }
        if (key == GLFW_KEY_8 || key == GLFW_KEY_KP_8) {
            g_rope_length_delta_ticks.fetch_add(1);
            return;
        }
    }

    if (action != GLFW_PRESS) {
        return;
    }
    if (key == GLFW_KEY_BACKSPACE) {
        g_reset_requested.store(true);
    } else if (key == GLFW_KEY_SPACE) {
        g_paused.store(!g_paused.load());
    } else if (key == GLFW_KEY_ENTER || key == GLFW_KEY_KP_ENTER) {
        g_start_policy_requested.store(true);
    } else if (key == GLFW_KEY_R) {
        g_toggle_rope_requested.store(true);
    } else if (key == GLFW_KEY_0 || key == GLFW_KEY_KP_0) {
        g_zero_command_requested.store(true);
    } else if (key == GLFW_KEY_H) {
        g_stand_suspend_requested.store(true);
    } else if (key == GLFW_KEY_ESCAPE) {
        g_stop_requested.store(true);
    }
}

class P1MujocoDeployApp {
public:
    int run(int argc, char** argv)
    {
        const int config_result = configure(argc, argv);
        if (config_result >= 0) {
            return config_result;
        }

        if (!validateInputFiles() || !loadModel() || !initializeRuntime()) {
            cleanup();
            return 1;
        }

        printStartupInfo();
        const int result = options_.headless ? runHeadlessLoop() : runViewerLoop();
        printExitInfo();
        cleanup();
        return result;
    }

private:
    RunnerOptions options_;
    PolicyConfig config_;
    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    std::unique_ptr<MujocoP1Backend> backend_;
    std::unique_ptr<P1RealDeployPolicy> policy_;
    P1JoystickVelocityReader joystick_velocity_;
    ElasticRope elastic_rope_;
    int rope_body_id_ = -1;
    bool rope_enabled_ = false;
    double policy_period_ = 0.02;
    double next_policy_time_ = 0.0;
    bool policy_active_ = false;
    bool stand_ready_reported_ = false;
    bool joystick_available_ = false;
    double command_vx_ = 0.0;
    double command_vy_ = 0.0;
    double command_yaw_rate_ = 0.0;
    std::uint64_t sim_steps_ = 0;
    std::uint64_t policy_steps_ = 0;
    std::chrono::steady_clock::time_point wall_start_;
    std::chrono::steady_clock::time_point last_report_;
    P1StateSnapshot last_state_;
    std::array<float, kPolicyDof> last_action_{};
    std::array<double, kPolicyDof> stand_motor_{};

    int configure(int argc, char** argv)
    {
        try {
            parseArgs(argc, argv, options_, config_);
        } catch (const std::exception& error) {
            std::cerr << "[ERROR] " << error.what() << "\n";
            printUsage(argv[0]);
            return 2;
        }

        if (options_.show_help) {
            printUsage(argv[0]);
            return 0;
        }

        if (options_.print_config) {
            printResolvedConfig(options_, config_);
            return 0;
        }
        return -1;
    }

    bool validateInputFiles() const
    {
        if (options_.model_xml_path.empty()) {
            std::cerr << "[ERROR] MuJoCo XML path is empty. Set model_xml_path in "
                      << "simulate/p1_mujoco_deploy.yaml or pass --model.\n";
            return false;
        }
        if (config_.policy_model_path.empty()) {
            std::cerr << "[ERROR] Policy model path is empty. Set policy_model_path in "
                      << "simulate/p1_mujoco_deploy.yaml or pass --policy.\n";
            return false;
        }
        if (!fileReadable(options_.model_xml_path)) {
            std::cerr << "[ERROR] MuJoCo XML is not readable: "
                      << options_.model_xml_path << "\n";
            return false;
        }
        if (!fileReadable(config_.policy_model_path)) {
            std::cerr << "[ERROR] Policy model is not readable: "
                      << config_.policy_model_path << "\n";
            return false;
        }
        return true;
    }

    bool loadModel()
    {
        char load_error[1024] = "Could not load XML";
        model_ = mj_loadXML(options_.model_xml_path.c_str(),
                            nullptr,
                            load_error,
                            sizeof(load_error));
        if (!model_) {
            std::cerr << "[ERROR] Load MuJoCo XML failed: " << load_error << "\n";
            return false;
        }

        data_ = mj_makeData(model_);
        if (!data_) {
            std::cerr << "[ERROR] mj_makeData failed.\n";
            return false;
        }

        return true;
    }

    bool initializeRuntime()
    {
        backend_ = std::make_unique<MujocoP1Backend>(model_, data_);
        if (!backend_->initialize()) {
            return false;
        }
        backend_->configureMotorDelay(options_.motor_delay_enabled,
                                      options_.motor_delay_min_seconds,
                                      options_.motor_delay_max_seconds);
        backend_->setMotorDelayActive(false);

        policy_ = std::make_unique<P1RealDeployPolicy>(config_);
        if (!policy_->load()) {
            return false;
        }
        configureJoystick();

        policy_->buildStandMotorTarget(stand_motor_);
        applyInitialPose();
        configureElasticRope();
        applyStandMitTargets();

        policy_period_ = 1.0 / options_.policy_hz;
        next_policy_time_ = 0.0;
        policy_active_ = false;
        stand_ready_reported_ = false;
        command_vx_ = 0.0;
        command_vy_ = 0.0;
        command_yaw_rate_ = 0.0;
        sim_steps_ = 0;
        policy_steps_ = 0;
        wall_start_ = std::chrono::steady_clock::now();
        last_report_ = wall_start_;
        backend_->readState(last_state_);
        return true;
    }

    void configureJoystick()
    {
        P1JoystickVelocityOptions joystick_options;
        joystick_options.enabled = options_.joystick_enabled;
        joystick_options.type = options_.joystick_type;
        joystick_options.device = options_.joystick_device;
        joystick_options.bits = options_.joystick_bits;
        joystick_options.deadzone = options_.joystick_deadzone;
        joystick_options.limits = options_.joystick_limits;
        joystick_options.signs = options_.joystick_signs;
        joystick_available_ = joystick_velocity_.open(joystick_options);
    }

    void applyInitialPose()
    {
        backend_->setJointPositionsFromMotorTarget(stand_motor_);
        backend_->setRootPose(options_.initial_base_pos,
                              options_.initial_base_quat);
    }

    void configureElasticRope()
    {
        elastic_rope_.length = options_.rope_length_m;
        elastic_rope_.stiffness = options_.rope_stiffness;
        elastic_rope_.damping = options_.rope_damping;
        elastic_rope_.anchor_z = options_.rope_anchor_z;
        elastic_rope_.side_offset_y = options_.rope_side_offset_y;
        elastic_rope_.local_attach_z = options_.rope_local_attach_z;

        rope_body_id_ = mj_name2id(model_, mjOBJ_BODY, "pelvis_link");
        rope_enabled_ = options_.elastic_rope_enabled && rope_body_id_ >= 0;
        if (options_.elastic_rope_enabled && rope_body_id_ < 0) {
            std::cerr << "[WARN] pelvis_link body not found; elastic rope is disabled.\n";
        }
        updateAutoRopeLength();
    }

    void printStartupInfo() const
    {
        std::cout << "[INFO] P1 single-process MuJoCo policy runner started.\n"
                  << "[INFO] model=" << options_.model_xml_path << "\n"
                  << "[INFO] runner_config=" << options_.runner_config_path << "\n"
                  << "[INFO] policy=" << config_.policy_model_path << "\n"
                  << "[INFO] timestep=" << model_->opt.timestep
                  << " policy_hz=" << options_.policy_hz
                  << " stand_seconds=" << options_.stand_seconds
                  << " cli_command=(" << options_.command_vx << ", "
                  << options_.command_vy << ", "
                  << options_.command_yaw_rate << ")"
                  << " runtime_command_initial=(0, 0, 0)"
                  << " gravity_z=" << model_->opt.gravity[2]
                  << " initial_joint_pose=deploy_action_offset"
                  << " elastic_rope=" << (rope_enabled_ ? "on" : "off")
                  << " override_initial_base_pos=("
                  << options_.initial_base_pos[0] << ", "
                  << options_.initial_base_pos[1] << ", "
                  << options_.initial_base_pos[2] << ")"
                  << " rope_length=" << elastic_rope_.length
                  << " rope_auto_length="
                  << (options_.rope_auto_length ? "true" : "false")
                  << " rope_support_ratio=" << options_.rope_support_ratio
                  << " rope_step=" << options_.rope_length_step_m
                  << " rope_length_range=(" << options_.rope_min_length_m
                  << ", " << options_.rope_max_length_m << ")"
                  << " rope_k=" << elastic_rope_.stiffness
                  << " rope_d=" << elastic_rope_.damping
                  << " joystick_velocity="
                  << (options_.joystick_enabled
                          ? (joystick_available_ ? "available" : "missing")
                          : "disabled")
                  << " joystick_type=" << options_.joystick_type
                  << " joystick_device=" << options_.joystick_device
                  << " joystick_deadzone=" << options_.joystick_deadzone
                  << " joystick_limits=("
                  << options_.joystick_limits[0] << ", "
                  << options_.joystick_limits[1] << ", "
                  << options_.joystick_limits[2] << ")"
                  << " viewer_overlay="
                  << (options_.viewer_overlay_enabled ? "on" : "off")
                  << " viewer_page="
                  << viewerOverlayPageName(options_.viewer_overlay_page)
                  << " viewer_angle_units="
                  << (options_.viewer_overlay_degrees ? "deg" : "rad")
                  << " viewer_curve="
                  << (options_.viewer_curve_enabled ? "on" : "off")
                  << " viewer_curve_signal="
                  << viewerCurveSignalName(options_.viewer_curve_signal)
                  << " viewer_curve_joint="
                  << options_.viewer_curve_joint_index
                  << " viewer_curve_window="
                  << options_.viewer_curve_window_seconds
                  << " motor_delay="
                  << (options_.motor_delay_enabled ? "on" : "off")
                  << " motor_delay_applies=policy"
                  << " motor_delay_range_ms=("
                  << options_.motor_delay_min_seconds * 1000.0
                  << ", "
                  << options_.motor_delay_max_seconds * 1000.0
                  << ")"
                  << "\n";
    }

    void zeroVelocityCommand()
    {
        command_vx_ = 0.0;
        command_vy_ = 0.0;
        command_yaw_rate_ = 0.0;
        g_zero_command_requested.store(false);
    }

    void resetSimulation()
    {
        mj_resetData(model_, data_);
        policy_->resetPolicyState();
        backend_->resetMotorDelay();
        backend_->setMotorDelayActive(false);
        applyInitialPose();
        elastic_rope_.length = options_.rope_length_m;
        rope_enabled_ = options_.elastic_rope_enabled && rope_body_id_ >= 0;
        updateAutoRopeLength();
        elastic_rope_.clear(data_, rope_body_id_);
        applyStandMitTargets();
        next_policy_time_ = 0.0;
        policy_active_ = false;
        zeroVelocityCommand();
        stand_ready_reported_ = false;
        sim_steps_ = 0;
        policy_steps_ = 0;
        wall_start_ = std::chrono::steady_clock::now();
        last_report_ = wall_start_;
        g_reset_requested.store(false);
        g_toggle_rope_requested.store(false);
        std::cout << "[INFO] Simulation reset to initial suspended pose. "
                  << "Command reset to (" << command_vx_ << ", "
                  << command_vy_ << ", " << command_yaw_rate_ << ").\n";
    }

    void applyElasticRope()
    {
        elastic_rope_.clear(data_, rope_body_id_);
        if (rope_enabled_) {
            elastic_rope_.advance(model_, data_, rope_body_id_);
            elastic_rope_.apply(data_, rope_body_id_);
        }
    }

    void startPolicy()
    {
        if (policy_active_) {
            return;
        }

        policy_->resetPolicyState();
        zeroVelocityCommand();
        backend_->setMotorDelayActive(true);
        next_policy_time_ = data_->time;
        policy_active_ = true;
        updateJoystickVelocityCommand();
        std::cout << "[INFO] Policy connected at t=" << data_->time << " s. "
                  << "elastic rope=" << (rope_enabled_ ? "enabled" : "disabled")
                  << ", length=" << elastic_rope_.length << " m.\n";
    }

    void applyStandMitTargets()
    {
        backend_->setMitTargets(stand_motor_,
                                policy_->policyKpMotor(),
                                policy_->policyKdMotor());
    }

    void returnToElasticStand()
    {
        policy_active_ = false;
        policy_->resetPolicyState();
        zeroVelocityCommand();
        backend_->setMotorDelayActive(false);
        applyStandMitTargets();
        rope_enabled_ = options_.elastic_rope_enabled && rope_body_id_ >= 0;
        stand_ready_reported_ = false;
        std::cout << "[INFO] Returned to elastic-rope stand hold at t="
                  << data_->time << " s, rope_length="
                  << elastic_rope_.length << " m.\n";
    }

    void handleInputRequests()
    {
        if (g_toggle_rope_requested.exchange(false)) {
            handleRopeToggle();
        }
        handleRopeLengthTicks();

        bool command_changed = false;
        if (g_zero_command_requested.exchange(false)) {
            command_vx_ = 0.0;
            command_vy_ = 0.0;
            command_yaw_rate_ = 0.0;
            command_changed = true;
        }

        if (policy_active_) {
            command_changed = updateJoystickVelocityCommand() || command_changed;
        }
        if (command_changed && sim_steps_ % 100 == 0) {
            printVelocityCommand();
        }

        if (g_stand_suspend_requested.exchange(false)) {
            returnToElasticStand();
        }
        if (g_start_policy_requested.exchange(false)) {
            startPolicy();
        }
    }

    void handleRopeToggle()
    {
        if (options_.elastic_rope_enabled && rope_body_id_ >= 0) {
            rope_enabled_ = !rope_enabled_;
            elastic_rope_.clear(data_, rope_body_id_);
            std::cout << "[INFO] Elastic rope "
                      << (rope_enabled_ ? "enabled" : "disabled")
                      << " at t=" << data_->time
                      << " s, rope_length=" << elastic_rope_.length << " m.\n";
        } else {
            rope_enabled_ = false;
            std::cout << "[WARN] Elastic rope cannot be enabled; pelvis_link or option missing.\n";
        }
    }

    void handleRopeLengthTicks()
    {
        const int rope_ticks = g_rope_length_delta_ticks.exchange(0);
        if (rope_ticks == 0) {
            return;
        }

        rope_enabled_ = options_.elastic_rope_enabled && rope_body_id_ >= 0;
        const double old_length = elastic_rope_.length;
        elastic_rope_.length = std::clamp(elastic_rope_.length +
                                          static_cast<double>(rope_ticks) *
                                          options_.rope_length_step_m,
                                          options_.rope_min_length_m,
                                          options_.rope_max_length_m);
        if (std::abs(elastic_rope_.length - old_length) > 1e-9) {
            std::cout << "[INFO] Rope length=" << std::fixed
                      << std::setprecision(3) << elastic_rope_.length
                      << " m at t=" << data_->time << " s.\n";
        }
    }

    double robotMass() const
    {
        double mass = 0.0;
        for (int body_id = 1; body_id < model_->nbody; ++body_id) {
            mass += model_->body_mass[body_id];
        }
        return mass;
    }

    void updateAutoRopeLength()
    {
        if (!options_.rope_auto_length || !rope_enabled_ ||
            rope_body_id_ < 0 || elastic_rope_.stiffness <= 1e-9) {
            return;
        }

        mj_forward(model_, data_);
        const double distance = elastic_rope_.averageDistanceToAnchors(
            model_, data_, rope_body_id_);
        if (distance <= 1e-9) {
            return;
        }

        const double support_ratio =
            std::clamp(options_.rope_support_ratio, 0.0, 1.5);
        const double gravity = std::abs(model_->opt.gravity[2]);
        const double target_force = support_ratio * robotMass() * gravity;
        const double target_extension = target_force / elastic_rope_.stiffness;
        elastic_rope_.length = std::clamp(distance - target_extension,
                                          options_.rope_min_length_m,
                                          options_.rope_max_length_m);
    }

    bool updateJoystickVelocityCommand()
    {
        if (!joystick_available_) {
            return false;
        }

        const double old_vx = command_vx_;
        const double old_vy = command_vy_;
        const double old_yaw_rate = command_yaw_rate_;
        joystick_velocity_.update(command_vx_, command_vy_, command_yaw_rate_);

        return std::abs(command_vx_ - old_vx) > 1e-4 ||
               std::abs(command_vy_ - old_vy) > 1e-4 ||
               std::abs(command_yaw_rate_ - old_yaw_rate) > 1e-4;
    }

    void printVelocityCommand() const
    {
        std::cout << std::fixed << std::setprecision(3)
                  << "[INFO] Command vx=" << command_vx_
                  << " vy=" << command_vy_
                  << " yaw_rate=" << command_yaw_rate_
                  << " at t=" << data_->time << " s.\n";
    }

    void reportTiming()
    {
        if (!options_.print_timing) {
            return;
        }
        const auto now = std::chrono::steady_clock::now();
        if (now - last_report_ < std::chrono::seconds(1)) {
            return;
        }
        const double wall_elapsed =
            std::chrono::duration<double>(now - wall_start_).count();
        const double realtime_factor =
            wall_elapsed > 1e-9 ? data_->time / wall_elapsed : 0.0;
        std::cout << std::fixed << std::setprecision(3)
                  << "[INFO] sim_t=" << data_->time
                  << " steps=" << sim_steps_
                  << " policy_steps=" << policy_steps_
                  << " rtf=" << realtime_factor
                  << " rope=" << (rope_enabled_ ? "true" : "false")
                  << " policy=" << (policy_active_ ? "true" : "false")
                  << " rope_len=" << elastic_rope_.length
                  << " rope_fz=" << elastic_rope_.force_world[2]
                  << " cmd=(" << command_vx_ << ", "
                  << command_vy_ << ", "
                  << command_yaw_rate_ << ")"
                  << " q0=" << last_state_.q[0]
                  << " ctrl0=" << backend_->controlOfMotor(0)
                  << " action0=" << last_action_[0] << "\n";
        last_report_ = now;
    }

    bool runOneStep()
    {
        if (g_reset_requested.load()) {
            resetSimulation();
        }
        handleInputRequests();

        if (!backend_->readState(last_state_)) {
            std::cerr << "[ERROR] Failed to read MuJoCo P1 state.\n";
            return false;
        }

        maybeReportStandReady();
        if (policy_active_ && data_->time + 1e-12 >= next_policy_time_) {
            if (!runPolicyStep()) {
                return false;
            }
        }

        backend_->applyMitTorques();
        applyElasticRope();
        mj_step(model_, data_);
        ++sim_steps_;
        reportTiming();

        if (options_.duration_seconds > 0.0 &&
            data_->time >= options_.duration_seconds) {
            g_stop_requested.store(true);
        }
        return !g_stop_requested.load();
    }

    void maybeReportStandReady()
    {
        if (policy_active_ || stand_ready_reported_ ||
            data_->time < options_.stand_seconds) {
            return;
        }

        stand_ready_reported_ = true;
        std::cout << "[INFO] Deploy initial pose with elastic rope is ready at t="
                  << data_->time
                  << " s. Press Enter to connect policy. "
                  << "Use 7/8 for rope length and R to toggle rope. "
                  << "Use Xbox left stick for vx/vy, right stick X for yaw, "
                  << "0 to clear command.\n";
    }

    bool runPolicyStep()
    {
        std::array<double, kPolicyDof> target_motor_rad{};
        if (!policy_->step(last_state_,
                           command_vx_,
                           command_vy_,
                           command_yaw_rate_,
                           target_motor_rad,
                           last_action_)) {
            return false;
        }

        backend_->setMitTargets(target_motor_rad,
                                policy_->policyKpMotor(),
                                policy_->policyKdMotor());
        next_policy_time_ += policy_period_;
        ++policy_steps_;
        return true;
    }

    int runViewerLoop()
    {
        MujocoViewer viewer;
        ViewerOverlayConfig overlay_config;
        overlay_config.enabled = options_.viewer_overlay_enabled;
        overlay_config.page = options_.viewer_overlay_page;
        overlay_config.use_degrees = options_.viewer_overlay_degrees;
        overlay_config.curve_enabled = options_.viewer_curve_enabled;
        overlay_config.curve_signal = options_.viewer_curve_signal;
        overlay_config.curve_joint_index = options_.viewer_curve_joint_index;
        overlay_config.curve_window_seconds =
            options_.viewer_curve_window_seconds;
        if (!viewer.initialize(model_, data_, keyboard, overlay_config)) {
            return 1;
        }

        bool running = true;
        bool step_error = false;
        while (!viewer.shouldClose() &&
               !g_stop_requested.load() &&
               running) {
            if (!g_paused.load()) {
                const mjtNum render_start = data_->time;
                while (data_->time - render_start < 1.0 / 60.0) {
                    if (!runOneStep()) {
                        step_error = !g_stop_requested.load();
                        running = false;
                        break;
                    }
                }
            }

            const ViewerTelemetry telemetry = buildViewerTelemetry();
            viewer.renderFrame(&telemetry);
        }

        viewer.shutdown();
        return step_error ? 1 : 0;
    }

    int runHeadlessLoop()
    {
        while (!g_stop_requested.load()) {
            if (!runOneStep()) {
                return g_stop_requested.load() ? 0 : 1;
            }
        }
        return 0;
    }

    ViewerTelemetry buildViewerTelemetry() const
    {
        ViewerTelemetry telemetry;
        telemetry.sim_time = data_ ? data_->time : 0.0;
        telemetry.sim_steps = sim_steps_;
        telemetry.policy_steps = policy_steps_;
        telemetry.paused = g_paused.load();
        telemetry.policy_active = policy_active_;
        telemetry.rope_enabled = rope_enabled_;
        telemetry.rope_length = elastic_rope_.length;
        telemetry.rope_force_z = elastic_rope_.force_world[2];
        telemetry.command_vx = command_vx_;
        telemetry.command_vy = command_vy_;
        telemetry.command_yaw_rate = command_yaw_rate_;
        telemetry.state = last_state_;
        telemetry.action = last_action_;

        const auto now = std::chrono::steady_clock::now();
        const double wall_elapsed =
            std::chrono::duration<double>(now - wall_start_).count();
        telemetry.realtime_factor =
            wall_elapsed > 1e-9 ? telemetry.sim_time / wall_elapsed : 0.0;

        if (backend_) {
            backend_->readState(telemetry.state);
            for (int i = 0; i < kPolicyDof; ++i) {
                telemetry.control[static_cast<std::size_t>(i)] =
                    backend_->controlOfMotor(i);
            }
        }
        return telemetry;
    }

    void printExitInfo() const
    {
        const double final_sim_time = data_ ? data_->time : 0.0;
        std::cout << "[INFO] Exiting P1 MuJoCo deploy sim. sim_t="
                  << std::fixed << std::setprecision(3) << final_sim_time
                  << " policy_steps=" << policy_steps_
                  << " cmd=(" << command_vx_ << ", "
                  << command_vy_ << ", "
                  << command_yaw_rate_ << ")\n";
    }

    void cleanup()
    {
        backend_.reset();
        policy_.reset();
        if (data_) {
            mj_deleteData(data_);
            data_ = nullptr;
        }
        if (model_) {
            mj_deleteModel(model_);
            model_ = nullptr;
        }
    }
};

}  // namespace

int main(int argc, char** argv)
{
    std::signal(SIGINT, signalHandler);
    std::signal(SIGTERM, signalHandler);

    P1MujocoDeployApp app;
    return app.run(argc, argv);
}
