#include "mujoco_viewer.h"

#include "p1_config.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdio>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <string>

namespace p1_sim {
namespace {

constexpr double kRadToDeg = 57.2957795130823208768;

const char* overlayPageName(ViewerOverlayPage page)
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

const char* curveSignalName(ViewerCurveSignal signal)
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

double displayAngle(double value, bool use_degrees)
{
    return use_degrees ? value * kRadToDeg : value;
}

std::string fixed(double value, int precision)
{
    std::ostringstream out;
    out << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string signedFixed(double value, int precision)
{
    std::ostringstream out;
    out << std::showpos << std::fixed << std::setprecision(precision) << value;
    return out.str();
}

std::string vector3(const std::array<double, 3>& values,
                    int precision,
                    bool show_sign = true,
                    double scale = 1.0)
{
    std::ostringstream out;
    if (show_sign) {
        out << std::showpos;
    }
    out << std::fixed << std::setprecision(precision)
        << values[0] * scale << " "
        << values[1] * scale << " "
        << values[2] * scale;
    return out.str();
}

std::string vector4(const std::array<double, 4>& values, int precision)
{
    std::ostringstream out;
    out << std::showpos << std::fixed << std::setprecision(precision)
        << values[0] << " "
        << values[1] << " "
        << values[2] << " "
        << values[3];
    return out.str();
}

void appendRow(std::ostringstream& left,
               std::ostringstream& right,
               const std::string& label,
               const std::string& value)
{
    if (left.tellp() > 0) {
        left << '\n';
        right << '\n';
    }
    left << label;
    right << value;
}

std::array<double, 4> normalizedQuat(const std::array<double, 4>& quat)
{
    const double norm = std::sqrt(quat[0] * quat[0] +
                                  quat[1] * quat[1] +
                                  quat[2] * quat[2] +
                                  quat[3] * quat[3]);
    if (norm < 1e-9 || !std::isfinite(norm)) {
        return {1.0, 0.0, 0.0, 0.0};
    }
    return {quat[0] / norm, quat[1] / norm, quat[2] / norm, quat[3] / norm};
}

std::array<double, 3> rpyFromQuat(const std::array<double, 4>& quat_wxyz)
{
    const std::array<double, 4> q = normalizedQuat(quat_wxyz);
    const double w = q[0];
    const double x = q[1];
    const double y = q[2];
    const double z = q[3];

    const double roll = std::atan2(2.0 * (w * x + y * z),
                                   1.0 - 2.0 * (x * x + y * y));
    const double pitch_arg = std::clamp(2.0 * (w * y - z * x), -1.0, 1.0);
    const double pitch = std::asin(pitch_arg);
    const double yaw = std::atan2(2.0 * (w * z + x * y),
                                  1.0 - 2.0 * (y * y + z * z));
    return {roll, pitch, yaw};
}

std::string statusText(const ViewerTelemetry& telemetry)
{
    if (telemetry.paused) {
        return "paused";
    }
    return telemetry.policy_active ? "policy" : "stand";
}

void buildStatusOverlay(const ViewerTelemetry& telemetry,
                        const ViewerOverlayConfig& config,
                        std::string& left_text,
                        std::string& right_text)
{
    std::ostringstream left;
    std::ostringstream right;
    appendRow(left, right, "P1 telemetry", statusText(telemetry));
    appendRow(left, right, "sim time", fixed(telemetry.sim_time, 3) + " s");
    appendRow(left, right, "steps",
              std::to_string(telemetry.sim_steps) + " sim / " +
                  std::to_string(telemetry.policy_steps) + " policy");
    appendRow(left, right, "rt factor", fixed(telemetry.realtime_factor, 2));
    appendRow(left, right, "command vx vy yaw",
              signedFixed(telemetry.command_vx, 2) + " " +
                  signedFixed(telemetry.command_vy, 2) + " " +
                  signedFixed(telemetry.command_yaw_rate, 2));
    appendRow(left, right, "elastic rope",
              std::string(telemetry.rope_enabled ? "on" : "off") +
                  " len=" + fixed(telemetry.rope_length, 3) +
                  " fz=" + signedFixed(telemetry.rope_force_z, 1));
    appendRow(left, right, "quat wxyz", vector4(telemetry.state.quat, 3));
    appendRow(left, right, "gyro xyz",
              vector3(telemetry.state.gyro,
                      3,
                      true,
                      config.use_degrees ? kRadToDeg : 1.0) +
                  (config.use_degrees ? " deg/s" : " rad/s"));
    appendRow(left, right, "gravity xyz",
              vector3(telemetry.state.projected_gravity, 3));
    const int curve_joint =
        std::clamp(config.curve_joint_index, 0, kPolicyDof - 1);
    appendRow(left, right, "joint curve",
              std::string(config.curve_enabled ? "on " : "off ") +
                  kMotorNamesP1RealOrder[static_cast<std::size_t>(curve_joint)] +
                  " " + curveSignalName(config.curve_signal));
    appendRow(left, right, "HUD page", overlayPageName(config.page));
    left_text = left.str();
    right_text = right.str();
}

void buildJointOverlay(const ViewerTelemetry& telemetry,
                       const ViewerOverlayConfig& config,
                       std::string& left_text,
                       std::string& right_text)
{
    std::ostringstream left;
    std::ostringstream right;
    const char* angle_unit = config.use_degrees ? "deg" : "rad";
    const char* velocity_unit = config.use_degrees ? "deg/s" : "rad/s";

    appendRow(left, right,
              std::string("joint state (") + angle_unit + ", " + velocity_unit + ")",
              "q        dq       tau    ctrl");

    for (int i = 0; i < kPolicyDof; ++i) {
        const auto idx = static_cast<std::size_t>(i);
        left << '\n' << std::left << std::setw(22) << kMotorNamesP1RealOrder[idx];
        right << '\n'
              << std::showpos << std::fixed
              << std::setw(8) << std::setprecision(3)
              << displayAngle(telemetry.state.q[idx], config.use_degrees)
              << " "
              << std::setw(8) << std::setprecision(3)
              << displayAngle(telemetry.state.dq[idx], config.use_degrees)
              << " "
              << std::setw(7) << std::setprecision(1)
              << telemetry.state.tau[idx]
              << " "
              << std::setw(7) << std::setprecision(1)
              << telemetry.control[idx];
    }

    left_text = left.str();
    right_text = right.str();
}

void buildImuOverlay(const ViewerTelemetry& telemetry,
                     const ViewerOverlayConfig& config,
                     std::string& left_text,
                     std::string& right_text)
{
    const double angle_scale = config.use_degrees ? kRadToDeg : 1.0;
    const char* angle_unit = config.use_degrees ? "deg" : "rad";
    const char* rate_unit = config.use_degrees ? "deg/s" : "rad/s";
    const std::array<double, 3> rpy = rpyFromQuat(telemetry.state.quat);

    std::ostringstream left;
    std::ostringstream right;
    appendRow(left, right, "IMU", "site imu");
    appendRow(left, right, "quat wxyz", vector4(telemetry.state.quat, 4));
    appendRow(left, right, std::string("rpy xyz (") + angle_unit + ")",
              vector3(rpy, 3, true, angle_scale));
    appendRow(left, right, std::string("gyro xyz (") + rate_unit + ")",
              vector3(telemetry.state.gyro, 3, true, angle_scale));
    appendRow(left, right, "acc xyz (m/s^2)",
              vector3(telemetry.state.accel, 3));
    appendRow(left, right, "projected gravity",
              vector3(telemetry.state.projected_gravity, 3));
    left_text = left.str();
    right_text = right.str();
}

void buildHelpOverlay(const ViewerOverlayConfig& config,
                      std::string& left_text,
                      std::string& right_text)
{
    std::ostringstream left;
    std::ostringstream right;
    appendRow(left, right, "HUD keys", config.enabled ? "visible" : "hidden");
    appendRow(left, right, "V", "show/hide");
    appendRow(left, right, "Tab", "next page");
    appendRow(left, right, "C/J/I/M", "summary/joints/imu/all");
    appendRow(left, right, "U", config.use_degrees ? "deg -> rad" : "rad -> deg");
    appendRow(left, right, "G", config.curve_enabled ? "curve hide" : "curve show");
    appendRow(left, right, "[ / ]", "previous / next joint");
    appendRow(left, right, "N", "next curve signal");
    appendRow(left, right, "X", "clear curve");
    left_text = left.str();
    right_text = right.str();
}

void drawOverlay(mjtGridPos position,
                 const mjrRect& viewport,
                 const std::string& left,
                 const std::string& right,
                 mjrContext* context)
{
    mjr_overlay(mjFONT_NORMAL,
                position,
                viewport,
                left.c_str(),
                right.c_str(),
                context);
}

mjrRect curveViewport(const mjrRect& viewport)
{
    const int width = std::clamp(viewport.width / 3, 360, 560);
    const int height = std::clamp(viewport.height / 3, 230, 360);
    const int margin = 16;
    return {viewport.left + viewport.width - width - margin,
            viewport.bottom + margin,
            width,
            height};
}

void setFigureText(char* target, std::size_t size, const std::string& text)
{
    if (size == 0) {
        return;
    }
    std::snprintf(target, size, "%s", text.c_str());
}

void setFigureLineColor(mjvFigure& figure, int line, float r, float g, float b)
{
    figure.linergb[line][0] = r;
    figure.linergb[line][1] = g;
    figure.linergb[line][2] = b;
}

void configureJointFigure(mjvFigure& figure,
                          const std::string& title,
                          double window_seconds)
{
    mjv_defaultFigure(&figure);
    figure.flg_legend = 1;
    figure.flg_ticklabel[0] = 1;
    figure.flg_ticklabel[1] = 1;
    figure.flg_extend = 0;
    figure.flg_symmetric = 0;
    figure.linewidth = 2.0F;
    figure.gridwidth = 1.0F;
    figure.gridsize[0] = 5;
    figure.gridsize[1] = 5;
    figure.figurergba[3] = 0.72F;
    figure.panergba[3] = 0.88F;
    figure.legendrgba[3] = 0.68F;
    figure.range[0][0] = static_cast<float>(-window_seconds);
    figure.range[0][1] = 0.0F;
    figure.range[1][0] = 1.0F;
    figure.range[1][1] = -1.0F;
    setFigureText(figure.title, sizeof(figure.title), title);
    setFigureText(figure.xlabel, sizeof(figure.xlabel), "time before now (s)");
    setFigureText(figure.xformat, sizeof(figure.xformat), "%.1f");
    setFigureText(figure.yformat, sizeof(figure.yformat), "%.2f");
    setFigureLineColor(figure, 0, 0.35F, 0.68F, 1.00F);
    setFigureLineColor(figure, 1, 1.00F, 0.58F, 0.24F);
    setFigureLineColor(figure, 2, 0.35F, 0.86F, 0.54F);
    setFigureLineColor(figure, 3, 0.95F, 0.38F, 0.50F);
}

using CurveValueGetter = double (*)(const MujocoViewer::CurveSample&, std::size_t);

double curvePosition(const MujocoViewer::CurveSample& sample, std::size_t joint)
{
    return sample.q[joint];
}

double curveVelocity(const MujocoViewer::CurveSample& sample, std::size_t joint)
{
    return sample.dq[joint];
}

double curveTorque(const MujocoViewer::CurveSample& sample, std::size_t joint)
{
    return sample.tau[joint];
}

double curveControl(const MujocoViewer::CurveSample& sample, std::size_t joint)
{
    return sample.control[joint];
}

void addCurveLine(mjvFigure& figure,
                  int line,
                  const std::deque<MujocoViewer::CurveSample>& history,
                  std::size_t joint,
                  double now,
                  double scale,
                  const std::string& name,
                  CurveValueGetter getter)
{
    setFigureText(figure.linename[line], sizeof(figure.linename[line]), name);

    const int available = static_cast<int>(history.size());
    const int points = std::min(available, mjMAXLINEPNT);
    figure.linepnt[line] = points;
    const int first = available - points;
    for (int i = 0; i < points; ++i) {
        const auto& sample = history[static_cast<std::size_t>(first + i)];
        figure.linedata[line][2 * i] = static_cast<float>(sample.time - now);
        figure.linedata[line][2 * i + 1] =
            static_cast<float>(getter(sample, joint) * scale);
    }
}

}  // namespace

bool MujocoViewer::initialize(mjModel* model,
                              mjData* data,
                              GLFWkeyfun keyboard_callback,
                              const ViewerOverlayConfig& overlay_config)
{
    model_ = model;
    data_ = data;
    keyboard_callback_ = keyboard_callback;
    overlay_config_ = overlay_config;
    overlay_config_.curve_joint_index =
        std::clamp(overlay_config_.curve_joint_index, 0, kPolicyDof - 1);
    if (!std::isfinite(overlay_config_.curve_window_seconds) ||
        overlay_config_.curve_window_seconds <= 0.0) {
        overlay_config_.curve_window_seconds = 5.0;
    }
    overlay_config_.curve_window_seconds =
        std::clamp(overlay_config_.curve_window_seconds, 0.5, 30.0);
    curve_history_.clear();
    last_curve_sample_time_ = -1.0;

    if (!glfwInit()) {
        std::cerr << "[ERROR] Could not initialize GLFW.\n";
        return false;
    }

    window_ = glfwCreateWindow(1280, 900, "P1 MuJoCo Deploy Sim", nullptr, nullptr);
    if (!window_) {
        std::cerr << "[ERROR] Could not create GLFW window.\n";
        glfwTerminate();
        return false;
    }

    glfwSetWindowUserPointer(window_, this);
    glfwMakeContextCurrent(window_);
    glfwSwapInterval(1);

    mjv_defaultCamera(&camera_);
    mjv_defaultOption(&visual_options_);
    visual_options_.geomgroup[3] = 0;
    mjv_defaultScene(&scene_);
    mjr_defaultContext(&context_);
    setFreeCamera(135.0, -18.0, 2.4);
    mjv_makeScene(model_, &scene_, 2000);
    mjr_makeContext(model_, &context_, mjFONTSCALE_150);

    glfwSetKeyCallback(window_, keyCallback);
    glfwSetCursorPosCallback(window_, mouseMoveCallback);
    glfwSetMouseButtonCallback(window_, mouseButtonCallback);
    glfwSetScrollCallback(window_, scrollCallback);
    return true;
}

bool MujocoViewer::shouldClose() const
{
    return !window_ || glfwWindowShouldClose(window_);
}

void MujocoViewer::renderFrame(const ViewerTelemetry* telemetry)
{
    if (!window_) {
        return;
    }

    mjrRect viewport = {0, 0, 0, 0};
    glfwGetFramebufferSize(window_, &viewport.width, &viewport.height);
    mjv_updateScene(model_,
                    data_,
                    &visual_options_,
                    nullptr,
                    &camera_,
                    mjCAT_ALL,
                    &scene_);
    mjr_render(viewport, &scene_, &context_);
    if (telemetry) {
        appendCurveSample(*telemetry);
        if (overlay_config_.curve_enabled) {
            renderCurveFigure(viewport, *telemetry);
        }
    }
    renderOverlay(viewport, telemetry);
    glfwSwapBuffers(window_);
    glfwPollEvents();
}

void MujocoViewer::shutdown()
{
    if (!window_) {
        return;
    }
    mjv_freeScene(&scene_);
    mjr_freeContext(&context_);
    glfwDestroyWindow(window_);
    window_ = nullptr;
    glfwTerminate();
}

MujocoViewer* MujocoViewer::fromWindow(GLFWwindow* window)
{
    return static_cast<MujocoViewer*>(glfwGetWindowUserPointer(window));
}

void MujocoViewer::keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods)
{
    if (MujocoViewer* viewer = fromWindow(window)) {
        viewer->handleKey(key, action);
        if (viewer->keyboard_callback_) {
            viewer->keyboard_callback_(window, key, scancode, action, mods);
        }
    }
}

void MujocoViewer::mouseButtonCallback(GLFWwindow* window, int, int, int)
{
    if (MujocoViewer* viewer = fromWindow(window)) {
        viewer->handleMouseButton(window);
    }
}

void MujocoViewer::mouseMoveCallback(GLFWwindow* window, double xpos, double ypos)
{
    if (MujocoViewer* viewer = fromWindow(window)) {
        viewer->handleMouseMove(window, xpos, ypos);
    }
}

void MujocoViewer::scrollCallback(GLFWwindow* window, double, double yoffset)
{
    if (MujocoViewer* viewer = fromWindow(window)) {
        viewer->handleScroll(yoffset);
    }
}

void MujocoViewer::handleKey(int key, int action)
{
    if (action != GLFW_PRESS) {
        return;
    }

    if (key == GLFW_KEY_1 || key == GLFW_KEY_KP_1) {
        setFreeCamera(135.0, -18.0, 2.4);
        std::cout << "[INFO] Camera: fixed oblique world view.\n";
    } else if (key == GLFW_KEY_2 || key == GLFW_KEY_KP_2) {
        setFreeCamera(90.0, -12.0, 2.2);
        std::cout << "[INFO] Camera: fixed side world view.\n";
    } else if (key == GLFW_KEY_3 || key == GLFW_KEY_KP_3) {
        setFreeCamera(180.0, -12.0, 2.2);
        std::cout << "[INFO] Camera: fixed rear world view.\n";
    } else if (key == GLFW_KEY_4 || key == GLFW_KEY_KP_4) {
        setFreeCamera(90.0, -75.0, 2.6);
        std::cout << "[INFO] Camera: fixed top world view.\n";
    } else if (key == GLFW_KEY_F) {
        toggleTrackingCamera();
    } else if (key == GLFW_KEY_V) {
        toggleOverlayEnabled();
    } else if (key == GLFW_KEY_TAB) {
        cycleOverlayPage();
    } else if (key == GLFW_KEY_C) {
        setOverlayPage(ViewerOverlayPage::kSummary);
    } else if (key == GLFW_KEY_J) {
        setOverlayPage(ViewerOverlayPage::kJoints);
    } else if (key == GLFW_KEY_I) {
        setOverlayPage(ViewerOverlayPage::kImu);
    } else if (key == GLFW_KEY_M) {
        setOverlayPage(ViewerOverlayPage::kAll);
    } else if (key == GLFW_KEY_U) {
        toggleOverlayUnits();
    } else if (key == GLFW_KEY_G) {
        toggleCurveEnabled();
    } else if (key == GLFW_KEY_N) {
        cycleCurveSignal();
    } else if (key == GLFW_KEY_LEFT_BRACKET) {
        selectCurveJoint(-1);
    } else if (key == GLFW_KEY_RIGHT_BRACKET) {
        selectCurveJoint(1);
    } else if (key == GLFW_KEY_X) {
        clearCurveHistory();
    }
}

void MujocoViewer::handleMouseButton(GLFWwindow* window)
{
    mouse_left_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_LEFT) == GLFW_PRESS;
    mouse_middle_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_MIDDLE) == GLFW_PRESS;
    mouse_right_ = glfwGetMouseButton(window, GLFW_MOUSE_BUTTON_RIGHT) == GLFW_PRESS;
    glfwGetCursorPos(window, &last_mouse_x_, &last_mouse_y_);
}

void MujocoViewer::handleMouseMove(GLFWwindow* window, double xpos, double ypos)
{
    if (!mouse_left_ && !mouse_middle_ && !mouse_right_) {
        return;
    }

    const double dx = xpos - last_mouse_x_;
    const double dy = ypos - last_mouse_y_;
    last_mouse_x_ = xpos;
    last_mouse_y_ = ypos;

    int width = 0;
    int height = 0;
    glfwGetWindowSize(window, &width, &height);
    if (height <= 0) {
        return;
    }

    const bool shift =
        glfwGetKey(window, GLFW_KEY_LEFT_SHIFT) == GLFW_PRESS ||
        glfwGetKey(window, GLFW_KEY_RIGHT_SHIFT) == GLFW_PRESS;

    mjtMouse action = mjMOUSE_ZOOM;
    if (mouse_right_) {
        action = shift ? mjMOUSE_MOVE_H : mjMOUSE_MOVE_V;
    } else if (mouse_left_) {
        action = shift ? mjMOUSE_ROTATE_H : mjMOUSE_ROTATE_V;
    }

    mjv_moveCamera(model_,
                   action,
                   dx / static_cast<double>(height),
                   dy / static_cast<double>(height),
                   &scene_,
                   &camera_);
}

void MujocoViewer::handleScroll(double yoffset)
{
    mjv_moveCamera(model_,
                   mjMOUSE_ZOOM,
                   0.0,
                   -0.05 * yoffset,
                   &scene_,
                   &camera_);
}

void MujocoViewer::setFreeCamera(double azimuth, double elevation, double distance)
{
    camera_.type = mjCAMERA_FREE;
    camera_.fixedcamid = -1;
    camera_.trackbodyid = -1;
    camera_.azimuth = azimuth;
    camera_.elevation = elevation;
    camera_.distance = distance;
    camera_.lookat[0] = 0.0;
    camera_.lookat[1] = 0.0;
    camera_.lookat[2] = 0.35;
}

void MujocoViewer::setTrackingCamera()
{
    const int pelvis_body_id = mj_name2id(model_, mjOBJ_BODY, "pelvis_link");
    if (pelvis_body_id < 0) {
        std::cerr << "[WARN] Camera tracking unavailable: pelvis_link body not found.\n";
        return;
    }

    camera_.type = mjCAMERA_TRACKING;
    camera_.fixedcamid = -1;
    camera_.trackbodyid = pelvis_body_id;
    camera_.azimuth = 135.0;
    camera_.elevation = -18.0;
    camera_.distance = 2.4;
    std::cout << "[INFO] Camera: tracking pelvis_link. Press F to return to fixed view.\n";
}

void MujocoViewer::toggleTrackingCamera()
{
    if (camera_.type == mjCAMERA_TRACKING) {
        setFreeCamera(135.0, -18.0, 2.4);
        std::cout << "[INFO] Camera: fixed world view. Press F to track pelvis_link.\n";
        return;
    }
    setTrackingCamera();
}

void MujocoViewer::setOverlayPage(ViewerOverlayPage page)
{
    overlay_config_.page = page;
    overlay_config_.enabled = true;
    std::cout << "[INFO] HUD page: " << overlayPageName(page) << ".\n";
}

void MujocoViewer::cycleOverlayPage()
{
    const int page_count = 4;
    const int page = static_cast<int>(overlay_config_.page);
    setOverlayPage(static_cast<ViewerOverlayPage>((page + 1) % page_count));
}

void MujocoViewer::toggleOverlayEnabled()
{
    overlay_config_.enabled = !overlay_config_.enabled;
    std::cout << "[INFO] HUD overlay "
              << (overlay_config_.enabled ? "enabled" : "disabled")
              << ".\n";
}

void MujocoViewer::toggleOverlayUnits()
{
    overlay_config_.use_degrees = !overlay_config_.use_degrees;
    std::cout << "[INFO] HUD angle units: "
              << (overlay_config_.use_degrees ? "deg" : "rad")
              << ".\n";
}

void MujocoViewer::toggleCurveEnabled()
{
    overlay_config_.curve_enabled = !overlay_config_.curve_enabled;
    std::cout << "[INFO] Joint curve "
              << (overlay_config_.curve_enabled ? "enabled" : "disabled")
              << ".\n";
}

void MujocoViewer::cycleCurveSignal()
{
    constexpr int kSignalCount = 6;
    const int signal = static_cast<int>(overlay_config_.curve_signal);
    overlay_config_.curve_signal =
        static_cast<ViewerCurveSignal>((signal + 1) % kSignalCount);
    std::cout << "[INFO] Joint curve signal: "
              << curveSignalName(overlay_config_.curve_signal) << ".\n";
}

void MujocoViewer::selectCurveJoint(int delta)
{
    const int current =
        std::clamp(overlay_config_.curve_joint_index, 0, kPolicyDof - 1);
    overlay_config_.curve_joint_index =
        (current + delta + kPolicyDof) % kPolicyDof;
    clearCurveHistory();
    std::cout << "[INFO] Joint curve selected: "
              << kMotorNamesP1RealOrder[static_cast<std::size_t>(
                     overlay_config_.curve_joint_index)]
              << ".\n";
}

void MujocoViewer::clearCurveHistory()
{
    curve_history_.clear();
    last_curve_sample_time_ = -1.0;
}

void MujocoViewer::appendCurveSample(const ViewerTelemetry& telemetry)
{
    if (telemetry.sim_time <= last_curve_sample_time_ + 1e-12) {
        return;
    }

    CurveSample sample;
    sample.time = telemetry.sim_time;
    sample.q = telemetry.state.q;
    sample.dq = telemetry.state.dq;
    sample.tau = telemetry.state.tau;
    sample.control = telemetry.control;
    curve_history_.push_back(sample);
    last_curve_sample_time_ = telemetry.sim_time;
    trimCurveHistory(telemetry.sim_time);
}

void MujocoViewer::trimCurveHistory(double current_time)
{
    const double window_seconds =
        std::clamp(overlay_config_.curve_window_seconds, 0.5, 30.0);
    const double oldest_time = current_time - window_seconds;
    while (!curve_history_.empty() && curve_history_.front().time < oldest_time) {
        curve_history_.pop_front();
    }
    while (curve_history_.size() > static_cast<std::size_t>(mjMAXLINEPNT)) {
        curve_history_.pop_front();
    }
}

void MujocoViewer::renderCurveFigure(const mjrRect& viewport,
                                     const ViewerTelemetry& telemetry)
{
    if (curve_history_.empty()) {
        return;
    }

    const int joint_index =
        std::clamp(overlay_config_.curve_joint_index, 0, kPolicyDof - 1);
    const auto joint = static_cast<std::size_t>(joint_index);
    const std::string joint_name = kMotorNamesP1RealOrder[joint];
    const std::string title =
        joint_name + "  " + curveSignalName(overlay_config_.curve_signal);
    const double angle_scale = overlay_config_.use_degrees ? kRadToDeg : 1.0;
    const char* angle_unit = overlay_config_.use_degrees ? "deg" : "rad";
    const char* rate_unit = overlay_config_.use_degrees ? "deg/s" : "rad/s";

    mjvFigure figure;
    configureJointFigure(figure, title, overlay_config_.curve_window_seconds);

    int line = 0;
    auto add_position = [&]() {
        addCurveLine(figure,
                     line++,
                     curve_history_,
                     joint,
                     telemetry.sim_time,
                     angle_scale,
                     std::string("q ") + angle_unit,
                     curvePosition);
    };
    auto add_velocity = [&]() {
        addCurveLine(figure,
                     line++,
                     curve_history_,
                     joint,
                     telemetry.sim_time,
                     angle_scale,
                     std::string("dq ") + rate_unit,
                     curveVelocity);
    };
    auto add_torque = [&]() {
        addCurveLine(figure,
                     line++,
                     curve_history_,
                     joint,
                     telemetry.sim_time,
                     1.0,
                     "tau Nm",
                     curveTorque);
    };
    auto add_control = [&]() {
        addCurveLine(figure,
                     line++,
                     curve_history_,
                     joint,
                     telemetry.sim_time,
                     1.0,
                     "ctrl Nm",
                     curveControl);
    };

    switch (overlay_config_.curve_signal) {
    case ViewerCurveSignal::kPosition:
        add_position();
        break;
    case ViewerCurveSignal::kVelocity:
        add_velocity();
        break;
    case ViewerCurveSignal::kTorque:
        add_torque();
        break;
    case ViewerCurveSignal::kControl:
        add_control();
        break;
    case ViewerCurveSignal::kPositionVelocity:
        add_position();
        add_velocity();
        break;
    case ViewerCurveSignal::kAll:
        add_position();
        add_velocity();
        add_torque();
        add_control();
        break;
    }

    if (line == 0) {
        return;
    }
    mjr_figure(curveViewport(viewport), &figure, &context_);
}

void MujocoViewer::renderOverlay(const mjrRect& viewport,
                                 const ViewerTelemetry* telemetry)
{
    if (!overlay_config_.enabled) {
        return;
    }

    std::string left;
    std::string right;
    if (!telemetry) {
        buildHelpOverlay(overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMLEFT, viewport, left, right, &context_);
        return;
    }

    switch (overlay_config_.page) {
    case ViewerOverlayPage::kSummary:
        buildStatusOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_TOPLEFT, viewport, left, right, &context_);
        buildHelpOverlay(overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMLEFT, viewport, left, right, &context_);
        break;
    case ViewerOverlayPage::kJoints:
        buildJointOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_TOPLEFT, viewport, left, right, &context_);
        buildHelpOverlay(overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMLEFT, viewport, left, right, &context_);
        break;
    case ViewerOverlayPage::kImu:
        buildImuOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_TOPLEFT, viewport, left, right, &context_);
        buildHelpOverlay(overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMLEFT, viewport, left, right, &context_);
        break;
    case ViewerOverlayPage::kAll:
        buildStatusOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_TOPLEFT, viewport, left, right, &context_);
        buildJointOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_TOPRIGHT, viewport, left, right, &context_);
        buildImuOverlay(*telemetry, overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMLEFT, viewport, left, right, &context_);
        buildHelpOverlay(overlay_config_, left, right);
        drawOverlay(mjGRID_BOTTOMRIGHT, viewport, left, right, &context_);
        break;
    }
}

}  // namespace p1_sim
