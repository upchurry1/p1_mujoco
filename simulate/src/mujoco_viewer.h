#pragma once

#include "p1_types.h"

#include <GLFW/glfw3.h>
#include <mujoco/mujoco.h>

#include <array>
#include <cstdint>
#include <deque>
#include <vector>

namespace p1_sim {

struct ViewerOverlayConfig {
    bool enabled = true;
    ViewerOverlayPage page = ViewerOverlayPage::kSummary;
    bool use_degrees = false;
    bool curve_enabled = true;
    ViewerCurveSignal curve_signal = ViewerCurveSignal::kPositionVelocity;
    int curve_joint_index = 0;
    double curve_window_seconds = 5.0;
};

struct ViewerTelemetry {
    double sim_time = 0.0;
    double realtime_factor = 0.0;
    std::uint64_t sim_steps = 0;
    std::uint64_t policy_steps = 0;
    bool paused = false;
    bool policy_active = false;
    bool rope_enabled = false;
    double rope_length = 0.0;
    double rope_force_z = 0.0;
    double command_vx = 0.0;
    double command_vy = 0.0;
    double command_yaw_rate = 0.0;
    P1StateSnapshot state{};
    std::array<float, kPolicyDof> action{};
    std::array<double, kPolicyDof> control{};
};

struct ViewerFootContactTelemetry {
    bool available = false;
    bool left_touching = false;
    bool right_touching = false;
    int left_contacts = 0;
    int right_contacts = 0;
    double left_normal_force = 0.0;
    double right_normal_force = 0.0;
};

class MujocoViewer {
public:
    struct CurveSample {
        double time = 0.0;
        std::array<double, kPolicyDof> q{};
        std::array<double, kPolicyDof> dq{};
        std::array<double, kPolicyDof> tau{};
        std::array<double, kPolicyDof> control{};
    };

    bool initialize(mjModel* model,
                    mjData* data,
                    GLFWkeyfun keyboard_callback,
                    const ViewerOverlayConfig& overlay_config);
    bool shouldClose() const;
    void renderFrame(const ViewerTelemetry* telemetry);
    void shutdown();

private:
    static MujocoViewer* fromWindow(GLFWwindow* window);
    static void keyCallback(GLFWwindow* window, int key, int scancode, int action, int mods);
    static void mouseButtonCallback(GLFWwindow* window, int button, int action, int mods);
    static void mouseMoveCallback(GLFWwindow* window, double xpos, double ypos);
    static void scrollCallback(GLFWwindow* window, double xoffset, double yoffset);

    void handleKey(int key, int action);
    void handleMouseButton(GLFWwindow* window);
    void handleMouseMove(GLFWwindow* window, double xpos, double ypos);
    void handleScroll(double yoffset);
    void setFreeCamera(double azimuth, double elevation, double distance);
    void setTrackingCamera();
    void toggleTrackingCamera();
    void setOverlayPage(ViewerOverlayPage page);
    void cycleOverlayPage();
    void toggleOverlayEnabled();
    void toggleOverlayUnits();
    void toggleCurveEnabled();
    void cycleCurveSignal();
    void selectCurveJoint(int delta);
    void clearCurveHistory();
    void appendCurveSample(const ViewerTelemetry& telemetry);
    void trimCurveHistory(double current_time);
    void renderCurveFigure(const mjrRect& viewport, const ViewerTelemetry& telemetry);
    void renderOverlay(const mjrRect& viewport, const ViewerTelemetry* telemetry);
    void cacheFootGeomIds();
    int footSideForGeom(int geom_id) const;
    bool isWorldContactGeom(int geom_id) const;
    void renderFootContactVisualization();
    void addContactSphere(const mjtNum pos[3],
                          const float rgba[4],
                          double radius);
    void addContactArrow(const mjtNum pos[3],
                         const mjtNum normal[3],
                         const float rgba[4],
                         double normal_force);

    mjModel* model_ = nullptr;
    mjData* data_ = nullptr;
    GLFWwindow* window_ = nullptr;
    GLFWkeyfun keyboard_callback_ = nullptr;
    ViewerOverlayConfig overlay_config_{};
    ViewerFootContactTelemetry foot_contacts_{};
    std::vector<int> left_foot_geom_ids_;
    std::vector<int> right_foot_geom_ids_;
    std::deque<CurveSample> curve_history_;
    double last_curve_sample_time_ = -1.0;
    mjvCamera camera_{};
    mjvOption visual_options_{};
    mjvScene scene_{};
    mjrContext context_{};
    bool mouse_left_ = false;
    bool mouse_middle_ = false;
    bool mouse_right_ = false;
    double last_mouse_x_ = 0.0;
    double last_mouse_y_ = 0.0;
};

}  // namespace p1_sim
