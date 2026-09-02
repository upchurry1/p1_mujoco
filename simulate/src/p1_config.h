#pragma once

#include "p1_types.h"

#include <array>
#include <string>

#ifndef P1_ENABLE_BUILTIN_POLICY_DEFAULTS
#define P1_ENABLE_BUILTIN_POLICY_DEFAULTS 0
#endif

namespace p1_sim {

struct AnkleParallelMap {
    int model_pitch_dof = 0;
    int model_roll_dof = 0;
    int upper_motor_index = 0;
    int lower_motor_index = 0;
};

struct PolicyConfig {
    std::string policy_model_path;

#if P1_ENABLE_BUILTIN_POLICY_DEFAULTS
    // Per-joint control arrays follow the policy/control order:
    // left/right pairs by joint type.
    std::array<std::array<double, 2>, kPolicyDof> action_clip{{
        {{-0.12, 0.12}},  // hip_roll_l
        {{-0.12, 0.12}},  // hip_roll_r
        {{-0.40, 0.40}},  // hip_pitch_l
        {{-0.40, 0.40}},  // hip_pitch_r
        {{-0.12, 0.12}},  // hip_yaw_l
        {{-0.12, 0.12}},  // hip_yaw_r
        {{ 0.00, 0.85}},  // knee_pitch_l
        {{ 0.00, 0.85}},  // knee_pitch_r
        {{-0.18, 0.18}},  // ankle_pitch_l
        {{-0.18, 0.18}},  // ankle_pitch_r
        {{-0.08, 0.08}},  // ankle_roll_l
        {{-0.08, 0.08}},  // ankle_roll_r
    }};

    std::array<double, kPolicyDof> action_scale{
        0.08, 0.08,
        0.45, 0.45,
        0.08, 0.08,
        0.45, 0.45,
        0.20, 0.20,
        0.08, 0.08
    };

    std::array<double, kPolicyDof> stand_pose_rad{
         0.0,  0.0,
        -0.2, -0.2,
         0.0,  0.0,
         0.2,  0.2,
        -0.05, -0.05,
        0.0,  0.0
    };

    std::array<double, kPolicyDof> dof_pos_scale = filledArray<double, kPolicyDof>(1.0);
    std::array<double, kPolicyDof> dof_vel_scale = filledArray<double, kPolicyDof>(0.05);
    std::array<double, 3> projected_gravity_scale{1.0, 1.0, 1.0};
    std::array<double, 2> gait_phase_scale{1.0, 1.0};
    std::array<double, kPolicyDof> last_action_scale =
        filledArray<double, kPolicyDof>(1.0);

    std::array<double, kPolicyDof> policy_mit_kp_model{
        180.0, 180.0,
        180.0, 180.0,
        187.0, 187.0,
        180.0, 180.0,
        180.0, 180.0,
        187.0, 187.0
    };

    std::array<double, kPolicyDof> policy_mit_kd_model{
        10.0, 10.0,
        10.0, 10.0,
        9.07, 9.07,
        10.0, 10.0,
        10.0, 10.0,
        9.07, 9.07
    };

    double raw_action_clip = 1.0;
    std::array<double, 3> command_scale{1.0, 1.0, 1.0};
    std::array<double, 3> body_ang_vel_scale{0.2, 0.2, 0.2};
    bool observation_scale_first = false;
    bool base_ang_vel_clip_enabled = false;
    std::array<double, 2> base_ang_vel_clip{-10.0, 10.0};
    bool projected_gravity_clip_enabled = false;
    std::array<double, 2> projected_gravity_clip{-1.0, 1.0};
    bool command_clip_enabled = false;
    std::array<double, 2> command_clip{-1.0, 1.0};
    bool gait_phase_clip_enabled = false;
    std::array<double, 2> gait_phase_clip{-1.0, 1.0};
    bool joint_pos_rel_clip_enabled = false;
    std::array<double, 2> joint_pos_rel_clip{-3.0, 3.0};
    bool joint_vel_rel_clip_enabled = false;
    std::array<double, 2> joint_vel_rel_clip{-100.0, 100.0};
    bool last_action_clip_enabled = false;
    std::array<double, 2> last_action_clip{-1.0, 1.0};
    std::array<int, kPolicyDof> control_to_motor_index{
        0, 6,
        1, 7,
        2, 8,
        3, 9,
        4, 10,
        5, 11
    };
#else
    std::array<std::array<double, 2>, kPolicyDof> action_clip{};
    std::array<double, kPolicyDof> action_scale{};
    std::array<double, kPolicyDof> stand_pose_rad{};
    std::array<double, kPolicyDof> dof_pos_scale{};
    std::array<double, kPolicyDof> dof_vel_scale{};
    std::array<double, 3> projected_gravity_scale{1.0, 1.0, 1.0};
    std::array<double, 2> gait_phase_scale{1.0, 1.0};
    std::array<double, kPolicyDof> last_action_scale =
        filledArray<double, kPolicyDof>(1.0);
    std::array<double, kPolicyDof> policy_mit_kp_model{};
    std::array<double, kPolicyDof> policy_mit_kd_model{};
    double raw_action_clip = 1.0;
    std::array<double, 3> command_scale{};
    std::array<double, 3> body_ang_vel_scale{};
    bool observation_scale_first = false;
    bool base_ang_vel_clip_enabled = false;
    std::array<double, 2> base_ang_vel_clip{-10.0, 10.0};
    bool projected_gravity_clip_enabled = false;
    std::array<double, 2> projected_gravity_clip{-1.0, 1.0};
    bool command_clip_enabled = false;
    std::array<double, 2> command_clip{-1.0, 1.0};
    bool gait_phase_clip_enabled = false;
    std::array<double, 2> gait_phase_clip{-1.0, 1.0};
    bool joint_pos_rel_clip_enabled = false;
    std::array<double, 2> joint_pos_rel_clip{-3.0, 3.0};
    bool joint_vel_rel_clip_enabled = false;
    std::array<double, 2> joint_vel_rel_clip{-100.0, 100.0};
    bool last_action_clip_enabled = false;
    std::array<double, 2> last_action_clip{-1.0, 1.0};
    std::array<int, kPolicyDof> control_to_motor_index{};
#endif
    double policy_step_dt_s = 0.02;
    double gait_phase_period_s = 0.74;
    double gait_phase_stand_threshold = 0.05;
    double gait_phase_move_threshold = 0.15;
    bool include_gait_phase_observation = true;
    std::array<int, kPolicyDof> model_to_motor_index{
        0, 6,
        1, 7,
        2, 8,
        3, 9,
        -1, -1,
        -1, -1
    };
    int model_to_motor_count = 8;
    AnkleParallelMap left_ankle_parallel{8, 10, 4, 5};
    AnkleParallelMap right_ankle_parallel{9, 11, 10, 11};
    std::array<int, kPolicyDof> motor_to_model_direction ={1,1,1,1,1,1,1,1,1,1,1,1};
        //filledArray<int, kPolicyDof>(1);
    std::array<double, kPolicyDof> joint_min_rad =
        filledArray<double, kPolicyDof>(-3.14159265358979323846);
    std::array<double, kPolicyDof> joint_max_rad =
        filledArray<double, kPolicyDof>(3.14159265358979323846);
};

struct RunnerOptions {
    std::string model_xml_path;
    std::string runner_config_path;
    std::string deploy_config_path;
    double policy_hz = 50.0;
    double stand_seconds = 2.0;
    double duration_seconds = 0.0;
    bool auto_start_policy = false;
    double command_vx = 0.0;
    double command_vy = 0.0;
    double command_yaw_rate = 0.0;
    std::string real_observation_log_path;
    bool use_cli_command_on_policy_start = false;
    bool joystick_enabled = true;
    std::string joystick_type = "xbox";
    std::string joystick_device = "/dev/input/js0";
    int joystick_bits = 16;
    double joystick_deadzone = 0.08;
    std::array<double, 3> joystick_limits{0.50, 0.30, 0.60};
    std::array<double, 3> joystick_signs{1.0, -1.0, -1.0};
    bool use_xml_initial_pose = true;
    bool elastic_rope_enabled = true;
    std::array<double, 3> initial_base_pos{0.0, 0.0, 0.72};
    std::array<double, 4> initial_base_quat{1.0, 0.0, 0.0, 0.0};
    bool rope_auto_length = true;
    double rope_support_ratio = 0.98;
    double rope_length_m = 0.0;
    double rope_length_step_m = 0.1;
    double rope_min_length_m = -1.0;
    double rope_max_length_m = 3.0;
    double rope_stiffness = 300.0;
    double rope_damping = 50.0;
    double rope_anchor_z = 3.0;
    double rope_side_offset_y = 0.18;
    double rope_local_attach_z = 0.35;
    bool headless = false;
    bool print_timing = true;
    bool viewer_overlay_enabled = true;
    ViewerOverlayPage viewer_overlay_page = ViewerOverlayPage::kSummary;
    bool viewer_overlay_degrees = false;
    bool viewer_curve_enabled = true;
    ViewerCurveSignal viewer_curve_signal = ViewerCurveSignal::kPositionVelocity;
    int viewer_curve_joint_index = 0;
    double viewer_curve_window_seconds = 5.0;
    bool motor_delay_enabled = false;
    double motor_delay_min_seconds = 0.010;
    double motor_delay_max_seconds = 0.020;
    bool observation_delay_enabled = false;
    ObservationDelaySource observation_delay_source = ObservationDelaySource::kImu;
    double observation_delay_seconds = 0.0;
    bool imu_noise_enabled = false;
    std::array<double, kPolicyDof> joint_zero_offset_rad{};
    bool motor_to_model_direction_override = false;
    std::array<int, kPolicyDof> motor_to_model_direction{
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };
    std::array<int, kPolicyDof> mujoco_joint_direction{
        1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1
    };
    bool sim_log_enabled = true;
    std::string sim_log_path;
    bool show_help = false;
    bool print_config = false;
};

bool fileReadable(const std::string& path);
bool parseArgs(int argc, char** argv, RunnerOptions& options, PolicyConfig& config);
void printUsage(const char* program);
void printResolvedConfig(const RunnerOptions& options, const PolicyConfig& config);

extern const std::array<const char*, kPolicyDof> kMotorNamesP1RealOrder;

void mapControlArrayToMotor(const PolicyConfig& config,
                            const std::array<double, kPolicyDof>& control_values,
                            std::array<double, kPolicyDof>& motor_values);

}  // namespace p1_sim
