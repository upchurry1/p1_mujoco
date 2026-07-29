# P1 MuJoCo

This repository is a P1-focused fork of `unitree_mujoco`. The main workflow is no
longer the original Unitree SDK/DDS bridge demo. Instead, it provides a
single-process MuJoCo runner for testing an IsaacLab-exported TorchScript walking
policy on the P1 model.

For Chinese documentation, see [readme_zh.md](./readme_zh.md).

## What This Project Runs

The primary executable is:

```bash
simulate/build/p1_mujoco_deploy_sim
```

It loads a P1 MJCF model, reads the same deploy configuration used by the real
robot inference project, runs a TorchScript policy through `TorchPolicyRunner`,
computes MIT-style joint torques, and writes torques directly to MuJoCo
`data->ctrl`.

The P1 runner does not require Unitree SDK, DDS, `p1_ctrl`, or `g1_ctrl`.

## Main Features

- P1 and P1 v2 MuJoCo models under `unitree_robots/`
- IsaacLab / `robot_deploy-main` deploy YAML loading
- TorchScript policy inference
- MIT PD torque control in MuJoCo
- Elastic rope support for humanoid standing and suspended tests
- Optional motor torque response delay
- Xbox/Switch joystick velocity commands
- Viewer HUD for summary, joint, and IMU data
- Joint motion curves for `q`, `dq`, torque, and control input
- Motor and ankle validation scripts under `motor_model_validation/`

## Directory Layout

- `simulate/`: C++ P1 MuJoCo runner and build files
- `simulate/p1_mujoco_deploy.yaml`: default runtime path and viewer settings
- `simulate/src/`: P1 runner, MuJoCo backend, viewer, joystick, and policy glue
- `unitree_robots/p1/`: first P1 MJCF model
- `unitree_robots/p1_v2/`: current P1 v2 MJCF model used by default
- `unitree_robots/p1_ankle_ik_check/`: ankle kinematics check scenes
- `simulate_python/`: Python utilities and legacy simulator scripts
- `motor_model_validation/`: motor and ankle validation helpers
- `terrain_tool/` and `example/`: upstream Unitree references

## Dependencies

Install the common system dependencies:

```bash
sudo apt update
sudo apt install cmake build-essential libyaml-cpp-dev libglfw3-dev libfmt-dev joystick
```

Install or link MuJoCo under `simulate/mujoco`:

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
```

The policy runner also needs `robot_deploy-main`, because the build reuses:

```text
robot_deploy-main/src/inference/include/torch_policy_runner.hpp
robot_deploy-main/src/inference/src/torch_policy_runner.cpp
```

The default path is:

```text
/home/hr/gongxunp1/robot_deploy-main
```

Override it at CMake time if your checkout is elsewhere.

## Build

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate
mkdir -p build
cd build
cmake .. -DP1_ROBOT_DEPLOY_ROOT=/home/hr/gongxunp1/robot_deploy-main
make -j$(nproc)
```

The build creates:

```text
p1_mujoco_deploy_sim
p1_ankle_kinematics_check
jstest
```

## Runtime Configuration

Edit [simulate/p1_mujoco_deploy.yaml](./simulate/p1_mujoco_deploy.yaml) before
running:

```yaml
model_xml_path: "../unitree_robots/p1_v2/scene.xml"
deploy_config_path: "/path/to/robot_deploy-main/src/inference/config/deploy.yaml"
policy_model_path: "/path/to/robot_deploy-main/src/inference/model/policy.pt"

joystick:
  enabled: true
  type: "xbox"
  device: "/dev/input/js0"
  bits: 16
  deadzone: 0.10
  limits: [0.50, 0.30, 0.60]
  signs: [1.0, -1.0, -1.0]

motor_delay:
  enabled: true
  min_ms: 0
  max_ms: 8

viewer:
  overlay: true
  page: "summary"
  angle_units: "rad"
  curve: true
  curve_signal: "q+dq"
  curve_joint_index: 0
  curve_window_seconds: 5.0
```

Runtime-only changes in this YAML do not require recompilation.

## Run

From the build directory:

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate/build
./p1_mujoco_deploy_sim
```

Useful alternatives:

```bash
./p1_mujoco_deploy_sim --print-config
./p1_mujoco_deploy_sim --headless --duration 5 --no-joystick --no-motor-delay
./p1_mujoco_deploy_sim --runner-config ../p1_mujoco_deploy.yaml
./p1_mujoco_deploy_sim --policy /path/to/policy.pt --deploy-config /path/to/deploy.yaml
```

Startup behavior:

1. The simulator loads the P1 model and deploy configuration.
2. It moves the robot to the deploy default joint pose.
3. It holds the robot with MIT PD control and optional elastic rope.
4. After the ready prompt appears, press `Enter` to connect the policy.
5. Joystick velocity commands take effect after the policy is connected.

## Keyboard And Viewer Controls

- `Enter`: connect policy
- `7` / `8`: shorten / lengthen elastic rope
- `R`: enable or disable elastic rope
- `0`: clear velocity command
- `H`: return to stand MIT hold under elastic rope
- `Space`: pause or resume
- `Backspace`: reset to deploy initial pose
- `Esc`: exit
- `V`: show or hide HUD
- `Tab`: cycle HUD page
- `C` / `J` / `I` / `M`: summary / joints / IMU / all page
- `U`: switch radian / degree display
- `G`: show or hide joint curve figure
- `[` / `]`: previous / next curve joint
- `N`: cycle curve signal
- `X`: clear curve history
- `1` / `2` / `3` / `4`: fixed camera views
- `F`: pelvis-follow camera

## Joystick

The runner currently supports Xbox and Switch layouts. With the default Xbox
mapping:

- Left stick Y: `vx`
- Left stick X: `vy`
- Right stick X: yaw rate

Use `jstest` to inspect your device:

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate/build
./jstest /dev/input/js0
```

If an axis is reversed, edit `joystick.signs` in
`simulate/p1_mujoco_deploy.yaml`.

## Where To Tune Parameters

- Rigid body mass, inertia, center of mass, joint axes, limits, actuator torque
  limits, and contact friction: `unitree_robots/p1_v2/scene.xml`
- Policy deploy parameters such as joint order, default pose, action scale,
  action clip, observation scale, stiffness, damping, and policy step time:
  the `deploy_config_path` YAML from `robot_deploy-main`
- Runtime model/policy paths, joystick, viewer, and motor delay:
  `simulate/p1_mujoco_deploy.yaml`
- Motor delay can also be changed with `--motor-delay-min-ms`,
  `--motor-delay-max-ms`, `--motor-delay`, and `--no-motor-delay`

## Debug Utilities

Useful scripts:

```bash
python3 simulate_python/verify_imu_axes.py
python3 simulate_python/verify_motor_directions.py
python3 simulate_python/project_gravity.py
python3 simulate_python/validate_closed_chain_ankle.py
```

Motor model checks:

```bash
cd motor_model_validation
python3 run_mujoco_motor_test.py
python3 analyze_motor_log.py
```

## Notes

- The current C++ runner is compiled with `POLICY_V3=0`, which corresponds to a
  45-dimensional single observation stacked for 5 frames, for a total policy
  input size of 225.
- The TorchScript model and deploy YAML must match. A mismatch usually appears
  as a TorchScript matrix shape error during load or dry run.
- Generated logs and plots are ignored by Git. Keep exported policy weights in
  `robot_deploy-main` or another external model directory unless you explicitly
  want to version them.

## Upstream

This project started from Unitree's MuJoCo simulator:

- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/google-deepmind/mujoco
