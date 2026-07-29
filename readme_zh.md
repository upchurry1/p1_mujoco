# P1 MuJoCo

这个仓库现在已经不是原版 `unitree_mujoco` 的通用 DDS 仿真入口，而是面向 P1 机器人的 MuJoCo 策略仿真工程。当前主流程是：加载 P1 MJCF 模型，读取 IsaacLab / `robot_deploy-main` 导出的部署配置，加载 TorchScript 策略，在 MuJoCo 里用 MIT PD 计算控制力矩，并直接写入 `data->ctrl`。

主程序是：

```bash
simulate/build/p1_mujoco_deploy_sim
```

这个程序不依赖 Unitree SDK、DDS、`p1_ctrl` 或 `g1_ctrl`。

## 主要功能

- P1 / P1 v2 MuJoCo 模型
- 读取实机部署侧的 deploy YAML
- 加载 TorchScript 策略模型
- MuJoCo 内部 MIT PD 力矩控制
- 人形机器人起步用弹性吊绳
- 可选电机控制响应延时
- Xbox / Switch 手柄速度控制
- 可视化 HUD，显示关节、IMU、控制量等信息
- 关节运动曲线，支持 `q`、`dq`、`tau`、`ctrl`
- 电机和踝关节验证脚本

## 目录结构

- `simulate/`：C++ P1 MuJoCo 主仿真程序
- `simulate/p1_mujoco_deploy.yaml`：默认运行配置
- `simulate/src/`：P1 runner、MuJoCo backend、viewer、手柄、策略接口
- `unitree_robots/p1/`：早期 P1 MJCF 模型
- `unitree_robots/p1_v2/`：当前默认使用的 P1 v2 MJCF 模型
- `unitree_robots/p1_ankle_ik_check/`：踝关节运动学检查模型
- `simulate_python/`：Python 辅助脚本和旧仿真脚本
- `motor_model_validation/`：电机和踝关节模型验证工具
- `terrain_tool/`、`example/`：上游 Unitree 工程保留参考

## 依赖

基础依赖：

```bash
sudo apt update
sudo apt install cmake build-essential libyaml-cpp-dev libglfw3-dev libfmt-dev joystick
```

MuJoCo 建议解压到 `~/.mujoco`，然后在 `simulate/` 下建立软链接：

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate
ln -s ~/.mujoco/mujoco-3.3.6 mujoco
```

策略加载复用了 `robot_deploy-main` 里的 `TorchPolicyRunner`：

```text
robot_deploy-main/src/inference/include/torch_policy_runner.hpp
robot_deploy-main/src/inference/src/torch_policy_runner.cpp
```

默认 `robot_deploy-main` 路径是：

```text
/home/hr/gongxunp1/robot_deploy-main
```

如果你的路径不同，编译时用 `-DP1_ROBOT_DEPLOY_ROOT=...` 指定。

## 编译

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate
mkdir -p build
cd build
cmake .. -DP1_ROBOT_DEPLOY_ROOT=/home/hr/gongxunp1/robot_deploy-main
make -j$(nproc)
```

编译后主要生成：

```text
p1_mujoco_deploy_sim
p1_ankle_kinematics_check
jstest
```

## 运行配置

运行前先检查 [simulate/p1_mujoco_deploy.yaml](./simulate/p1_mujoco_deploy.yaml)：

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

这里的运行参数改完后不需要重新编译，重新运行程序即可。

## 运行

从 build 目录启动：

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate/build
./p1_mujoco_deploy_sim
```

常用命令：

```bash
./p1_mujoco_deploy_sim --print-config
./p1_mujoco_deploy_sim --headless --duration 5 --no-joystick --no-motor-delay
./p1_mujoco_deploy_sim --runner-config ../p1_mujoco_deploy.yaml
./p1_mujoco_deploy_sim --policy /path/to/policy.pt --deploy-config /path/to/deploy.yaml
```

启动流程：

1. 程序加载 P1 模型和部署配置。
2. 机器人进入 deploy 默认关节姿态。
3. 策略接入前，用 MIT PD 保持站姿，可配合弹性吊绳调试。
4. 界面提示 ready 后，按 `Enter` 接入策略。
5. 策略接入后，Xbox 手柄速度指令生效。

## 键盘和界面控制

- `Enter`：接入策略
- `7` / `8`：缩短 / 放长弹性吊绳
- `R`：启用或关闭弹性吊绳
- `0`：清零速度指令
- `H`：回到弹性吊绳站姿保持
- `Space`：暂停 / 继续
- `Backspace`：重置到 deploy 初始姿态
- `Esc`：退出
- `V`：显示 / 隐藏 HUD
- `Tab`：切换 HUD 页面
- `C` / `J` / `I` / `M`：summary / joints / IMU / all 页面
- `U`：切换弧度 / 角度显示
- `G`：显示 / 隐藏关节曲线
- `[` / `]`：切换上一 / 下一个曲线关节
- `N`：切换曲线信号
- `X`：清空曲线历史
- `1` / `2` / `3` / `4`：固定相机视角
- `F`：骨盆跟随相机

## 手柄

当前支持 Xbox 和 Switch 布局。默认 Xbox 控制方式：

- 左摇杆 Y：`vx`
- 左摇杆 X：`vy`
- 右摇杆 X：yaw rate

可以用 `jstest` 检查手柄设备：

```bash
cd /home/hr/gongxunp1/unitree_mujoco/simulate/build
./jstest /dev/input/js0
```

如果某个方向反了，修改 `simulate/p1_mujoco_deploy.yaml` 里的 `joystick.signs`。

## 参数在哪里调

- 刚体质量、惯量、质心、关节轴、关节限位、执行器最大力矩、脚底接触摩擦：`unitree_robots/p1_v2/scene.xml`
- 策略部署参数，例如关节顺序、默认姿态、action scale、action clip、observation scale、PD 刚度、PD 阻尼、策略周期：`deploy_config_path` 指向的 deploy YAML
- 运行时路径、手柄、HUD、曲线、电机延时：`simulate/p1_mujoco_deploy.yaml`
- 电机延时也可以用命令行覆盖：`--motor-delay-min-ms`、`--motor-delay-max-ms`、`--motor-delay`、`--no-motor-delay`

## 调试工具

常用脚本：

```bash
python3 simulate_python/verify_imu_axes.py
python3 simulate_python/verify_motor_directions.py
python3 simulate_python/project_gravity.py
python3 simulate_python/validate_closed_chain_ankle.py
```

电机模型验证：

```bash
cd motor_model_validation
python3 run_mujoco_motor_test.py
python3 analyze_motor_log.py
```

## 注意事项

- 当前 C++ runner 编译时使用 `POLICY_V3=0`，对应单帧 45 维观测、5 帧历史，总输入维度是 225。
- TorchScript 模型必须和 deploy YAML 匹配。二者不匹配时，常见表现是 TorchScript dry-run 或推理时报矩阵维度错误。
- `simulate/p1_mujoco_deploy.yaml` 里的模型路径和策略路径是本机运行路径，换机器后需要改成对应位置。
- MuJoCo 日志、build 目录、电机验证结果图表默认不会提交到 Git。

## 上游来源

本项目基于 Unitree 官方 MuJoCo 工程修改：

- https://github.com/unitreerobotics/unitree_mujoco
- https://github.com/google-deepmind/mujoco
