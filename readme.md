# P1 MuJoCo 仿真与 Real2Sim 工程

本仓库是面向 P1 机器人的sim2sim。当前是一个单进程 MuJoCo 策略仿真与真机日志分析工具链。

核心目标有两个：

1. 在 MuJoCo 中运行和实机部署侧一致的 TorchScript 策略，观察 P1 策略表现。
2. 使用真机 CSV 日志复现策略输入、重新推理、对比策略输出和真机记录结果，用于定位 sim2sim / real2sim 差异。

## 主要可执行程序

编译后主要使用以下程序：

```text
simulate/build/p1_mujoco_deploy_sim
simulate/build/real2sim_replay
simulate/build/ankle_kinematics_check
simulate/build/jstest
```

其中：

- `p1_mujoco_deploy_sim`：打开 MuJoCo，可视化运行 P1 策略。
- `real2sim_replay`：读取真机 CSV 日志，重新推理策略，并输出对比 CSV。
- `plot_real2sim.py`：读取 `real2sim_replay` 输出结果，绘制分析图。
- `ankle_kinematics_check`：检查 P1 并联脚踝 IK / FK 一致性。
- `jstest`：检查 Linux 手柄输入。

## 当前能力

- 加载 P1 MuJoCo 模型：默认使用 `unitree_robots/p1_803_nobattery/scene.xml`
- 加载实机部署侧 `deploy.yaml`
- 加载 TorchScript 策略模型 `policy.pt`
- 在 MuJoCo 中执行 MIT PD 控制，并将力矩写入 `data->ctrl`
- 支持弹性吊绳：启动后保持默认站姿，可通过键盘下放，再按 `Enter` 接入策略
- 支持 Xbox / Switch 手柄速度指令
- 支持观测延时模拟：可以单独延迟 IMU 或 motor 观测
- 支持电机控制响应延时模拟
- 支持 IMU 噪声开关
- 支持真机 CSV observation 驱动策略推理，并在 MuJoCo 中渲染策略效果
- 支持 real2sim 离线分析：对比模型重新推理动作、真机记录动作、电机目标、脚踝 FK 后的虚拟关节目标
- 支持 P1 真机电机坐标系与 MuJoCo 关节坐标系符号映射

## 目录结构

```text
unitree_mujoco/
├── simulate/                         # 当前主要使用的 C++ 仿真与分析工程
│   ├── CMakeLists.txt
│   ├── p1_mujoco_deploy.yaml          # p1_mujoco_deploy_sim 默认运行配置
│   ├── config.yaml                    # 其他兼容配置
│   ├── src/
│   │   ├── main.cpp                   # p1_mujoco_deploy_sim 主入口
│   │   ├── real2sim_replay.cpp         # 真机日志 replay / 策略重推理
│   │   ├── p1_policy_controller.*      # 策略后处理、观测构建、MIT 控制
│   │   ├── mujoco_p1_backend.*         # MuJoCo 状态读写和电机接口
│   │   ├── mujoco_viewer.*             # MuJoCo 可视化和 HUD
│   │   ├── p1_config.*                 # YAML / CLI 配置解析
│   │   ├── p1_sim_log.*                # 仿真日志记录
│   │   ├── ankle_motor_ik.hpp          # 本地并联脚踝逆解
│   │   └── ankle_motor_fk.hpp          # 本地并联脚踝正解
│   └── tools/
│       └── plot_real2sim.py            # real2sim 结果绘图
├── unitree_robots/
│   ├── p1/
│   ├── p1_v2/
│   ├── p1_803_nobattery/              # 当前默认 P1 模型
│   └── p1_ankle_ik_check/
├── simulate_python/                   # Python 辅助脚本和旧版仿真脚本
├── terrain_tool/                      # 上游地形工具
├── example/                           # 上游 Unitree 示例，当前 P1 流程不使用
└── doc/
```

## 依赖

建议环境：

- Ubuntu
- CMake
- g++
- MuJoCo
- yaml-cpp
- GLFW
- fmt
- LibTorch / TorchScript 运行环境
- `robot_deploy-main`，用于复用实机部署侧的策略 runner 和配置

安装常用系统依赖：

```bash
sudo apt update
sudo apt install cmake build-essential libyaml-cpp-dev libglfw3-dev libfmt-dev joystick
```

MuJoCo 建议放在：

```text
~/.mujoco/
```

然后在 `simulate/` 下建立软链接，例如：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate
ln -s /home/ubuntu/.mujoco/mujoco-3.5.0 mujoco
```

如果你的 MuJoCo 版本或路径不同，软链接指向自己的实际路径即可。

## 编译

进入 `simulate` 目录编译：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate
mkdir -p build
cd build
cmake .. -DP1_ROBOT_DEPLOY_ROOT=/home/ubuntu/gongxunp1/robot_deploy-main
make -j$(nproc)
```

`P1_ROBOT_DEPLOY_ROOT` 指向 `robot_deploy-main` 根目录。当前工程会复用其中的 TorchScript 推理代码和部署配置。

如果只想重新编译主要程序：

```bash
make -j$(nproc) p1_mujoco_deploy_sim real2sim_replay
```

## 默认配置文件

主配置文件是：

```text
simulate/p1_mujoco_deploy.yaml
```

当前关键字段：

```yaml
model_xml_path: "../unitree_robots/p1_803_nobattery/scene.xml"
deploy_config_path: "/home/ubuntu/gongxunp1/robot_deploy-main/src/inference/config/deploy.yaml"
policy_model_path: "/home/ubuntu/gongxunp1/robot_deploy-main/src/inference/model/policy.pt"

policy_observation:
  gait_phase: true

raw_action_clip: 1.0

mujoco_motor_to_model_direction: [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1]
mujoco_joint_direction: [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1]

auto_start_policy: false
use_cli_command_on_policy_start: false

elastic_rope:
  enabled: true

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
  max_ms: 5

observation_delay:
  enabled: false
  source: "imu"
  delay_ms: 0

imu_noise:
  enabled: true

joint_zero_offset_deg: 0

logging:
  enabled: true
  path: "log"

viewer:
  overlay: true
  page: "summary"
  angle_units: "deg"
  curve: true
  curve_signal: "q+dq"
  curve_joint_index: 0
  curve_window_seconds: 10
```

修改这个 YAML 后不需要重新编译，重新运行程序即可。

## 坐标系映射说明

当前工程里要区分三套量：

- 策略模型输出的虚拟关节目标：`target_q_model_rad`
- P1 真机电机坐标系下的电机目标：`target_q_motor_rad`
- MuJoCo XML 关节坐标系下的关节位置：`qpos`

配置中有两个符号映射：

```yaml
mujoco_motor_to_model_direction: [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1]
mujoco_joint_direction: [-1, -1, 1, 1, -1, -1, -1, 1, 1, -1, -1, -1]
```

含义：

- `mujoco_motor_to_model_direction`：用于策略模型坐标和真机电机坐标之间的符号转换。
- `mujoco_joint_direction`：用于真机电机坐标和 MuJoCo XML 关节坐标之间的符号转换。

这两个字段都按 P1 真机电机顺序排列：

```text
M0  L hip roll
M1  L hip pitch
M2  L hip yaw
M3  L knee
M4  L ankle pitch motor
M5  L ankle roll motor
M6  R hip roll
M7  R hip pitch
M8  R hip yaw
M9  R knee
M10 R ankle pitch motor
M11 R ankle roll motor
```

注意：脚踝比较特殊。策略侧输出的是虚拟 ankle pitch / ankle roll，真机日志里的 `target_pos_rad_M4/M5/M10/M11` 是驱动并联脚踝的电机位置。`real2sim_replay` 的分析图会使用本地 `ankle_motor_fk.hpp` 将真机脚踝电机角度正解为虚拟脚踝关节角，用于和策略输出对比。

## 启动 MuJoCo 策略仿真

从 build 目录启动：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate/build
./p1_mujoco_deploy_sim
```

默认流程：

1. 打开 MuJoCo viewer。
2. 机器人进入 deploy 默认站立姿态。
3. 策略未接入前，使用 MIT PD 保持站姿。
4. 弹性吊绳默认开启，可以用键盘调整吊装高度。
5. 按 `Enter` 后接入策略。
6. 接入策略后，手柄或命令行速度指令开始生效。

常用启动方式：

```bash
# 打印解析后的配置并退出
./p1_mujoco_deploy_sim --print-config

# 不启用手柄
./p1_mujoco_deploy_sim --no-joystick

# 自动接入策略，适合无界面或批量测试
./p1_mujoco_deploy_sim --start-policy

# 无界面运行 5 秒
./p1_mujoco_deploy_sim --headless --duration 5 --start-policy --no-joystick

# 临时指定策略和 deploy 配置
./p1_mujoco_deploy_sim \
  --policy /path/to/policy.pt \
  --deploy-config /path/to/deploy.yaml
```

## 使用真机 observation 驱动 MuJoCo 渲染

这个模式用于验证：同一套策略在 MuJoCo 中渲染运行，但策略输入不再来自 MuJoCo 传感器，而是来自真机 CSV 日志里的 policy observation。

启动命令：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate/build

./p1_mujoco_deploy_sim \
  --real-observation-log /home/ubuntu/gongxunp1/0831_1757.csv \
  --no-joystick
```

行为说明：

- MuJoCo viewer 仍然会打开。
- 初始状态仍然是默认站立姿态。
- 策略接入前仍然保持吊装站姿。
- 可以用 `7` / `8` 调整绳索高度。
- 按 `Enter` 后接入策略。
- 接入策略后，每个策略周期读取一帧 CSV 中的 `policy_obs_0..policy_obs_704`。
- 策略使用 CSV observation 推理动作。
- MuJoCo 使用该动作经过后处理后的电机目标执行 PD / torque 控制。
- 时间推进按 CSV 时间戳的相邻帧 dt 执行，更接近真机控制节奏。

该模式不是运动学回放。也就是说，MuJoCo 里的机器人状态可能会逐渐偏离真机日志中的实际状态，因为策略输入来自日志，而 MuJoCo 本体仍然按照自己的动力学和电机响应运动。

## 真机 CSV 格式

当前推荐真机日志至少记录 policy 输入，也就是策略实际看到的 observation。

必需列：

```text
timestamp
policy_obs_0
policy_obs_1
...
policy_obs_704
```

`timestamp` 也可以使用兼容名称：

```text
time
time_s
elapsed_s
sim_time
elapsed_us
```

其中：

- `policy_obs_0..704` 是已经拼好的策略输入。
- 当前默认是 47 维单帧 observation，堆叠 15 帧，总维度 705。
- observation 排列为按 term 分块，每个 term 内部按 oldest 到 newest 排列。

当前 705 维 layout：

```text
[0:45)    base_ang_vel_history       latest=[42:45)
[45:90)   projected_gravity_history  latest=[87:90)
[90:135)  velocity_commands_history  latest=[132:135)
[135:165) gait_phase_sin_cos_history latest=[163:165)
[165:345) joint_pos_rel_history      latest=[333:345)
[345:525) joint_vel_rel_history      latest=[513:525)
[525:705) last_action_history        latest=[693:705)
```

建议同时记录以下列，方便做分析图：

```text
raw_action_0 ... raw_action_11
target_pos_rad_M0 ... target_pos_rad_M11
```

含义：

- `raw_action_*`：真机策略输出的原始 action，用于和 MuJoCo 重新推理出的 action 对比。
- `target_pos_rad_M*`：真机最终下发给电机的目标位置，已经经过实机侧坐标和脚踝处理。

如果 CSV 某一行缺列或字段为空，`real2sim_replay` 会跳过不完整行或报错提示具体列名。

## real2sim 离线分析

`real2sim_replay` 用于读取真机日志、重新构造策略输入、再次运行同一个 TorchScript 策略，并输出对比结果。

推荐命令：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate/build

./real2sim_replay \
  --log /home/ubuntu/gongxunp1/0831_1757.csv \
  --log-observation-stage policy_input \
  --output /home/ubuntu/gongxunp1/real_logs/real2sim_result.csv
```

常用参数：

```text
--config PATH                  指定 runner 配置，默认使用 simulate/p1_mujoco_deploy.yaml
--model PATH                   覆盖 MuJoCo XML 路径
--deploy-config PATH           覆盖 deploy.yaml 路径
--policy PATH                  覆盖 TorchScript policy.pt 路径
--policy-hz VALUE              覆盖策略频率
--log PATH                     输入真机 CSV
--output PATH                  输出结果 CSV
--mode offline|mujoco          offline 只重推理；mujoco 会推进 MuJoCo
--log-observation-stage raw|policy_input
--use-log-timestamp            使用日志时间戳 dt 推进，默认开启
--fixed-dt VALUE               使用固定 dt
--start-time VALUE             指定分析起始时间
--duration VALUE               指定分析时长
```

当前最常用的是：

```text
--log-observation-stage policy_input
```

因为这会直接使用真机记录的 `policy_obs_0..704`，用于验证：

```text
同一帧 observation
    -> 同一个 policy.pt
    -> MuJoCo 侧重新推理出的 raw action
```

如果重新推理出的 raw action 和真机日志里的 raw action 几乎完全重合，说明 observation 拼接、归一化、历史帧顺序、策略模型加载基本一致。

## 绘制 real2sim 分析图

运行 `real2sim_replay` 后，可以绘图：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco

python3 simulate/tools/plot_real2sim.py \
  --input /home/ubuntu/gongxunp1/real_logs/real2sim_result.csv \
  --output-dir /home/ubuntu/gongxunp1/real_logs/real2sim_result_plots
```

输出图包括：

```text
policy_raw_action_compare.png
model_joint_target_compare.png
motor_fk_model_target_compare.png
comparison_error_summary.png
comparison_summary.txt
```

图的含义：

- `policy_raw_action_compare.png`：MuJoCo 侧重新推理出的 raw action vs 真机日志记录的 raw action。
- `model_joint_target_compare.png`：策略模型虚拟关节目标对比。
- `motor_fk_model_target_compare.png`：真机脚踝电机目标经过 FK 转成虚拟踝关节后，再和策略虚拟关节目标对比。
- `comparison_error_summary.png`：各关节误差统计。
- `comparison_summary.txt`：文本形式误差摘要。

目前不再保留未经过脚踝 FK / 坐标处理的旧版 `motor_output_compare.png`，避免误判脚踝并联机构导致的差异。

## 观测延时模拟

用于模拟真机部署中 IMU 数据和电机数据不是同一时刻到达的问题。

配置文件方式：

```yaml
observation_delay:
  enabled: true
  source: "imu"
  delay_ms: 20
```

`source` 可选：

```text
imu
motor
```

命令行方式：

```bash
# 延迟 IMU 20 ms
./p1_mujoco_deploy_sim \
  --observation-delay-ms 20 \
  --observation-delay-source imu

# 延迟电机观测 20 ms
./p1_mujoco_deploy_sim \
  --observation-delay-ms 20 \
  --observation-delay-source motor

# 关闭观测延时
./p1_mujoco_deploy_sim --no-observation-delay
```

含义：

- `imu`：延迟 gyro / projected gravity 等 IMU 相关观测。
- `motor`：延迟关节位置、速度、力矩等 motor 相关观测。

这个功能只影响策略 observation 的构建，不直接改变 MuJoCo 本体动力学。

## 电机响应延时模拟

配置文件：

```yaml
motor_delay:
  enabled: true
  min_ms: 0
  max_ms: 5
```

命令行：

```bash
./p1_mujoco_deploy_sim --motor-delay --motor-delay-min-ms 0 --motor-delay-max-ms 8
./p1_mujoco_deploy_sim --no-motor-delay
```

这个功能模拟 MIT 命令到电机响应之间的延迟，会影响 MuJoCo 里电机实际执行的控制目标。

## 日志输出

默认配置：

```yaml
logging:
  enabled: true
  path: "log"
```

命令行：

```bash
# 写入指定 CSV
./p1_mujoco_deploy_sim --log-csv /tmp/p1_sim.csv

# 写入指定目录，自动生成带时间戳文件名
./p1_mujoco_deploy_sim --log-dir /tmp/p1_logs

# 不记录日志
./p1_mujoco_deploy_sim --no-log
```

日志会记录仿真中关节位置、速度、力矩、目标位置、策略动作等信息，便于后续对比分析。

## 键盘控制

MuJoCo viewer 中可用：

```text
7             缩短弹性吊绳，机器人上提
8             放长弹性吊绳，机器人下放
R             开关弹性吊绳
Enter         接入策略
0             速度指令清零
H             回到吊装站姿保持
Space         暂停 / 继续
Backspace     重置到 deploy 初始姿态
Esc           退出
```

HUD：

```text
V             显示 / 隐藏 HUD
Tab           切换 HUD 页面
C / J / I / M summary / joints / IMU / all 页面
U             切换弧度 / 角度显示
```

曲线窗口：

```text
G             显示 / 隐藏关节曲线
[ / ]         上一个 / 下一个关节
N             切换曲线信号
X             清空曲线历史
```

相机：

```text
1             固定斜视角
2             固定侧视角
3             固定后视角
4             固定俯视角
F             骨盆跟随相机
```

## 手柄控制

默认 Xbox 映射：

```text
左摇杆 Y      vx 前后速度
左摇杆 X      vy 横向速度
右摇杆 X      yaw rate 转向角速度
```

检查手柄输入：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate/build
./jstest /dev/input/js0
```

如果方向反了，修改：

```yaml
joystick:
  signs: [1.0, -1.0, -1.0]
```

如果暂时不用手柄：

```bash
./p1_mujoco_deploy_sim --no-joystick
```

## 脚踝 IK / FK

P1 脚踝是并联机构，需要区分：

```text
策略输出：虚拟 ankle pitch / ankle roll
真机日志：驱动脚踝的两个电机位置
```

当前仓库里脚踝运动学代码是本地实现：

```text
simulate/src/ankle_motor_ik.hpp
simulate/src/ankle_motor_fk.hpp
```


目前：

- `real2sim_replay` 已经使用 `ankle_motor_fk.hpp`，把真机记录的脚踝驱动电机目标正解为虚拟踝关节角度，用于绘图分析。
- `p1_mujoco_deploy_sim` 的主控制链路中，普通关节会做坐标符号映射；脚踝是否需要进一步接入 IK，应根据当前实机部署侧和 MuJoCo XML 执行器建模方式继续确认。

检查 IK / FK：

```bash
cd /home/ubuntu/gongxunp1/unitree_mujoco/simulate/build
./ankle_kinematics_check
```

## 常见问题

### 1. 为什么 raw action 对比几乎完全重合？

如果 `policy_raw_action_compare.png` 中两条线几乎完全重合，说明：

```text
真机记录的 policy_obs
    -> 当前加载的 policy.pt
    -> 重新推理 raw action
```

这一链路和实机推理基本一致。

这通常说明 observation 拼接、历史帧顺序、归一化和模型加载没有明显问题。

### 2. 为什么电机目标位置对不上？

可能原因包括：

- 真机电机坐标系和 MuJoCo XML 关节坐标系符号不同。
- 脚踝是并联机构，真机日志记录的是驱动电机位置，不是虚拟 ankle pitch / roll。
- MuJoCo 电机响应、PD 参数、零偏、力矩限制和实机不同。
- 使用了未经过 FK / 坐标处理的旧图进行比较。

优先查看：

```text
motor_fk_model_target_compare.png
```

而不是旧的直接电机位置对比图。

### 3. `empty numeric field: policy_obs_xxx`

说明 CSV 中某一行的该列为空。需要检查真机日志写入是否有断行、末尾逗号、字段数量不足或 NaN 被写成空字符串。

可以先只保留完整行，或者重新记录 CSV。

### 4. MuJoCo viewer 没有打开

确认没有使用：

```bash
--headless
```

并确认系统有图形界面和 OpenGL / GLFW 环境。

## 上游来源

本工程基于 Unitree 官方 MuJoCo 工程修改：

```text
https://github.com/unitreerobotics/unitree_mujoco
https://github.com/google-deepmind/mujoco
```
