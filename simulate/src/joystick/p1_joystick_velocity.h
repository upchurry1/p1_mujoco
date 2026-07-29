#pragma once

#include <array>
#include <memory>
#include <string>

class Joystick;

namespace p1_sim {

struct P1JoystickVelocityOptions {
    bool enabled = true;
    std::string type = "xbox";
    std::string device = "/dev/input/js0";
    int bits = 16;
    double deadzone = 0.08;
    std::array<double, 3> limits{0.5, 0.3, 0.6};
    std::array<double, 3> signs{1.0, -1.0, -1.0};
};

class P1JoystickVelocityReader {
public:
    P1JoystickVelocityReader();
    ~P1JoystickVelocityReader();

    bool open(const P1JoystickVelocityOptions& options);
    bool available() const;
    void update(double& vx, double& vy, double& yaw_rate);

private:
    struct AxisLayout {
        int lx = 0;
        int ly = 1;
        int rx = 2;
        int ry = 3;
        bool valid = true;
    };

    static AxisLayout axisLayoutFor(const std::string& type);
    void drainEvents();
    double normalizedAxis(int axis_index, bool invert) const;

    std::unique_ptr<Joystick> joystick_;
    AxisLayout layout_;
    std::array<int, 20> buttons_{};
    std::array<int, 10> axes_{};
    double max_axis_value_ = 32768.0;
    double deadzone_ = 0.08;
    double vx_limit_ = 1.0;
    double vy_limit_ = 0.45;
    double yaw_rate_limit_ = 0.5;
    double vx_sign_ = 1.0;
    double vy_sign_ = -1.0;
    double yaw_rate_sign_ = -1.0;
};

}  // namespace p1_sim
