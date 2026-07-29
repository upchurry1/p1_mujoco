#include "p1_joystick_velocity.h"

#include "joystick.h"

#include <algorithm>
#include <cmath>
#include <iostream>

namespace p1_sim {

P1JoystickVelocityReader::P1JoystickVelocityReader() = default;
P1JoystickVelocityReader::~P1JoystickVelocityReader() = default;

P1JoystickVelocityReader::AxisLayout
P1JoystickVelocityReader::axisLayoutFor(const std::string& type)
{
    if (type == "switch") {
        return {0, 1, 2, 3, true};
    }
    if (type == "xbox") {
        return {0, 1, 3, 4, true};
    }
    return {0, 1, 2, 3, false};
}

bool P1JoystickVelocityReader::open(const P1JoystickVelocityOptions& options)
{
    if (!options.enabled) {
        return false;
    }

    layout_ = axisLayoutFor(options.type);
    if (!layout_.valid) {
        std::cerr << "[WARN] Unsupported joystick_type=" << options.type
                  << "; joystick velocity command is disabled.\n";
        return false;
    }

    joystick_ = std::make_unique<Joystick>(options.device);
    if (!joystick_->isFound()) {
        std::cerr << "[WARN] Joystick device " << options.device
                  << " is not available; joystick velocity command is disabled.\n";
        joystick_.reset();
        return false;
    }

    const int safe_bits = std::clamp(options.bits, 2, 30);
    max_axis_value_ = static_cast<double>(1 << (safe_bits - 1));
    deadzone_ = std::clamp(options.deadzone, 0.0, 0.95);
    vx_limit_ = std::max(0.0, options.limits[0]);
    vy_limit_ = std::max(0.0, options.limits[1]);
    yaw_rate_limit_ = std::max(0.0, options.limits[2]);
    vx_sign_ = options.signs[0] >= 0.0 ? 1.0 : -1.0;
    vy_sign_ = options.signs[1] >= 0.0 ? 1.0 : -1.0;
    yaw_rate_sign_ = options.signs[2] >= 0.0 ? 1.0 : -1.0;
    drainEvents();
    return true;
}

bool P1JoystickVelocityReader::available() const
{
    return joystick_ != nullptr;
}

void P1JoystickVelocityReader::update(double& vx,
                                      double& vy,
                                      double& yaw_rate)
{
    if (!available()) {
        return;
    }

    drainEvents();
    const double lx = normalizedAxis(layout_.lx, false);
    const double ly = normalizedAxis(layout_.ly, true);
    const double rx = normalizedAxis(layout_.rx, false);

    vx = std::clamp(vx_sign_ * ly * vx_limit_, -vx_limit_, vx_limit_);
    vy = std::clamp(vy_sign_ * lx * vy_limit_, -vy_limit_, vy_limit_);
    yaw_rate = std::clamp(yaw_rate_sign_ * rx * yaw_rate_limit_,
                          -yaw_rate_limit_,
                          yaw_rate_limit_);
}

void P1JoystickVelocityReader::drainEvents()
{
    if (!joystick_) {
        return;
    }

    JoystickEvent event{};
    for (int i = 0; i < 128 && joystick_->sample(&event); ++i) {
        const auto index = static_cast<std::size_t>(event.number);
        if (event.isButton() && index < buttons_.size()) {
            buttons_[index] = event.value;
        } else if (event.isAxis() && index < axes_.size()) {
            axes_[index] = event.value;
        }
    }
}

double P1JoystickVelocityReader::normalizedAxis(int axis_index,
                                                bool invert) const
{
    if (axis_index < 0 ||
        static_cast<std::size_t>(axis_index) >= axes_.size() ||
        max_axis_value_ <= 1.0) {
        return 0.0;
    }

    double value =
        static_cast<double>(axes_[static_cast<std::size_t>(axis_index)]) /
        max_axis_value_;
    value = std::clamp(value, -1.0, 1.0);
    if (invert) {
        value = -value;
    }

    const double abs_value = std::abs(value);
    if (abs_value <= deadzone_) {
        return 0.0;
    }
    return std::copysign((abs_value - deadzone_) / (1.0 - deadzone_),
                         value);
}

}  // namespace p1_sim
