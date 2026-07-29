#include "ankle_motor_fk.hpp"
#include "ankle_motor_ik.hpp"

#include <array>
#include <cmath>
#include <iomanip>
#include <iostream>

namespace {

struct Sample {
    double roll;
    double pitch;
};

}  // namespace

int main()
{
    ankle_motor_ik::Solver ik_solver;
    ankle_motor_fk::Solver fk_solver;

    constexpr std::array<Sample, 13> kSamples{{
        { 0.00,  0.00},
        { 0.00, -0.05},
        { 0.00,  0.05},
        { 0.00, -0.15},
        { 0.00,  0.15},
        { 0.04,  0.00},
        {-0.04,  0.00},
        { 0.08,  0.00},
        {-0.08,  0.00},
        { 0.04, -0.10},
        {-0.04, -0.10},
        { 0.04,  0.10},
        {-0.04,  0.10},
    }};

    int ik_failures = 0;
    int fk_failures = 0;
    double max_abs_roll_error = 0.0;
    double max_abs_pitch_error = 0.0;

    std::cout << "P1 ankle IK/FK standalone check. This tool is not wired into "
              << "MuJoCo control.\n";
    std::cout << std::fixed << std::setprecision(6)
              << "roll_rad,pitch_rad,upper_motor_rad,lower_motor_rad,"
              << "fk_roll_rad,fk_pitch_rad,roll_error_rad,pitch_error_rad,"
              << "ik_reachable,fk_reachable\n";

    for (const Sample& sample : kSamples) {
        const ankle_motor_ik::MotorAngles motors =
            ik_solver.solve(sample.roll, sample.pitch);

        double fk_roll = 0.0;
        double fk_pitch = 0.0;
        double roll_error = 0.0;
        double pitch_error = 0.0;
        bool fk_reachable = false;

        if (!motors.reachable()) {
            ++ik_failures;
        } else {
            const ankle_motor_fk::FootAngles foot =
                fk_solver.solve(motors.motor1, motors.motor2);
            fk_reachable = foot.reachable;
            if (foot.reachable) {
                fk_roll = foot.roll;
                fk_pitch = foot.pitch;
                roll_error = ankle_motor_ik::wrap_to_pi(fk_roll - sample.roll);
                pitch_error = ankle_motor_ik::wrap_to_pi(fk_pitch - sample.pitch);
                max_abs_roll_error =
                    std::max(max_abs_roll_error, std::abs(roll_error));
                max_abs_pitch_error =
                    std::max(max_abs_pitch_error, std::abs(pitch_error));
            } else {
                ++fk_failures;
            }
        }

        std::cout << sample.roll << ','
                  << sample.pitch << ','
                  << motors.motor1 << ','
                  << motors.motor2 << ','
                  << fk_roll << ','
                  << fk_pitch << ','
                  << roll_error << ','
                  << pitch_error << ','
                  << (motors.reachable() ? 1 : 0) << ','
                  << (fk_reachable ? 1 : 0) << '\n';
    }

    std::cout << "summary,max_abs_roll_error_rad="
              << max_abs_roll_error
              << ",max_abs_pitch_error_rad="
              << max_abs_pitch_error
              << ",ik_failures="
              << ik_failures
              << ",fk_failures="
              << fk_failures
              << '\n';

    return (ik_failures == 0 && fk_failures == 0) ? 0 : 1;
}
