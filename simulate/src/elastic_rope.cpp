#include "elastic_rope.h"

#include <mujoco/mujoco.h>

#include <algorithm>
#include <cmath>

namespace p1_sim {

void ElasticRope::advance(const mjModel* model, const mjData* data, int body_id)
{
    resetWrench();
    if (!model || !data || body_id < 0 || body_id >= model->nbody) {
        return;
    }

    mjtNum velocity[6] = {0.0};
    mj_objectVelocity(model, data, mjOBJ_BODY, body_id, velocity, 0);
    const Vec3 angular_velocity{velocity[0], velocity[1], velocity[2]};
    const Vec3 linear_velocity{velocity[3], velocity[4], velocity[5]};

    const mjtNum* body_origin = data->xpos + 3 * body_id;
    const mjtNum* body_com = data->xipos + 3 * body_id;
    const mjtNum* body_xmat = data->xmat + 9 * body_id;
    const double rope_stiffness = stiffness / static_cast<double>(kRopeCount);
    const double rope_damping = damping / static_cast<double>(kRopeCount);

    for (int i = 0; i < kRopeCount; ++i) {
        const Vec3 attachment =
            transformPoint(body_origin, body_xmat, localAttachmentPoint(i));
        const Vec3 body_offset = subtract(attachment, toVec3(body_origin));
        const Vec3 com_offset = subtract(attachment, toVec3(body_com));
        const Vec3 point_velocity =
            add(linear_velocity, cross(angular_velocity, body_offset));
        const Vec3 delta_x = subtract(anchorPoint(i), attachment);
        const double distance = norm(delta_x);
        if (distance < 1e-6) {
            continue;
        }

        const Vec3 direction = scale(delta_x, 1.0 / distance);
        const double velocity_along_rope = dot(point_velocity, direction);
        const double magnitude =
            std::max(0.0,
                     rope_stiffness * (distance - length) -
                     rope_damping * velocity_along_rope);
        const Vec3 force = scale(direction, magnitude);

        force_world = add(force_world, force);
        torque_world = add(torque_world, cross(com_offset, force));
    }
}

void ElasticRope::apply(mjData* data, int body_id) const
{
    if (!data || body_id < 0) {
        return;
    }
    const int offset = 6 * body_id;
    data->xfrc_applied[offset + 0] = force_world[0];
    data->xfrc_applied[offset + 1] = force_world[1];
    data->xfrc_applied[offset + 2] = force_world[2];
    data->xfrc_applied[offset + 3] = torque_world[0];
    data->xfrc_applied[offset + 4] = torque_world[1];
    data->xfrc_applied[offset + 5] = torque_world[2];
}

void ElasticRope::clear(mjData* data, int body_id) const
{
    if (!data || body_id < 0) {
        return;
    }
    const int offset = 6 * body_id;
    for (int i = 0; i < 6; ++i) {
        data->xfrc_applied[offset + i] = 0.0;
    }
}

double ElasticRope::averageDistanceToAnchors(const mjModel* model,
                                             const mjData* data,
                                             int body_id) const
{
    if (!model || !data || body_id < 0 || body_id >= model->nbody) {
        return 0.0;
    }

    const mjtNum* body_origin = data->xpos + 3 * body_id;
    const mjtNum* body_xmat = data->xmat + 9 * body_id;
    double distance_sum = 0.0;
    int valid_count = 0;
    for (int i = 0; i < kRopeCount; ++i) {
        const Vec3 attachment =
            transformPoint(body_origin, body_xmat, localAttachmentPoint(i));
        const Vec3 delta_x = subtract(anchorPoint(i), attachment);
        const double distance = norm(delta_x);
        if (distance > 1e-6) {
            distance_sum += distance;
            ++valid_count;
        }
    }

    return valid_count > 0 ? distance_sum / static_cast<double>(valid_count) : 0.0;
}

void ElasticRope::resetWrench()
{
    force_world = {0.0, 0.0, 0.0};
    torque_world = {0.0, 0.0, 0.0};
}

ElasticRope::Vec3 ElasticRope::anchorPoint(int index) const
{
    const double y = index == 0 ? side_offset_y : -side_offset_y;
    return {0.0, y, anchor_z};
}

ElasticRope::Vec3 ElasticRope::localAttachmentPoint(int index) const
{
    const double y = index == 0 ? side_offset_y : -side_offset_y;
    return {0.0, y, local_attach_z};
}

ElasticRope::Vec3 ElasticRope::toVec3(const mjtNum* value)
{
    return {value[0], value[1], value[2]};
}

ElasticRope::Vec3 ElasticRope::transformPoint(const mjtNum* origin,
                                              const mjtNum* xmat,
                                              const Vec3& local)
{
    return {
        origin[0] + xmat[0] * local[0] + xmat[1] * local[1] + xmat[2] * local[2],
        origin[1] + xmat[3] * local[0] + xmat[4] * local[1] + xmat[5] * local[2],
        origin[2] + xmat[6] * local[0] + xmat[7] * local[1] + xmat[8] * local[2],
    };
}

ElasticRope::Vec3 ElasticRope::add(const Vec3& a, const Vec3& b)
{
    return {a[0] + b[0], a[1] + b[1], a[2] + b[2]};
}

ElasticRope::Vec3 ElasticRope::subtract(const Vec3& a, const Vec3& b)
{
    return {a[0] - b[0], a[1] - b[1], a[2] - b[2]};
}

ElasticRope::Vec3 ElasticRope::scale(const Vec3& a, double value)
{
    return {a[0] * value, a[1] * value, a[2] * value};
}

double ElasticRope::dot(const Vec3& a, const Vec3& b)
{
    return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

ElasticRope::Vec3 ElasticRope::cross(const Vec3& a, const Vec3& b)
{
    return {
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    };
}

double ElasticRope::norm(const Vec3& a)
{
    return std::sqrt(dot(a, a));
}

}  // namespace p1_sim
