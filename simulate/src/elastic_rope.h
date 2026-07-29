#pragma once

#include <mujoco/mujoco.h>

#include <array>

namespace p1_sim {

class ElasticRope {
public:
    using Vec3 = std::array<double, 3>;

    void advance(const mjModel* model, const mjData* data, int body_id);
    void apply(mjData* data, int body_id) const;
    void clear(mjData* data, int body_id) const;
    double averageDistanceToAnchors(const mjModel* model, const mjData* data, int body_id) const;

    double length = 0.0;
    double stiffness = 250.0;
    double damping = 100.0;
    double anchor_z = 3.0;
    double side_offset_y = 0.18;
    double local_attach_z = 0.5;
    Vec3 force_world{0.0, 0.0, 0.0};
    Vec3 torque_world{0.0, 0.0, 0.0};

private:
    static constexpr int kRopeCount = 2;

    void resetWrench();
    Vec3 anchorPoint(int index) const;
    Vec3 localAttachmentPoint(int index) const;

    static Vec3 toVec3(const mjtNum* value);
    static Vec3 transformPoint(const mjtNum* origin,
                               const mjtNum* xmat,
                               const Vec3& local);
    static Vec3 add(const Vec3& a, const Vec3& b);
    static Vec3 subtract(const Vec3& a, const Vec3& b);
    static Vec3 scale(const Vec3& a, double value);
    static double dot(const Vec3& a, const Vec3& b);
    static Vec3 cross(const Vec3& a, const Vec3& b);
    static double norm(const Vec3& a);
};

}  // namespace p1_sim
