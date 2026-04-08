#pragma once
#include "Vector3.h"

struct Rotation {
    double m[3][3];

    static Rotation Identity() {
        Rotation r;
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                r.m[i][j] = (i == j ? 1.0 : 0.0);
        return r;
    }

    static Rotation FromEuler(double x, double y, double z) {
        double cx = std::cos(x), sx = std::sin(x);
        double cy = std::cos(y), sy = std::sin(y);
        double cz = std::cos(z), sz = std::sin(z);

        Rotation r;
        r.m[0][0] = cy * cz;                  r.m[0][1] = -cy * sz;                  r.m[0][2] = sy;
        r.m[1][0] = cx * sz + sx * sy * cz;   r.m[1][1] = cx * cz - sx * sy * sz;    r.m[1][2] = -sx * cy;
        r.m[2][0] = sx * sz - cx * sy * cz;   r.m[2][1] = sx * cz + cx * sy * sz;    r.m[2][2] = cx * cy;
        return r;
    }

    Vector3 Apply(const Vector3& v) const {
        return Vector3(
            m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z,
            m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z,
            m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z
        );
    }
};