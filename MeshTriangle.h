#pragma once
#include "Vector3.h"
#include "Rotation.h"

struct MeshTriangle {
    Vector3 localCenter;
    Vector3 localNormal;
    double area;

    double temperature;
    double gasProductionQ;
    double localGasVelocity;
    double gasFluxZ;

    MeshTriangle(Vector3 p1, Vector3 p2, Vector3 p3) {
        localCenter = (p1 + p2 + p3) * (1.0 / 3.0);

        Vector3 edge1 = p2 - p1;
        Vector3 edge2 = p3 - p1;
        Vector3 crossProd = edge1.cross(edge2);

        double crossMag = crossProd.magnitude();
        area = 0.5 * crossMag;

        localNormal = (crossMag > 1e-9) ? crossProd * (1.0 / crossMag) : Vector3(0, 1, 0);

        temperature = 20.0;
        gasProductionQ = 0.0;
        localGasVelocity = 0.0;
        gasFluxZ = 0.0;
    }

    Vector3 GetWorldNormal(const Rotation& rot) const {
        return rot.Apply(localNormal);
    }

    Vector3 GetWorldCenter(const Vector3& cometPos, const Rotation& rot) const {
        return rot.Apply(localCenter) + cometPos;
    }
};