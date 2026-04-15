#pragma once
#include <glm/glm.hpp>
#include "Core/PhysicsConsts.h"

struct MeshTriangle {
    glm::dvec3 localCenter;
    glm::dvec3 localNormal;
    double area;

    double temperature;
    double gasProductionQ;
    double localGasVelocity;
    double gasFluxZ;

    MeshTriangle(glm::dvec3 p1, glm::dvec3 p2, glm::dvec3 p3) {
        localCenter = (p1 + p2 + p3) * (1.0 / 3.0);

        glm::dvec3 edge1 = p2 - p1;
        glm::dvec3 edge2 = p3 - p1;
        glm::dvec3 crossProd = glm::cross(edge1, edge2);

        double crossMag = glm::length(crossProd);
        area = 0.5 * crossMag * PhysicsConsts::IncreaseAreaCometBy;

        localNormal = (crossMag > 1e-9) ? crossProd * (1.0 / crossMag) : glm::dvec3(0, 1, 0);

        temperature = 20.0;
        gasProductionQ = 0.0;
        localGasVelocity = 0.0;
        gasFluxZ = 0.0;
    }

    glm::dvec3 GetWorldNormal(const glm::dmat3& rot) const {
        return rot * localNormal;
    }

    glm::dvec3 GetWorldCenter(const glm::dvec3& cometPos, const glm::dmat3& rot) const {
        return (rot * localCenter) + cometPos;
    }
};