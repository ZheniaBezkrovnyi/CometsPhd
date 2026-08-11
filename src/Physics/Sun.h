#pragma once
#include <glm/glm.hpp>
#include "Core/PhysicsConsts.h" 

class Sun {
public:
    glm::dvec3 position;
    double luminosity;
    double solarConstant;

    Sun(glm::dvec3 startPosition, double solarConst) : position(startPosition), solarConstant(solarConst) {
        luminosity = 1.0;
    }

    glm::dvec3 GetDirectionFrom(const glm::dvec3& point) const {
        return glm::normalize(position - point);
    }

    double GetDistanceFrom(const glm::dvec3& point) const {
        return glm::length(position - point);
    }

    double GetSolarFlux(const glm::dvec3& point) const {
        double dist = GetDistanceFrom(point);
        double r_AU = dist / PhysicsConsts::AU_METERS;
        if (r_AU < 1e-5) return solarConstant;
        return solarConstant / (r_AU * r_AU);
    }
};