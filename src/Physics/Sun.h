#pragma once
#include <glm/glm.hpp>
#include "PhysicsConsts.h" 

class Sun {
public:
    glm::dvec3 position;
    double luminosity;

    Sun(glm::dvec3 startPosition) : position(startPosition) {
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
        if (r_AU < 1e-5) return PhysicsConsts::SolarConst;
        return PhysicsConsts::SolarConst / (r_AU * r_AU);
    }
};