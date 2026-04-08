#pragma once
#include "PhysicsConsts.h" 
#include "Vector3.h"


class Sun {
public:
    Vector3 position;
    double luminosity;

    Sun(Vector3 startPosition) : position(startPosition) {
        luminosity = 1.0;
    }

    Vector3 GetDirectionFrom(const Vector3& point) const {
        return (position - point).normalized();
    }

    double GetDistanceFrom(const Vector3& point) const {
        Vector3 vec = position - point;
        return vec.magnitude();
    }

    double GetSolarFlux(const Vector3& point) const {
        double dist = GetDistanceFrom(point);
        double r_AU = dist / PhysicsConsts::AU_METERS;
        if (r_AU < 1e-5) return PhysicsConsts::SolarConst;
        return PhysicsConsts::SolarConst / (r_AU * r_AU);
    }
};