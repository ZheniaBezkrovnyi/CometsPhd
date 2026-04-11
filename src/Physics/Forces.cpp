#define GLM_ENABLE_EXPERIMENTAL

#include "Forces.h"
#include <glm/gtx/norm.hpp>


GasDragForce::GasDragForce(glm::dvec3 center, double Q_gas, double v_terminal)
    : cometCenter(center), gasProductionRate(Q_gas), v_gas(v_terminal) {
}

void GasDragForce::UpdateForce(ParticleData& data, double dt) {
    for (size_t i = 0; i < data.count; i++) {
        if (!data.active[i]) continue;

        glm::dvec3 r_vec = data.positions[i] - cometCenter;
        double r = glm::length(r_vec);

        if (r < 1e-9) continue;

        double surfaceAreaAtR = 4 * PhysicsConsts::PI * r * r;
        double rho_gas = gasProductionRate / (surfaceAreaAtR * v_gas);

        double A_cross = PhysicsConsts::PI * data.radii[i] * data.radii[i];
        double Cd = 2.0;

        glm::dvec3 dragDir = glm::normalize(r_vec);
        double f_drag_mag = 0.5 * Cd * A_cross * rho_gas * v_gas * v_gas;

        data.forces[i] += dragDir * f_drag_mag;
    }
}

SolarPressureForce::SolarPressureForce(const Sun& sunRef, double efficiency)
    : sun(sunRef), Q_pr(efficiency) {
}

void SolarPressureForce::UpdateForce(ParticleData& data, double dt) {
    for (size_t i = 0; i < data.count; i++) {
        if (!data.active[i]) continue;

        double solarFlux = sun.GetSolarFlux(data.positions[i]);
        double pressure = solarFlux / PhysicsConsts::SpeedOfLight;

        glm::dvec3 dir = sun.GetDirectionFrom(data.positions[i]) * -1.0;
        double area = PhysicsConsts::PI * data.radii[i] * data.radii[i];
        double force = pressure * area * Q_pr;

        data.forces[i] += dir * force;
    }
}

GravityForce::GravityForce(glm::dvec3 cometCenter, double cometMass)
    : center(cometCenter), M_comet(cometMass) {
}

void GravityForce::UpdateForce(ParticleData& data, double dt) {
    for (size_t i = 0; i < data.count; i++) {
        if (!data.active[i]) continue;

        glm::dvec3 dir = data.positions[i] - center;
        double distSq = glm::length2(dir);

        if (distSq < 1.0) continue;

        double forceMag = (PhysicsConsts::G * M_comet * data.masses[i]) / distSq;
        glm::dvec3 force = glm::normalize(dir) * (-forceMag);

        data.forces[i] += force;
    }
}