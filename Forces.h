#pragma once
#include "Particle.h"
#include "PhysicsConsts.h"
#include "Sun.h"

class IForceGenerator {
public:
    virtual ~IForceGenerator() = default;
    virtual void UpdateForce(Particle& particle, double dt) = 0;
};


class GasDragForce : public IForceGenerator {
    Vector3 cometCenter;
    double gasProductionRate;
    double v_gas;   

public:
    GasDragForce(Vector3 center, double Q_gas, double v_terminal)
        : cometCenter(center), gasProductionRate(Q_gas), v_gas(v_terminal) {
    }

    void UpdateForce(Particle& p, double dt) override {
        if (!p.active || true) return;

        Vector3 r_vec = p.position - cometCenter;
        double r = r_vec.magnitude();

        double surfaceAreaAtR = 4 * PhysicsConsts::PI * r * r;
        double rho_gas = gasProductionRate / (surfaceAreaAtR * v_gas);

        double A_cross = PhysicsConsts::PI * p.radius * p.radius;
        double Cd = 2.0; 

        Vector3 dragDir = r_vec.normalized();
        double f_drag_mag = 0.5 * Cd * A_cross * rho_gas * v_gas * v_gas;

        p.forceAccum = p.forceAccum + (dragDir * f_drag_mag);
    }
};


class SolarPressureForce : public IForceGenerator {
    const Sun& sun;
    double Q_pr; //efficiency (1.0 for black)

public:
    SolarPressureForce(const Sun& sunRef, double efficiency = 1.0)
        : sun(sunRef), Q_pr(efficiency) {
    }

    void UpdateForce(Particle& p, double dt) override {
        if (!p.active) return;

        double solarFlux = sun.GetSolarFlux(p.position);
        double pressure = solarFlux / PhysicsConsts::SpeedOfLight;

        Vector3 dir = sun.GetDirectionFrom(p.position) * -1.0;

        double area = PhysicsConsts::PI * p.radius * p.radius;
        double force = pressure * area * Q_pr;

        p.forceAccum = p.forceAccum + (dir * force);
    }
};

class GravityForce : public IForceGenerator {
    Vector3 center;
    double M_comet;
public:
    GravityForce(Vector3 cometCenter, double cometMass)
        : center(cometCenter), M_comet(cometMass) {
    }

    void UpdateForce(Particle& p, double dt) override {
        if (!p.active) return;
        Vector3 dir = p.position - center;
        double distSq = dir.x * dir.x + dir.y * dir.y + dir.z * dir.z;
        double dist = std::sqrt(distSq);

        if (dist < 1.0) return;

        double forceMag = (PhysicsConsts::G * M_comet * p.mass) / distSq;

        Vector3 force = dir * (-1.0 / dist) * forceMag;

        p.forceAccum = p.forceAccum + force;
    }
};