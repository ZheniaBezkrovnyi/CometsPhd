#pragma once
#include "Particle.h"
#include "PhysicsConsts.h"

class IForceGenerator {
public:
    virtual ~IForceGenerator() = default;
    virtual void UpdateForce(Particle& particle, double dt) = 0;
};

class GravityForce : public IForceGenerator {
    Vector3 center;
    double M_comet;
public:
    GravityForce(Vector3 cometCenter, double cometMass) : center(cometCenter), M_comet(cometMass) {}

    void UpdateForce(Particle& p, double dt) override {
        if (!p.active) return;
        Vector3 dir = p.position - center;
        double dist = dir.magnitude();
        if (dist < 1.0) return;

        double forceMag = (PhysicsConsts::G * M_comet * p.mass) / (dist * dist);
        Vector3 force = dir.normalized() * (-forceMag);

        p.forceAccum = p.forceAccum + force;
    }
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
        if (!p.active) return;

        Vector3 r_vec = p.position - cometCenter;
        double r = r_vec.magnitude();

        double surfaceAreaAtR = 4 * PhysicsConsts::PI * r * r;
        double rho_gas = gasProductionRate / (surfaceAreaAtR * v_gas);

        double A_cross = 1e-10;
        double Cd = 2.0; 

        Vector3 dragDir = r_vec.normalized();
        double f_drag_mag = 0.5 * Cd * A_cross * rho_gas * v_gas * v_gas;

        p.forceAccum = p.forceAccum + (dragDir * f_drag_mag);
    }
};