#pragma once
#include <glm/glm.hpp>
#include "ParticleSystem.h"
#include "PhysicsConsts.h"
#include "Sun.h"

class IForceGenerator {
public:
    virtual ~IForceGenerator() = default;
    virtual void UpdateForce(ParticleData& data, double dt) = 0;
};

class GasDragForce : public IForceGenerator {
    glm::dvec3 cometCenter;
    double gasProductionRate;
    double v_gas;

public:
    GasDragForce(glm::dvec3 center, double Q_gas, double v_terminal);
    void UpdateForce(ParticleData& data, double dt) override;
};

class SolarPressureForce : public IForceGenerator {
    const Sun& sun;
    double Q_pr;

public:
    SolarPressureForce(const Sun& sunRef, double efficiency = 1.0);
    void UpdateForce(ParticleData& data, double dt) override;
};

class GravityForce : public IForceGenerator {
    glm::dvec3 center;
    double M_comet;
public:
    GravityForce(glm::dvec3 cometCenter, double cometMass);
    void UpdateForce(ParticleData& data, double dt) override;
};