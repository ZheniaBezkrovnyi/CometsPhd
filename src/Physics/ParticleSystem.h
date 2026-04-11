#pragma once
#include <vector>
#include <memory>
#include <glm/glm.hpp>
#include "Particle.h"

class IForceGenerator;

struct ParticleData {
    std::vector<glm::dvec3> positions;
    std::vector<glm::dvec3> velocities;
    std::vector<glm::dvec3> forces;
    std::vector<double> masses;
    std::vector<float> radii;
    std::vector<bool> active;

    size_t count = 0;

    void Add(const Particle& p) {
        positions.push_back(p.position);
        velocities.push_back(p.velocity);
        forces.push_back(p.forceAccum);
        masses.push_back(p.mass);
        radii.push_back(p.radius);
        active.push_back(p.active);
        count++;
    }

    void Reserve(size_t capacity) {
        positions.reserve(capacity);
        velocities.reserve(capacity);
        forces.reserve(capacity);
        masses.reserve(capacity);
        radii.reserve(capacity);
        active.reserve(capacity);
    }
};

class ParticleSystem {
public:
    ParticleData data;
    std::vector<std::unique_ptr<IForceGenerator>> forces;

    void AddParticle(const Particle& p);
    void AddForce(std::unique_ptr<IForceGenerator> force);
    void Update(double dt);
    size_t GetCount() const;
};