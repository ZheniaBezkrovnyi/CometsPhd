#include "ParticleSystem.h"
#include "Forces.h"

void ParticleSystem::AddParticle(const Particle& p) {
    data.Add(p);
}

void ParticleSystem::AddForce(std::unique_ptr<IForceGenerator> force) {
    forces.push_back(std::move(force));
}

void ParticleSystem::Update(double dt) {
    for (size_t i = 0; i < data.count; i++) {
        if (data.active[i]) {
            data.forces[i] = glm::dvec3(0.0);
        }
    }

    for (auto& f : forces) {
        f->UpdateForce(data, dt);
    }

    for (size_t i = 0; i < data.count; i++) {
        if (!data.active[i]) continue;

        glm::dvec3 acceleration = data.forces[i] * (1.0 / data.masses[i]);
        data.velocities[i] += acceleration * dt;
        data.positions[i] += data.velocities[i] * dt;
    }
}

size_t ParticleSystem::GetCount() const {
    return data.count;
}