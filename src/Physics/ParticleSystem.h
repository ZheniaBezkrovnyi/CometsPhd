#pragma once
#include <vector>
#include <memory>
#include "Particle.h"
#include "Forces.h"
class ParticleSystem {
    std::vector<Particle> particles;
    std::vector<std::unique_ptr<IForceGenerator>> forces;

public:
    void AddParticle(const Particle& p) {
        particles.push_back(p);
    }

    void AddForce(std::unique_ptr<IForceGenerator> force) {
        forces.push_back(std::move(force));
    }

    void Update(double dt) {
        for (auto& p : particles) {
            if (!p.active) continue;

            p.forceAccum = Vector3(0, 0, 0);

            for (auto& f : forces) {
                f->UpdateForce(p, dt);
            }

            Vector3 acceleration = p.forceAccum * (1.0 / p.mass);
            p.velocity = p.velocity + (acceleration * dt);
            p.position = p.position + (p.velocity * dt);
        }
    }

    size_t GetCount() const { return particles.size(); }
    Particle GetParticle(size_t i) const { return particles[i]; }
};