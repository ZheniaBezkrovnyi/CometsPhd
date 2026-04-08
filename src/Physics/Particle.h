#pragma once
#include <glm/glm.hpp>

struct Particle {
    glm::dvec3 position;
    glm::dvec3 velocity;
    glm::dvec3 forceAccum;
    double mass;
    float radius;
    bool active;

    Particle(glm::dvec3 pos, double m, float r)
        : position(pos), velocity(0.0), forceAccum(0.0), mass(m), radius(r), active(true) {
    }
};