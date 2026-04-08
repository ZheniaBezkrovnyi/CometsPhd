#pragma once
#include "Vector3.h"

struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 forceAccum; 
    double mass;
    float radius;
    bool active;

    Particle(Vector3 pos, double m, float r) : position(pos), mass(m), radius(r), active(true) {}
};