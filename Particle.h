#pragma once
#include "Vector3.h"

struct Particle {
    Vector3 position;
    Vector3 velocity;
    Vector3 forceAccum; 
    double mass;
    bool active;

    Particle(Vector3 pos, double m) : position(pos), mass(m), active(true) {}
};