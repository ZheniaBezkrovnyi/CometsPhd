#pragma once
#include <iostream>
#include <cmath>
#include <chrono>
#include <vector>
#include "ParticleSystem.h"
#include "CometSurface.h"
#include "CometModel.h"

int main() {
    CometModel comet;
    ParticleSystem particleSys;

    const double dt = 1.0;       
    const double particleMass = 1e-5;
    const double gasVelocity = 600.0;

    try {
        std::cout << "Loading mesh...\n";
        comet.LoadFromPLY("plyexample");
    }
    catch (const std::exception& e) {
        std::cerr << "Error loading PLY: " << e.what() << std::endl;
        return -1;
    }

    Vector3 cometCenter(0, 0, 0);
    double cometMass = 1.0e13;
    double solarMass = 1.989e30;

    double au_meters = 1.496e11;
    Vector3 sunPos(1.5 * au_meters, 0, 0);

    particleSys.AddForce(std::make_unique<GravityForce>(cometCenter, cometMass));
    particleSys.AddForce(std::make_unique<GasDragForce>(cometCenter, comet.totalGasProductionRate, gasVelocity));


    std::cout << "Starting Simulation Loop...\n";
    int frames = 10;

    for (int i = 0; i < frames; i++) {
        std::cout << "\n--- Frame " << i << " (Time: " << i * dt << "s) ---\n";

        comet.UpdateSurfacePhysics(sunPos);

        std::cout << "Total Gas Q: " << comet.totalGasProductionRate << " kg/s\n";

        int spawnedThisFrame = 0;

        for (auto& tri : comet.triangles) {
            if (tri.gasProductionQ > 1e-9) {
                double dustMassFlow = tri.gasProductionQ * 1.0;
                double totalDustMass = dustMassFlow * dt;

                int count = static_cast<int>(totalDustMass / particleMass);

                if (count > 5) count = 5;

                for (int k = 0; k < count; k++) {
                    Vector3 initVel = tri.normal * (gasVelocity * 0.1);

                    Particle p(tri.center, particleMass);
                    p.velocity = initVel;

                    particleSys.AddParticle(p);
                    spawnedThisFrame++;
                }
            }
        }
        std::cout << "Spawned particles: " << spawnedThisFrame << ". Total count: " << particleSys.GetCount() << "\n";

        particleSys.Update(dt);

        if (particleSys.GetCount() > 0) {
            Particle p = particleSys.GetParticle(0);
            double dist = (p.position - cometCenter).magnitude();
            std::cout << "P[0] Dist: " << dist << "m | Vel: " << p.velocity.magnitude() << " m/s\n";

            if (dist < 100.0) std::cout << "Status: Near Surface (Landed/Starting)\n";
            else if (dist > 10000.0) std::cout << "Status: Escaping (Tail)\n";
            else std::cout << "Status: Flying\n";
        }
    }

    return 0;
}