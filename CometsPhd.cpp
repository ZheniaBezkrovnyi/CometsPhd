#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <iomanip>

#include "PhysicsConsts.h"
#include "PLY.h"
#include "Sun.h"    
#include "Forces.h"
#include "CometModel.h"
#include "ParticleSystem.h"

int main() {
    std::cout << "=== COMET SIMULATION: INITIALIZATION ===\n";

    double mComet = 1.0e13;
    double distanceCometToSun = 1.5 * PhysicsConsts::AU_METERS;
    double t = 0.0;
    double dt = 10.0;
    int totalFrames = 100;

    double dustRadius = 1e-4;
    double dustDensity = 1000.0;
    double dustMass = (4.0 / 3.0) * PhysicsConsts::PI * std::pow(dustRadius, 3) * dustDensity;

    Sun sun(Vector3(distanceCometToSun, 0, 0));

    CometModel comet;
    try {
        comet.LoadFromPLY("PLY`s/plyexample.ply");
    }
    catch (const std::exception& e) {
        std::cerr << "CRITICAL ERROR: " << e.what() << "\n";
        return -1;
    }

    ParticleSystem particleSys;

    particleSys.AddForce(std::make_unique<GravityForce>(Vector3(0, 0, 0), mComet));
    particleSys.AddForce(std::make_unique<SolarPressureForce>(sun, 1.0));
    particleSys.AddForce(std::make_unique<GasDragForce>());

    std::cout << "Particle Mass: " << dustMass << " kg\n";
    std::cout << "Starting Loop...\n\n";

    for (int frame = 0; frame < totalFrames; frame++) {
        double rotationSpeed = 2.0 * PhysicsConsts::PI / (12.0 * 3600.0);
        double currentAngle = t * rotationSpeed;

        Rotation currentRot = Rotation::FromEuler(0, currentAngle, 0);
        comet.SetTransform(Vector3(0, 0, 0), currentRot);
        comet.UpdateSurfacePhysics(sun.position);

        int spawnedCount = 0;

        static std::mt19937 rng(42);
        std::uniform_real_distribution<double> dist01(0.0, 1.0);

        for (const auto& tri : comet.triangles) {
            if (tri.gasProductionQ > 1e-12) {
                double dustFlowRate = tri.gasProductionQ * 1.0;
                double dustMassThisStep = dustFlowRate * dt;
                double numParticlesFloat = dustMassThisStep / dustMass;

                int numParticles = static_cast<int>(numParticlesFloat);
                if (dist01(rng) < (numParticlesFloat - numParticles)) {
                    numParticles++;
                }

                for (int i = 0; i < numParticles; i++) {
                    Vector3 spawnPos = tri.GetWorldCenter(comet.currentPosition, comet.currentRotation);

                    Particle p(spawnPos, dustMass, dustRadius);

                    p.velocity = tri.GetWorldNormal(comet.currentRotation) * 10; // need gasForce

                    particleSys.AddParticle(p);
                    spawnedCount++;
                }
            }
        }

        particleSys.Update(dt);
        t += dt;

        if (frame % 10 == 0) {
            std::cout << "Frame " << frame << " (t=" << t << "s) | ";
            std::cout << "Active Particles: " << particleSys.GetCount() << " | ";
            std::cout << "Comet Gas Rate: " << comet.totalGasProductionRate << " kg/s\n";

            if (particleSys.GetCount() > 0) {
                Particle p = particleSys.GetParticle(0);
                double dist = p.position.magnitude();
                std::cout << "   [TEST SAMPLE] Dist: " << std::fixed << std::setprecision(2) << dist
                    << "m | Vel: " << p.velocity.magnitude() << " m/s";

                std::cout << "\n";
            }
        }
    }

    std::cout << "=== SIMULATION FINISHED ===\n";
    return 0;
}