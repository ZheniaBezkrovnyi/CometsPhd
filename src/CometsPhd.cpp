#include <iostream>
#include <vector>
#include <memory>
#include <cmath>
#include <random>
#include <iomanip>
#include <string>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>

#include "PhysicsConsts.h"
#include "PLY.h"
#include "Sun.h"    
#include "Forces.h"
#include "CometModel.h"
#include "ParticleSystem.h"


std::string FormatTime(double seconds) {
    int days = static_cast<int>(seconds / 86400);
    int hours = static_cast<int>(std::fmod(seconds, 86400) / 3600);
    int mins = static_cast<int>(std::fmod(seconds, 3600) / 60);

    std::string res = "";
    if (days > 0) res += std::to_string(days) + "d ";
    if (hours > 0 || days > 0) res += std::to_string(hours) + "h ";
    res += std::to_string(mins) + "m";
    return res;
}

void PrintSimulationConfig(double mComet, double dustMass, int totalFrames, double dt) {
    std::cout << "=================================================\n";
    std::cout << "         COMET DUST EMISSION SIMULATOR           \n";
    std::cout << "=================================================\n";
    std::cout << "[CONFIG] Comet Mass:      " << std::scientific << mComet << " kg\n";
    std::cout << "[CONFIG] Dust Ptl Mass:   " << dustMass << " kg\n";
    std::cout << "[CONFIG] Time Step (dt):  " << std::fixed << std::setprecision(1) << dt << " s\n";
    std::cout << "[CONFIG] Total Frames:    " << totalFrames << "\n";
    std::cout << "=================================================\n\n";
}

void LogSimulationStep(int frame, int totalFrames, double t, size_t activeParticles, double gasRate) {
    std::cout << "\r[RUNNING] Frame: " << std::setw(5) << frame << "/" << totalFrames
        << " | Time: " << std::setw(8) << FormatTime(t)
        << " | Particles: " << std::setw(7) << activeParticles
        << " | Total Gas Rate: " << std::scientific << std::setprecision(2) << gasRate << " kg/s"
        << std::flush;
}


void SetupPhysicsForces(ParticleSystem& particleSys, const Sun& sun, double mComet) {
    particleSys.AddForce(std::make_unique<GravityForce>(glm::dvec3(0, 0, 0), mComet));
    particleSys.AddForce(std::make_unique<SolarPressureForce>(sun, 1.0));
}

void UpdateCometState(CometModel& comet, const Sun& sun, double t) {
    double rotationSpeed = 2.0 * PhysicsConsts::PI / (12.0 * 3600.0);
    double currentAngle = t * rotationSpeed;

    glm::dmat4 rot4 = glm::rotate(glm::dmat4(1.0), currentAngle, glm::dvec3(0.0, 1.0, 0.0));
    comet.SetTransform(glm::dvec3(0, 0, 0), glm::dmat3(rot4));
    comet.UpdateSurfacePhysics(sun.position);
}

void EmitDustFromSurface(CometModel& comet, ParticleSystem& particleSys, double dt, double dustMass, double dustRadius, std::mt19937& rng) {
    std::uniform_real_distribution<double> dist01(0.0, 1.0);

    for (const auto& tri : comet.triangles) {
        if (tri.gasProductionQ > 1e-12) {
            double dustMassThisStep = tri.gasProductionQ * dt;
            double numParticlesFloat = dustMassThisStep / dustMass;

            int numParticles = static_cast<int>(numParticlesFloat);
            if (dist01(rng) < (numParticlesFloat - numParticles)) {
                numParticles++;
            }

            for (int i = 0; i < numParticles; i++) {
                glm::dvec3 spawnPos = tri.GetWorldCenter(comet.currentPosition, comet.currentRotation);
                Particle p(spawnPos, dustMass, dustRadius);
                p.velocity = tri.GetWorldNormal(comet.currentRotation) * tri.localGasVelocity;
                particleSys.AddParticle(p);
            }
        }
    }
}

int main() {
    double mComet = 1.0e13;
    double distanceCometToSun = 1.5 * PhysicsConsts::AU_METERS;
    double t = 0.0;
    double dt = 10.0;
    int totalFrames = 1000;

    double dustRadius = 1e-4;
    double dustDensity = 1000.0;
    double dustMass = (4.0 / 3.0) * PhysicsConsts::PI * std::pow(dustRadius, 3) * dustDensity;

    PrintSimulationConfig(mComet, dustMass, totalFrames, dt);

    std::cout << "[INIT] Booting simulation systems...\n";

    Sun sun(glm::dvec3(distanceCometToSun, 0.0, 0.0));
    CometModel comet;

    try {
        comet.LoadFromPLY("data/plyexample.ply");
    }
    catch (const std::exception& e) {
        std::cerr << "[ERROR] Failed to load shape model: " << e.what() << "\n";
        return -1;
    }

    ParticleSystem particleSys;
    particleSys.data.Reserve(1000000);
    SetupPhysicsForces(particleSys, sun, mComet);

    std::cout << "[INIT] Ready. Starting integration loop.\n";
    std::cout << "-------------------------------------------------\n";

    std::mt19937 rng(42);

    for (int frame = 0; frame <= totalFrames; frame++) {

        UpdateCometState(comet, sun, t);
        EmitDustFromSurface(comet, particleSys, dt, dustMass, dustRadius, rng);

        particleSys.Update(dt);

        t += dt;
        LogSimulationStep(frame, totalFrames, t, particleSys.GetCount(), comet.totalGasProductionRate);
    }

    std::cout << "\n-------------------------------------------------\n";
    std::cout << "[FINISH] Simulation completed successfully.\n";

    return 0;
}