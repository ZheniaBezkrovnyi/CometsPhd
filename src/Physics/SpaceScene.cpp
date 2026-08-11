#include "SpaceScene.h"
#include <glm/gtc/matrix_transform.hpp>

void SpaceScene::Init(const AppSettings& config) {
    KeplerianElements cometElems = {
        config.physics.cometOrbit.a,
        config.physics.cometOrbit.e,
        config.physics.cometOrbit.i,
        config.physics.cometOrbit.Omega,
        config.physics.cometOrbit.w,
        config.physics.cometOrbit.M0,
        config.physics.cometOrbit.epoch
    };
    cometOrbit = OrbitalBody(cometElems);
}

void SpaceScene::Update(const SimulationTime& simTime, const AppSettings& config) {
    UpdateHeliocentricKinematics(simTime);
    UpdateSpinKinematics(simTime, config);
    UpdateLocalCoordinateSystems();
}


void SpaceScene::UpdateHeliocentricKinematics(const SimulationTime& simTime) {
    current_heliocentricPos = cometOrbit.CalculatePosition(simTime.GetCurrentJD());
    current_rh_AU = glm::length(current_heliocentricPos);
}

void SpaceScene::UpdateSpinKinematics(const SimulationTime& simTime, const AppSettings& config) {
    double rotationPeriodSeconds = config.physics.rotationPeriodHours * PhysicsConsts::SECONDS_PER_HOUR;
    double rotationSpeed = 2.0 * PhysicsConsts::PI / rotationPeriodSeconds;
    current_Angle = (float)std::fmod(simTime.GetElapsedSeconds() * rotationSpeed, 2.0 * PhysicsConsts::PI);
}

void SpaceScene::UpdateLocalCoordinateSystems() {
    glm::dvec3 dirToSunWorld = glm::normalize(-current_heliocentricPos);

    glm::mat3 invRot = glm::mat3(glm::rotate(glm::mat4(1.0f), -current_Angle, glm::vec3(0.0f, 1.0f, 0.0f)));
    current_sunLocalDir = invRot * glm::vec3(dirToSunWorld);
}