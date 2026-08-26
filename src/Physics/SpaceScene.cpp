#include "SpaceScene.h"
#include <glm/gtc/matrix_transform.hpp>
#include <algorithm>

void SpaceScene::Init(const AppSettings& config) {
    KeplerianElements cometElems = {
        config.physics.cometOrbit.a, config.physics.cometOrbit.e, config.physics.cometOrbit.i,
        config.physics.cometOrbit.Omega, config.physics.cometOrbit.w, config.physics.cometOrbit.M0, config.physics.cometOrbit.epoch
    };
    cometOrbit = OrbitalBody(cometElems);

    KeplerianElements earthElems = {
        config.physics.earthOrbit.a, config.physics.earthOrbit.e, config.physics.earthOrbit.i,
        config.physics.earthOrbit.Omega, config.physics.earthOrbit.w, config.physics.earthOrbit.M0, config.physics.earthOrbit.epoch
    };
    earthOrbit = OrbitalBody(earthElems);
}

void SpaceScene::Update(const SimulationTime& simTime, const AppSettings& config) {
    UpdateHeliocentricKinematics(simTime);
    UpdateObserverMetrics();
    UpdateSpinKinematics(simTime, config);
    UpdateLocalCoordinateSystems(config);
}


void SpaceScene::UpdateHeliocentricKinematics(const SimulationTime& simTime) {
    current_cometPos = cometOrbit.CalculatePosition(simTime.GetCurrentJD());
    current_earthPos = earthOrbit.CalculatePosition(simTime.GetCurrentJD());

    current_rh_AU = glm::length(current_cometPos);
}

void SpaceScene::UpdateObserverMetrics() {
    glm::dvec3 earthToComet = current_cometPos - current_earthPos;
    current_delta_AU = glm::length(earthToComet);

    glm::dvec3 cometToSun = glm::normalize(-current_cometPos);
    glm::dvec3 cometToEarth = glm::normalize(-earthToComet);

    double dotProduct = glm::dot(cometToSun, cometToEarth);
    dotProduct = std::clamp(dotProduct, -1.0, 1.0);

    current_phaseAngle_deg = std::acos(dotProduct) * (180.0 / PhysicsConsts::PI);
}

void SpaceScene::UpdateSpinKinematics(const SimulationTime& simTime, const AppSettings& config) {
    double rotationPeriodSeconds = config.physics.rotationPeriodHours * PhysicsConsts::SECONDS_PER_HOUR;
    double rotationSpeed = 2.0 * PhysicsConsts::PI / rotationPeriodSeconds;
    current_Angle = (float)std::fmod(simTime.GetElapsedSeconds() * rotationSpeed, 2.0 * PhysicsConsts::PI);
}

void SpaceScene::UpdateLocalCoordinateSystems(const AppSettings& config) {
    glm::dvec3 dirToSunWorld = glm::normalize(-current_cometPos);
    glm::dvec3 dirToEarthWorld = glm::normalize(current_earthPos - current_cometPos);

    float ra = glm::radians((float)config.physics.poleRA);
    float dec = glm::radians((float)config.physics.poleDEC);

    glm::mat4 spin = glm::rotate(glm::mat4(1.0f), current_Angle, glm::vec3(0.0f, 0.0f, 1.0f));

    glm::mat4 tilt = glm::rotate(glm::mat4(1.0f), ra, glm::vec3(0.0f, 0.0f, 1.0f)) *
        glm::rotate(glm::mat4(1.0f), glm::radians(90.0f) - dec, glm::vec3(0.0f, 1.0f, 0.0f));

    current_modelMatrix = tilt * spin;

    glm::mat3 invRot = glm::mat3(glm::inverse(current_modelMatrix));

    current_sunLocalDir = invRot * glm::vec3(dirToSunWorld);
    current_earthLocalDir = invRot * glm::vec3(dirToEarthWorld);
}