#pragma once
#include <glm/glm.hpp>
#include "Core/AppSettings.h"
#include "Core/Timer.h"
#include "Physics/OrbitalBody.h"

class SpaceScene {
public:
    SpaceScene() = default;

    void Init(const AppSettings& config);
    void Update(const SimulationTime& simTime, const AppSettings& config);

    double GetCometHeliocentricDist() const { return current_rh_AU; }
    double GetCometGeocentricDist() const { return current_delta_AU; }
    double GetPhaseAngleDeg() const { return current_phaseAngle_deg; }

    glm::vec3 GetSunLocalDir() const { return current_sunLocalDir; }
    glm::vec3 GetEarthLocalDir() const { return current_earthLocalDir; }
    glm::mat4 GetCometModelMatrix() const { return current_modelMatrix; }


private:
    void UpdateHeliocentricKinematics(const SimulationTime& simTime);
    void UpdateObserverMetrics();
    void UpdateSpinKinematics(const SimulationTime& simTime, const AppSettings& config);
    void UpdateLocalCoordinateSystems(const AppSettings& config);

    OrbitalBody cometOrbit;
    OrbitalBody earthOrbit;

    glm::dvec3 current_cometPos = glm::dvec3(0.0);
    glm::dvec3 current_earthPos = glm::dvec3(0.0);
    glm::mat4 current_modelMatrix = glm::mat4(1.0f);

    double current_rh_AU = 0.0;
    double current_delta_AU = 0.0;
    double current_phaseAngle_deg = 0.0;

    glm::vec3 current_sunLocalDir = glm::vec3(0.0f);
    glm::vec3 current_earthLocalDir = glm::vec3(0.0f);
    float current_Angle = 0.0f;
};