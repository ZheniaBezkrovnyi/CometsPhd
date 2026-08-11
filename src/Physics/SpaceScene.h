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
    glm::vec3 GetSunLocalDir() const { return current_sunLocalDir; }
    float GetCometRotationAngle() const { return current_Angle; }

private:
    void UpdateHeliocentricKinematics(const SimulationTime& simTime);
    void UpdateSpinKinematics(const SimulationTime& simTime, const AppSettings& config);
    void UpdateLocalCoordinateSystems();

    OrbitalBody cometOrbit;
    glm::dvec3 current_heliocentricPos = glm::dvec3(0.0);

    double current_rh_AU = 0.0;
    glm::vec3 current_sunLocalDir = glm::vec3(0.0f);
    float current_Angle = 0.0f;
};