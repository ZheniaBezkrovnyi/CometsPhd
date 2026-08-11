#pragma once
#include <glm/glm.hpp>
#include "Core/AppSettings.h"

class Camera {
public:
    void Init(const AppSettings& config);
    void ApplyClearColor() const;
    glm::mat4 GetProjectionMatrix(int width, int height, float maxCoord) const;
    glm::mat4 GetViewMatrix(float currentAngle, float maxCoord) const;

private:
    float fov = 60.0f;
    float heightMultiplier = 0.5f;
    float distanceMultiplier = 1.8f;
    float farPlaneMultiplier = 10.0f;
    float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
};