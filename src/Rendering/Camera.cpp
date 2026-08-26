#include "Camera.h"
#include <GL/glew.h>
#include <glm/gtc/matrix_transform.hpp>

void Camera::Init(const AppSettings& config) {
    fov = config.camera.fov;
    heightMultiplier = config.camera.heightMultiplier;
    distanceMultiplier = config.camera.distanceMultiplier;
    farPlaneMultiplier = config.camera.farPlaneMultiplier;
    clearColor[0] = config.camera.clearColor[0];
    clearColor[1] = config.camera.clearColor[1];
    clearColor[2] = config.camera.clearColor[2];
    clearColor[3] = config.camera.clearColor[3];
}

void Camera::ApplyClearColor() const {
    glClearColor(clearColor[0], clearColor[1], clearColor[2], clearColor[3]);
}

glm::mat4 Camera::GetProjectionMatrix(int width, int height, float maxCoord) const {
    return glm::perspective(glm::radians(fov), (float)width / (float)height, 0.1f, maxCoord * farPlaneMultiplier);
}

glm::mat4 Camera::GetViewMatrix(float maxCoord) const {
    return glm::lookAt(
        glm::vec3(0, maxCoord * heightMultiplier, maxCoord * distanceMultiplier),
        glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)
    );
}