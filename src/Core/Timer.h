#pragma once
#include <GLFW/glfw3.h>
#include <string>
#include <iomanip>
#include <sstream>

class FPSCounter {
public:
    void Update(GLFWwindow* window, const std::string& baseTitle) {
        double currentTime = glfwGetTime();
        frameCount++;

        if (currentTime - lastTime >= 1.0) {
            double msPerFrame = 1000.0 / double(frameCount);

            std::stringstream ss;
            ss << baseTitle << " | FPS: " << frameCount
                << " | " << std::fixed << std::setprecision(2) << msPerFrame << " ms";

            glfwSetWindowTitle(window, ss.str().c_str());

            frameCount = 0;
            lastTime = currentTime;
        }
    }

private:
    double lastTime = 0.0;
    int frameCount = 0;
};