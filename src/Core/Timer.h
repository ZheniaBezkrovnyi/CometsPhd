#pragma once
#include <GLFW/glfw3.h>
#include <string>
#include <iomanip>
#include <sstream>
#include "PhysicsConsts.h"

class SimulationTime {
public:
    void Init(double scale, double startJulianDate) {
        timeScale = scale;
        baseJD = startJulianDate;
        elapsedSeconds = 0.0;
    }

    void Advance(double realDt) {
        elapsedSeconds += realDt * timeScale;
    }

    double GetElapsedSeconds() const { return elapsedSeconds; }
    double GetCurrentJD() const { return baseJD + (elapsedSeconds / PhysicsConsts::SECONDS_PER_DAY); }

    void SetTimeScale(double newScale) { timeScale = newScale; }
    double GetTimeScale() const { return timeScale; }

private:
    double timeScale = 1.0;
    double baseJD = 0.0;
    double elapsedSeconds = 0.0;
};

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