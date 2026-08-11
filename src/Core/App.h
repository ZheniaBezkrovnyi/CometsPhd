#pragma once
#include "GLContext.h"
#include "Rendering/OptixRenderer.h"
#include "Rendering/Camera.h"
#include "Rendering/GLMesh.h"
#include "Geometry/Geometry.h"
#include "Core/AppSettings.h"
#include "Core/Timer.h"
#include "Physics/SpaceScene.h"
#include "Utils/ScreenshotCapture.h"
#include <memory>

class App {
public:
    App();
    ~App();

    bool Init();
    void Run();

private:
    void Update(double dt);
    void RunOptixSimulation();
    void RenderOpenGL();

    AppSettings config;
    std::unique_ptr<GLContext> glContext;
    std::unique_ptr<OptixRenderer> optixRenderer;
    FPSCounter fpsCounter;
    ScreenshotCaptureState screenshotState;

    Camera camera;
    GLMesh cometMesh;
    float4* d_prevTemperature = nullptr;
    float maxCoord = 0.0f;

    SimulationTime simTime;
    SpaceScene spaceScene;
    unsigned int frameCount = 0;
};