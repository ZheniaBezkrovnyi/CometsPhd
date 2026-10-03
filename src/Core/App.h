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
#include <fstream>
#include <iomanip>
#include <filesystem>

class App {
public:
    App(const std::filesystem::path& configPath,
        const std::filesystem::path& outputDirectory,
        const std::filesystem::path& exeDir);
    ~App();

    bool Init();
    void Run();

private:
    void Update(double dt);
    void RunOptixThermal(InteropVertex* d_vertices);
    float RunOptixPhotometry(InteropVertex* d_vertices);
    void RenderOpenGL();
    void OnSimulationComplete();
    void RecordData();
    void RunPostProcessing();

    AppSettings config;
    std::filesystem::path outputDir;
    std::string screenshotDir;

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

    std::ofstream photometryLog;
    bool simulationFinished = false;
};