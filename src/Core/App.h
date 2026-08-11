#pragma once
#include "GLContext.h"
#include "Rendering/OptixRenderer.h"
#include "Geometry/Geometry.h"
#include "Core/AppSettings.h"
#include "Core/Timer.h"
#include "Physics/OrbitalBody.h"
#include "Utils/ScreenshotCapture.h"
#include <cuda_gl_interop.h>
#include <vector>
#include <memory>

class App {
public:
    App();
    ~App();

    bool Init();
    void Run();

private:
    void Update(double dt);
    void Draw();
    std::vector<InteropVertex> LoadAndPrepareGeometry(const std::string& filepath, float& outMaxCoord, float baseTemp);

    void UpdateTransformations();
    void RunOptixSimulation();
    void RenderOpenGL();

    AppSettings config;
    std::unique_ptr<GLContext> glContext;
    std::unique_ptr<OptixRenderer> optixRenderer;
    FPSCounter fpsCounter;
    ScreenshotCaptureState screenshotState;

    GLuint vbo = 0;
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    float4* d_prevTemperature = nullptr;

    std::vector<InteropVertex> hostVertices;
    int totalVertices = 0;
    float maxCoord = 0.0f;

    SimulationTime simTime;
    OrbitalBody cometBody;
    unsigned int frameCount = 0;

    double current_rh_AU = 0.0;
    glm::vec3 current_sunLocalDir = glm::vec3(0.0f);
    float current_Angle = 0.0f;
};