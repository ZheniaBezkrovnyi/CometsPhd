#pragma once
#include "GLContext.h"
#include "Rendering/OptixRenderer.h"
#include "Geometry/Geometry.h"
#include "Core/AppSettings.h"
#include "Core/Timer.h"
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

    double simulationTime = 0.0;
    unsigned int frameCount = 0;
};