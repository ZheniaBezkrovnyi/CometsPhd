#include "App.h"
#include <iostream>
#include <filesystem>
#include "Physics/Photometry.h"
#include <glm/gtc/type_ptr.hpp>
#include "Geometry/ModelLoader.h"

#include <cstdlib>

namespace fs = std::filesystem;

App::App() {
    std::string configPath = "";
    std::string configDir = "C:/Users/Yevhen/Projects/Univ/CometsPhd/Configs";

    if (fs::exists(configDir) && fs::is_directory(configDir)) {
        for (const auto& entry : fs::directory_iterator(configDir)) {
            if (entry.is_regular_file() && entry.path().extension() == ".json") {
                configPath = entry.path().string();
                break;
            }
        }
    }

    if (!configPath.empty()) {
        config.LoadFromJson(configPath);
    }
    else {
        std::cerr << "[WARNING] No .json file found in '" << configDir << "' directory. Using hardcoded defaults.\n";
    }

    glContext = std::make_unique<GLContext>();
    optixRenderer = std::make_unique<OptixRenderer>();

    spaceScene.Init(config);
}

App::~App() {
    if (d_prevTemperature) cudaFree(d_prevTemperature);

    if (photometryLog.is_open()) photometryLog.close();
}

bool App::Init() {
    if (!glContext->Init(config)) { std::cerr << "Failed: glContext->Init" << std::endl; return false; }
    if (!optixRenderer->Init(config)) { std::cerr << "Failed: optixRenderer->Init" << std::endl; return false; }

    camera.Init(config);

    std::vector<InteropVertex> vertices = ModelLoader::LoadPLY(config.modelPath, maxCoord, config.thermal.minTemp);
    if (vertices.empty()) { std::cerr << "Failed: Model is empty or PLY path is wrong!" << std::endl; return false; }

    cometMesh.Upload(vertices);
    simTime.Init(config.physics.timeScale, config.physics.startJulianDate);

    cudaMalloc(&d_prevTemperature, cometMesh.GetVertexCount() * sizeof(float4));

    InteropVertex* d_vertices = cometMesh.MapToCUDA();
    if (!optixRenderer->BuildGAS((CUdeviceptr)d_vertices, cometMesh.GetVertexCount())) { std::cerr << "Failed: OptiX BuildGAS" << std::endl; return false; }
    if (!optixRenderer->BuildPipeline(config.ptxPath)) { std::cerr << "Failed: OptiX BuildPipeline" << std::endl; return false; }
    cometMesh.UnmapFromCUDA();

    photometryLog.open("C:\\Users\\Yevhen\\Projects\\Univ\\CometsPhd\\Dop\\Graphs\\photometry_log.csv");
    if (photometryLog.is_open()) {
        photometryLog << "JD,PhaseAngle_deg,Distance_AU,ApparentMagnitude" << std::endl;
    }

    return true;
}

void App::Run() {
    const double FIXED_DT = 1.0 / 60.0;

    while (!glContext->ShouldClose()) {
        glContext->PollEvents();

        Update(FIXED_DT);
        RenderOpenGL();

        CaptureScreenshotIfNeeded(
            config.screenshotCapture.outputDir,
            config.screenshotCapture.enabled,
            config.screenshotCapture.maxFrames,
            config.screenshotCapture.frameStride,
            screenshotState,
            frameCount,
            glContext->GetWidth(),
            glContext->GetHeight()
        );

        glContext->SwapBuffers();
        frameCount++;
        fpsCounter.Update(glContext->GetWindow(), config.window.title);
    }
}

void App::Update(double dt) {
    simTime.Advance(dt);
    spaceScene.Update(simTime, config);

    if (simulationFinished) return; 

    InteropVertex* d_vertices = cometMesh.MapToCUDA();
    RunOptixThermal(d_vertices);

    if (frameCount % 60 == 0 || true) {
        float visibleArea = RunOptixPhotometry(d_vertices) * 1000000.0;
        double distanceAU = spaceScene.GetCometGeocentricDist();
        double currentMag = Photometry::CalculateMagnitudeFromVisibleArea(
            visibleArea,
            spaceScene.GetCometHeliocentricDist(),
            distanceAU,
            config.thermal.albedo
        );

        double jd = simTime.GetCurrentJD();
        double phaseAngle = spaceScene.GetPhaseAngleDeg();
        double realTimeHours = simTime.GetElapsedSeconds() / 3600.0;

        if (realTimeHours < config.physics.rotationPeriodHours * 3) {
            std::cout << "[Metrics] JD: " << std::fixed << std::setprecision(2) << jd
                << " | Phase Angle: " << phaseAngle << " deg"
                << " | Dist: " << distanceAU << " AU"
                << " | Mag: " << currentMag << ""
                << " | ElapsedHours: " << realTimeHours << "\n";

            if (photometryLog.is_open()) {
                photometryLog << std::fixed << std::setprecision(6)
                    << jd << "," << phaseAngle << "," << distanceAU << "," << currentMag << std::endl;
            }
        }
        else {
            OnSimulationComplete();
        }
    }

    cometMesh.UnmapFromCUDA();
}
void App::OnSimulationComplete() {
    simulationFinished = true;

    if (photometryLog.is_open()) {
        photometryLog.close();
    }

    std::cout << "\n Running Python scripts...\n";
    std::system("python C:\\Users\\Yevhen\\Projects\\Univ\\CometsPhd\\Dop\\Graphs\\plot_lightcurve.py");
    //std::system("python C:\\Users\\Yevhen\\Projects\\Univ\\CometsPhd\\Dop\\Graphs\\plot_double_lightcurve.py");

    glfwSetWindowShouldClose(glContext->GetWindow(), GLFW_TRUE);
}

void App::RunOptixThermal(InteropVertex* d_vertices) {
    optixRenderer->CopyPreviousTemperatures(d_vertices, d_prevTemperature, cometMesh.GetVertexCount());

    OptixParams params = {};
    params.vertices = d_vertices;
    params.prevTemperature = d_prevTemperature;
    params.numVertices = cometMesh.GetVertexCount();

    glm::vec3 sunDir = spaceScene.GetSunLocalDir();
    params.sunDir = make_float3(sunDir.x, sunDir.y, sunDir.z);

    params.rh_AU = (float)spaceScene.GetCometHeliocentricDist();
    params.solarConstant = config.thermal.solarConstant;
    params.albedo = config.thermal.albedo;
    params.emissivity = config.thermal.emissivity;
    params.activeFraction = config.thermal.activeFraction;
    params.minTemp = config.thermal.minTemp;
    params.maxTempForColor = config.thermal.maxTempForColor;
    params.frameCount = frameCount;
    params.indirectSamples = config.thermal.indirectSamples;
    params.indirectSeed = config.thermal.indirectSeed;
    params.indirectSolarScale = config.thermal.indirectSolarScale;
    params.indirectIRScale = config.thermal.indirectIRScale;
    params.maxIndirectFractionOfSolarFlux = config.thermal.maxIndirectFractionOfSolarFlux;
    params.rayEpsilon = config.thermal.rayEpsilon;

    optixRenderer->RenderThermal(params, cometMesh.GetVertexCount() / 3);
}

float App::RunOptixPhotometry(InteropVertex* d_vertices) {
    OptixParams params = {};
    params.vertices = d_vertices;
    params.numVertices = cometMesh.GetVertexCount();

    glm::vec3 sunDir = spaceScene.GetSunLocalDir();
    params.sunDir = make_float3(sunDir.x, sunDir.y, sunDir.z);

    glm::vec3 earthDir = spaceScene.GetEarthLocalDir();
    params.earthDir = make_float3(earthDir.x, earthDir.y, earthDir.z);

    params.rayEpsilon = config.thermal.rayEpsilon;

    return optixRenderer->RenderPhotometry(params, cometMesh.GetVertexCount() / 3);
}

void App::RenderOpenGL() {
    int width = glContext->GetWidth();
    int height = glContext->GetHeight();

    glViewport(0, 0, width, height);

    camera.ApplyClearColor();
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glm::mat4 proj = camera.GetProjectionMatrix(width, height, maxCoord);
    glLoadMatrixf(glm::value_ptr(proj));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();

    glm::mat4 view = camera.GetViewMatrix(maxCoord);
    glm::mat4 model = spaceScene.GetCometModelMatrix();
    glm::mat4 modelView = view * model;
    glLoadMatrixf(glm::value_ptr(modelView));

    cometMesh.BindAndDraw();
}