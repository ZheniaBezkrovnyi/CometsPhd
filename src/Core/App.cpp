#include "App.h"
#include <iostream>
#include <filesystem>
#include "Physics/Photometry.h"
#include <glm/gtc/type_ptr.hpp>
#include "Geometry/ModelLoader.h"
#include <cstdlib>
#include <stdexcept>

namespace fs = std::filesystem;

App::App(const fs::path& configPath, const fs::path& outputDirectory, const fs::path& exeDir)
    : outputDir(outputDirectory) {
    if (!fs::exists(configPath)) {
        throw std::runtime_error("Config not found: " + configPath.string());
    }
    config.LoadFromJson(configPath.string());

    if (fs::path(config.ptxPath).is_relative()) {
        config.ptxPath = (exeDir / config.ptxPath).string();
    }

    const fs::path shots = config.screenshotCapture.outputDir;
    screenshotDir = (shots.is_relative() ? outputDir / shots : shots).string();

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

    fs::create_directories(outputDir);
    std::ofstream(outputDir / "config.json") << config.ToJson() << "\n";

    photometryLog.open(outputDir / "photometry_log.csv");
    if (!photometryLog.is_open()) {
        throw std::runtime_error("Cannot open " + (outputDir / "photometry_log.csv").string());
    }
    photometryLog << "JD,PhaseAngle_deg,Distance_AU,ApparentMagnitude" << std::endl;

    return true;
}

void App::Run() {
    const double FIXED_DT = 1.0 / 60.0;

    while (!glContext->ShouldClose()) {
        glContext->PollEvents();

        Update(FIXED_DT);
        RenderOpenGL();

        CaptureScreenshotIfNeeded(
            screenshotDir,
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

    if (simulationFinished) {
        RunPostProcessing();
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

        if (realTimeHours < config.physics.rotationPeriodHours * config.physics.durationRotations) {
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

    glfwSetWindowShouldClose(glContext->GetWindow(), GLFW_TRUE);
}

void App::RunPostProcessing() {
    for (const std::string& name : config.postprocess.scripts) {
        fs::path script = name;
        if (script.is_relative()) {
            script = fs::path(COMET_SOURCE_DIR) / script;   // скрипти лежать у репо, а не поруч з exe
        }
        const std::string command = config.postprocess.python + " \"" + script.string() + "\" \""
            + fs::absolute(outputDir).string() + "\"";
        std::cout << "[INFO] " << command << "\n";
        if (std::system(command.c_str()) != 0) {
            std::cerr << "[WARNING] Post-processing failed: " << script.string() << "\n";
        }
    }
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