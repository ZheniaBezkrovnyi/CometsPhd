#include "App.h"
#include <iostream>
#include <filesystem>
#include "Stages/PhotometryStage.h"
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

    BuildStages();
    for (auto& stage : stages) {
        stage->Begin(outputDir);
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
    if (config.thermal.enabled) {
        RunOptixThermal(d_vertices);
    }

    const double elapsedHours = simTime.GetElapsedSeconds() / 3600.0;
    if (elapsedHours < config.physics.rotationPeriodHours * config.physics.durationRotations) {
        const StepContext ctx{ simTime, spaceScene, *optixRenderer, d_vertices, cometMesh.GetVertexCount(), frameCount };
        for (auto& stage : stages) {
            stage->Step(ctx);
        }
    }
    else {
        OnSimulationComplete();
    }

    cometMesh.UnmapFromCUDA();
}
void App::OnSimulationComplete() {
    simulationFinished = true;

    for (auto& stage : stages) {
        stage->End();
    }

    glfwSetWindowShouldClose(glContext->GetWindow(), GLFW_TRUE);
}

void App::RunPostProcessing() {
    std::vector<std::string> scripts;
    for (const auto& stage : stages) {
        const std::vector<std::string> plots = stage->Plots();
        scripts.insert(scripts.end(), plots.begin(), plots.end());
    }
    scripts.insert(scripts.end(), config.postprocess.scripts.begin(), config.postprocess.scripts.end());

    for (const std::string& name : scripts) {
        fs::path script = name;
        if (script.is_relative()) {
            script = fs::path(COMET_SOURCE_DIR) / script; 
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


void App::BuildStages() {
    if (config.photometry.enabled) stages.push_back(std::make_unique<PhotometryStage>(config));
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