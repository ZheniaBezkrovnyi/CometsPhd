#include "App.h"
#include <iostream>
#include <filesystem>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include "IO/PLY.h"
#include "Core/PhysicsConsts.h"

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

    KeplerianElements cometElems = {
        config.physics.cometOrbit.a,
        config.physics.cometOrbit.e,
        config.physics.cometOrbit.i,
        config.physics.cometOrbit.Omega,
        config.physics.cometOrbit.w,
        config.physics.cometOrbit.M0,
        config.physics.cometOrbit.epoch
    };
    cometBody = OrbitalBody(cometElems);
}

App::~App() {
    if (cuda_vbo_resource) cudaGraphicsUnregisterResource(cuda_vbo_resource);
    if (vbo) glDeleteBuffers(1, &vbo);
    if (d_prevTemperature) cudaFree(d_prevTemperature);
}

std::vector<InteropVertex> App::LoadAndPrepareGeometry(const std::string& filepath, float& outMaxCoord, float baseTemp) {
    PLY plyModel(filepath);
    std::vector<Face> faces = plyModel.getFaces();
    std::vector<Vertex> verts = plyModel.getVertices();

    std::vector<InteropVertex> vertices;
    vertices.reserve(faces.size() * 3);

    outMaxCoord = 0.0f;
    for (const auto& face : faces) {
        glm::vec3 p1(verts[face.v1].x, verts[face.v1].y, verts[face.v1].z);
        glm::vec3 p2(verts[face.v2].x, verts[face.v2].y, verts[face.v2].z);
        glm::vec3 p3(verts[face.v3].x, verts[face.v3].y, verts[face.v3].z);

        glm::vec3 edge1 = p2 - p1;
        glm::vec3 edge2 = p3 - p1;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

        outMaxCoord = fmaxf(outMaxCoord, glm::length(p1));
        outMaxCoord = fmaxf(outMaxCoord, glm::length(p2));
        outMaxCoord = fmaxf(outMaxCoord, glm::length(p3));

        int indices[3] = { face.v1, face.v2, face.v3 };
        for (int i = 0; i < 3; ++i) {
            InteropVertex v;
            v.position = make_float4(verts[indices[i]].x, verts[indices[i]].y, verts[indices[i]].z, 0.5f);
            v.normal = make_float4(normal.x, normal.y, normal.z, 0.5f);
            v.color = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
            v.temperature = make_float4(baseTemp, 0.5f, 0.5f, 0.5f);
            vertices.push_back(v);
        }
    }
    return vertices;
}

bool App::Init() {
    if (!glContext->Init(config)) { std::cerr << "Failed: glContext->Init" << std::endl; return false; }
    if (!optixRenderer->Init(config)) { std::cerr << "Failed: optixRenderer->Init" << std::endl; return false; }

    hostVertices = LoadAndPrepareGeometry(config.modelPath, maxCoord, config.thermal.minTemp);
    if (hostVertices.empty()) { std::cerr << "Failed: Model is empty or PLY path is wrong!" << std::endl; return false; }

    totalVertices = hostVertices.size();

    simTime.Init(config.physics.timeScale, config.physics.startJulianDate);

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, totalVertices * sizeof(InteropVertex), hostVertices.data(), GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    cudaMalloc(&d_prevTemperature, totalVertices * sizeof(float4));

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, position));
    glNormalPointer(GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, normal));
    glColorPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, color));

    InteropVertex* d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);

    if (!optixRenderer->BuildGAS((CUdeviceptr)d_vertices, totalVertices)) { std::cerr << "Failed: OptiX BuildGAS" << std::endl; return false; }
    if (!optixRenderer->BuildPipeline(config.ptxPath)) { std::cerr << "Failed: OptiX BuildPipeline" << std::endl; return false; }

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

    return true;
}

void App::Run() {
    double previousRealTime = glfwGetTime();

    while (!glContext->ShouldClose()) {
        glContext->PollEvents();

        double currentRealTime = glfwGetTime();
        double dt = currentRealTime - previousRealTime;
        previousRealTime = currentRealTime;

        if (dt > 0.25) dt = 0.25;

        Update(dt);
        Draw();

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

    UpdateTransformations();
    RunOptixSimulation();
}

void App::Draw() {
    RenderOpenGL();
}

void App::UpdateTransformations() {
    glm::dvec3 heliocentricPosAU = cometBody.CalculatePosition(simTime.GetCurrentJD());
    current_rh_AU = glm::length(heliocentricPosAU);
    glm::dvec3 dirToSunWorld = glm::normalize(-heliocentricPosAU);

    double rotationPeriodSeconds = config.physics.rotationPeriodHours * PhysicsConsts::SECONDS_PER_HOUR;
    double rotationSpeed = 2.0 * PhysicsConsts::PI / rotationPeriodSeconds;
    current_Angle = (float)fmod(simTime.GetElapsedSeconds() * rotationSpeed, 2.0 * PhysicsConsts::PI);

    glm::mat3 invRot = glm::mat3(glm::rotate(glm::mat4(1.0f), -current_Angle, glm::vec3(0.0f, 1.0f, 0.0f)));
    current_sunLocalDir = invRot * glm::vec3(dirToSunWorld);
}

void App::RunOptixSimulation() {
    InteropVertex* d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);

    optixRenderer->CopyPreviousTemperatures(d_vertices, d_prevTemperature, totalVertices);

    OptixParams params = {};
    params.vertices = d_vertices;
    params.prevTemperature = d_prevTemperature;
    params.numVertices = totalVertices;

    params.sunDir = make_float3(current_sunLocalDir.x, current_sunLocalDir.y, current_sunLocalDir.z);
    params.rh_AU = (float)current_rh_AU;

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

    optixRenderer->Render(params, totalVertices / 3);

    cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
}

void App::RenderOpenGL() {
    int width = glContext->GetWidth();
    int height = glContext->GetHeight();

    glViewport(0, 0, width, height);
    glClearColor(config.camera.clearColor[0], config.camera.clearColor[1], config.camera.clearColor[2], config.camera.clearColor[3]);
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

    glMatrixMode(GL_PROJECTION);
    glLoadIdentity();
    glm::mat4 proj = glm::perspective(glm::radians(config.camera.fov), (float)width / (float)height, 0.1f, maxCoord * config.camera.farPlaneMultiplier);
    glLoadMatrixf(glm::value_ptr(proj));

    glMatrixMode(GL_MODELVIEW);
    glLoadIdentity();
    glm::mat4 view = glm::lookAt(
        glm::vec3(0, maxCoord * config.camera.heightMultiplier, maxCoord * config.camera.distanceMultiplier),
        glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)
    );
    glm::mat4 model = glm::rotate(glm::mat4(1.0f), current_Angle, glm::vec3(0, 1, 0));
    glLoadMatrixf(glm::value_ptr(view * model));

    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glDrawArrays(GL_TRIANGLES, 0, totalVertices);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}