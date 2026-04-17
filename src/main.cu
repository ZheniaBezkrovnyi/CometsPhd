#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "IO/PLY.h"
#include "Physics/Sun.h"
#include "Geometry/Geometry.h"
#include "Core/PhysicsConsts.h"
#include "Core/Timer.h"

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}


struct AppSettings {
    std::string modelPath = "C:/Users/ebbez/source/repos/CometsPhd/data/Churyumov-Gerasimenko SPC 2017 - 199k poly.ply";

    struct {
        int width = 1100;
        int height = 1100;
        const char* title = "Comet CUDA Interop";
    } window;

    struct {
        float fov = 60.0f;
        float heightMultiplier = 0.5f;
        float distanceMultiplier = 1.8f;
        float farPlaneMultiplier = 10.0f;
        float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    } camera;

    struct {
        double dt = 1.0;
        double rotationPeriodHours = 12.0;
    } physics;

    struct {
        int blockSize = 256;
        float baseTemp = 20.0f;
        float tempScale = 380.0f;
    } thermal;
};

// =====================================================================
// APPLICATION CONTEXT
// =====================================================================

struct AppContext {
    GLFWwindow* window = nullptr;
    GLuint vbo = 0;
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    int totalVertices = 0;
    float maxCoord = 0.0f;

    AppSettings config; 
};

// =====================================================================
// CUDA KERNELS & DEVICE FUNCTIONS
// =====================================================================

__device__ void GetHeatmapColor(float temp, float baseTemp, float tempScale, float* r, float* g, float* b) {
    float t = fminf(fmaxf((temp - baseTemp) / tempScale, 0.0f), 1.0f);
    if (t < 0.5f) {
        *r = 0.0f;
        *g = t * 2.0f;
        *b = 1.0f - t * 2.0f;
    }
    else {
        *r = (t - 0.5f) * 2.0f;
        *g = 1.0f - (t - 0.5f) * 2.0f;
        *b = 0.0f;
    }
}

__global__ void UpdateCometTemperatureKernel(InteropVertex* vertices, int numVertices,
    float sunX, float sunY, float sunZ,
    float baseTemp, float tempScale) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    float4 norm = vertices[idx].normal;

    float cosTheta = (norm.x * sunX + norm.y * sunY + norm.z * sunZ);
    cosTheta = fmaxf(cosTheta, 0.0f);

    float newTemp = baseTemp + cosTheta * tempScale;
    float r, g, b;
    GetHeatmapColor(newTemp, baseTemp, tempScale, &r, &g, &b);

    vertices[idx].temperature = make_float4(newTemp, 0.5f, 0.5f, 0.5f);
    vertices[idx].color = make_float4(r, g, b, 1.0f);
}

// =====================================================================
// SYSTEMS & LOGIC
// =====================================================================

void CheckCUDAError(const char* msg, bool fatal = false) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): code " << (int)err << " - " << cudaGetErrorString(err) << std::endl;
        if (fatal) exit(-1);
    }
}

std::vector<InteropVertex> LoadAndPrepareGeometry(const std::string& filepath, float& outMaxCoord, float baseTemp) {
    std::cout << "[INIT] Loading PLY Model from: " << filepath << "\n";
    PLY plyModel(filepath);
    std::vector<Face> faces = plyModel.getFaces();
    std::vector<Vertex> verts = plyModel.getVertices();

    std::vector<InteropVertex> hostVertices;
    hostVertices.reserve(faces.size() * 3);

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
            hostVertices.push_back(v);
        }
    }
    return hostVertices;
}

bool InitGraphicsAndInterop(AppContext& ctx, const std::vector<InteropVertex>& vertices) {
    if (!glfwInit()) return false;

    ctx.window = glfwCreateWindow(ctx.config.window.width, ctx.config.window.height, ctx.config.window.title, NULL, NULL);
    if (!ctx.window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(ctx.window);
    //glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    ctx.totalVertices = vertices.size();

    glGenBuffers(1, &ctx.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo);
    glBufferData(GL_ARRAY_BUFFER, ctx.totalVertices * sizeof(InteropVertex), vertices.data(), GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(&ctx.cuda_vbo_resource, ctx.vbo, cudaGraphicsMapFlagsNone);
    CheckCUDAError("GLRegisterBuffer", true);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, position));
    glNormalPointer(GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, normal));
    glColorPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, color));

    glEnable(GL_DEPTH_TEST);
    return true;
}

void RunSimulationLoop(AppContext& ctx) {
    std::cout << "[RUN] Starting Rendering Loop...\n";

    FPSCounter fpsCounter;
    double t = 0.0;
    int gridSize = (ctx.totalVertices + ctx.config.thermal.blockSize - 1) / ctx.config.thermal.blockSize;

    while (!glfwWindowShouldClose(ctx.window)) {
        glfwPollEvents();

        InteropVertex* d_vertices;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &ctx.cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, ctx.cuda_vbo_resource);

        double rotationSpeed = 2.0 * PhysicsConsts::PI / (ctx.config.physics.rotationPeriodHours * 3600.0);
        float currentAngle = (float)(t * rotationSpeed);

        glm::mat3 invRot = glm::mat3(glm::rotate(glm::mat4(1.0f), -currentAngle, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 sunLocalDir = glm::normalize(invRot * glm::vec3(1.0f, 0.0f, 0.0f));

        UpdateCometTemperatureKernel << <gridSize, ctx.config.thermal.blockSize >> > (
            d_vertices, ctx.totalVertices,
            sunLocalDir.x, sunLocalDir.y, sunLocalDir.z,
            ctx.config.thermal.baseTemp, ctx.config.thermal.tempScale
            );

        cudaDeviceSynchronize();
        CheckCUDAError("Kernel Launch");
        cudaGraphicsUnmapResources(1, &ctx.cuda_vbo_resource, 0);

        int width, height;
        glfwGetFramebufferSize(ctx.window, &width, &height);
        if (height == 0) height = 1;
        if (width == 0) width = 1;

        glViewport(0, 0, width, height);
        glClearColor(ctx.config.camera.clearColor[0], ctx.config.camera.clearColor[1],
            ctx.config.camera.clearColor[2], ctx.config.camera.clearColor[3]);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

        glMatrixMode(GL_PROJECTION);
        glLoadIdentity();
        glm::mat4 proj = glm::perspective(glm::radians(ctx.config.camera.fov),
            (float)width / (float)height, 0.1f,
            ctx.maxCoord * ctx.config.camera.farPlaneMultiplier);
        glLoadMatrixf(glm::value_ptr(proj));

        glMatrixMode(GL_MODELVIEW);
        glLoadIdentity();
        glm::mat4 view = glm::lookAt(
            glm::vec3(0, ctx.maxCoord * ctx.config.camera.heightMultiplier, ctx.maxCoord * ctx.config.camera.distanceMultiplier),
            glm::vec3(0, 0, 0), glm::vec3(0, 1, 0)
        );
        glm::mat4 model = glm::rotate(glm::mat4(1.0f), currentAngle, glm::vec3(0, 1, 0));
        glLoadMatrixf(glm::value_ptr(view * model));

        glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo);
        glDrawArrays(GL_TRIANGLES, 0, ctx.totalVertices);
        glBindBuffer(GL_ARRAY_BUFFER, 0);

        glfwSwapBuffers(ctx.window);
        t += ctx.config.physics.dt;

        fpsCounter.Update(ctx.window, ctx.config.window.title);
    }
}

void Cleanup(AppContext& ctx) {
    if (ctx.cuda_vbo_resource) cudaGraphicsUnregisterResource(ctx.cuda_vbo_resource);
    if (ctx.vbo) glDeleteBuffers(1, &ctx.vbo);
    if (ctx.window) {
        glfwDestroyWindow(ctx.window);
        glfwTerminate();
    }
}


int main() {
    try {
        cudaSetDevice(0);
        cudaFree(0);

        AppContext app;

        std::vector<InteropVertex> vertices = LoadAndPrepareGeometry(app.config.modelPath, app.maxCoord, app.config.thermal.baseTemp);

        if (!InitGraphicsAndInterop(app, vertices)) {
            std::cerr << "Failed to initialize graphics!" << std::endl;
            return -1;
        }

        RunSimulationLoop(app);

        Cleanup(app);

        return 0;
    }
    catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
        return -1;
    }
}