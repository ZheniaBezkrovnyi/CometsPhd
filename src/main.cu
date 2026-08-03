#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <optix.h>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>

#include <iostream>
#include <vector>
#include <cmath>
#include <fstream>
#include <sstream>
#include <limits>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "IO/PLY.h"
#include "Physics/Sun.h"
#include "Geometry/Geometry.h"
#include "Core/PhysicsConsts.h"
#include "Core/Timer.h"
#include "Shaders/OptixParams.h"

#include "Utils/ScreenshotCapture.h"

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

struct AppSettings {
    std::string modelPath = "C:/Users/ebbez/source/repos/CometsPhd/data/Churyumov-Geras_SPC 2017 - 199k.ply";
    std::string ptxPath = "OptixKernels.ptx";

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
        double timeScale = 300.0;
        double rotationPeriodHours = 12.0;
        double rh_AU = 1.3;
    } physics;

    struct {
        float solarConstant = 1361.0f;
        float albedo = 0.04f;
        float emissivity = 0.95f;
        float activeFraction = 1.0f;
        float minTemp = 40.0f;
        float maxTempForColor = 230.0f;

        int indirectSamples = 256;
        unsigned int indirectSeed = 1337u;
        float indirectSolarScale = 0.15f;
        float indirectIRScale = 0.05f;
        float maxIndirectFractionOfSolarFlux = 0.10f;
        float rayEpsilon = 0.01f;
    } thermal;

    struct {
        bool startupLogs = false;
        bool optixLogs = false;

        bool temperatureDebug = false;
        int temperatureDebugIntervalFrames = 300;

        bool cudaErrorChecks = false;
        bool syncAfterKernels = false;
    } diagnostics;

    struct {
        bool enabled = true;
        std::string outputDir = "capture_frames";
        int maxFrames = 500;
        int frameStride = 1;
    } screenshotCapture;
};

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<int> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;
typedef SbtRecord<int> HitGroupSbtRecord;


struct AppContext {
    GLFWwindow* window = nullptr;
    GLuint vbo = 0;
    cudaGraphicsResource* cuda_vbo_resource = nullptr;

    OptixDeviceContext optixContext = nullptr;
    CUcontext cudaContext = nullptr;

    OptixTraversableHandle gasHandle = 0;
    CUdeviceptr d_gas_output_buffer = 0;

    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_params = 0;
    float4* d_prevTemperature = nullptr;
    OptixParams params;

    int totalVertices = 0;
    float maxCoord = 0.0f;

    AppSettings config;
};


void PrintTemperatureDebug(const InteropVertex* d_vertices, int totalVertices, unsigned int frameCount, int intervalFrames = 30) {
    if (frameCount % intervalFrames != 0) return;
    if (totalVertices <= 0) return;

    std::vector<InteropVertex> hostVertices(totalVertices);

    cudaMemcpy(
        hostVertices.data(),
        d_vertices,
        totalVertices * sizeof(InteropVertex),
        cudaMemcpyDeviceToHost
    );

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "[TEMP DEBUG] cudaMemcpy failed: " << cudaGetErrorString(err) << "\n";
        return;
    }

    double tempSum = 0.0;
    double fluxSum = 0.0;

    float minTemp = std::numeric_limits<float>::max();
    float maxTemp = -std::numeric_limits<float>::max();

    float minFlux = std::numeric_limits<float>::max();
    float maxFlux = -std::numeric_limits<float>::max();

    int validCount = 0;

    for (int i = 0; i < totalVertices; i += 3) {
        float temp = hostVertices[i].temperature.x;
        float flux = hostVertices[i].temperature.y;

        if (!std::isfinite(temp) || !std::isfinite(flux)) continue;

        minTemp = std::min(minTemp, temp);
        maxTemp = std::max(maxTemp, temp);

        minFlux = std::min(minFlux, flux);
        maxFlux = std::max(maxFlux, flux);

        tempSum += temp;
        fluxSum += flux;

        validCount++;
    }

    if (validCount == 0) {
        std::cout << "[TEMP DEBUG] no valid values\n";
        return;
    }

    double avgTemp = tempSum / validCount;
    double avgFlux = fluxSum / validCount;

    std::cout
        << "[TEMP DEBUG] frame=" << frameCount
        << " faces=" << validCount
        << " T_min=" << minTemp << " K"
        << " T_avg=" << avgTemp << " K"
        << " T_max=" << maxTemp << " K"
        << " Flux_min=" << minFlux << " W/m2"
        << " Flux_avg=" << avgFlux << " W/m2"
        << " Flux_max=" << maxFlux << " W/m2"
        << "\n";
}

void CheckCUDAError(const AppContext& ctx, const char* msg, bool fatal = false) {
    if (!ctx.config.diagnostics.cudaErrorChecks) return;

    cudaError_t err = cudaPeekAtLastError();
    if (err != cudaSuccess) {
        std::cerr
            << "[CUDA ERROR] " << msg
            << " | code=" << (int)err
            << " | " << cudaGetErrorString(err)
            << "\n";

        if (fatal) exit(-1);
    }
}

static void context_log_cb(unsigned int level, const char* tag, const char* message, void* /*cbdata */) {
    std::cerr << "[" << std::setw(2) << level << "][" << std::setw(12) << tag << "]: " << message << "\n";
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

bool BuildOptiXGAS(AppContext& ctx, CUdeviceptr d_vertices) {
    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[OPTIX] Building GAS...\n";
    }
    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr vertexBuffers[1] = { d_vertices };
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(InteropVertex);
    buildInput.triangleArray.numVertices = ctx.totalVertices;
    buildInput.triangleArray.vertexBuffers = vertexBuffers;

    buildInput.triangleArray.indexFormat = OPTIX_INDICES_FORMAT_NONE;
    buildInput.triangleArray.numIndexTriplets = 0;
    buildInput.triangleArray.indexBuffer = 0;

    uint32_t triangleInputFlags[1] = { OPTIX_GEOMETRY_FLAG_DISABLE_ANYHIT };
    buildInput.triangleArray.flags = triangleInputFlags;
    buildInput.triangleArray.numSbtRecords = 1;

    OptixAccelBuildOptions accelOptions = {};
    accelOptions.buildFlags = OPTIX_BUILD_FLAG_ALLOW_COMPACTION | OPTIX_BUILD_FLAG_PREFER_FAST_TRACE;
    accelOptions.operation = OPTIX_BUILD_OPERATION_BUILD;

    OptixAccelBufferSizes bufferSizes;
    optixAccelComputeMemoryUsage(ctx.optixContext, &accelOptions, &buildInput, 1, &bufferSizes);

    CUdeviceptr d_temp_buffer;
    cudaMalloc((void**)&d_temp_buffer, bufferSizes.tempSizeInBytes);
    cudaMalloc((void**)&ctx.d_gas_output_buffer, bufferSizes.outputSizeInBytes);

    OptixResult res = optixAccelBuild(
        ctx.optixContext, 0, &accelOptions, &buildInput, 1,
        d_temp_buffer, bufferSizes.tempSizeInBytes,
        ctx.d_gas_output_buffer, bufferSizes.outputSizeInBytes,
        &ctx.gasHandle, nullptr, 0
    );

    cudaFree((void*)d_temp_buffer);

    if (res != OPTIX_SUCCESS) {
        std::cerr << "[OPTIX ERROR] Failed to build GAS!\n";
        return false;
    }

    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[OPTIX] GAS built successfully.\n";
    }
    return true;
}

std::string ReadPTX(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.good()) {
        std::cerr << "Failed to find PTX file: " << filepath << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

bool BuildOptiXPipeline(AppContext& ctx) {
    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[OPTIX] Building pipeline...\n";
    }

    OptixModuleCompileOptions moduleCompileOptions = {};
    moduleCompileOptions.maxRegisterCount = OPTIX_COMPILE_DEFAULT_MAX_REGISTER_COUNT;
    moduleCompileOptions.optLevel = OPTIX_COMPILE_OPTIMIZATION_DEFAULT;
    moduleCompileOptions.debugLevel = OPTIX_COMPILE_DEBUG_LEVEL_MINIMAL;

    OptixPipelineCompileOptions pipelineCompileOptions = {};
    pipelineCompileOptions.usesMotionBlur = false;
    pipelineCompileOptions.traversableGraphFlags = OPTIX_TRAVERSABLE_GRAPH_FLAG_ALLOW_SINGLE_GAS;
    pipelineCompileOptions.numPayloadValues = 1;
    pipelineCompileOptions.numAttributeValues = 2;
    pipelineCompileOptions.exceptionFlags = OPTIX_EXCEPTION_FLAG_NONE;
    pipelineCompileOptions.pipelineLaunchParamsVariableName = "params";

    std::string ptx = ReadPTX(ctx.config.ptxPath);
    if (ptx.empty()) return false;

    char log[2048];
    size_t sizeof_log = sizeof(log);
    log[0] = '\0';

    OptixResult res = optixModuleCreate(
        ctx.optixContext,
        &moduleCompileOptions,
        &pipelineCompileOptions,
        ptx.c_str(),
        ptx.size(),
        log,
        &sizeof_log,
        &ctx.module
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " Module creation failed! Log:\n" << log << "\n";
        return false;
    }

    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module = ctx.module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";

    sizeof_log = sizeof(log);
    log[0] = '\0';

    res = optixProgramGroupCreate(
        ctx.optixContext,
        &rgDesc,
        1,
        &pgOptions,
        log,
        &sizeof_log,
        &ctx.raygenPG
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " RayGen PG creation failed! Log:\n" << log << "\n";
        return false;
    }

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module = ctx.module;
    msDesc.miss.entryFunctionName = "__miss__ms";

    sizeof_log = sizeof(log);
    log[0] = '\0';

    res = optixProgramGroupCreate(
        ctx.optixContext,
        &msDesc,
        1,
        &pgOptions,
        log,
        &sizeof_log,
        &ctx.missPG
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " Miss PG creation failed! Log:\n" << log << "\n";
        return false;
    }

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH = ctx.module;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";

    sizeof_log = sizeof(log);
    log[0] = '\0';

    res = optixProgramGroupCreate(
        ctx.optixContext,
        &hgDesc,
        1,
        &pgOptions,
        log,
        &sizeof_log,
        &ctx.hitgroupPG
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " HitGroup PG creation failed! Log:\n" << log << "\n";
        return false;
    }

    OptixProgramGroup programGroups[] = {
        ctx.raygenPG,
        ctx.missPG,
        ctx.hitgroupPG
    };

    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    sizeof_log = sizeof(log);
    log[0] = '\0';

    res = optixPipelineCreate(
        ctx.optixContext,
        &pipelineCompileOptions,
        &pipelineLinkOptions,
        programGroups,
        3,
        log,
        &sizeof_log,
        &ctx.pipeline
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " Pipeline link failed! Log:\n" << log << "\n";
        return false;
    }

    res = optixPipelineSetStackSize(
        ctx.pipeline,
        2048,
        2048,
        2048,
        1
    );

    if (res != OPTIX_SUCCESS) {
        std::cerr << " Pipeline stack setup failed!\n";
        return false;
    }

    CUdeviceptr d_raygen = 0;
    CUdeviceptr d_miss = 0;
    CUdeviceptr d_hitgroup = 0;

    RayGenSbtRecord rgSBT = {};
    optixSbtRecordPackHeader(ctx.raygenPG, &rgSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_raygen), sizeof(RayGenSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_raygen), &rgSBT, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice);

    MissSbtRecord msSBT = {};
    optixSbtRecordPackHeader(ctx.missPG, &msSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_miss), sizeof(MissSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_miss), &msSBT, sizeof(MissSbtRecord), cudaMemcpyHostToDevice);

    HitGroupSbtRecord hgSBT = {};
    optixSbtRecordPackHeader(ctx.hitgroupPG, &hgSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_hitgroup), sizeof(HitGroupSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_hitgroup), &hgSBT, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice);

    ctx.sbt.raygenRecord = d_raygen;
    ctx.sbt.missRecordBase = d_miss;
    ctx.sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    ctx.sbt.missRecordCount = 1;
    ctx.sbt.hitgroupRecordBase = d_hitgroup;
    ctx.sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    ctx.sbt.hitgroupRecordCount = 1;

    cudaMalloc(reinterpret_cast<void**>(&ctx.d_params), sizeof(OptixParams));
    CheckCUDAError(ctx, "cudaMalloc d_params", true);

    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[OPTIX] Pipeline and SBT built successfully.\n";
    }
    return true;
}

bool InitOptiX(AppContext& ctx) {
    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[INIT] Initializing OptiX...\n";
    }

    cudaFree(0);

    CUcontext cuCtx = 0;
    if (optixInit() != OPTIX_SUCCESS) return false;

    OptixDeviceContextOptions options = {};

    if (ctx.config.diagnostics.optixLogs) {
        options.logCallbackFunction = &context_log_cb;
        options.logCallbackLevel = 4;
    }
    else {
        options.logCallbackFunction = nullptr;
        options.logCallbackLevel = 0;
    }

    if (optixDeviceContextCreate(cuCtx, &options, &ctx.optixContext) != OPTIX_SUCCESS) {
        return false;
    }

    return true;
}

bool InitGraphicsAndInterop(AppContext& ctx, const std::vector<InteropVertex>& vertices) {
    if (!glfwInit()) return false;
    ctx.window = glfwCreateWindow(ctx.config.window.width, ctx.config.window.height, ctx.config.window.title, NULL, NULL);
    if (!ctx.window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(ctx.window);
    glfwSwapInterval(0);
    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    ctx.totalVertices = vertices.size();
    cudaMalloc(&ctx.d_prevTemperature, ctx.totalVertices * sizeof(float4));
    CheckCUDAError(ctx, "cudaMalloc d_prevTemperature", true);

    glGenBuffers(1, &ctx.vbo);
    glBindBuffer(GL_ARRAY_BUFFER, ctx.vbo);
    glBufferData(GL_ARRAY_BUFFER, ctx.totalVertices * sizeof(InteropVertex), vertices.data(), GL_DYNAMIC_DRAW);

    cudaGraphicsGLRegisterBuffer(&ctx.cuda_vbo_resource, ctx.vbo, cudaGraphicsMapFlagsNone);
    CheckCUDAError(ctx, "GLRegisterBuffer", true);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);
    glVertexPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, position));
    glNormalPointer(GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, normal));
    glColorPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, color));

    glEnable(GL_DEPTH_TEST);

    InteropVertex* d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &ctx.cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, ctx.cuda_vbo_resource);

    bool gasOk = BuildOptiXGAS(ctx, (CUdeviceptr)d_vertices);
    if (!gasOk) {
        cudaGraphicsUnmapResources(1, &ctx.cuda_vbo_resource, 0);
        return false;
    }

    bool pipelineOk = BuildOptiXPipeline(ctx);
    if (!pipelineOk) {
        cudaGraphicsUnmapResources(1, &ctx.cuda_vbo_resource, 0);
        return false;
    }

    cudaGraphicsUnmapResources(1, &ctx.cuda_vbo_resource, 0);
    return true;
}

__global__ void CopyTemperatureKernel(const InteropVertex* vertices, float4* prevTemperature, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    prevTemperature[idx] = vertices[idx].temperature;
}

void CopyPreviousTemperature(
    const AppContext& ctx,
    const InteropVertex* d_vertices,
    float4* d_prevTemperature,
    int totalVertices
) {
    int blockSize = 256;
    int gridSize = (totalVertices + blockSize - 1) / blockSize;

    CopyTemperatureKernel << <gridSize, blockSize >> > (d_vertices, d_prevTemperature, totalVertices);

    if (ctx.config.diagnostics.syncAfterKernels) {
        cudaDeviceSynchronize();
    }

    CheckCUDAError(ctx, "CopyTemperatureKernel", true);
}

void RunSimulationLoop(AppContext& ctx) {
    if (ctx.config.diagnostics.startupLogs) {
        std::cout << "[RUN] Starting rendering loop...\n";
    }

    FPSCounter fpsCounter;

    double simulationTime = 0.0;
    double previousRealTime = glfwGetTime();
    unsigned int frameCount = 0;

    ScreenshotCaptureState screenshotState;

    while (!glfwWindowShouldClose(ctx.window)) {
        glfwPollEvents();

        double currentRealTime = glfwGetTime();
        double realDt = currentRealTime - previousRealTime;
        previousRealTime = currentRealTime;

        if (realDt > 0.25) {
            realDt = 0.25;
        }

        simulationTime += realDt * ctx.config.physics.timeScale;

        InteropVertex* d_vertices;
        size_t num_bytes;
        cudaGraphicsMapResources(1, &ctx.cuda_vbo_resource, 0);
        cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, ctx.cuda_vbo_resource);

        double rotationPeriodSeconds = ctx.config.physics.rotationPeriodHours * 3600.0;
        double rotationSpeed = 2.0 * PhysicsConsts::PI / rotationPeriodSeconds;
        float currentAngle = (float)fmod(simulationTime * rotationSpeed, 2.0 * PhysicsConsts::PI);

        glm::mat3 invRot = glm::mat3(glm::rotate(glm::mat4(1.0f), -currentAngle, glm::vec3(0.0f, 1.0f, 0.0f)));
        glm::vec3 sunLocalDir = glm::normalize(invRot * glm::vec3(1.0f, 0.0f, 0.0f));


        CopyPreviousTemperature(ctx, d_vertices, ctx.d_prevTemperature, ctx.totalVertices);

        ctx.params.handle = ctx.gasHandle;
        ctx.params.vertices = d_vertices;
        ctx.params.prevTemperature = ctx.d_prevTemperature;
        ctx.params.numVertices = ctx.totalVertices;

        ctx.params.sunDir = make_float3(sunLocalDir.x, sunLocalDir.y, sunLocalDir.z);

        ctx.params.rh_AU = (float)ctx.config.physics.rh_AU;
        ctx.params.solarConstant = ctx.config.thermal.solarConstant;
        ctx.params.albedo = ctx.config.thermal.albedo;
        ctx.params.emissivity = ctx.config.thermal.emissivity;
        ctx.params.activeFraction = ctx.config.thermal.activeFraction;

        ctx.params.minTemp = ctx.config.thermal.minTemp;
        ctx.params.maxTempForColor = ctx.config.thermal.maxTempForColor;

        ctx.params.frameCount = frameCount;

        ctx.params.indirectSamples = ctx.config.thermal.indirectSamples;
        ctx.params.indirectSeed = ctx.config.thermal.indirectSeed;
        ctx.params.indirectSolarScale = ctx.config.thermal.indirectSolarScale;
        ctx.params.indirectIRScale = ctx.config.thermal.indirectIRScale;
        ctx.params.maxIndirectFractionOfSolarFlux = ctx.config.thermal.maxIndirectFractionOfSolarFlux;
        ctx.params.rayEpsilon = ctx.config.thermal.rayEpsilon;

        cudaMemcpy(
            reinterpret_cast<void*>(ctx.d_params),
            &ctx.params,
            sizeof(OptixParams),
            cudaMemcpyHostToDevice
        );
        CheckCUDAError(ctx, "copy OptixParams", true);

        int numTriangles = ctx.totalVertices / 3;
        optixLaunch(ctx.pipeline, 0, ctx.d_params, sizeof(OptixParams), &ctx.sbt, numTriangles, 1, 1);

        if (ctx.config.diagnostics.syncAfterKernels) {
            cudaDeviceSynchronize();
        }

        CheckCUDAError(ctx, "OptiX Launch");

        if (ctx.config.diagnostics.temperatureDebug) {
            PrintTemperatureDebug(
                d_vertices,
                ctx.totalVertices,
                frameCount,
                ctx.config.diagnostics.temperatureDebugIntervalFrames
            );
        }

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

        CaptureScreenshotIfNeeded(
            ctx.config.screenshotCapture.outputDir,
            ctx.config.screenshotCapture.enabled,
            ctx.config.screenshotCapture.maxFrames,
            ctx.config.screenshotCapture.frameStride,
            screenshotState,
            frameCount,
            width,
            height
        );

        glfwSwapBuffers(ctx.window);

        frameCount++;

        fpsCounter.Update(ctx.window, ctx.config.window.title);
    }
}

void Cleanup(AppContext& ctx) {
    if (ctx.d_prevTemperature) cudaFree(ctx.d_prevTemperature);
    if (ctx.d_params) cudaFree(reinterpret_cast<void*>(ctx.d_params));
    if (ctx.d_gas_output_buffer) cudaFree(reinterpret_cast<void*>(ctx.d_gas_output_buffer));
    if (ctx.pipeline) optixPipelineDestroy(ctx.pipeline);
    if (ctx.raygenPG) optixProgramGroupDestroy(ctx.raygenPG);
    if (ctx.missPG) optixProgramGroupDestroy(ctx.missPG);
    if (ctx.hitgroupPG) optixProgramGroupDestroy(ctx.hitgroupPG);
    if (ctx.module) optixModuleDestroy(ctx.module);
    if (ctx.optixContext) optixDeviceContextDestroy(ctx.optixContext);

    if (ctx.cuda_vbo_resource) cudaGraphicsUnregisterResource(ctx.cuda_vbo_resource);
    if (ctx.vbo) glDeleteBuffers(1, &ctx.vbo);
    if (ctx.window) {
        glfwDestroyWindow(ctx.window);
        glfwTerminate();
    }

    if (ctx.sbt.raygenRecord) cudaFree(reinterpret_cast<void*>(ctx.sbt.raygenRecord));
    if (ctx.sbt.missRecordBase) cudaFree(reinterpret_cast<void*>(ctx.sbt.missRecordBase));
    if (ctx.sbt.hitgroupRecordBase) cudaFree(reinterpret_cast<void*>(ctx.sbt.hitgroupRecordBase));
}

int main() {
    try {
        cudaSetDevice(0);
        cudaFree(0);

        AppContext app;

        if (!InitOptiX(app)) {
            std::cerr << "Failed to initialize OptiX!" << std::endl;
            return -1;
        }

        std::vector<InteropVertex> vertices = LoadAndPrepareGeometry(
            app.config.modelPath,
            app.maxCoord,
            app.config.thermal.minTemp
        );

        if (!InitGraphicsAndInterop(app, vertices)) {
            std::cerr << "Failed to initialize graphics or OptiX pipeline!" << std::endl;
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