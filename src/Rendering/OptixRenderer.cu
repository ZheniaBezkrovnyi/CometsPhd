#include "OptixRenderer.h"
#include <iostream>
#include <fstream>
#include <sstream>
#include <optix_stubs.h>
#include <optix_function_table_definition.h>
#include <optix_stack_size.h>


__global__ void CopyTemperatureKernel(const InteropVertex* vertices, float4* prevTemperature, int numVertices) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;
    prevTemperature[idx] = vertices[idx].temperature;
}

template <typename T>
struct SbtRecord {
    __align__(OPTIX_SBT_RECORD_ALIGNMENT) char header[OPTIX_SBT_RECORD_HEADER_SIZE];
    T data;
};
typedef SbtRecord<int> RayGenSbtRecord;
typedef SbtRecord<int> MissSbtRecord;
typedef SbtRecord<int> HitGroupSbtRecord;

std::string ReadPTX(const std::string& filepath) {
    std::ifstream file(filepath);
    if (!file.good()) {
        std::cerr << "\n[ПОМИЛКА] Не знайдено PTX файл за шляхом: " << filepath << std::endl;
        std::cerr << "Перевір Working Directory або пропиши абсолютний шлях в AppSettings.h!\n" << std::endl;
        return "";
    }
    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

OptixRenderer::~OptixRenderer() {
    if (d_params) cudaFree(reinterpret_cast<void*>(d_params));
    if (d_gas_output_buffer) cudaFree(reinterpret_cast<void*>(d_gas_output_buffer));
    if (pipeline) optixPipelineDestroy(pipeline);
    if (raygenPG) optixProgramGroupDestroy(raygenPG);
    if (missPG) optixProgramGroupDestroy(missPG);
    if (hitgroupPG) optixProgramGroupDestroy(hitgroupPG);
    if (module) optixModuleDestroy(module);
    if (optixContext) optixDeviceContextDestroy(optixContext);

    if (sbt.raygenRecord) cudaFree(reinterpret_cast<void*>(sbt.raygenRecord));
    if (sbt.missRecordBase) cudaFree(reinterpret_cast<void*>(sbt.missRecordBase));
    if (sbt.hitgroupRecordBase) cudaFree(reinterpret_cast<void*>(sbt.hitgroupRecordBase));
}

bool OptixRenderer::Init(const AppSettings& config) {
    cudaFree(0);
    CUcontext cuCtx = 0;
    if (optixInit() != OPTIX_SUCCESS) return false;

    OptixDeviceContextOptions options = {};
    if (optixDeviceContextCreate(cuCtx, &options, &optixContext) != OPTIX_SUCCESS) return false;

    cudaMalloc(reinterpret_cast<void**>(&d_params), sizeof(OptixParams));
    return true;
}

bool OptixRenderer::BuildGAS(CUdeviceptr d_vertices, int numVertices) {
    if (!d_vertices) {
        std::cerr << "[OptiX Error] d_vertices is NULL! Перевір OpenGL-CUDA interop." << std::endl;
        return false;
    }

    OptixBuildInput buildInput = {};
    buildInput.type = OPTIX_BUILD_INPUT_TYPE_TRIANGLES;

    CUdeviceptr vertexBuffers[1] = { d_vertices };
    buildInput.triangleArray.vertexFormat = OPTIX_VERTEX_FORMAT_FLOAT3;
    buildInput.triangleArray.vertexStrideInBytes = sizeof(InteropVertex);
    buildInput.triangleArray.numVertices = numVertices;
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
    OptixResult resMem = optixAccelComputeMemoryUsage(optixContext, &accelOptions, &buildInput, 1, &bufferSizes);
    if (resMem != OPTIX_SUCCESS) {
        std::cerr << "[OptiX Error] ComputeMemoryUsage failed: " << optixGetErrorName(resMem) << std::endl;
        return false;
    }

    CUdeviceptr d_temp_buffer;
    cudaMalloc((void**)&d_temp_buffer, bufferSizes.tempSizeInBytes);
    cudaMalloc((void**)&d_gas_output_buffer, bufferSizes.outputSizeInBytes);

    OptixResult resBuild = optixAccelBuild(
        optixContext, 0, &accelOptions, &buildInput, 1,
        d_temp_buffer, bufferSizes.tempSizeInBytes,
        d_gas_output_buffer, bufferSizes.outputSizeInBytes,
        &gasHandle, nullptr, 0
    );

    cudaFree((void*)d_temp_buffer);

    if (resBuild != OPTIX_SUCCESS) {
        std::cerr << "[OptiX Error] AccelBuild failed: " << optixGetErrorName(resBuild) << std::endl;
        return false;
    }

    return true;
}

bool OptixRenderer::BuildPipeline(const std::string& ptxPath) {
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

    std::string ptx = ReadPTX(ptxPath);
    if (ptx.empty()) return false;

    char log[2048];
    size_t sizeof_log = sizeof(log);

    if (optixModuleCreate(optixContext, &moduleCompileOptions, &pipelineCompileOptions,
        ptx.c_str(), ptx.size(), log, &sizeof_log, &module) != OPTIX_SUCCESS) return false;

    OptixProgramGroupOptions pgOptions = {};

    OptixProgramGroupDesc rgDesc = {};
    rgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_RAYGEN;
    rgDesc.raygen.module = module;
    rgDesc.raygen.entryFunctionName = "__raygen__rg";
    if (optixProgramGroupCreate(optixContext, &rgDesc, 1, &pgOptions, log, &sizeof_log, &raygenPG) != OPTIX_SUCCESS) return false;

    OptixProgramGroupDesc msDesc = {};
    msDesc.kind = OPTIX_PROGRAM_GROUP_KIND_MISS;
    msDesc.miss.module = module;
    msDesc.miss.entryFunctionName = "__miss__ms";
    if (optixProgramGroupCreate(optixContext, &msDesc, 1, &pgOptions, log, &sizeof_log, &missPG) != OPTIX_SUCCESS) return false;

    OptixProgramGroupDesc hgDesc = {};
    hgDesc.kind = OPTIX_PROGRAM_GROUP_KIND_HITGROUP;
    hgDesc.hitgroup.moduleCH = module;
    hgDesc.hitgroup.entryFunctionNameCH = "__closesthit__ch";
    if (optixProgramGroupCreate(optixContext, &hgDesc, 1, &pgOptions, log, &sizeof_log, &hitgroupPG) != OPTIX_SUCCESS) return false;

    OptixProgramGroup programGroups[] = { raygenPG, missPG, hitgroupPG };
    OptixPipelineLinkOptions pipelineLinkOptions = {};
    pipelineLinkOptions.maxTraceDepth = 1;

    if (optixPipelineCreate(optixContext, &pipelineCompileOptions, &pipelineLinkOptions,
        programGroups, 3, log, &sizeof_log, &pipeline) != OPTIX_SUCCESS) return false;

    if (optixPipelineSetStackSize(pipeline, 2048, 2048, 2048, 1) != OPTIX_SUCCESS) return false;

    return BuildSBT();
}

bool OptixRenderer::BuildSBT() {
    CUdeviceptr d_raygen = 0, d_miss = 0, d_hitgroup = 0;

    RayGenSbtRecord rgSBT = {};
    optixSbtRecordPackHeader(raygenPG, &rgSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_raygen), sizeof(RayGenSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_raygen), &rgSBT, sizeof(RayGenSbtRecord), cudaMemcpyHostToDevice);

    MissSbtRecord msSBT = {};
    optixSbtRecordPackHeader(missPG, &msSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_miss), sizeof(MissSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_miss), &msSBT, sizeof(MissSbtRecord), cudaMemcpyHostToDevice);

    HitGroupSbtRecord hgSBT = {};
    optixSbtRecordPackHeader(hitgroupPG, &hgSBT);
    cudaMalloc(reinterpret_cast<void**>(&d_hitgroup), sizeof(HitGroupSbtRecord));
    cudaMemcpy(reinterpret_cast<void*>(d_hitgroup), &hgSBT, sizeof(HitGroupSbtRecord), cudaMemcpyHostToDevice);

    sbt.raygenRecord = d_raygen;
    sbt.missRecordBase = d_miss;
    sbt.missRecordStrideInBytes = sizeof(MissSbtRecord);
    sbt.missRecordCount = 1;
    sbt.hitgroupRecordBase = d_hitgroup;
    sbt.hitgroupRecordStrideInBytes = sizeof(HitGroupSbtRecord);
    sbt.hitgroupRecordCount = 1;

    return true;
}

void OptixRenderer::CopyPreviousTemperatures(const InteropVertex* d_vertices, float4* d_prevTemperature, int numVertices) {
    int blockSize = 256;
    int gridSize = (numVertices + blockSize - 1) / blockSize;
    CopyTemperatureKernel << <gridSize, blockSize >> > (d_vertices, d_prevTemperature, numVertices);
}

void OptixRenderer::Render(OptixParams& hostParams, int numTriangles) {
    hostParams.handle = gasHandle;
    cudaMemcpy(reinterpret_cast<void*>(d_params), &hostParams, sizeof(OptixParams), cudaMemcpyHostToDevice);
    optixLaunch(pipeline, 0, d_params, sizeof(OptixParams), &sbt, numTriangles, 1, 1);
}