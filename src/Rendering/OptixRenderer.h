#pragma once
#include <optix.h>
#include <cuda_runtime.h>
#include <string>
#include "Shaders/OptixParams.h"
#include "Geometry/Geometry.h"

struct AppSettings;

class OptixRenderer {
public:
    OptixRenderer() = default;
    ~OptixRenderer();

    bool Init(const AppSettings& config);
    bool BuildGAS(CUdeviceptr d_vertices, int numVertices);
    bool BuildPipeline(const std::string& ptxPath);
    void Render(OptixParams& hostParams, int numTriangles);

    void CopyPreviousTemperatures(const InteropVertex* d_vertices, float4* d_prevTemperature, int numVertices);

private:
    bool BuildSBT();

    OptixDeviceContext optixContext = nullptr;
    OptixTraversableHandle gasHandle = 0;
    CUdeviceptr d_gas_output_buffer = 0;

    OptixModule module = nullptr;
    OptixPipeline pipeline = nullptr;
    OptixProgramGroup raygenPG = nullptr;
    OptixProgramGroup missPG = nullptr;
    OptixProgramGroup hitgroupPG = nullptr;
    OptixShaderBindingTable sbt = {};

    CUdeviceptr d_params = 0;
};