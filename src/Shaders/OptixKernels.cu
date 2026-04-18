#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "OptixParams.h"

extern "C" {
    __constant__ OptixParams params;
}

__device__ void GetHeatmapColor(float temp, float baseTemp, float tempScale, float* r, float* g, float* b) {
    float t = fminf(fmaxf((temp - baseTemp) / tempScale, 0.0f), 1.0f);
    if (t < 0.5f) {
        *r = 0.0f; *g = t * 2.0f; *b = 1.0f - t * 2.0f;
    }
    else {
        *r = (t - 0.5f) * 2.0f; *g = 1.0f - (t - 0.5f) * 2.0f; *b = 0.0f;
    }
}


extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    if (idx.x >= params.numVertices) return;

    float4 norm = params.vertices[idx.x].normal;
    float4 pos = params.vertices[idx.x].position;

    float3 normal = make_float3(norm.x, norm.y, norm.z);
    float3 position = make_float3(pos.x, pos.y, pos.z);

    float cosTheta = (normal.x * params.sunDir.x + normal.y * params.sunDir.y + normal.z * params.sunDir.z);

    unsigned int isVisible = 0;

    if (cosTheta > 0.0f) {
        optixTrace(
            params.handle,
            position,           
            params.sunDir,      
            0.001f,             
            1e16f,         
            0.0f,          
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0, 1, 0,     
            isVisible
        );
    }
    else {
        cosTheta = 0.0f;
    }

    float newTemp = params.baseTemp + (cosTheta * params.tempScale * (float)isVisible);

    float r, g, b;
    GetHeatmapColor(newTemp, params.baseTemp, params.tempScale, &r, &g, &b);

    params.vertices[idx.x].temperature = make_float4(newTemp, 0.5f, 0.5f, 0.5f);
    params.vertices[idx.x].color = make_float4(r, g, b, 1.0f);
}


extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(1);
}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(0);
}