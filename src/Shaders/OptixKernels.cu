#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "OptixParams.h"

extern "C" {
    __constant__ OptixParams params;
}


static __forceinline__ __device__ unsigned int lcg(unsigned int& prev) {
    const unsigned int LCG_A = 1664525u;
    const unsigned int LCG_C = 1013904223u;
    prev = (LCG_A * prev + LCG_C);
    return prev & 0x00FFFFFF;
}

static __forceinline__ __device__ float rnd(unsigned int& prev) {
    return ((float)lcg(prev) / (float)0x01000000);
}

static __forceinline__ __device__ float3 cosine_sample_hemisphere(const float3& n, unsigned int& seed) {
    float r1 = rnd(seed);
    float r2 = rnd(seed);
    float z = sqrtf(1.0f - r2);
    float phi = 2.0f * 3.1415926f * r1;
    float x = cosf(phi) * sqrtf(r2);
    float y = sinf(phi) * sqrtf(r2);

    float3 u = (fabsf(n.x) > 0.1f) ? make_float3(0.0f, 1.0f, 0.0f) : make_float3(1.0f, 0.0f, 0.0f);
    float3 v = make_float3(n.y * u.z - n.z * u.y, n.z * u.x - n.x * u.z, n.x * u.y - n.y * u.x);
    float invLen = 1.0f / sqrtf(v.x * v.x + v.y * v.y + v.z * v.z);
    v = make_float3(v.x * invLen, v.y * invLen, v.z * invLen);
    u = make_float3(v.y * n.z - v.z * n.y, v.z * n.x - v.x * n.z, v.x * n.y - v.y * n.x);

    return make_float3(
        x * u.x + y * v.x + z * n.x,
        x * u.y + y * v.y + z * n.y,
        x * u.z + y * v.z + z * n.z
    );
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


extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(0xFFFFFFFF);
}

extern "C" __global__ void __closesthit__ch() {
    optixSetPayload_0(optixGetPrimitiveIndex());
}

extern "C" __global__ void __raygen__rg() {
    const uint3 idx = optixGetLaunchIndex();
    int triIdx = idx.x;
    int numTriangles = params.numVertices / 3;

    if (triIdx >= numTriangles) return;

    int v0 = triIdx * 3;
    int v1 = triIdx * 3 + 1;
    int v2 = triIdx * 3 + 2;

    float4 norm = params.vertices[v0].normal;
    float3 normal = make_float3(norm.x, norm.y, norm.z);

    float4 p0 = params.vertices[v0].position;
    float4 p1 = params.vertices[v1].position;
    float4 p2 = params.vertices[v2].position;

    float3 position = make_float3((p0.x + p1.x + p2.x) / 3.0f,
        (p0.y + p1.y + p2.y) / 3.0f,
        (p0.z + p1.z + p2.z) / 3.0f);

    unsigned int seed = triIdx + params.frameCount * numTriangles;

    float cosTheta = (normal.x * params.sunDir.x + normal.y * params.sunDir.y + normal.z * params.sunDir.z);
    unsigned int hit_idx = 0xFFFFFFFF;

    if (cosTheta > 0.0f) {
        optixTrace(
            params.handle, position, params.sunDir,
            0.001f, 1e16f, 0.0f, OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0, 1, 0, hit_idx
        );
    }

    float direct_energy = (cosTheta > 0.0f && hit_idx == 0xFFFFFFFF) ? cosTheta : 0.0f;

    float gathered_heat = 0.0f;
    const int NUM_BOUNCE_RAYS = 10;

    for (int i = 0; i < NUM_BOUNCE_RAYS; i++) {
        float3 bounce_dir = cosine_sample_hemisphere(normal, seed);
        unsigned int bounce_hit_idx = 0xFFFFFFFF;

        optixTrace(
            params.handle, position, bounce_dir,
            0.001f, 1e16f, 0.0f, OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT,
            0, 1, 0, bounce_hit_idx
        );

        if (bounce_hit_idx != 0xFFFFFFFF) {
            float neighborTemp = params.vertices[bounce_hit_idx * 3].temperature.x;
            gathered_heat += neighborTemp;
        }
    }

    gathered_heat /= (float)NUM_BOUNCE_RAYS;

    float newTemp = params.baseTemp + (direct_energy * params.tempScale) + (gathered_heat * 0.1f);

    float r, g, b;
    GetHeatmapColor(newTemp, params.baseTemp, params.tempScale, &r, &g, &b);

    float4 finalTemp = make_float4(newTemp, 0.5f, 0.5f, 0.5f);
    float4 finalColor = make_float4(r, g, b, 1.0f);

    params.vertices[v0].temperature = finalTemp;
    params.vertices[v1].temperature = finalTemp;
    params.vertices[v2].temperature = finalTemp;

    params.vertices[v0].color = finalColor;
    params.vertices[v1].color = finalColor;
    params.vertices[v2].color = finalColor;
}