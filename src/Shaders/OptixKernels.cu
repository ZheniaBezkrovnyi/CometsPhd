#include <optix.h>
#include <optix_device.h>
#include <cuda_runtime.h>
#include "OptixParams.h"

extern "C" {
    __constant__ OptixParams params;
}

static __forceinline__ __device__ float Dot3(float3 a, float3 b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

static __forceinline__ __device__ float VaporPressureH2O(float T) {
    return 3.56e12f * expf(-6141.0f / T);
}

static __forceinline__ __device__ float SublimationNumberFluxH2O(float T, float activeFraction) {
    float P = VaporPressureH2O(T);

    const float sqrt_const = 1.6113e-24f;

    return activeFraction * P / (sqrt_const * sqrtf(T));
}

static __forceinline__ __device__ float SublimationMassFluxH2O(float T, float activeFraction) {
    const float m = 2.9915e-26f;
    return SublimationNumberFluxH2O(T, activeFraction) * m;
}

static __forceinline__ __device__ float SolveSurfaceTemperature(float absorbedFlux) {
    const float sigma = 5.670374419e-8f;
    const float latentHeat = 2.83e6f;

    if (absorbedFlux <= 1e-6f) return params.minTemp;

    float T = 180.0f;

    for (int i = 0; i < 20; ++i) {
        float massFlux = SublimationMassFluxH2O(T, params.activeFraction);

        float radiation = params.emissivity * sigma * T * T * T * T;
        float sublimation = latentHeat * massFlux;

        float F = radiation + sublimation - absorbedFlux;

        float dRadiation = 4.0f * params.emissivity * sigma * T * T * T;
        float dlnZ = 6141.0f / (T * T) - 0.5f / T;
        float dSublimation = sublimation * dlnZ;

        float dF = dRadiation + dSublimation;

        T -= F / dF;
        T = fminf(fmaxf(T, params.minTemp), 400.0f);
    }

    return T;
}

__device__ float SmoothStep(float edge0, float edge1, float x) {
    float t = fminf(fmaxf((x - edge0) / (edge1 - edge0), 0.0f), 1.0f);
    return t * t * (3.0f - 2.0f * t);
}

__device__ float3 LerpColor(float3 a, float3 b, float t) {
    return make_float3(
        a.x + (b.x - a.x) * t,
        a.y + (b.y - a.y) * t,
        a.z + (b.z - a.z) * t
    );
}

__device__ void GetHeatmapColor(float temp, float minTemp, float maxTemp, float* r, float* g, float* b) {
    float range = fmaxf(maxTemp - minTemp, 1.0f);
    float t = fminf(fmaxf((temp - minTemp) / range, 0.0f), 1.0f);

    t = powf(t, 0.45f);

    float3 c0 = make_float3(0.030f, 0.045f, 0.180f);
    float3 c1 = make_float3(0.000f, 0.120f, 0.420f);
    float3 c2 = make_float3(0.000f, 0.380f, 0.850f);
    float3 c3 = make_float3(0.000f, 0.850f, 0.720f);
    float3 c4 = make_float3(0.950f, 0.900f, 0.120f);
    float3 c5 = make_float3(1.000f, 0.300f, 0.020f);
    float3 c6 = make_float3(1.000f, 0.950f, 0.760f);

    float3 color;

    if (t < 0.14f) {
        color = LerpColor(c0, c1, SmoothStep(0.00f, 0.14f, t));
    }
    else if (t < 0.34f) {
        color = LerpColor(c1, c2, SmoothStep(0.14f, 0.34f, t));
    }
    else if (t < 0.54f) {
        color = LerpColor(c2, c3, SmoothStep(0.34f, 0.54f, t));
    }
    else if (t < 0.74f) {
        color = LerpColor(c3, c4, SmoothStep(0.54f, 0.74f, t));
    }
    else if (t < 0.91f) {
        color = LerpColor(c4, c5, SmoothStep(0.74f, 0.91f, t));
    }
    else {
        color = LerpColor(c5, c6, SmoothStep(0.91f, 1.00f, t));
    }

    float brightness = 1.35f;

    color.x = fminf(color.x * brightness, 1.0f);
    color.y = fminf(color.y * brightness, 1.0f);
    color.z = fminf(color.z * brightness, 1.0f);

    *r = color.x;
    *g = color.y;
    *b = color.z;
}

extern "C" __global__ void __miss__ms() {
    optixSetPayload_0(0xFFFFFFFFu);
}

extern "C" __global__ void __closesthit__ch() {
    unsigned int primitiveIndex = optixGetPrimitiveIndex();
    optixSetPayload_0(primitiveIndex);
}

extern "C" __global__ void __raygen__rg() {
    const uint3 launchIndex = optixGetLaunchIndex();

    int triIdx = static_cast<int>(launchIndex.x);
    int numTriangles = params.numVertices / 3;

    if (triIdx >= numTriangles) return;

    int v0 = triIdx * 3;
    int v1 = triIdx * 3 + 1;
    int v2 = triIdx * 3 + 2;

    float4 n0 = params.vertices[v0].normal;
    float3 normal = make_float3(n0.x, n0.y, n0.z);

    float4 p0 = params.vertices[v0].position;
    float4 p1 = params.vertices[v1].position;
    float4 p2 = params.vertices[v2].position;

    float3 position = make_float3(
        (p0.x + p1.x + p2.x) * 0.3333333333f,
        (p0.y + p1.y + p2.y) * 0.3333333333f,
        (p0.z + p1.z + p2.z) * 0.3333333333f
    );

    float mu = fmaxf(0.0f, Dot3(normal, params.sunDir));

    unsigned int payload0 = 0xFFFFFFFFu;

    if (mu > 0.0f) {
        float epsilon = 0.1f;
        float3 rayOrigin = make_float3(
            position.x + normal.x * epsilon,
            position.y + normal.y * epsilon,
            position.z + normal.z * epsilon
        );

        optixTrace(
            params.handle,
            rayOrigin,   
            params.sunDir,
            0.1f,       
            1.0e16f,    
            0.0f,
            OptixVisibilityMask(255),
            OPTIX_RAY_FLAG_DISABLE_ANYHIT | OPTIX_RAY_FLAG_TERMINATE_ON_FIRST_HIT,
            0,
            1,
            0,
            payload0
        );
    }

    float shadow = (mu > 0.0f && payload0 == 0xFFFFFFFFu) ? 1.0f : 0.0f;

    float solarFlux = params.solarConstant / (params.rh_AU * params.rh_AU);
    float absorbedFlux = (1.0f - params.albedo) * solarFlux * mu * shadow;

    float newTemp = SolveSurfaceTemperature(absorbedFlux);

    float r, g, b;
    GetHeatmapColor(newTemp, params.minTemp, params.maxTempForColor, &r, &g, &b);

    float4 finalTemp = make_float4(newTemp, absorbedFlux, 0.0f, 0.0f);
    float4 finalColor = make_float4(r, g, b, 1.0f);

    params.vertices[v0].temperature = finalTemp;
    params.vertices[v1].temperature = finalTemp;
    params.vertices[v2].temperature = finalTemp;

    params.vertices[v0].color = finalColor;
    params.vertices[v1].color = finalColor;
    params.vertices[v2].color = finalColor;
}