#pragma once
#include <optix.h>
#include <vector_types.h>
#include "Geometry/Geometry.h"

struct OptixParams {
    OptixTraversableHandle handle;

    InteropVertex* vertices;
    float4* prevTemperature;

    int numVertices;

    float3 sunDir;
    float3 earthDir;
    float* d_totalVisibleArea;

    float rh_AU;
    float solarConstant;
    float albedo;
    float emissivity;
    float activeFraction;

    float minTemp;
    float maxTempForColor;

    unsigned int frameCount;

    int indirectSamples;
    unsigned int indirectSeed;

    float indirectSolarScale;
    float indirectIRScale;
    float maxIndirectFractionOfSolarFlux;
    float rayEpsilon;
};