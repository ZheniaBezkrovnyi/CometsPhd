#pragma once
#include <optix.h>
#include <vector_types.h>
#include "Geometry/Geometry.h"

struct OptixParams {
    OptixTraversableHandle handle;
    InteropVertex* vertices;
    int numVertices;

    float3 sunDir;

    float rh_AU;
    float solarConstant;
    float albedo;
    float emissivity;
    float activeFraction;

    float minTemp;
    float maxTempForColor;

    unsigned int frameCount;
};