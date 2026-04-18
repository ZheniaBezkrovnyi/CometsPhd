#pragma once
#include <optix.h>
#include <vector_types.h>
#include "Geometry/Geometry.h"

struct OptixParams {
    OptixTraversableHandle handle; 
    InteropVertex* vertices;     
    int numVertices;     

    float3 sunDir;  
    float baseTemp;
    float tempScale;
};