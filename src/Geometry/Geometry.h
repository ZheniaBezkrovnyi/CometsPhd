#pragma once
#include <iostream>
#include <vector>
#include <set>
#include <cstdint>
#include <unordered_set>
#include <vector_types.h>


struct alignas(16) InteropVertex {
    float4 position;
    float4 normal;
    float4 color;
    float4 temperature;
};

struct Vertex {
    int index;

    float x, y, z;
    std::set<int> neighbors;
    std::set<int> faceIndices;

    Vertex operator-(const Vertex& other) const {
        Vertex result;
        result.x = this->x - other.x;
        result.y = this->y - other.y;
        result.z = this->z - other.z;

        return result;
    }
    bool operator==(const Vertex& other) const {
        if (x == other.x && y == other.y && z == other.z) {
            return true;
        }
        return false;
    }
};

struct Face {
    int v1, v2, v3;
};