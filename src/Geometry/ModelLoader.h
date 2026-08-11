#pragma once
#include <string>
#include <vector>
#include "Geometry/Geometry.h"

class ModelLoader {
public:
    static std::vector<InteropVertex> LoadPLY(const std::string& filepath, float& outMaxCoord, float baseTemp);
};