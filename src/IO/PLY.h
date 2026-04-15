#pragma once
#include <string>
#include <vector>
#include <typeindex>
#include "Geometry/Geometry.h"
#include "plycpp.h"


class PLY {
public:
    PLY(const std::string& fileName);
    std::vector<Vertex> getVertices() const;
    std::vector<Face> getFaces() const;

private:
    std::string fileName;
    mutable std::type_index typeIndex = typeid(void);
};
