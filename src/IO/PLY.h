#pragma once
#include <string>
#include <vector>
#include <typeindex>
#include "Geometry.h"
#include "plycpp.h"

using namespace std;

class PLY {
public:
    PLY(const string& fileName);
    vector<Vertex> getVertices() const;
    vector<Face> getFaces() const;

private:
    string fileName;
    mutable type_index typeIndex = typeid(void);
};
