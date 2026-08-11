#pragma once
#include <GL/glew.h>
#include <cuda_gl_interop.h>
#include <vector>
#include "Geometry/Geometry.h"

class GLMesh {
public:
    GLMesh() = default;
    ~GLMesh();

    void Upload(const std::vector<InteropVertex>& vertices);
    void BindAndDraw() const;

    InteropVertex* MapToCUDA();
    void UnmapFromCUDA();

    int GetVertexCount() const { return vertexCount; }

private:
    GLuint vbo = 0;
    cudaGraphicsResource* cuda_vbo_resource = nullptr;
    int vertexCount = 0;
};