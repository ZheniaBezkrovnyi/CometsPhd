#include "GLMesh.h"
#include <cstddef> 

GLMesh::~GLMesh() {
    if (cuda_vbo_resource) cudaGraphicsUnregisterResource(cuda_vbo_resource);
    if (vbo) glDeleteBuffers(1, &vbo);
}

void GLMesh::Upload(const std::vector<InteropVertex>& vertices) {
    vertexCount = static_cast<int>(vertices.size());

    glGenBuffers(1, &vbo);
    glBindBuffer(GL_ARRAY_BUFFER, vbo);
    glBufferData(GL_ARRAY_BUFFER, vertexCount * sizeof(InteropVertex), vertices.data(), GL_STATIC_DRAW);

    cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

void GLMesh::BindAndDraw() const {
    glBindBuffer(GL_ARRAY_BUFFER, vbo);

    glEnableClientState(GL_VERTEX_ARRAY);
    glEnableClientState(GL_COLOR_ARRAY);
    glEnableClientState(GL_NORMAL_ARRAY);

    glVertexPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, position));
    glNormalPointer(GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, normal));
    glColorPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, color));

    glDrawArrays(GL_TRIANGLES, 0, vertexCount);

    glDisableClientState(GL_VERTEX_ARRAY);
    glDisableClientState(GL_COLOR_ARRAY);
    glDisableClientState(GL_NORMAL_ARRAY);

    glBindBuffer(GL_ARRAY_BUFFER, 0);
}

InteropVertex* GLMesh::MapToCUDA() {
    if (!cuda_vbo_resource) return nullptr;
    InteropVertex* d_vertices;
    size_t num_bytes;
    cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
    cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);
    return d_vertices;
}

void GLMesh::UnmapFromCUDA() {
    if (cuda_vbo_resource) {
        cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);
    }
}