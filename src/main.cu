#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cuda_gl_interop.h>

#include <iostream>
#include <vector>
#include <cmath>

#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>

#include "IO/PLY.h"
#include "Physics/Sun.h"
#include "Geometry/Geometry.h"
#include "Core/PhysicsConsts.h"


__device__ void GetHeatmapColor(float temp, float* r, float* g, float* b) {
    float t = fminf(fmaxf((temp - 20.0f) / 380.0f, 0.0f), 1.0f);

    if (t < 0.5f) {
        *r = 0.0f;
        *g = t * 2.0f;
        *b = 1.0f - t * 2.0f;
    }
    else {
        *r = (t - 0.5f) * 2.0f;
        *g = 1.0f - (t - 0.5f) * 2.0f;
        *b = 0.0f;
    }
}

__global__ void UpdateCometTemperatureKernel(InteropVertex* vertices, int numVertices, glm::vec3 sunLocalPos) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= numVertices) return;

    glm::vec3 pos(vertices[idx].x, vertices[idx].y, vertices[idx].z);
    glm::vec3 norm(vertices[idx].nx, vertices[idx].ny, vertices[idx].nz);

    glm::vec3 dirToSun = glm::normalize(sunLocalPos - pos);

    float cosTheta = fmaxf(glm::dot(norm, dirToSun), 0.0f);

    float newTemp = 20.0f + cosTheta * 380.0f;
    vertices[idx].temperature = newTemp;

    GetHeatmapColor(newTemp, &vertices[idx].r, &vertices[idx].g, &vertices[idx].b);
}

// =====================================================================
// MAIN APPLICATION
// =====================================================================

GLuint vbo;
cudaGraphicsResource* cuda_vbo_resource;

void CheckCUDAError(const char* msg) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "CUDA Error (" << msg << "): code " << (int)err << std::endl;
    }
}

int main() {

    try {
        std::cout << "[INIT] Loading PLY Model...\n";
        PLY plyModel("C:/Users/ebbez/source/repos/CometsPhd/data/plyexample.ply");
        std::vector<Face> faces = plyModel.getFaces();
        std::vector<Vertex> verts = plyModel.getVertices();

        std::vector<InteropVertex> hostVertices;
        hostVertices.reserve(faces.size() * 3);

        float maxCoord = 0.0f;

        for (const auto& face : faces) {
            glm::vec3 p1(verts[face.v1].x, verts[face.v1].y, verts[face.v1].z);
            glm::vec3 p2(verts[face.v2].x, verts[face.v2].y, verts[face.v2].z);
            glm::vec3 p3(verts[face.v3].x, verts[face.v3].y, verts[face.v3].z);

            glm::vec3 edge1 = p2 - p1;
            glm::vec3 edge2 = p3 - p1;
            glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

            maxCoord = fmaxf(maxCoord, glm::length(p1));
            maxCoord = fmaxf(maxCoord, glm::length(p2));
            maxCoord = fmaxf(maxCoord, glm::length(p3));

            int indices[3] = { face.v1, face.v2, face.v3 };
            for (int i = 0; i < 3; ++i) {
                InteropVertex v;
                v.x = verts[indices[i]].x; v.y = verts[indices[i]].y; v.z = verts[indices[i]].z;
                v.nx = normal.x; v.ny = normal.y; v.nz = normal.z;
                v.r = 0.5f; v.g = 0.5f; v.b = 0.5f;
                v.temperature = 20.0f;
                hostVertices.push_back(v);
            }
        }
        int totalVertices = hostVertices.size();
        std::cout << "[INIT] Prepared " << totalVertices << " unrolled vertices.\n";

        if (!glfwInit()) return -1;
        GLFWwindow* window = glfwCreateWindow(1024, 768, "Comet CUDA Interop", NULL, NULL);
        if (!window) { glfwTerminate(); return -1; }
        glfwMakeContextCurrent(window);
        glewExperimental = GL_TRUE;

        if (glewInit() != GLEW_OK) return -1;

        glGenBuffers(1, &vbo);
        glBindBuffer(GL_ARRAY_BUFFER, vbo);
        glBufferData(GL_ARRAY_BUFFER, totalVertices * sizeof(InteropVertex), hostVertices.data(), GL_DYNAMIC_DRAW);

        cudaGraphicsGLRegisterBuffer(&cuda_vbo_resource, vbo, cudaGraphicsMapFlagsNone);
        CheckCUDAError("GLRegisterBuffer");

        glEnableClientState(GL_VERTEX_ARRAY);
        glEnableClientState(GL_COLOR_ARRAY);
        glEnableClientState(GL_NORMAL_ARRAY);

        glVertexPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, x));
        glNormalPointer(GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, nx));
        glColorPointer(3, GL_FLOAT, sizeof(InteropVertex), (void*)offsetof(InteropVertex, r));

        glEnable(GL_DEPTH_TEST);

        double t = 0.0;
        double dt = 10.0;

        Sun sun(glm::dvec3(PhysicsConsts::AU_METERS, 0.0, 0.0));

        std::cout << "[RUN] Starting Rendering Loop...\n";
        while (!glfwWindowShouldClose(window)) {
            glfwPollEvents();

            InteropVertex* d_vertices;
            size_t num_bytes;

            cudaGraphicsMapResources(1, &cuda_vbo_resource, 0);
            cudaGraphicsResourceGetMappedPointer((void**)&d_vertices, &num_bytes, cuda_vbo_resource);

            double rotationSpeed = 2.0 * PhysicsConsts::PI / (12.0 * 3600.0);
            float currentAngle = (float)(t * rotationSpeed);

            glm::mat3 invRot = glm::mat3(glm::rotate(glm::mat4(1.0f), -currentAngle, glm::vec3(0.0f, 1.0f, 0.0f)));
            glm::vec3 sunLocalPos = invRot * glm::vec3(1.0f, 0.0f, 0.0f);

            int blockSize = 256;
            int gridSize = (totalVertices + blockSize - 1) / blockSize;
            UpdateCometTemperatureKernel<<<gridSize, blockSize >>>(d_vertices, totalVertices, sunLocalPos);
            cudaDeviceSynchronize();

            CheckCUDAError("Kernel Launch");

            cudaGraphicsUnmapResources(1, &cuda_vbo_resource, 0);

            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            if (height == 0) height = 1;
            if (width == 0) width = 1;

            glViewport(0, 0, width, height);
            glClearColor(0.05f, 0.05f, 0.1f, 1.0f);
            glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

            glMatrixMode(GL_PROJECTION);
            glLoadIdentity();
            glm::mat4 proj = glm::perspective(glm::radians(45.0f), (float)width / (float)height, 0.1f, maxCoord * 10.0f);
            glLoadMatrixf(glm::value_ptr(proj));

            glMatrixMode(GL_MODELVIEW);
            glLoadIdentity();
            glm::mat4 view = glm::lookAt(glm::vec3(0, maxCoord * 0.5f, maxCoord * 2.5f), glm::vec3(0, 0, 0), glm::vec3(0, 1, 0));

            glm::mat4 model = glm::rotate(glm::mat4(1.0f), currentAngle, glm::vec3(0, 1, 0));
            glm::mat4 mv = view * model;
            glLoadMatrixf(glm::value_ptr(mv));

            glBindBuffer(GL_ARRAY_BUFFER, vbo);
            glDrawArrays(GL_TRIANGLES, 0, totalVertices);

            glfwSwapBuffers(window);
            t += dt;
        }

        cudaGraphicsUnregisterResource(cuda_vbo_resource);
        glDeleteBuffers(1, &vbo);
        glfwDestroyWindow(window);
        glfwTerminate();

        return 0;
    }
    catch (const std::exception& e) { std::cerr << e.what(); }
}