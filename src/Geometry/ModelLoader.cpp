#include "ModelLoader.h"
#include "IO/PLY.h"
#include <glm/glm.hpp>
#include <algorithm>

std::vector<InteropVertex> ModelLoader::LoadPLY(const std::string& filepath, float& outMaxCoord, float baseTemp) {
    PLY plyModel(filepath);
    std::vector<Face> faces = plyModel.getFaces();
    std::vector<Vertex> verts = plyModel.getVertices();

    std::vector<InteropVertex> vertices;
    vertices.reserve(faces.size() * 3);

    outMaxCoord = 0.0f;
    for (const auto& face : faces) {
        glm::vec3 p1(verts[face.v1].x, verts[face.v1].y, verts[face.v1].z);
        glm::vec3 p2(verts[face.v2].x, verts[face.v2].y, verts[face.v2].z);
        glm::vec3 p3(verts[face.v3].x, verts[face.v3].y, verts[face.v3].z);

        glm::vec3 edge1 = p2 - p1;
        glm::vec3 edge2 = p3 - p1;
        glm::vec3 normal = glm::normalize(glm::cross(edge1, edge2));

        outMaxCoord = std::fmax(outMaxCoord, glm::length(p1));
        outMaxCoord = std::fmax(outMaxCoord, glm::length(p2));
        outMaxCoord = std::fmax(outMaxCoord, glm::length(p3));

        int indices[3] = { face.v1, face.v2, face.v3 };
        for (int i = 0; i < 3; ++i) {
            InteropVertex v;
            v.position = make_float4(verts[indices[i]].x, verts[indices[i]].y, verts[indices[i]].z, 0.5f);
            v.normal = make_float4(normal.x, normal.y, normal.z, 0.5f);
            v.color = make_float4(0.5f, 0.5f, 0.5f, 0.5f);
            v.temperature = make_float4(baseTemp, 0.5f, 0.5f, 0.5f);
            vertices.push_back(v);
        }
    }
    return vertices;
}