#include "PLY.h"
#include "plycpp.h"
#include <iostream>
#include <stdexcept>

using namespace std;
using namespace plycpp;

PLY::PLY(const string& file) : fileName(file), typeIndex(typeid(void)) {}


vector<Vertex> PLY::getVertices() const {
    PLYData plyData;
    load(fileName, plyData);

    vector<Vertex> vertices;
    for (const auto& element : plyData) {
        if (element.key == "vertex") {
            auto plyVertex = element.data;
            int size = plyVertex->size();

            for (size_t i = 0; i < size; i++) {

                Vertex vertex;
                vertex.x = plyVertex->properties["x"]->at<float>(i);
                vertex.y = plyVertex->properties["y"]->at<float>(i);
                vertex.z = plyVertex->properties["z"]->at<float>(i);

                //cout << vertex.x << "   " << vertex.y << "   " << vertex.z << "   " << endl;
                vertices.push_back(vertex);
            }
        }
    }
    return vertices;
}


vector<Face> PLY::getFaces() const {
    PLYData plyData;
    load(fileName, plyData);

    vector<Face> faces;
    for (const auto& element : plyData) {
        for (const auto& prop : element.data->properties)
        {
            if (prop.key == string("vertex_indices")) {
                typeIndex = prop.data->type;
            }
        }
    }
    for (const auto& element : plyData) {
        if (element.key == "face") {
            auto plyFace = plyData["face"];
            auto faceProperty = plyFace->properties["vertex_indices"];

            bool isUnsigned = typeIndex.name() == string("unsigned int");

            cout << plyFace->size() << "plyFace->size()" << endl;

            for (int i = 0; i + 2 < faceProperty->size(); i += 3) {
                Face face;
                if (isUnsigned) {
                    face.v1 = static_cast<int>(faceProperty->at<unsigned int>(i));
                    face.v2 = static_cast<int>(faceProperty->at<unsigned int>(i + 1));
                    face.v3 = static_cast<int>(faceProperty->at<unsigned int>(i + 2));

                    //cout << face.v1 << "   " << face.v2 << "   " << face.v3 << "   " << endl;
                }
                else {
                    face.v1 = faceProperty->at<int>(i);
                    face.v2 = faceProperty->at<int>(i + 1);
                    face.v3 = faceProperty->at<int>(i + 2);
                }

                faces.push_back(face);
            }
        }
    }

    return faces;
}
