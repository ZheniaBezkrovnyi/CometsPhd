#pragma once
#include "Vector3.h"
struct MeshTriangle {
    Vector3 v1, v2, v3; 
    Vector3 center;
    Vector3 normal;
    double area;   

    double temperature; 
    double gasFluxZ;
    double gasProductionQ;  

    MeshTriangle(Vector3 _v1, Vector3 _v2, Vector3 _v3) : v1(_v1), v2(_v2), v3(_v3) {
        center = (v1 + v2 + v3) * (1.0 / 3.0);

        Vector3 edge1 = v2 - v1;
        Vector3 edge2 = v3 - v1;
        Vector3 crossProd = edge1.cross(edge2);

        double crossMag = crossProd.magnitude();
        area = 0.5 * crossMag;
        normal = (crossMag > 0) ? crossProd * (1.0 / crossMag) : Vector3(0, 1, 0);

        temperature = 20.0;
        gasProductionQ = 0.0;
    }
};