#pragma once
#define GLM_ENABLE_EXPERIMENTAL

#include <vector>
#include <cmath>
#include <iostream>
#include <algorithm>
#include <string>
#include <glm/glm.hpp>
#include <glm/gtx/norm.hpp>
#include "IO/PLY.h"           
#include "Core/PhysicsConsts.h" 
#include "MeshTriangle.h"

class CometModel {
public:
    std::vector<MeshTriangle> triangles;
    glm::dmat3 currentRotation;
    glm::dvec3 currentPosition;

    double totalGasProductionRate;

    CometModel() {
        currentRotation = glm::dmat3(1.0);
        currentPosition = glm::dvec3(0.0);
        totalGasProductionRate = 0.0;
    }

    void LoadFromPLY(const std::string& filename) {
        PLY plyLoader(filename);
        auto rawVertices = plyLoader.getVertices();
        auto rawFaces = plyLoader.getFaces();

        triangles.clear();
        triangles.reserve(rawFaces.size());

        for (const auto& f : rawFaces) {
            glm::dvec3 p1(rawVertices[f.v1].x, rawVertices[f.v1].y, rawVertices[f.v1].z);
            glm::dvec3 p2(rawVertices[f.v2].x, rawVertices[f.v2].y, rawVertices[f.v2].z);
            glm::dvec3 p3(rawVertices[f.v3].x, rawVertices[f.v3].y, rawVertices[f.v3].z);

            triangles.emplace_back(p1, p2, p3);
        }
        std::cout << "Model loaded: " << triangles.size() << " triangles.\n";
    }

    void SetTransform(glm::dvec3 pos, glm::dmat3 rot) {
        currentPosition = pos;
        currentRotation = rot;
    }

    void UpdateSurfacePhysics(glm::dvec3 sunPositionWorld) {
        totalGasProductionRate = 0.0;

        for (auto& tri : triangles) {
            glm::dvec3 worldCenter = tri.GetWorldCenter(currentPosition, currentRotation);
            glm::dvec3 worldNormal = tri.GetWorldNormal(currentRotation);

            glm::dvec3 sunVec = sunPositionWorld - worldCenter;
            double distSq = glm::length2(sunVec);
            double dist = std::sqrt(distSq);

            glm::dvec3 sunDir = sunVec * (1.0 / dist);

            double cosTheta = glm::dot(sunDir, worldNormal);
            double effectiveCos = std::max(0.0, cosTheta);

            double r_h_AU = dist / PhysicsConsts::AU_METERS;
            double E_input = (PhysicsConsts::SolarConst / (r_h_AU * r_h_AU))
                * (1.0 - PhysicsConsts::Albedo) * effectiveCos;

            tri.temperature = SolveFaceTemperature(E_input);

            double P_vap = 3.56e12 * std::exp(-6141.0 / tri.temperature);
            double sqrtT = std::sqrt(tri.temperature);

            double denominator = std::sqrt(2 * PhysicsConsts::PI * PhysicsConsts::m_gas_H2O * PhysicsConsts::kB) * sqrtT;
            tri.gasFluxZ = P_vap / denominator;

            tri.localGasVelocity = std::sqrt((8.0 * PhysicsConsts::kB * tri.temperature)
                / (PhysicsConsts::PI * PhysicsConsts::m_gas_H2O));

            tri.gasProductionQ = tri.gasFluxZ * PhysicsConsts::m_gas_H2O * tri.area;

            totalGasProductionRate += tri.gasProductionQ;
        }
    }

private:
    double SolveFaceTemperature(double EnergyInput) {
        double E_avail = EnergyInput;

        if (E_avail <= 1e-6) return 40.0;

        double T = 180.0;
        double epsilon = PhysicsConsts::Emissivity;

        double activeFraction = 1.0;

        const double sigma = PhysicsConsts::Sigma;
        const double L_sub = PhysicsConsts::L_sub;
        const double m_gas = PhysicsConsts::m_gas_H2O;
        const double kB = PhysicsConsts::kB;
        const double PI = PhysicsConsts::PI;

        const double sqrt_const = 1.6113e-24;

        for (int i = 0; i < 20; i++) {
            double P_vap = 3.56e12 * std::exp(-6141.0 / T);
            double sqrtT = std::sqrt(T);

            double numFlux = (activeFraction * P_vap) / (sqrt_const * sqrtT);
            double massFlux = numFlux * m_gas;

            double Rad = epsilon * sigma * (T * T * T * T);
            double Sub = L_sub * massFlux;
            double F = Rad + Sub - E_avail;

            double dRad = 4.0 * epsilon * sigma * (T * T * T);

            double dlnZ = 6141.0 / (T * T) - 0.5 / T;
            double dSub = Sub * dlnZ;
            double dF = dRad + dSub;

            double T_new = T - F / dF;

            if (std::abs(T_new - T) < 0.001) return T_new;

            if (T_new < 40.0) T_new = 40.0;
            if (T_new > 400.0) T_new = 400.0;

            T = T_new;
        }
        return T;
    }
};