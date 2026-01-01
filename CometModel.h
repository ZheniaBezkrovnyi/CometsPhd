#pragma once
#include <vector>
#include "MeshTriangle.h"
#include "PLY.h"
#include "PhysicsConsts.h"
class CometModel {
public:
    std::vector<MeshTriangle> triangles;
    double totalGasProductionRate;

    void LoadFromPLY(const std::string& filename) {
        PLY plyLoader(filename);

        auto rawVertices = plyLoader.getVertices();
        auto rawFaces = plyLoader.getFaces();

        triangles.clear();

        for (const auto& f : rawFaces) {
            const auto& rv1 = rawVertices[f.v1];
            const auto& rv2 = rawVertices[f.v2];
            const auto& rv3 = rawVertices[f.v3];

            Vector3 p1(rv1.x, rv1.y, rv1.z);
            Vector3 p2(rv2.x, rv2.y, rv2.z);
            Vector3 p3(rv3.x, rv3.y, rv3.z);

            triangles.emplace_back(p1, p2, p3);
        }

        std::cout << "Loaded " << triangles.size() << " triangles from " << filename << "\n";
    }

    void UpdateSurfacePhysics(Vector3 sunPosition) {
        totalGasProductionRate = 0.0;

        for (auto& tri : triangles) {
            Vector3 sunDir = (sunPosition - tri.center).normalized();
            double r_h = (sunPosition - tri.center).magnitude() / 1.496e11;
            double cosTheta = sunDir.dot(tri.normal);

            tri.temperature = SolveFaceTemperature(r_h, cosTheta);

            double P_vap = 3.56e12 * std::exp(-6141.0 / tri.temperature);

            tri.gasFluxZ = P_vap / std::sqrt(2 * PhysicsConsts::PI * PhysicsConsts::m_gas_H2O * PhysicsConsts::kB * tri.temperature);
            tri.gasProductionQ = tri.gasFluxZ * PhysicsConsts::m_gas_H2O * tri.area;

            totalGasProductionRate += tri.gasProductionQ;
        }
    }

private:
    double SolveFaceTemperature(double r_h_AU, double cosTheta) {
        double effectiveCos = std::max(0.0, cosTheta);

        double Albedo = 0.04;
        double E_input = (PhysicsConsts::SolarConst / (r_h_AU * r_h_AU)) * (1.0 - Albedo) * effectiveCos;
        double E_avail = E_input * (1.0 - 0.1); 

        if (E_avail < 1e-5) return 20.0;

        double T = 200.0;
        double epsilon = 0.95;

        for (int i = 0; i < 8; i++) {
            double P_vap = 3.56e12 * std::exp(-6141.0 / T);
            double Z = P_vap / std::sqrt(2 * PhysicsConsts::PI * PhysicsConsts::m_gas_H2O * PhysicsConsts::kB * T);

            double Rad = epsilon * PhysicsConsts::Sigma * std::pow(T, 4);
            double Sub = PhysicsConsts::L_sub * PhysicsConsts::m_gas_H2O * Z;

            double F = Rad + Sub - E_avail;

            double dRad = 4 * epsilon * PhysicsConsts::Sigma * std::pow(T, 3);
            double dSub = Sub * (6141.0 / (T * T));

            double T_new = T - F / (dRad + dSub);

            if (std::abs(T_new - T) < 0.01) return T_new;
            T = T_new;
        }
        return T;
    }
};