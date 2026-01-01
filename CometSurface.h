#pragma once
#include <iostream>
#include "PhysicsConsts.h"

class CometSurface {
public:
    static double SolveTemperature(double r_h_AU, double cosTheta) {
        double S_sun = 1361.0;
        double r_meters = r_h_AU * 1.496e11;

        double E_input = (S_sun / (r_h_AU * r_h_AU)) * (1.0 - 0.04) * std::max(0.0, cosTheta);

        double eta = 0.1;
        double E_avail = E_input * (1.0 - eta);

        double T = 200.0;
        double epsilon = 0.95; 

        for (int i = 0; i < 10; i++) {
            double P_vap = 3.56e12 * std::exp(-6141.0 / T);

            double Z = P_vap / std::sqrt(2 * PhysicsConsts::PI * PhysicsConsts::m_gas_H2O * PhysicsConsts::kB * T);

            double Rad = epsilon * PhysicsConsts::Sigma * std::pow(T, 4);
            double Sub = PhysicsConsts::L_sub * PhysicsConsts::m_gas_H2O * Z;

            double F = Rad + Sub - E_avail;

            double dRad = 4 * epsilon * PhysicsConsts::Sigma * std::pow(T, 3);
            double dPvap = P_vap * (6141.0 / (T * T));
            double dZ = (dPvap * std::sqrt(T) - P_vap * 0.5 / std::sqrt(T)) /
                (std::sqrt(2 * PhysicsConsts::PI * PhysicsConsts::m_gas_H2O * PhysicsConsts::kB) * T);
            double dSub = PhysicsConsts::L_sub * PhysicsConsts::m_gas_H2O * dZ;

            double dF = dRad + dSub;

            double T_new = T - F / dF;
            if (abs(T_new - T) < 0.01) return T_new;
            T = T_new;
        }
        return T;
    }
};