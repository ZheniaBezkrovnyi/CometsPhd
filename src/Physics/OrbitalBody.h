#pragma once
#include <glm/glm.hpp>
#include "Core/PhysicsConsts.h"

struct KeplerianElements {
    double a;      // Велика піввісь (Астрономічні одиниці - АО)
    double e;      // Ексцентриситет
    double i;      // Нахил орбіти (градуси)
    double Omega;  // Довгота висхідного вузла (градуси)
    double w;      // Аргумент перицентру (градуси)
    double M0;     // Середня аномалія в епоху (градуси)
    double epoch;  // Епоха (Юліанський день - JD)
};

class OrbitalBody {
public:
    OrbitalBody() = default;
    OrbitalBody(const KeplerianElements& elements);

    // Повертає 3D координати в Астрономічних Одиницях (АО) відносно Сонця
    glm::dvec3 CalculatePosition(double currentJD) const;

private:
    KeplerianElements orbParams;
    double meanMotion; // Середній рух (радіани на день)

    double SolveKepler(double M, double e) const;
};