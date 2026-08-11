#include "OrbitalBody.h"
#include <cmath>

OrbitalBody::OrbitalBody(const KeplerianElements& elements) : orbParams(elements) {
    // Третій закон Кеплера: n = sqrt(G * M_sun / a^3). 
    // У системі одиниць сонячних мас, АО та днів, k = 0.01720209895 (стала Гаусса)
    const double k = 0.01720209895;
    meanMotion = k / std::sqrt(orbParams.a * orbParams.a * orbParams.a);
}

double OrbitalBody::SolveKepler(double M, double e) const {
    double E = M;
    // Метод Ньютона-Рафсона для рівняння Кеплера: E - e*sin(E) = M
    for (int iter = 0; iter < 15; iter++) {
        double dE = (E - e * std::sin(E) - M) / (1.0 - e * std::cos(E));
        E -= dE;
        if (std::abs(dE) < 1e-7) break;
    }
    return E;
}

glm::dvec3 OrbitalBody::CalculatePosition(double currentJD) const {
    // 1. Рахуємо середню аномалію для поточного часу
    double dt = currentJD - orbParams.epoch;
    double M = orbParams.M0 * (PhysicsConsts::PI / 180.0) + meanMotion * dt;

    // Приводимо M до діапазону [0, 2*PI]
    M = std::fmod(M, 2.0 * PhysicsConsts::PI);
    if (M < 0.0) M += 2.0 * PhysicsConsts::PI;

    // 2. Вирішуємо рівняння Кеплера для ексцентричної аномалії E
    double E = SolveKepler(M, orbParams.e);

    // 3. Координати в орбітальній площині (вісь X направлена в перицентр)
    double x_orb = orbParams.a * (std::cos(E) - orbParams.e);
    double y_orb = orbParams.a * std::sqrt(1.0 - orbParams.e * orbParams.e) * std::sin(E);

    // 4. Перехід до тривимірного геліоцентричного простору (екліптики)
    double i_rad = orbParams.i * (PhysicsConsts::PI / 180.0);
    double Omega_rad = orbParams.Omega * (PhysicsConsts::PI / 180.0);
    double w_rad = orbParams.w * (PhysicsConsts::PI / 180.0);

    double cw = std::cos(w_rad);
    double sw = std::sin(w_rad);
    double cO = std::cos(Omega_rad);
    double sO = std::sin(Omega_rad);
    double ci = std::cos(i_rad);
    double si = std::sin(i_rad);

    // Матриця повороту: P (вектор на перицентр) та Q (вектор справжньої аномалії 90 град)
    glm::dvec3 P(
        cw * cO - sw * sO * ci,
        cw * sO + sw * cO * ci,
        sw * si
    );

    glm::dvec3 Q(
        -sw * cO - cw * sO * ci,
        -sw * sO + cw * cO * ci,
        cw * si
    );

    return P * x_orb + Q * y_orb;
}