#include "Photometry.h"
#include "Core/PhysicsConsts.h"
#include <cmath>

namespace Photometry {
    double CalculateMagnitudeFromVisibleArea(double equivalentArea, double rh_AU, double delta_AU, double albedo) {
        if (equivalentArea <= 1e-12 || rh_AU <= 1e-6 || delta_AU <= 1e-6) return 99.0;

        double F_rel = (albedo / PhysicsConsts::PI) * equivalentArea;

        double AU_m = PhysicsConsts::AU_METERS;
        double delta_m = delta_AU * AU_m;

        double F_ratio = F_rel / (delta_m * delta_m * rh_AU * rh_AU);

        double V_sun = -26.74;
        return V_sun - 2.5 * std::log10(F_ratio);
    }
}