#include "Stages/PhotometryStage.h"
#include "Physics/Photometry.h"
#include <iomanip>
#include <iostream>
#include <stdexcept>

void PhotometryStage::Begin(const std::filesystem::path& runDir) {
    log.open(runDir / "photometry_log.csv");
    if (!log.is_open()) {
        throw std::runtime_error("Cannot open " + (runDir / "photometry_log.csv").string());
    }
    log << "JD,PhaseAngle_deg,Distance_AU,ApparentMagnitude" << std::endl;
}

void PhotometryStage::Step(const StepContext& ctx) {
    const int every = config.photometry.everyNFrames;
    if (every > 1 && ctx.frame % every != 0) return;

    OptixParams params = {};
    params.vertices = ctx.vertices;
    params.numVertices = ctx.vertexCount;

    glm::vec3 sunDir = ctx.scene.GetSunLocalDir();
    params.sunDir = make_float3(sunDir.x, sunDir.y, sunDir.z);

    glm::vec3 earthDir = ctx.scene.GetEarthLocalDir();
    params.earthDir = make_float3(earthDir.x, earthDir.y, earthDir.z);

    params.rayEpsilon = config.thermal.rayEpsilon;

    float visibleArea = ctx.optix.RenderPhotometry(params, ctx.vertexCount / 3) * 1000000.0;
    double distanceAU = ctx.scene.GetCometGeocentricDist();
    double currentMag = Photometry::CalculateMagnitudeFromVisibleArea(
        visibleArea,
        ctx.scene.GetCometHeliocentricDist(),
        distanceAU,
        config.thermal.albedo
    );

    double jd = ctx.time.GetCurrentJD();
    double phaseAngle = ctx.scene.GetPhaseAngleDeg();
    double realTimeHours = ctx.time.GetElapsedSeconds() / 3600.0;

    std::cout << "[Metrics] JD: " << std::fixed << std::setprecision(2) << jd
        << " | Phase Angle: " << phaseAngle << " deg"
        << " | Dist: " << distanceAU << " AU"
        << " | Mag: " << currentMag
        << " | ElapsedHours: " << realTimeHours << "\n";

    log << std::fixed << std::setprecision(6)
        << jd << "," << phaseAngle << "," << distanceAU << "," << currentMag << std::endl;
}

void PhotometryStage::End() {
    if (log.is_open()) log.close();
}