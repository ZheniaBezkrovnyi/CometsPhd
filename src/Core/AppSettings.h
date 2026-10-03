#pragma once
#include <array>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>
#include "External/json.hpp"

using json = nlohmann::json;

struct WindowSettings {
    int width = 1100;
    int height = 1100;
    const char* title = "Comet CUDA Interop";
};

struct CameraSettings {
    float fov = 60.0f;
    float heightMultiplier = 0.5f;
    float distanceMultiplier = 1.8f;
    float farPlaneMultiplier = 10.0f;
    std::array<float, 4> clearColor = { 0.0f, 0.0f, 0.0f, 1.0f };
};

struct OrbitSettings {
    double a = 0.0;
    double e = 0.0;
    double i = 0.0;
    double Omega = 0.0;
    double w = 0.0;
    double M0 = 0.0;
    double epoch = 0.0;
};

struct PhysicsSettings {
    double timeScale = 86400.0;
    double rotationPeriodHours = 12.4;
    double durationRotations = 3.0;
    bool freezeOrbits = true;
    double poleRA = 0.0;
    double poleDEC = 90.0;
    double startJulianDate = 2457248.5;
    OrbitSettings cometOrbit = { 3.463, 0.641, 7.04, 50.14, 12.78, 0.0, 2457248.5 };
    OrbitSettings earthOrbit = { 1.00000011, 0.01671022, 0.00005, -11.26064, 102.94719, 100.46435, 2451545.0 };
};

struct ThermalSettings {
    bool enabled = true;
    float solarConstant = 1361.0f;
    float albedo = 0.04f;
    float emissivity = 0.95f;
    float activeFraction = 1.0f;
    float minTemp = 40.0f;
    float maxTempForColor = 230.0f;

    int indirectSamples = 256;
    unsigned int indirectSeed = 1337u;
    float indirectSolarScale = 0.15f;
    float indirectIRScale = 0.05f;
    float maxIndirectFractionOfSolarFlux = 0.10f;
    float rayEpsilon = 0.01f;
};

struct PhotometrySettings {
    bool enabled = true;
    int everyNFrames = 1;
    std::vector<std::string> plots = { "checks/photometry/plot_lightcurve.py" };
    double absoluteMagnitudeH = 0.0;
    double phaseCoefficientBeta = 0.0;
};

struct DiagnosticsSettings {
    bool startupLogs = false;
    bool optixLogs = false;
    bool temperatureDebug = false;
    int temperatureDebugIntervalFrames = 300;
    bool cudaErrorChecks = false;
    bool syncAfterKernels = false;
    bool timing = false;
};

struct ScreenshotSettings {
    bool enabled = true;
    std::string outputDir = "capture_frames";
    int maxFrames = 50;
    int frameStride = 1;
};


struct PostprocessSettings {
    std::string python = "python";
    std::vector<std::string> scripts;  
};

struct AppSettings {
    std::string modelPath = "C:/Users/Yevhen/Projects/Univ/CometsPhd/data/Churyumov-Geras_SPC 2017 - 199k.ply";
    std::string ptxPath = "OptixKernels.ptx";

    WindowSettings window;
    CameraSettings camera;
    PhysicsSettings physics;
    ThermalSettings thermal;
    PhotometrySettings photometry;
    DiagnosticsSettings diagnostics;
    ScreenshotSettings screenshotCapture;
    PostprocessSettings postprocess;

    void LoadFromJson(const std::string& filepath);
    std::string ToJson() const;
};












NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(WindowSettings, width, height)
NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(CameraSettings, fov, heightMultiplier, distanceMultiplier,
    farPlaneMultiplier, clearColor)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(OrbitSettings, a, e, i, Omega, w, M0, epoch)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PhysicsSettings, timeScale, rotationPeriodHours, durationRotations,
        freezeOrbits, poleRA, poleDEC, startJulianDate, cometOrbit, earthOrbit)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ThermalSettings, enabled, solarConstant, albedo, emissivity, activeFraction,
        minTemp, maxTempForColor, indirectSamples, indirectSeed, indirectSolarScale, indirectIRScale,
        maxIndirectFractionOfSolarFlux, rayEpsilon)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PhotometrySettings, enabled, everyNFrames, plots,
        absoluteMagnitudeH, phaseCoefficientBeta)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(DiagnosticsSettings, startupLogs, optixLogs, temperatureDebug,
        temperatureDebugIntervalFrames, cudaErrorChecks, syncAfterKernels, timing)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(ScreenshotSettings, enabled, outputDir, maxFrames, frameStride)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(PostprocessSettings, python, scripts)
    NLOHMANN_DEFINE_TYPE_NON_INTRUSIVE(AppSettings, modelPath, ptxPath, window, camera, physics, thermal,
        photometry, diagnostics, screenshotCapture, postprocess)


    inline void WarnUnknownKeys(const json& user, const json& known, const std::string& path) {
    for (auto it = user.begin(); it != user.end(); ++it) {
        const std::string keyPath = path.empty() ? it.key() : path + "." + it.key();
        const auto match = known.find(it.key());
        if (match == known.end()) {
            std::cerr << "[WARNING] Unknown config key ignored: " << keyPath << "\n";
        }
        else if (it->is_object() && match->is_object()) {
            WarnUnknownKeys(*it, *match, keyPath);
        }
    }
}


inline json ReadConfigJson(const std::filesystem::path& path) {
    std::ifstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open config: " + path.string());
    }

    json j = json::parse(file, nullptr, true, true);  
    if (!j.contains("base")) {
        return j;
    }
    json merged = ReadConfigJson(std::filesystem::path(COMET_SOURCE_DIR) / j["base"].get<std::string>());
    j.erase("base");
    merged.merge_patch(j);
    return merged;
}

inline void AppSettings::LoadFromJson(const std::string& filepath) {
    const json user = ReadConfigJson(filepath);
    json merged = json(*this);
    WarnUnknownKeys(user, merged, "");

    merged.merge_patch(user);
    *this = merged.get<AppSettings>();

    std::cout << "[INFO] Loaded settings from " << filepath << "\n";
}

inline std::string AppSettings::ToJson() const {
    return json(*this).dump(2);
}