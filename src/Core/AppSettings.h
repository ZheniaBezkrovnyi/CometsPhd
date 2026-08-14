#pragma once
#include <string>
#include <fstream>
#include <iostream>
#include "External/json.hpp" 

using json = nlohmann::json;

struct AppSettings {
    std::string modelPath = "C:/Users/Yevhen/Projects/Univ/CometsPhd/data/Churyumov-Geras_SPC 2017 - 199k.ply";
    std::string ptxPath = "OptixKernels.ptx";

    struct {
        int width = 1100;
        int height = 1100;
        const char* title = "Comet CUDA Interop";
    } window;

    struct {
        float fov = 60.0f;
        float heightMultiplier = 0.5f;
        float distanceMultiplier = 1.8f;
        float farPlaneMultiplier = 10.0f;
        float clearColor[4] = { 0.0f, 0.0f, 0.0f, 1.0f };
    } camera;

    struct {
        double timeScale = 86400.0;
        double rotationPeriodHours = 12.4;
        double startJulianDate = 2457248.5;

        struct {
            double a = 3.463;
            double e = 0.641;
            double i = 7.04;
            double Omega = 50.14;
            double w = 12.78;
            double M0 = 0.0;
            double epoch = 2457248.5;
        } cometOrbit;

        struct {
            double a = 1.00000011;
            double e = 0.01671022;
            double i = 0.00005;
            double Omega = -11.26064;
            double w = 102.94719;
            double M0 = 100.46435;
            double epoch = 2451545.0;
        } earthOrbit;

    } physics;

    struct {
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
    } thermal;

    struct {
        bool startupLogs = false;
        bool optixLogs = false;
        bool temperatureDebug = false;
        int temperatureDebugIntervalFrames = 300;
        bool cudaErrorChecks = false;
        bool syncAfterKernels = false;
    } diagnostics;

    struct {
        bool enabled = true;
        std::string outputDir = "capture_frames";
        int maxFrames = 50;
        int frameStride = 1;
    } screenshotCapture;

    void LoadFromJson(const std::string& filepath) {
        std::ifstream file(filepath);
        if (!file.is_open()) {
            std::cerr << "[WARNING] Config file " << filepath << " not found. Using defaults.\n";
            return;
        }

        json j;
        file >> j;

        if (j.contains("modelPath")) modelPath = j["modelPath"];
        if (j.contains("ptxPath")) ptxPath = j["ptxPath"];

        if (j.contains("window")) {
            auto& win = j["window"];
            if (win.contains("width")) window.width = win["width"];
            if (win.contains("height")) window.height = win["height"];
        }

        if (j.contains("camera")) {
            auto& cam = j["camera"];
            if (cam.contains("fov")) camera.fov = cam["fov"];
            if (cam.contains("heightMultiplier")) camera.heightMultiplier = cam["heightMultiplier"];
            if (cam.contains("distanceMultiplier")) camera.distanceMultiplier = cam["distanceMultiplier"];
            if (cam.contains("farPlaneMultiplier")) camera.farPlaneMultiplier = cam["farPlaneMultiplier"];
            if (cam.contains("clearColor")) {
                camera.clearColor[0] = cam["clearColor"][0];
                camera.clearColor[1] = cam["clearColor"][1];
                camera.clearColor[2] = cam["clearColor"][2];
                camera.clearColor[3] = cam["clearColor"][3];
            }
        }

        if (j.contains("physics")) {
            auto& phys = j["physics"];
            if (phys.contains("timeScale")) physics.timeScale = phys["timeScale"];
            if (phys.contains("rotationPeriodHours")) physics.rotationPeriodHours = phys["rotationPeriodHours"];
            if (phys.contains("startJulianDate")) physics.startJulianDate = phys["startJulianDate"];

            if (phys.contains("cometOrbit")) {
                auto& orb = phys["cometOrbit"];
                if (orb.contains("a")) physics.cometOrbit.a = orb["a"];
                if (orb.contains("e")) physics.cometOrbit.e = orb["e"];
                if (orb.contains("i")) physics.cometOrbit.i = orb["i"];
                if (orb.contains("Omega")) physics.cometOrbit.Omega = orb["Omega"];
                if (orb.contains("w")) physics.cometOrbit.w = orb["w"];
                if (orb.contains("M0")) physics.cometOrbit.M0 = orb["M0"];
                if (orb.contains("epoch")) physics.cometOrbit.epoch = orb["epoch"];
            }

            if (phys.contains("earthOrbit")) {
                auto& orb = phys["earthOrbit"];
                if (orb.contains("a")) physics.earthOrbit.a = orb["a"];
                if (orb.contains("e")) physics.earthOrbit.e = orb["e"];
                if (orb.contains("i")) physics.earthOrbit.i = orb["i"];
                if (orb.contains("Omega")) physics.earthOrbit.Omega = orb["Omega"];
                if (orb.contains("w")) physics.earthOrbit.w = orb["w"];
                if (orb.contains("M0")) physics.earthOrbit.M0 = orb["M0"];
                if (orb.contains("epoch")) physics.earthOrbit.epoch = orb["epoch"];
            }
        }

        if (j.contains("thermal")) {
            auto& th = j["thermal"];
            if (th.contains("solarConstant")) thermal.solarConstant = th["solarConstant"];
            if (th.contains("albedo")) thermal.albedo = th["albedo"];
            if (th.contains("emissivity")) thermal.emissivity = th["emissivity"];
            if (th.contains("activeFraction")) thermal.activeFraction = th["activeFraction"];
            if (th.contains("minTemp")) thermal.minTemp = th["minTemp"];
            if (th.contains("maxTempForColor")) thermal.maxTempForColor = th["maxTempForColor"];
            if (th.contains("indirectSamples")) thermal.indirectSamples = th["indirectSamples"];
            if (th.contains("indirectSeed")) thermal.indirectSeed = th["indirectSeed"];
            if (th.contains("indirectSolarScale")) thermal.indirectSolarScale = th["indirectSolarScale"];
            if (th.contains("indirectIRScale")) thermal.indirectIRScale = th["indirectIRScale"];
            if (th.contains("maxIndirectFractionOfSolarFlux")) thermal.maxIndirectFractionOfSolarFlux = th["maxIndirectFractionOfSolarFlux"];
            if (th.contains("rayEpsilon")) thermal.rayEpsilon = th["rayEpsilon"];
        }

        if (j.contains("diagnostics")) {
            auto& diag = j["diagnostics"];
            if (diag.contains("startupLogs")) diagnostics.startupLogs = diag["startupLogs"];
            if (diag.contains("optixLogs")) diagnostics.optixLogs = diag["optixLogs"];
            if (diag.contains("temperatureDebug")) diagnostics.temperatureDebug = diag["temperatureDebug"];
            if (diag.contains("temperatureDebugIntervalFrames")) diagnostics.temperatureDebugIntervalFrames = diag["temperatureDebugIntervalFrames"];
            if (diag.contains("cudaErrorChecks")) diagnostics.cudaErrorChecks = diag["cudaErrorChecks"];
            if (diag.contains("syncAfterKernels")) diagnostics.syncAfterKernels = diag["syncAfterKernels"];
        }

        if (j.contains("screenshotCapture")) {
            auto& sc = j["screenshotCapture"];
            if (sc.contains("enabled")) screenshotCapture.enabled = sc["enabled"];
            if (sc.contains("outputDir")) screenshotCapture.outputDir = sc["outputDir"];
            if (sc.contains("maxFrames")) screenshotCapture.maxFrames = sc["maxFrames"];
            if (sc.contains("frameStride")) screenshotCapture.frameStride = sc["frameStride"];
        }

        std::cout << "[INFO] Loaded settings from " << filepath << "\n";
    }
};