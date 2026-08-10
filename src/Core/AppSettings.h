#pragma once
#include <string>

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
        double timeScale = 300.0;
        double rotationPeriodHours = 12.0;
        double rh_AU = 1.3;
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
};