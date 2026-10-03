#include "Core/App.h"
#include <algorithm>
#include <iostream>
#include <filesystem>
#include <ctime>
#include <iomanip>
#include <sstream>
#include <stdexcept>
#include <vector>
#include <cuda_runtime.h>

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

namespace fs = std::filesystem;

static const fs::path kSourceDir = COMET_SOURCE_DIR;

static std::string Timestamp() {
    const std::time_t now = std::time(nullptr);
    std::tm local{};
#ifdef _WIN32
    localtime_s(&local, &now);
#else
    localtime_r(&now, &local);
#endif
    std::ostringstream ss;
    ss << std::put_time(&local, "%Y-%m-%d_%H%M%S");
    return ss.str();
}

static fs::path DefaultConfig() {
    std::vector<fs::path> configs;
    for (const auto& entry : fs::directory_iterator(kSourceDir / "Configs")) {
        if (entry.is_regular_file() && entry.path().extension() == ".json") {
            configs.push_back(entry.path());
        }
    }
    if (configs.empty()) {
        throw std::runtime_error("No .json file in " + (kSourceDir / "Configs").string());
    }
    std::sort(configs.begin(), configs.end());
    return configs.front();
}

int main(int argc, char** argv) {
    try {
        const fs::path configPath = argc > 1 ? fs::path(argv[1]) : DefaultConfig();
        const fs::path outputDir = argc > 2
            ? fs::path(argv[2])
            : kSourceDir / "runs" / (Timestamp() + "_" + configPath.stem().string());
        const fs::path exeDir = fs::absolute(argv[0]).parent_path();

        std::cout << "[INFO] Config: " << configPath.string() << "\n";
        std::cout << "[INFO] Output: " << outputDir.string() << "\n";

        cudaSetDevice(0);
        cudaFree(0);

        App app(configPath, outputDir, exeDir);
        if (!app.Init()) {
            std::cerr << "Initialization failed!" << std::endl;
            return -1;
        }
        app.Run();
    }
    catch (const std::exception& e) {
        std::cerr << "Fatal error: " << e.what() << std::endl;
        return -1;
    }
    return 0;
}