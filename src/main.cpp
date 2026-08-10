#include "Core/App.h"
#include <iostream>
#include <cuda_runtime.h>

extern "C" {
    __declspec(dllexport) unsigned long NvOptimusEnablement = 0x00000001;
    __declspec(dllexport) int AmdPowerXpressRequestHighPerformance = 1;
}

int main() {
    try {
        cudaSetDevice(0);
        cudaFree(0);

        App app;
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