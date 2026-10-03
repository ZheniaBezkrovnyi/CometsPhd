#pragma once
#include <filesystem>


namespace Timing {
    void Enable(bool on);
    void Begin(const char* name);
    void End();
    void Report(const std::filesystem::path& runDir);
}