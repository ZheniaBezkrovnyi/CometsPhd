#include "Utils/Timing.h"
#include <cuda_runtime.h>
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

namespace Timing {

    namespace {
        using Clock = std::chrono::steady_clock;

        struct Section {
            std::string name;
            cudaEvent_t start = nullptr;
            cudaEvent_t stop = nullptr;
            bool gpuPending = false;
            Clock::time_point cpuStart{};
            int calls = 0;
            double cpuMs = 0.0;
            double gpuMs = 0.0;
        };

        bool enabled = false;
        std::vector<std::unique_ptr<Section>> sections;
        std::vector<Section*> open;

        Section& Find(const char* name) {
            for (auto& s : sections) {
                if (s->name == name) return *s;
            }
            sections.push_back(std::make_unique<Section>());
            Section& s = *sections.back();
            s.name = name;
            cudaEventCreate(&s.start);
            cudaEventCreate(&s.stop);
            return s;
        }

        void CollectGpu(Section& s) {
            float ms = 0.0f;
            cudaEventSynchronize(s.stop);
            cudaEventElapsedTime(&ms, s.start, s.stop);
            s.gpuMs += ms;
            s.gpuPending = false;
        }
    }

    void Enable(bool on) {
        enabled = on;
    }

    void Begin(const char* name) {
        if (!enabled) return;
        Section& s = Find(name);
        if (s.gpuPending) CollectGpu(s);  
        cudaEventRecord(s.start);
        s.cpuStart = Clock::now();
        open.push_back(&s);
    }

    void End() {
        if (!enabled || open.empty()) return;
        Section& s = *open.back();
        open.pop_back();
        s.cpuMs += std::chrono::duration<double, std::milli>(Clock::now() - s.cpuStart).count();
        cudaEventRecord(s.stop);
        s.gpuPending = true;
        ++s.calls;
    }

    void Report(const std::filesystem::path& runDir) {
        if (!enabled || sections.empty()) return;

        std::ofstream csv(runDir / "timings.csv");
        csv << "section,calls,cpu_ms_per_call,gpu_ms_per_call,cpu_total_s,gpu_total_s\n";
        std::printf("\n[Timing] %-14s %8s %14s %14s %12s\n", "section", "calls", "CPU ms/call", "GPU ms/call", "CPU total s");

        for (auto& s : sections) {
            if (s->gpuPending) CollectGpu(*s);
            const int n = std::max(1, s->calls);
            std::printf("[Timing] %-14s %8d %14.3f %14.3f %12.2f\n",
                s->name.c_str(), s->calls, s->cpuMs / n, s->gpuMs / n, s->cpuMs / 1000.0);
            csv << s->name << "," << s->calls << "," << s->cpuMs / n << "," << s->gpuMs / n << ","
                << s->cpuMs / 1000.0 << "," << s->gpuMs / 1000.0 << "\n";
        }
    }

}