#pragma once
#include <fstream>
#include "Stages/IStage.h"
#include "Core/AppSettings.h"


class PhotometryStage : public IStage {
public:
    explicit PhotometryStage(const AppSettings& settings) : config(settings) {}

    void Begin(const std::filesystem::path& runDir) override;
    void Step(const StepContext& ctx) override;
    void End() override;
    std::vector<std::string> Plots() const override { return config.photometry.plots; }

private:
    const AppSettings& config;
    std::ofstream log;
};