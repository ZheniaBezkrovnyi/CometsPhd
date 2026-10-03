#pragma once
#include <filesystem>
#include <string>
#include <vector>
#include "Core/Timer.h"
#include "Geometry/Geometry.h"
#include "Physics/SpaceScene.h"
#include "Rendering/OptixRenderer.h"


struct StepContext {
    const SimulationTime& time;
    const SpaceScene& scene;
    OptixRenderer& optix;
    InteropVertex* vertices;   
    int vertexCount;
    unsigned int frame;
};

class IStage {
public:
    virtual ~IStage() = default;

    virtual void Begin(const std::filesystem::path& /*runDir*/) {}
    virtual void Step(const StepContext& ctx) = 0;
    virtual void End() {}

    virtual std::vector<std::string> Plots() const { return {}; }
};