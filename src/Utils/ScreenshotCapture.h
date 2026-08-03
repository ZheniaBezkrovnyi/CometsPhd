#pragma once

#include <GL/glew.h>
#include <string>

struct ScreenshotCaptureState {
    int savedFrames = 0;
};

bool EnsureScreenshotDirectory(const std::string& path);

void CaptureScreenshotIfNeeded(
    const std::string& outputDir,
    bool enabled,
    int maxFrames,
    int frameStride,
    ScreenshotCaptureState& state,
    int renderedFrameIndex,
    int width,
    int height
);