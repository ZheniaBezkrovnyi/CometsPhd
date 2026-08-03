#include "Utils/ScreenshotCapture.h"

#include <vector>
#include <sstream>
#include <iomanip>
#include <filesystem>
#include <cstring>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "External/stb_image_write.h"

bool EnsureScreenshotDirectory(const std::string& path) {
    std::error_code ec;

    if (std::filesystem::exists(path, ec)) {
        return std::filesystem::is_directory(path, ec);
    }

    return std::filesystem::create_directories(path, ec);
}

static std::string BuildScreenshotPath(const std::string& outputDir, int frameIndex) {
    std::ostringstream ss;
    ss << outputDir << "/frame_"
        << std::setw(5) << std::setfill('0') << frameIndex
        << ".png";

    return ss.str();
}

static bool SaveFramebufferToPNG(const std::string& path, int width, int height) {
    if (width <= 0 || height <= 0) return false;

    std::vector<unsigned char> pixels(width * height * 3);
    std::vector<unsigned char> flipped(width * height * 3);

    glPixelStorei(GL_PACK_ALIGNMENT, 1);
    glReadBuffer(GL_BACK);
    glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels.data());

    if (glGetError() != GL_NO_ERROR) return false;

    const int rowSize = width * 3;

    for (int y = 0; y < height; ++y) {
        std::memcpy(
            flipped.data() + y * rowSize,
            pixels.data() + (height - 1 - y) * rowSize,
            rowSize
        );
    }

    return stbi_write_png(path.c_str(), width, height, 3, flipped.data(), rowSize) != 0;
}

void CaptureScreenshotIfNeeded(
    const std::string& outputDir,
    bool enabled,
    int maxFrames,
    int frameStride,
    ScreenshotCaptureState& state,
    int renderedFrameIndex,
    int width,
    int height
) {
    if (!enabled) return;
    if (state.savedFrames >= maxFrames) return;
    if (frameStride <= 0) return;
    if (renderedFrameIndex % frameStride != 0) return;

    if (!EnsureScreenshotDirectory(outputDir)) return;

    std::string path = BuildScreenshotPath(outputDir, state.savedFrames);

    if (SaveFramebufferToPNG(path, width, height)) {
        state.savedFrames++;
    }
}