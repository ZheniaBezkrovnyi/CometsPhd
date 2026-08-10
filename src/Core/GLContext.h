#pragma once
#include <GL/glew.h>
#include <GLFW/glfw3.h>
#include <string>

struct AppSettings;

class GLContext {
public:
    GLContext() = default;
    ~GLContext();

    bool Init(const AppSettings& config);
    void PollEvents();
    void SwapBuffers();
    bool ShouldClose() const;

    GLFWwindow* GetWindow() const { return window; }
    int GetWidth() const;
    int GetHeight() const;

private:
    GLFWwindow* window = nullptr;
};