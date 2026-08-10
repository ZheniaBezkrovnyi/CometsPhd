#include "Core/AppSettings.h"
#include "GLContext.h"
#include <iostream>

GLContext::~GLContext() {
    if (window) {
        glfwDestroyWindow(window);
        glfwTerminate();
    }
}

bool GLContext::Init(const AppSettings& config) {
    if (!glfwInit()) return false;

    window = glfwCreateWindow(config.window.width, config.window.height, config.window.title, NULL, NULL);
    if (!window) {
        glfwTerminate();
        return false;
    }

    glfwMakeContextCurrent(window);
    glfwSwapInterval(0);

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) return false;

    glEnable(GL_DEPTH_TEST);

    return true;
}

void GLContext::PollEvents() {
    glfwPollEvents();
}

void GLContext::SwapBuffers() {
    glfwSwapBuffers(window);
}

bool GLContext::ShouldClose() const {
    return glfwWindowShouldClose(window);
}

int GLContext::GetWidth() const {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return w == 0 ? 1 : w;
}

int GLContext::GetHeight() const {
    int w, h;
    glfwGetFramebufferSize(window, &w, &h);
    return h == 0 ? 1 : h;
}