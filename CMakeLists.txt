cmake_minimum_required(VERSION 3.18)

project(CometSimulator LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

find_package(glfw3 CONFIG REQUIRED)
find_package(GLEW REQUIRED)
find_package(glm CONFIG REQUIRED)

file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu" 
    "src/*.h"
)

source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${SOURCES})

add_executable(CometSimulator ${SOURCES})

target_include_directories(CometSimulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(CometSimulator PRIVATE
    CUDA::cudart   
    OpenGL::GL     
    glfw       
    GLEW::GLEW   
    glm::glm 
)