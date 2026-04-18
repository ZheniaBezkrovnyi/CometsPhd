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

set(OPTIX_INCLUDE_DIR "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0/include")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx
            -I${OPTIX_INCLUDE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu
            -o ${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixParams.h
    COMMENT "Compiling OptiX PTX Shaders..."
)


target_sources(CometSimulator PRIVATE ${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx)

target_include_directories(CometSimulator PRIVATE ${OPTIX_INCLUDE_DIR})

target_include_directories(CometSimulator PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)

target_link_libraries(CometSimulator PRIVATE
    CUDA::cudart   
    OpenGL::GL     
    glfw       
    GLEW::GLEW   
    glm::glm 
)