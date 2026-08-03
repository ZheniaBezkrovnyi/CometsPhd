cmake_minimum_required(VERSION 3.18)
project(CometSimulator LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

file(GLOB_RECURSE SOURCES 
    "src/*.cpp" 
    "src/*.cu" 
    "src/*.h"
    "src/*.hpp"
)

set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu" PROPERTIES HEADER_FILE_ONLY TRUE)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${SOURCES})

add_executable(CometSimulator ${SOURCES})

target_include_directories(CometSimulator PRIVATE
    "C:/Libraries/C++/glm"
    "C:/Libraries/C++/glfw/include"
    "C:/Libraries/C++/glew/include"
)

target_link_directories(CometSimulator PRIVATE
    "C:/Libraries/C++/glfw/lib-vc2026" 
    "C:/Libraries/C++/glew/lib/Release/x64"
)

target_link_libraries(CometSimulator PRIVATE
    glfw3   
    glew32s 
    OpenGL::GL
    CUDA::cudart 
)
target_compile_definitions(CometSimulator PRIVATE GLEW_STATIC)

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

add_custom_command(TARGET CometSimulator POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy
    "${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx"
    "$<TARGET_FILE_DIR:CometSimulator>/OptixKernels.ptx"
    COMMENT "Copying PTX file to executable directory..."
)