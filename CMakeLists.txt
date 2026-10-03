cmake_minimum_required(VERSION 3.18)
project(CometSimulator LANGUAGES CXX CUDA)

set(CMAKE_CUDA_ARCHITECTURES "75;86;89")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(CUDAToolkit REQUIRED)
find_package(OpenGL REQUIRED)

file(GLOB_RECURSE SOURCES CONFIGURE_DEPENDS
    "src/*.cpp" 
    "src/*.cu" 
    "src/*.h"
    "src/*.hpp"
)

set_source_files_properties("${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu" PROPERTIES HEADER_FILE_ONLY TRUE)
source_group(TREE ${CMAKE_CURRENT_SOURCE_DIR} PREFIX "" FILES ${SOURCES})

add_executable(CometSimulator ${SOURCES})

set(COMET_LIBS_DIR "C:/Libraries/C++" CACHE PATH "Folder with glm, glfw and glew")

target_include_directories(CometSimulator PRIVATE
    "${COMET_LIBS_DIR}/glm"
    "${COMET_LIBS_DIR}/glfw/include"
    "${COMET_LIBS_DIR}/glew/include"
)

target_link_directories(CometSimulator PRIVATE
    "${COMET_LIBS_DIR}/glfw/lib-vc2026"
    "${COMET_LIBS_DIR}/glew/lib/Release/x64"
)

target_link_libraries(CometSimulator PRIVATE
    glfw3   
    glew32s 
    OpenGL::GL
    CUDA::cudart 
)
target_compile_definitions(CometSimulator PRIVATE GLEW_STATIC COMET_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}")

set(OPTIX_ROOT "C:/ProgramData/NVIDIA Corporation/OptiX SDK 9.1.0" CACHE PATH "OptiX SDK root")
set(OPTIX_INCLUDE_DIR "${OPTIX_ROOT}/include")
if(NOT EXISTS "${OPTIX_INCLUDE_DIR}/optix.h")
    message(FATAL_ERROR "optix.h not found in ${OPTIX_INCLUDE_DIR}. Set OPTIX_ROOT.")
endif()

file(GLOB_RECURSE PTX_HEADERS CONFIGURE_DEPENDS "src/*.h" "src/*.hpp" "src/*.cuh")

add_custom_command(
    OUTPUT ${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx
    COMMAND ${CMAKE_CUDA_COMPILER} -ptx
            -I${OPTIX_INCLUDE_DIR}
            -I${CMAKE_CURRENT_SOURCE_DIR}/src
            ${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu
            -o ${CMAKE_CURRENT_BINARY_DIR}/OptixKernels.ptx
    DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/Shaders/OptixKernels.cu
            ${PTX_HEADERS}
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

