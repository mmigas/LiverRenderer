include_directories(
  ${ASMJIT_INCLUDE_DIRS}
)

# Include FetchContent module
include(FetchContent)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/findCmake")
# System libraries like OpenGL should still use find_package
find_package(OpenGL REQUIRED)
find_package(CUDA REQUIRED)
find_package(GLEW REQUIRED)
find_package(glfw REQUIRED)
add_executable(mitsuba-bin mitsuba.cpp realtime.cpp realtime.hpp NIS_Config.h)

target_link_libraries(mitsuba-bin PRIVATE mitsuba       
        glfw                  # GLFW library target
        OpenGL::GL            # OpenGL library target
        ${CUDA_LIBRARIES}
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cuda.lib  # Windows path to driver lib
        ${CUDA_TOOLKIT_ROOT_DIR}/lib/x64/cudart.lib  # Runtime lib
        glew
)

target_include_directories(mitsuba-bin PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}
        ${CMAKE_CURRENT_BINARY_DIR}
        ${GLFW_INCLUDE_DIRS}  # Include directories for GLFW
        ${CUDA_INCLUDE_DIRS}  # Include directories for CUDA
)


if (CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64")
  target_link_libraries(mitsuba-bin PRIVATE asmjit)
endif()

if (UNIX AND NOT APPLE)
  target_link_libraries(mitsuba-bin PRIVATE dl)
endif()

set_target_properties(mitsuba-bin PROPERTIES OUTPUT_NAME mitsuba)
