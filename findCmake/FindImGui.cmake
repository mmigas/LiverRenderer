# findCmake/FindImGui.cmake

# - Find GLM
# Find the native imgui includes and library
#
#  IMGUI_INCLUDE_DIRS - Where to find GLM include directory
#  IMGUI_FOUND       - True if GLM found.

find_path(IMGUI_INCLUDE_DIR
        NAMES imgui.h
        PATHS ${CMAKE_SOURCE_DIR}/libs/imgui/include
        PATH_SUFFIXES imgui
)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ImGui DEFAULT_MSG IMGUI_INCLUDE_DIR)
# Check if the include directory was found
if (IMGUI_INCLUDE_DIR)
    find_package(OpenGL REQUIRED)
    find_package(glew REQUIRED)
    find_package(glfw REQUIRED)

    set(IMGUI_SOURCES
            ${IMGUI_INCLUDE_DIR}/imgui.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_demo.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_draw.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_widgets.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_impl_glfw.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_impl_opengl3.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_tables.cpp
            ${IMGUI_INCLUDE_DIR}/imgui_stdlib.cpp
    )

    # Create an interface target for ImGui
    add_library(imgui STATIC ${IMGUI_SOURCES})

    target_link_libraries(imgui PUBLIC OpenGL::GL glew glfw)

    # Set the include directories for the target
    target_include_directories(imgui INTERFACE ${IMGUI_INCLUDE_DIR})

    # Optionally export the variables for parent scope use
    set(IMGUI_INCLUDE_DIRS ${IMGUI_INCLUDE_DIR} PARENT_SCOPE)
    set(IMGUI_FOUND TRUE PARENT_SCOPE)
else ()
    set(IMGUI_FOUND FALSE)
    message(FATAL_ERROR "Could not find imgui")
endif ()