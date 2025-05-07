# findCmake/FindImGuizmo.cmake

# - Find imguizmo
# Find the native imguizmo includes and library
#
#  IMGUIZMO_INCLUDE_DIRS - Where to find GLM include directory
#  IMGUIZMO_FOUND       - True if GLM found.

find_path(IMGUIZMO_INCLUDE_DIR
        NAMES ImGuizmo.h
        PATHS ${CMAKE_SOURCE_DIR}/libs/imguizmo/include
        PATH_SUFFIXES imguizmo
)
include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(ImGuizmo DEFAULT_MSG IMGUIZMO_INCLUDE_DIR)
# Check if the include directory was found
if (IMGUIZMO_INCLUDE_DIR)

    set(IMGUIZMO_SOURCES
            ${IMGUIZMO_INCLUDE_DIR}/ImGuizmo.h
            ${IMGUIZMO_INCLUDE_DIR}/ImGuizmo.cpp
    )

    # Create an interface target for ImGui
    add_library(imguizmo STATIC ${IMGUIZMO_SOURCES})

    #target_compile_definitions(imguizmo PUBLIC IMGUI_DEFINE_MATH_OPERATORS)

    target_link_libraries(imguizmo PUBLIC imgui)

    # Set the include directories for the target
    target_include_directories(imguizmo INTERFACE ${IMGUIZMO_INCLUDE_DIR})

    # Optionally export the variables for parent scope use
    set(IMGUIZMO_INCLUDE_DIRS ${IMGUIZMO_INCLUDE_DIR} PARENT_SCOPE)
    set(IMGUIZMO_FOUND TRUE PARENT_SCOPE)
else ()
    set(IMGUIZMO_FOUND FALSE)
    message(FATAL_ERROR "Could not find imguizmo")
endif ()