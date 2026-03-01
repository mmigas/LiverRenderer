# findCmake/FindGLM.cmake

# - Find GLM
# Find the native GLM includes and library
#
#  GLM_INCLUDE_DIRS - Where to find GLM include directory
#  GLM_FOUND       - True if GLM found.

find_path(GLM_INCLUDE_DIR
        NAMES glm/glm.hpp
        PATHS ${CMAKE_SOURCE_DIR}/libs/glm
        PATH_SUFFIXES glm
)

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(GLM DEFAULT_MSG GLM_INCLUDE_DIR)
# Check if the include directory was found
if (GLM_INCLUDE_DIR)
    # Create an interface target for GLM
    add_library(glm INTERFACE)

    # Set the include directories for the target
    target_include_directories(glm INTERFACE
            ${GLM_INCLUDE_DIR}
    )

    # Optionally export the variables for parent scope use
    set(GLM_INCLUDE_DIRS ${GLM_INCLUDE_DIR} PARENT_SCOPE)
    set(GLM_FOUND TRUE PARENT_SCOPE)
else ()
    set(GLM_FOUND FALSE)
    message(FATAL_ERROR "Could not find GLM")
endif ()