# findCmake/FindGLFW.cmake

# - Find glfw
# Find the native GLFW includes and library
#
#  GLFW_INCLUDE_DIRS - Where to find GLFW include directories
#  GLFW_LIBRARIES     - List of GLFW libraries (full paths)
#  GLFW_FOUND        - True if GLFW is found.

find_path(GLFW_INCLUDE_DIR
        NAMES GLFW/glfw3.h
        PATHS ${CMAKE_SOURCE_DIR}/libs/GLFW/include
        PATH_SUFFIXES GLFW
)

find_library(GLFW_LIBRARY
        NAMES glfw3dll
        PATHS ${CMAKE_SOURCE_DIR}/libs/GLFW/lib
)

# Assuming the DLL is in the same directory as the library
get_filename_component(GLFW_DLL_DIR ${GLFW_LIBRARY} DIRECTORY)
find_file(GLFW_DLL
        NAMES glfw3.dll
        PATHS ${GLFW_DLL_DIR}
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(glfw DEFAULT_MSG GLFW_INCLUDE_DIR GLFW_LIBRARY GLFW_DLL)


mark_as_advanced(GLFW_INCLUDE_DIR GLFW_LIBRARY GLFW_DLL)

set(GLFW_FOUND TRUE)

# Create a normal library target for GLFW
add_library(glfw STATIC IMPORTED)

# Set the include directories and library for the target
set_target_properties(glfw PROPERTIES
        IMPORTED_LOCATION ${GLFW_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${GLFW_INCLUDE_DIR})