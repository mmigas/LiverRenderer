# findCmake/FindGlew.cmake

# - Find glew
# Find the native glew includes and library
#
#  glew_INCLUDE_DIRS - Where to find glew include directories
#  glew_LIBRARIES     - List of glew libraries (full paths)
#  glew_FOUND        - True if glew is found.

find_path(glew_INCLUDE_DIR
        NAMES GL/glew.h
        PATHS ${CMAKE_SOURCE_DIR}/libs/glew/include
        PATH_SUFFIXES glew
)

find_library(glew_LIBRARY
        NAMES glew32s
        PATHS ${CMAKE_SOURCE_DIR}/libs/glew/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(glew DEFAULT_MSG glew_INCLUDE_DIR glew_LIBRARY)

mark_as_advanced(glew_INCLUDE_DIR glew_LIBRARY)

set(glew_FOUND TRUE)
if (glew_INCLUDE_DIR)
    find_package(OpenGL REQUIRED)
    # Create a normal library target for glew
    add_library(glew STATIC IMPORTED)

    # Set the include directories and library for the target
    set_target_properties(glew PROPERTIES
            IMPORTED_LOCATION ${glew_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${glew_INCLUDE_DIR})

    target_compile_definitions(glew INTERFACE GLEW_STATIC)

    target_link_libraries(glew INTERFACE OpenGL::GL)
endif ()