# FindZLIB.cmake

# - Try to find Zlib
# Once done, this will define
#
#  ZLIB_FOUND - system has Zlib
#  ZLIB_INCLUDE_DIR - the Zlib include directory
#  ZLIB_LIBRARIES - The libraries needed to use Zlib
#  ZLIB_VERSION_STRING - The version of Zlib found

#[[find_path(ZLIB_INCLUDE_DIR
    NAMES zlib.h
    PATHS
        ${CMAKE_SOURCE_DIR}/libs/zlib/include
        /usr/local/include
        /usr/include
)]]

find_library(ZLIB_LIBRARY
    NAMES zlib zlibstaticd
    PATHS
        ${CMAKE_SOURCE_DIR}/libs/zlib
        /usr/local/lib
        /usr/lib
)

# Check that both the include directory and the library were found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZLIB DEFAULT_MSG ZLIB_LIBRARY)

# If Zlib was found, set the ZLIB_LIBRARIES and ZLIB_FOUND variables
if (ZLIB_FOUND)
    set(ZLIB_LIBRARIES ${ZLIB_LIBRARY})

    # Optionally, try to get the Zlib version
    if (EXISTS "${ZLIB_INCLUDE_DIR}/zlib.h")
        file(READ "${ZLIB_INCLUDE_DIR}/zlib.h" _zlib_h_content)
        string(REGEX MATCH "#define ZLIB_VERSION \"[^\"]+\"" _zlib_version_def "${_zlib_h_content}")
        string(REGEX REPLACE ".*#define ZLIB_VERSION \"([^\"]+)\".*" "\\1" ZLIB_VERSION_STRING "${_zlib_version_def}")
    endif ()
endif ()

mark_as_advanced(ZLIB_INCLUDE_DIR ZLIB_LIBRARY)

# Create a target for Zlib (modern CMake approach)
if (ZLIB_FOUND)
    add_library(ZLIB::ZLIB INTERFACE IMPORTED)
    set_target_properties(ZLIB::ZLIB PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ZLIB_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${ZLIB_LIBRARIES}"
    )
endif ()
