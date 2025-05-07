# findCmake/Findspdlog.cmake

# - Find spdlog
# Find the native spdlog includes and library
#
#  spdlog_INCLUDE_DIRS - Where to find spdlog include directories
#  spdlog_LIBRARIES     - List of spdlog libraries (full paths)
#  spdlog_FOUND        - True if spdlog is found.

find_path(spdlog_INCLUDE_DIR
        NAMES spdlog/spdlog.h
        PATHS ${CMAKE_SOURCE_DIR}/libs/spdlog/include
        PATH_SUFFIXES spdlog
)

find_library(spdlog_LIBRARY
        NAMES spdlog
        PATHS ${CMAKE_SOURCE_DIR}/libs/spdlog/lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(spdlog DEFAULT_MSG spdlog_INCLUDE_DIR spdlog_LIBRARY)

mark_as_advanced(spdlog_INCLUDE_DIR spdlog_LIBRARY)

set(spdlog_FOUND TRUE)

# Create a normal library target for spdlog
add_library(spdlog STATIC IMPORTED)

# Set the include directories and library for the target
set_target_properties(spdlog PROPERTIES
        IMPORTED_LOCATION ${spdlog_LIBRARY}
        INTERFACE_INCLUDE_DIRECTORIES ${spdlog_INCLUDE_DIR})