# FindCmake/FindAssimp.cmake

# - Find Assimp
# Find the native assimp includes and library
#
#  assimp_INCLUDE_DIRS - Where to find assimp include directories
#  assimp_LIBRARIES     - List of assimp libraries (full paths)
#  assimp_FOUND        - True if assimp is found.

find_path(assimp_INCLUDE_DIR
        NAMES assimp/Importer.hpp
        PATHS ${CMAKE_SOURCE_DIR}/libs/assimp/include
        PATH_SUFFIXES assimp
)

find_library(assimp_LIBRARY
        NAMES assimp-vc143-mt
        PATHS ${CMAKE_SOURCE_DIR}/libs/assimp/lib/Release
)

get_filename_component(assimp_DLL_DIR ${assimp_LIBRARY} DIRECTORY)
find_file(assimp_DLL
        NAMES assimp-vc143-mt.dll
        PATHS ${CMAKE_SOURCE_DIR}/libs/assimp/bin/Release
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(assimp DEFAULT_MSG assimp_INCLUDE_DIR assimp_LIBRARY assimp_DLL)

mark_as_advanced(assimp_INCLUDE_DIR assimp_LIBRARY assimp_DLL)

if (assimp_FOUND)
    # Set the Assimp found flag
    set(assimp_FOUND TRUE)

    # Find OpenGL as it is required by Assimp
    find_package(OpenGL REQUIRED)

    # Create an interface target for Assimp
    add_library(assimp STATIC IMPORTED)
    set_target_properties(assimp PROPERTIES
            IMPORTED_LOCATION ${assimp_LIBRARY}
            INTERFACE_INCLUDE_DIRECTORIES ${assimp_INCLUDE_DIR}
    )

    # Assuming Assimp requires zlib and other dependencies; you might need to locate them
    target_link_libraries(assimp INTERFACE OpenGL::GL)  # Link Zlib and OpenGL with Assimp

else ()
    set(assimp_FOUND FALSE)
endif ()
